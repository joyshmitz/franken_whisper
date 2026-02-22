//! Speculative cancel-correct streaming pipeline.
//!
//! Orchestrates [`WindowManager`](crate::speculation::WindowManager),
//! [`CorrectionTracker`](crate::speculation::CorrectionTracker), and
//! [`ConcurrentTwoLaneExecutor`](crate::backend::ConcurrentTwoLaneExecutor)
//! to run fast + quality models in parallel with real-time correction.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use crate::backend::{ConcurrentTwoLaneExecutor, QualitySelector, TranscriptSegment};
use crate::error::FwResult;
use crate::model::{BackendKind, RunEvent, TranscriptionResult, TranscriptionSegment};
use crate::speculation::{
    CorrectionDecision, CorrectionTolerance, CorrectionTracker, PartialTranscript,
    SpeculationStats, WindowManager,
};

// ---------------------------------------------------------------------------
// bd-qlt.6: SpeculativeStreamingPipeline
// ---------------------------------------------------------------------------

/// Configuration for a speculative streaming run.
#[derive(Debug, Clone)]
pub struct SpeculativeConfig {
    pub window_size_ms: u64,
    pub overlap_ms: u64,
    pub fast_model_name: String,
    pub quality_model_name: String,
    pub tolerance: CorrectionTolerance,
    pub adaptive: bool,
    pub emit_events: bool,
}

impl Default for SpeculativeConfig {
    fn default() -> Self {
        Self {
            window_size_ms: 3000,
            overlap_ms: 500,
            fast_model_name: "whisper-tiny".to_owned(),
            quality_model_name: "whisper-large".to_owned(),
            tolerance: CorrectionTolerance::default(),
            adaptive: true,
            emit_events: true,
        }
    }
}

/// Bridge a `TranscriptionSegment` (model) to a `TranscriptSegment` (backend).
fn to_backend_segment(s: &TranscriptionSegment) -> TranscriptSegment {
    TranscriptSegment {
        start_ms: s.start_sec.map(|v| (v * 1000.0) as u64).unwrap_or(0),
        end_ms: s.end_sec.map(|v| (v * 1000.0) as u64).unwrap_or(0),
        text: s.text.clone(),
        confidence: s.confidence.unwrap_or(0.0),
    }
}

/// Bridge a `TranscriptSegment` (backend) to a `TranscriptionSegment` (model).
fn to_model_segment(s: &TranscriptSegment) -> TranscriptionSegment {
    TranscriptionSegment {
        start_sec: Some(s.start_ms as f64 / 1000.0),
        end_sec: Some(s.end_ms as f64 / 1000.0),
        text: s.text.clone(),
        speaker: None,
        confidence: Some(s.confidence),
    }
}

/// The speculative streaming pipeline orchestrator.
///
/// Rather than holding engine references directly, the pipeline accepts
/// closures for fast and quality model inference. This makes it testable
/// with mock functions and avoids object-safety complications with the
/// `Engine` trait.
pub struct SpeculativeStreamingPipeline {
    config: SpeculativeConfig,
    window_manager: WindowManager,
    correction_tracker: CorrectionTracker,
    next_seq: AtomicU64,
    events: Vec<RunEvent>,
    run_id: String,
}

impl SpeculativeStreamingPipeline {
    /// Create a new pipeline with the given configuration.
    #[must_use]
    pub fn new(config: SpeculativeConfig, run_id: String) -> Self {
        let window_manager = WindowManager::new(&run_id, config.window_size_ms, config.overlap_ms);
        let correction_tracker = CorrectionTracker::new(config.tolerance.clone());
        Self {
            config,
            window_manager,
            correction_tracker,
            next_seq: AtomicU64::new(0),
            events: Vec::new(),
            run_id,
        }
    }

    fn next_seq(&self) -> u64 {
        self.next_seq.fetch_add(1, Ordering::Relaxed)
    }

    /// Process a single window using provided model closures.
    ///
    /// Both closures are run in parallel on separate threads via
    /// [`ConcurrentTwoLaneExecutor`]. The fast model result is captured
    /// via the early-emit callback, then compared with the quality result
    /// to produce a [`CorrectionDecision`].
    pub fn process_window<F, Q>(
        &mut self,
        audio_hash: &str,
        audio_position_ms: u64,
        fast_fn: F,
        quality_fn: Q,
    ) -> FwResult<CorrectionDecision>
    where
        F: FnOnce() -> Vec<TranscriptionSegment> + Send + 'static,
        Q: FnOnce() -> Vec<TranscriptionSegment> + Send + 'static,
    {
        let window = self
            .window_manager
            .next_window(audio_position_ms, audio_hash);
        let window_id = window.window_id;
        let seq = self.next_seq();
        let fast_model_name = self.config.fast_model_name.clone();
        let quality_model_name = self.config.quality_model_name.clone();

        let executor = ConcurrentTwoLaneExecutor::new(QualitySelector::SpeculativeCorrect);

        // Bridge closures: model segments -> backend segments
        let fast_bridge = move || -> Vec<TranscriptSegment> {
            fast_fn().iter().map(to_backend_segment).collect()
        };
        let quality_bridge = move || -> Vec<TranscriptSegment> {
            quality_fn().iter().map(to_backend_segment).collect()
        };

        // Capture fast results via early-emit callback
        let fast_holder: Arc<Mutex<Vec<TranscriptionSegment>>> = Arc::new(Mutex::new(Vec::new()));
        let fast_holder_clone = fast_holder.clone();

        let result = executor.execute_with_early_emit(
            fast_bridge,
            quality_bridge,
            move |primary_result: &[TranscriptSegment], _latency_ms: u64| {
                let converted: Vec<TranscriptionSegment> =
                    primary_result.iter().map(to_model_segment).collect();
                *fast_holder_clone.lock().expect("lock poisoned") = converted;
            },
            |_primary, _secondary, _p_lat, _q_lat| {},
        );

        let fast_segments = fast_holder.lock().expect("lock poisoned").clone();

        // Register with tracker and window manager
        let partial = PartialTranscript::new(
            seq,
            window_id,
            fast_model_name,
            fast_segments.clone(),
            result.primary_latency_ms,
            chrono::Utc::now().to_rfc3339(),
        );
        self.correction_tracker.register_partial(partial);

        let fast_partial_for_wm = PartialTranscript::new(
            seq,
            window_id,
            self.config.fast_model_name.clone(),
            fast_segments,
            result.primary_latency_ms,
            chrono::Utc::now().to_rfc3339(),
        );
        self.window_manager
            .record_fast_result(window_id, fast_partial_for_wm);

        // Convert quality results
        let quality_segments: Vec<TranscriptionSegment> = result
            .secondary_result
            .iter()
            .map(to_model_segment)
            .collect();
        self.window_manager
            .record_quality_result(window_id, quality_segments.clone());

        // Submit to correction tracker
        let decision = self.correction_tracker.submit_quality_result(
            window_id,
            &quality_model_name,
            quality_segments,
            result.secondary_latency_ms,
        )?;

        self.window_manager.resolve_window(window_id);
        Ok(decision)
    }

    /// Get merged, deduplicated transcript from all resolved windows.
    #[must_use]
    pub fn merged_transcript(&self) -> Vec<TranscriptionSegment> {
        self.window_manager.merge_segments()
    }

    /// Build a full `TranscriptionResult` from the pipeline state.
    #[must_use]
    pub fn build_result(&self) -> TranscriptionResult {
        let segments = self.merged_transcript();
        let transcript = segments
            .iter()
            .map(|s| s.text.as_str())
            .collect::<Vec<_>>()
            .join(" ");
        TranscriptionResult {
            backend: BackendKind::Auto,
            transcript,
            language: Some("en".to_owned()),
            segments,
            acceleration: None,
            raw_output: serde_json::json!({}),
            artifact_paths: vec![],
        }
    }

    /// Get speculation statistics.
    #[must_use]
    pub fn stats(&self) -> SpeculationStats {
        let tracker_stats = self.correction_tracker.stats();
        SpeculationStats {
            windows_processed: tracker_stats.windows_processed,
            corrections_emitted: tracker_stats.corrections_emitted,
            correction_rate: self.correction_tracker.correction_rate(),
            mean_fast_latency_ms: if tracker_stats.windows_processed > 0 {
                tracker_stats.total_fast_latency_ms as f64 / tracker_stats.windows_processed as f64
            } else {
                0.0
            },
            mean_quality_latency_ms: if tracker_stats.windows_processed > 0 {
                tracker_stats.total_quality_latency_ms as f64
                    / tracker_stats.windows_processed as f64
            } else {
                0.0
            },
            current_window_size_ms: self.window_manager.current_window_size(),
            mean_drift_wer: self.correction_tracker.mean_wer(),
        }
    }

    /// All events.
    #[must_use]
    pub fn events(&self) -> &[RunEvent] {
        &self.events
    }

    /// Reference to the correction tracker.
    #[must_use]
    pub fn correction_tracker(&self) -> &CorrectionTracker {
        &self.correction_tracker
    }

    /// Reference to the window manager.
    #[must_use]
    pub fn window_manager(&self) -> &WindowManager {
        &self.window_manager
    }

    /// Run ID for this pipeline.
    #[must_use]
    pub fn run_id(&self) -> &str {
        &self.run_id
    }
}
