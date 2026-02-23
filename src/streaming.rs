//! Speculative cancel-correct streaming pipeline.
//!
//! Orchestrates [`WindowManager`](crate::speculation::WindowManager),
//! [`CorrectionTracker`](crate::speculation::CorrectionTracker), and
//! [`ConcurrentTwoLaneExecutor`](crate::backend::ConcurrentTwoLaneExecutor)
//! to run fast + quality models in parallel with real-time correction.

use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use crate::audio;
use crate::backend::{ConcurrentTwoLaneExecutor, QualitySelector, TranscriptSegment};
use crate::error::{FwError, FwResult};
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

/// The speculative streaming pipeline orchestrator.
///
/// Rather than holding engine references directly, the pipeline accepts
/// closures for fast and quality model inference. This keeps it testable
/// with deterministic mocks and ready for later engine-object integration.
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

    fn push_event(&mut self, code: &str, message: &str, payload: serde_json::Value) {
        self.events.push(RunEvent {
            seq: self.events.len() as u64,
            ts_rfc3339: chrono::Utc::now().to_rfc3339(),
            stage: "speculation".to_owned(),
            code: code.to_owned(),
            message: message.to_owned(),
            payload,
        });
    }

    fn process_window_by_id<F, Q>(
        &mut self,
        window_id: u64,
        fast_fn: F,
        quality_fn: Q,
    ) -> FwResult<CorrectionDecision>
    where
        F: FnOnce() -> Vec<TranscriptionSegment> + Send + 'static,
        Q: FnOnce() -> Vec<TranscriptionSegment> + Send + 'static,
    {
        let seq = self.next_seq();
        let fast_model_name = self.config.fast_model_name.clone();
        let quality_model_name = self.config.quality_model_name.clone();

        let executor = ConcurrentTwoLaneExecutor::new(QualitySelector::SpeculativeCorrect);

        // Capture original model segments directly in bridge closures to avoid
        // precision-lossy round-trip (model → backend → model) that turns
        // `confidence: None` into `Some(0.0)` and truncates sub-ms timestamps.
        let fast_holder: Arc<Mutex<Vec<TranscriptionSegment>>> = Arc::new(Mutex::new(Vec::new()));
        let quality_holder: Arc<Mutex<Vec<TranscriptionSegment>>> =
            Arc::new(Mutex::new(Vec::new()));
        let fast_holder_bridge = fast_holder.clone();
        let quality_holder_bridge = quality_holder.clone();

        let fast_bridge = move || -> Vec<TranscriptSegment> {
            let original = fast_fn();
            *fast_holder_bridge.lock().expect("lock poisoned") = original.clone();
            original.iter().map(to_backend_segment).collect()
        };
        let quality_bridge = move || -> Vec<TranscriptSegment> {
            let original = quality_fn();
            *quality_holder_bridge.lock().expect("lock poisoned") = original.clone();
            original.iter().map(to_backend_segment).collect()
        };

        let result = executor.execute_with_early_emit(
            fast_bridge,
            quality_bridge,
            |_primary_result, _latency_ms| {},
            |_primary, _secondary, _p_lat, _q_lat| {},
        );

        let fast_segments = fast_holder.lock().expect("lock poisoned").clone();

        // Register with tracker and window manager.
        let fast_ts = chrono::Utc::now().to_rfc3339();
        let partial = PartialTranscript::new(
            seq,
            window_id,
            fast_model_name,
            fast_segments.clone(),
            result.primary_latency_ms,
            fast_ts.clone(),
        );
        self.correction_tracker.register_partial(partial);

        let fast_partial_for_wm = PartialTranscript::new(
            seq,
            window_id,
            self.config.fast_model_name.clone(),
            fast_segments.clone(),
            result.primary_latency_ms,
            fast_ts.clone(),
        );
        self.window_manager
            .record_fast_result(window_id, fast_partial_for_wm);

        if self.config.emit_events {
            for segment in &fast_segments {
                let payload =
                    crate::robot::transcript_partial_value(&self.run_id, seq, &fast_ts, segment);
                self.push_event(
                    "transcript.partial",
                    "fast model emitted speculative partial segment",
                    payload,
                );
            }
        }

        // Use captured original model segments (no round-trip conversion).
        let quality_segments = quality_holder.lock().expect("lock poisoned").clone();
        self.window_manager
            .record_quality_result(window_id, quality_segments.clone());

        // Submit to correction tracker.
        let decision = self.correction_tracker.submit_quality_result(
            window_id,
            &quality_model_name,
            quality_segments,
            result.secondary_latency_ms,
        )?;

        self.window_manager.resolve_window(window_id);

        if self.config.emit_events {
            match &decision {
                CorrectionDecision::Confirm { seq, drift } => {
                    let payload = crate::robot::transcript_confirm_value(
                        &self.run_id,
                        *seq,
                        window_id,
                        drift,
                        result.secondary_latency_ms,
                        &self.config.quality_model_name,
                    );
                    self.push_event(
                        "transcript.confirm",
                        "quality model confirmed speculative transcript",
                        payload,
                    );
                }
                CorrectionDecision::Correct { correction } => {
                    let retract_payload = crate::robot::transcript_retract_value(
                        &self.run_id,
                        correction.retracted_seq,
                        correction.window_id,
                        "quality_correction",
                        &correction.quality_model_id,
                    );
                    self.push_event(
                        "transcript.retract",
                        "quality model retracted speculative transcript",
                        retract_payload,
                    );

                    let correct_payload =
                        crate::robot::transcript_correct_value(&self.run_id, correction);
                    self.push_event(
                        "transcript.correct",
                        "quality model emitted correction transcript",
                        correct_payload,
                    );
                }
            }
        }

        Ok(decision)
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
        self.process_window_by_id(window.window_id, fast_fn, quality_fn)
    }

    fn process_duration_loop<C, M>(
        &mut self,
        total_duration_ms: u64,
        audio_hash_seed: &str,
        checkpoint: &mut C,
        model_runner: &mut M,
    ) -> FwResult<TranscriptionResult>
    where
        C: FnMut() -> FwResult<()>,
        M: FnMut(u64, u64) -> FwResult<(Vec<TranscriptionSegment>, Vec<TranscriptionSegment>)>,
    {
        if total_duration_ms == 0 {
            checkpoint()?;
            let result = self.build_result();
            if self.config.emit_events {
                let stats = self.stats();
                self.push_event(
                    "transcript.speculation_stats",
                    "speculative pipeline aggregate statistics",
                    crate::robot::speculation_stats_value(&self.run_id, &stats),
                );
            }
            return Ok(result);
        }

        let mut position_ms = 0u64;
        while position_ms < total_duration_ms {
            checkpoint()?;

            let window_size_ms = self.window_manager.current_window_size().max(1);
            let step_ms = window_size_ms.saturating_sub(self.config.overlap_ms).max(1);

            let audio_hash = format!("{audio_hash_seed}:{position_ms}:{window_size_ms}");
            let Some(window) = self.window_manager.next_window_bounded(
                position_ms,
                total_duration_ms,
                &audio_hash,
            ) else {
                break;
            };

            let (fast_segments, quality_segments) = model_runner(window.start_ms, window.end_ms)?;
            self.process_window_by_id(
                window.window_id,
                move || fast_segments,
                move || quality_segments,
            )?;

            if window.end_ms >= total_duration_ms {
                break;
            }

            position_ms = position_ms.saturating_add(step_ms);
        }

        let result = self.build_result();
        if self.config.emit_events {
            let stats = self.stats();
            self.push_event(
                "transcript.speculation_stats",
                "speculative pipeline aggregate statistics",
                crate::robot::speculation_stats_value(&self.run_id, &stats),
            );
        }
        Ok(result)
    }

    /// Process an audio duration by repeatedly invoking model callbacks for each
    /// speculation window.
    pub fn process_duration_with_models<C, M>(
        &mut self,
        total_duration_ms: u64,
        audio_hash_seed: &str,
        mut checkpoint: C,
        mut model_runner: M,
    ) -> FwResult<TranscriptionResult>
    where
        C: FnMut() -> FwResult<()>,
        M: FnMut(u64, u64) -> FwResult<(Vec<TranscriptionSegment>, Vec<TranscriptionSegment>)>,
    {
        self.process_duration_loop(
            total_duration_ms,
            audio_hash_seed,
            &mut checkpoint,
            &mut model_runner,
        )
    }

    /// Convenience wrapper with no cancellation hook.
    pub fn process_duration_with_models_no_checkpoint<M>(
        &mut self,
        total_duration_ms: u64,
        audio_hash_seed: &str,
        model_runner: M,
    ) -> FwResult<TranscriptionResult>
    where
        M: FnMut(u64, u64) -> FwResult<(Vec<TranscriptionSegment>, Vec<TranscriptionSegment>)>,
    {
        self.process_duration_with_models(
            total_duration_ms,
            audio_hash_seed,
            || Ok(()),
            model_runner,
        )
    }

    /// Process an audio file by probing duration and invoking model callbacks
    /// for each bounded speculation window.
    pub fn process_file_with_models<C, M>(
        &mut self,
        audio_path: &Path,
        checkpoint: C,
        mut model_runner: M,
    ) -> FwResult<TranscriptionResult>
    where
        C: FnMut() -> FwResult<()>,
        M: FnMut(
            &Path,
            u64,
            u64,
        ) -> FwResult<(Vec<TranscriptionSegment>, Vec<TranscriptionSegment>)>,
    {
        let duration_sec =
            audio::probe_duration_seconds_with_timeout(audio_path, Duration::from_secs(10))
                .ok_or_else(|| {
                    FwError::InvalidRequest(format!(
                        "failed to probe audio duration for {}",
                        audio_path.display()
                    ))
                })?;
        let total_duration_ms = (duration_sec * 1000.0).round() as u64;
        let hash_seed = audio_path.display().to_string();
        self.process_duration_with_models(
            total_duration_ms,
            &hash_seed,
            checkpoint,
            |start_ms, end_ms| model_runner(audio_path, start_ms, end_ms),
        )
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
            confirmations_emitted: tracker_stats.confirmations_emitted,
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

    /// All events generated by the speculative pipeline.
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

#[cfg(test)]
mod tests {
    use super::*;

    fn seg(
        text: &str,
        start: Option<f64>,
        end: Option<f64>,
        conf: Option<f64>,
    ) -> TranscriptionSegment {
        TranscriptionSegment {
            text: text.to_owned(),
            start_sec: start,
            end_sec: end,
            confidence: conf,
            speaker: None,
        }
    }

    #[test]
    fn to_backend_segment_converts_seconds_to_ms_and_defaults() {
        // Normal conversion: seconds → ms.
        let s = seg("hello", Some(1.5), Some(2.75), Some(0.9));
        let bs = to_backend_segment(&s);
        assert_eq!(bs.start_ms, 1500);
        assert_eq!(bs.end_ms, 2750);
        assert_eq!(bs.text, "hello");
        assert!((bs.confidence - 0.9).abs() < f64::EPSILON);

        // None fields → 0.
        let s2 = seg("world", None, None, None);
        let bs2 = to_backend_segment(&s2);
        assert_eq!(bs2.start_ms, 0);
        assert_eq!(bs2.end_ms, 0);
        assert!((bs2.confidence - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn speculative_config_defaults() {
        let cfg = SpeculativeConfig::default();
        assert_eq!(cfg.window_size_ms, 3000);
        assert_eq!(cfg.overlap_ms, 500);
        assert_eq!(cfg.fast_model_name, "whisper-tiny");
        assert_eq!(cfg.quality_model_name, "whisper-large");
        assert!(cfg.adaptive);
        assert!(cfg.emit_events);
        // Tolerance defaults.
        assert!((cfg.tolerance.max_wer - 0.1).abs() < f64::EPSILON);
        assert!(!cfg.tolerance.always_correct);
    }

    #[test]
    fn pipeline_fresh_build_result_is_empty() {
        let pipeline =
            SpeculativeStreamingPipeline::new(SpeculativeConfig::default(), "test-run".to_owned());
        let result = pipeline.build_result();
        assert!(result.transcript.is_empty());
        assert!(result.segments.is_empty());
        assert_eq!(result.backend, BackendKind::Auto);
        assert_eq!(result.language, Some("en".to_owned()));
        assert_eq!(pipeline.run_id(), "test-run");
    }

    #[test]
    fn pipeline_process_window_identical_segments_confirms() {
        let mut pipeline = SpeculativeStreamingPipeline::new(
            SpeculativeConfig::default(),
            "test-confirm".to_owned(),
        );
        let segments = vec![seg("hello world", Some(0.0), Some(1.0), Some(0.95))];
        let fast_segs = segments.clone();
        let quality_segs = segments;

        let decision = pipeline
            .process_window("hash1", 0, move || fast_segs, move || quality_segs)
            .expect("process_window should succeed");

        assert!(matches!(decision, CorrectionDecision::Confirm { .. }));
        // Stats should reflect one window processed, zero corrections.
        let stats = pipeline.stats();
        assert_eq!(stats.windows_processed, 1);
        assert_eq!(stats.corrections_emitted, 0);
        assert_eq!(stats.confirmations_emitted, 1);
        // Events should include transcript.partial and transcript.confirm.
        let events = pipeline.events();
        assert!(events.iter().any(|e| e.code == "transcript.partial"));
        assert!(events.iter().any(|e| e.code == "transcript.confirm"));
    }

    #[test]
    fn pipeline_zero_duration_returns_immediately() {
        let mut pipeline =
            SpeculativeStreamingPipeline::new(SpeculativeConfig::default(), "test-zero".to_owned());
        let result = pipeline
            .process_duration_with_models_no_checkpoint(0, "seed", |_start, _end| {
                panic!("model_runner should not be called for zero duration");
            })
            .expect("should succeed");
        assert!(result.transcript.is_empty());
        // Should have emitted speculation_stats event.
        assert!(
            pipeline
                .events()
                .iter()
                .any(|e| e.code == "transcript.speculation_stats")
        );
    }

    // ── Task #207 — streaming pass 2 edge-case tests ────────────────

    #[test]
    fn event_seq_numbers_are_contiguous_across_windows() {
        let mut pipeline =
            SpeculativeStreamingPipeline::new(SpeculativeConfig::default(), "test-seq".to_owned());
        // Process two windows.
        for i in 0..2 {
            let s = vec![seg("word", Some(0.0), Some(1.0), Some(0.9))];
            let f = s.clone();
            let q = s;
            let hash = format!("h{i}");
            pipeline
                .process_window(&hash, i * 1000, move || f, move || q)
                .unwrap();
        }
        let events = pipeline.events();
        assert!(
            events.len() >= 4,
            "expected at least 4 events, got {}",
            events.len()
        );
        // Verify seq values are 0, 1, 2, 3, ...
        for (idx, event) in events.iter().enumerate() {
            assert_eq!(event.seq, idx as u64, "event {idx} has wrong seq");
        }
    }

    #[test]
    fn emit_events_false_suppresses_all_events() {
        let mut config = SpeculativeConfig::default();
        config.emit_events = false;

        let mut pipeline = SpeculativeStreamingPipeline::new(config, "test-no-events".to_owned());

        let s = vec![seg("hello", Some(0.0), Some(1.0), Some(0.9))];
        let f = s.clone();
        let q = s;
        pipeline
            .process_window("h", 0, move || f, move || q)
            .unwrap();

        assert!(
            pipeline.events().is_empty(),
            "no events should be emitted when emit_events is false"
        );
    }

    #[test]
    fn correction_emits_retract_and_correct_events() {
        let mut config = SpeculativeConfig::default();
        config.tolerance.always_correct = true; // force correction

        let mut pipeline = SpeculativeStreamingPipeline::new(config, "test-correct".to_owned());

        let fast = vec![seg("hello", Some(0.0), Some(1.0), Some(0.9))];
        let quality = vec![seg("hello", Some(0.0), Some(1.0), Some(0.9))];
        let f = fast;
        let q = quality;
        let decision = pipeline
            .process_window("h", 0, move || f, move || q)
            .unwrap();

        let is_correct = matches!(decision, CorrectionDecision::Correct { .. });
        assert!(is_correct, "always_correct should force Correct decision");

        let events = pipeline.events();
        assert!(
            events.iter().any(|e| e.code == "transcript.retract"),
            "should emit transcript.retract"
        );
        assert!(
            events.iter().any(|e| e.code == "transcript.correct"),
            "should emit transcript.correct"
        );
    }

    #[test]
    fn build_result_joins_segments_with_space() {
        let mut pipeline =
            SpeculativeStreamingPipeline::new(SpeculativeConfig::default(), "test-join".to_owned());

        let result = pipeline
            .process_duration_with_models_no_checkpoint(
                5000, // 5 seconds → at least one window
                "seed",
                |_start, _end| {
                    let segs = vec![seg("hello world", Some(0.0), Some(1.0), Some(0.9))];
                    Ok((segs.clone(), segs))
                },
            )
            .unwrap();

        assert!(
            !result.transcript.is_empty(),
            "transcript should not be empty"
        );
        assert!(!result.segments.is_empty(), "segments should not be empty");
        assert_eq!(result.language, Some("en".to_owned()));
        assert_eq!(result.backend, BackendKind::Auto);
    }

    #[test]
    fn checkpoint_error_cancels_duration_loop() {
        let mut pipeline = SpeculativeStreamingPipeline::new(
            SpeculativeConfig::default(),
            "test-cancel".to_owned(),
        );

        let result = pipeline.process_duration_with_models(
            10_000,
            "seed",
            || Err(crate::error::FwError::Cancelled("cancelled".to_owned())),
            |_start, _end| {
                panic!("model_runner should not be called when checkpoint fails");
            },
        );

        assert!(result.is_err());
        let err = result.unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("cancelled"),
            "error should mention cancelled: {msg}"
        );
    }
}
