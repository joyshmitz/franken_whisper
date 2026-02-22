//! Core speculative streaming types for dual-model transcription.
//!
//! The speculation module implements the "speculate then verify" pattern:
//! a fast model emits [`PartialTranscript`]s in real time, and a slower
//! quality model either confirms them or triggers a [`CorrectionEvent`].

use serde::{Deserialize, Serialize};

use crate::error::{FwError, FwResult};
use crate::model::TranscriptionSegment;
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// SpeculationWindow
// ---------------------------------------------------------------------------

/// Represents a sliding window over the audio stream that is sent to both the
/// fast and quality models.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeculationWindow {
    /// Monotonically increasing window identifier.
    pub window_id: u64,
    /// Links to `RunReport.run_id`.
    pub run_id: String,
    /// Absolute position in the audio stream (milliseconds).
    pub start_ms: u64,
    /// `start_ms + window_size_ms`.
    pub end_ms: u64,
    /// Overlap with the previous window (typically 200–500 ms).
    pub overlap_ms: u64,
    /// SHA-256 hex digest of the raw audio bytes in this window.
    pub audio_hash: String,
}

impl SpeculationWindow {
    /// Create a new speculation window.
    #[must_use]
    pub fn new(
        window_id: u64,
        run_id: String,
        start_ms: u64,
        end_ms: u64,
        overlap_ms: u64,
        audio_hash: String,
    ) -> Self {
        Self {
            window_id,
            run_id,
            start_ms,
            end_ms,
            overlap_ms,
            audio_hash,
        }
    }

    /// Duration of this window in milliseconds.
    #[must_use]
    pub fn duration_ms(&self) -> u64 {
        self.end_ms.saturating_sub(self.start_ms)
    }

    /// Returns `true` if the given millisecond offset falls within this window.
    #[must_use]
    pub fn contains_ms(&self, ms: u64) -> bool {
        ms >= self.start_ms && ms < self.end_ms
    }
}

// ---------------------------------------------------------------------------
// PartialStatus
// ---------------------------------------------------------------------------

/// Lifecycle status of a [`PartialTranscript`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PartialStatus {
    /// Emitted by the fast model, waiting for the quality model.
    Pending,
    /// The quality model agreed (within tolerance).
    Confirmed,
    /// The quality model disagreed; a correction has been emitted.
    Retracted,
}

// ---------------------------------------------------------------------------
// PartialTranscript
// ---------------------------------------------------------------------------

/// A speculative transcript produced by the fast model for a single window.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PartialTranscript {
    /// Global sequence number used for retraction targeting.
    pub seq: u64,
    /// Links to [`SpeculationWindow::window_id`].
    pub window_id: u64,
    /// Identifier of the model that produced this transcript.
    pub model_id: String,
    /// Transcription segments produced by the fast model.
    pub segments: Vec<TranscriptionSegment>,
    /// Wall-clock latency of the fast model inference (ms).
    pub latency_ms: u64,
    /// Mean confidence across all segments (0.0 when no segments carry confidence).
    pub confidence_mean: f64,
    /// RFC 3339 timestamp of when this partial was emitted.
    pub emitted_at_rfc3339: String,
    /// Current lifecycle status.
    pub status: PartialStatus,
}

impl PartialTranscript {
    /// Create a new `PartialTranscript`.
    ///
    /// `confidence_mean` is computed automatically from the segments and `status`
    /// is set to [`PartialStatus::Pending`].
    #[must_use]
    pub fn new(
        seq: u64,
        window_id: u64,
        model_id: String,
        segments: Vec<TranscriptionSegment>,
        latency_ms: u64,
        emitted_at_rfc3339: String,
    ) -> Self {
        let confidence_mean = mean_confidence(&segments);
        Self {
            seq,
            window_id,
            model_id,
            segments,
            latency_ms,
            confidence_mean,
            emitted_at_rfc3339,
            status: PartialStatus::Pending,
        }
    }

    /// Mark this partial as confirmed by the quality model.
    pub fn confirm(&mut self) {
        self.status = PartialStatus::Confirmed;
    }

    /// Mark this partial as retracted (a correction has been issued).
    pub fn retract(&mut self) {
        self.status = PartialStatus::Retracted;
    }
}

// ---------------------------------------------------------------------------
// CorrectionDrift
// ---------------------------------------------------------------------------

/// Quantifies the difference between the fast and quality transcriptions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorrectionDrift {
    /// Approximate word error rate: 0.0 = identical, 1.0 = completely different.
    pub wer_approx: f64,
    /// Absolute difference in mean confidence.
    pub confidence_delta: f64,
    /// `quality_segment_count - fast_segment_count`.
    pub segment_count_delta: i32,
    /// Levenshtein edit distance on the concatenated text.
    pub text_edit_distance: usize,
}

impl CorrectionDrift {
    /// Compute drift metrics between fast and quality segment sets.
    #[must_use]
    pub fn compute(
        fast_segments: &[TranscriptionSegment],
        quality_segments: &[TranscriptionSegment],
    ) -> Self {
        let fast_text = concat_segment_text(fast_segments);
        let quality_text = concat_segment_text(quality_segments);

        let text_edit_distance = levenshtein(&fast_text, &quality_text);

        // Word-level approximate WER.
        let fast_words: Vec<&str> = fast_text.split_whitespace().collect();
        let quality_words: Vec<&str> = quality_text.split_whitespace().collect();
        let word_edit_dist = levenshtein_words(&fast_words, &quality_words);
        let max_words = fast_words.len().max(quality_words.len()).max(1);
        let wer_approx = word_edit_dist as f64 / max_words as f64;

        let confidence_delta =
            (mean_confidence(fast_segments) - mean_confidence(quality_segments)).abs();

        let segment_count_delta = quality_segments.len() as i32 - fast_segments.len() as i32;

        Self {
            wer_approx,
            confidence_delta,
            segment_count_delta,
            text_edit_distance,
        }
    }
}

// ---------------------------------------------------------------------------
// CorrectionEvent
// ---------------------------------------------------------------------------

/// Emitted when the quality model disagrees with a [`PartialTranscript`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorrectionEvent {
    /// Unique correction identifier.
    pub correction_id: u64,
    /// The [`PartialTranscript::seq`] being retracted.
    pub retracted_seq: u64,
    /// The window this correction applies to.
    pub window_id: u64,
    /// Identifier of the quality model.
    pub quality_model_id: String,
    /// Corrected transcription segments from the quality model.
    pub corrected_segments: Vec<TranscriptionSegment>,
    /// Wall-clock latency of the quality model inference (ms).
    pub quality_latency_ms: u64,
    /// Mean confidence across the corrected segments.
    pub quality_confidence_mean: f64,
    /// Drift metrics between the fast and quality outputs.
    pub drift: CorrectionDrift,
    /// RFC 3339 timestamp of when the correction was produced.
    pub corrected_at_rfc3339: String,
}

impl CorrectionEvent {
    /// Create a new `CorrectionEvent`.
    ///
    /// `quality_confidence_mean` is computed from `corrected_segments` and
    /// `drift` is computed from `fast_segments` and `corrected_segments`.
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        correction_id: u64,
        retracted_seq: u64,
        window_id: u64,
        quality_model_id: String,
        corrected_segments: Vec<TranscriptionSegment>,
        quality_latency_ms: u64,
        corrected_at_rfc3339: String,
        fast_segments: &[TranscriptionSegment],
    ) -> Self {
        let quality_confidence_mean = mean_confidence(&corrected_segments);
        let drift = CorrectionDrift::compute(fast_segments, &corrected_segments);
        Self {
            correction_id,
            retracted_seq,
            window_id,
            quality_model_id,
            corrected_segments,
            quality_latency_ms,
            quality_confidence_mean,
            drift,
            corrected_at_rfc3339,
        }
    }

    /// Returns `true` when the approximate WER exceeds the given threshold.
    #[must_use]
    pub fn is_significant(&self, threshold: f64) -> bool {
        self.drift.wer_approx > threshold
    }
}

// ---------------------------------------------------------------------------
// SpeculationStats
// ---------------------------------------------------------------------------

/// Aggregate statistics for the speculation pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeculationStats {
    /// Number of windows processed so far.
    pub windows_processed: u64,
    /// Number of corrections emitted.
    pub corrections_emitted: u64,
    /// `corrections_emitted / windows_processed` (0.0 when no windows).
    pub correction_rate: f64,
    /// Mean latency of the fast model across all partials (ms).
    pub mean_fast_latency_ms: f64,
    /// Mean latency of the quality model across all corrections (ms).
    pub mean_quality_latency_ms: f64,
    /// Current window size being used (ms).
    pub current_window_size_ms: u64,
    /// Mean approximate WER across all corrections.
    pub mean_drift_wer: f64,
}

// ---------------------------------------------------------------------------
// Helper functions (private)
// ---------------------------------------------------------------------------

/// Compute the mean confidence across a slice of segments. Returns 0.0 when
/// no segment carries a confidence value.
fn mean_confidence(segments: &[TranscriptionSegment]) -> f64 {
    let (sum, count) = segments.iter().fold((0.0_f64, 0_u64), |(s, c), seg| {
        if let Some(conf) = seg.confidence {
            (s + conf, c + 1)
        } else {
            (s, c)
        }
    });
    if count == 0 { 0.0 } else { sum / count as f64 }
}

/// Concatenate all segment texts separated by a single space.
fn concat_segment_text(segments: &[TranscriptionSegment]) -> String {
    segments
        .iter()
        .map(|s| s.text.as_str())
        .collect::<Vec<_>>()
        .join(" ")
}

/// Character-level Levenshtein distance.
fn levenshtein(a: &str, b: &str) -> usize {
    let a_chars: Vec<char> = a.chars().collect();
    let b_chars: Vec<char> = b.chars().collect();
    let m = a_chars.len();
    let n = b_chars.len();

    if m == 0 {
        return n;
    }
    if n == 0 {
        return m;
    }

    // Two-row optimisation.
    let mut prev = (0..=n).collect::<Vec<usize>>();
    let mut curr = vec![0usize; n + 1];

    for i in 1..=m {
        curr[0] = i;
        for j in 1..=n {
            let cost = if a_chars[i - 1] == b_chars[j - 1] {
                0
            } else {
                1
            };
            curr[j] = (prev[j] + 1).min(curr[j - 1] + 1).min(prev[j - 1] + cost);
        }
        std::mem::swap(&mut prev, &mut curr);
    }

    prev[n]
}

/// Word-level Levenshtein distance.
fn levenshtein_words(a: &[&str], b: &[&str]) -> usize {
    let m = a.len();
    let n = b.len();

    if m == 0 {
        return n;
    }
    if n == 0 {
        return m;
    }

    let mut prev = (0..=n).collect::<Vec<usize>>();
    let mut curr = vec![0usize; n + 1];

    for i in 1..=m {
        curr[0] = i;
        for j in 1..=n {
            let cost = if a[i - 1] == b[j - 1] { 0 } else { 1 };
            curr[j] = (prev[j] + 1).min(curr[j - 1] + 1).min(prev[j - 1] + cost);
        }
        std::mem::swap(&mut prev, &mut curr);
    }

    prev[n]
}

// ---------------------------------------------------------------------------
// bd-qlt.4: WindowManager — audio stream windowing with configurable overlap
// ---------------------------------------------------------------------------

/// Status of a speculation window as it moves through the pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WindowStatus {
    /// Window created, no results yet.
    Pending,
    /// Fast model inference in progress.
    FastInProgress,
    /// Fast model result recorded.
    FastComplete,
    /// Quality model result recorded.
    QualityComplete,
    /// Window fully resolved (fast + quality compared).
    Resolved,
}

/// Tracks the state of a single speculation window through both model passes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WindowState {
    pub window: SpeculationWindow,
    pub fast_result: Option<PartialTranscript>,
    pub quality_result: Option<Vec<TranscriptionSegment>>,
    pub status: WindowStatus,
}

/// Manages the lifecycle of overlapping audio windows for speculative
/// dual-model transcription.
pub struct WindowManager {
    window_size_ms: u64,
    overlap_ms: u64,
    min_window_ms: u64,
    max_window_ms: u64,
    next_window_id: u64,
    run_id: String,
    windows: Vec<WindowState>,
}

impl WindowManager {
    /// Create a new `WindowManager`.
    ///
    /// `min_window_ms` defaults to 1000 and `max_window_ms` defaults to 30000.
    #[must_use]
    pub fn new(run_id: &str, window_size_ms: u64, overlap_ms: u64) -> Self {
        Self {
            window_size_ms,
            overlap_ms,
            min_window_ms: 1000,
            max_window_ms: 30_000,
            next_window_id: 0,
            run_id: run_id.to_owned(),
            windows: Vec::new(),
        }
    }

    /// Create the next speculation window starting at `audio_position_ms`.
    pub fn next_window(&mut self, audio_position_ms: u64, audio_hash: &str) -> SpeculationWindow {
        let id = self.next_window_id;
        self.next_window_id += 1;

        let window = SpeculationWindow::new(
            id,
            self.run_id.clone(),
            audio_position_ms,
            audio_position_ms + self.window_size_ms,
            self.overlap_ms,
            audio_hash.to_owned(),
        );

        self.windows.push(WindowState {
            window: window.clone(),
            fast_result: None,
            quality_result: None,
            status: WindowStatus::Pending,
        });

        window
    }

    /// Record the fast model result for a window.
    pub fn record_fast_result(&mut self, window_id: u64, partial: PartialTranscript) {
        if let Some(ws) = self.get_window_mut(window_id) {
            ws.fast_result = Some(partial);
            ws.status = WindowStatus::FastComplete;
        }
    }

    /// Record the quality model result for a window.
    pub fn record_quality_result(&mut self, window_id: u64, segments: Vec<TranscriptionSegment>) {
        if let Some(ws) = self.get_window_mut(window_id) {
            ws.quality_result = Some(segments);
            ws.status = WindowStatus::QualityComplete;
        }
    }

    /// Mark a window as fully resolved.
    pub fn resolve_window(&mut self, window_id: u64) {
        if let Some(ws) = self.get_window_mut(window_id) {
            ws.status = WindowStatus::Resolved;
        }
    }

    /// Update the window size, clamped to `[min_window_ms, max_window_ms]`.
    pub fn set_window_size(&mut self, new_size_ms: u64) {
        self.window_size_ms = new_size_ms.clamp(self.min_window_ms, self.max_window_ms);
    }

    /// Current window size in milliseconds.
    #[must_use]
    pub fn current_window_size(&self) -> u64 {
        self.window_size_ms
    }

    /// Count of windows with [`WindowStatus::Resolved`].
    #[must_use]
    pub fn windows_resolved(&self) -> usize {
        self.windows
            .iter()
            .filter(|ws| ws.status == WindowStatus::Resolved)
            .count()
    }

    /// Count of windows that are **not** yet resolved.
    #[must_use]
    pub fn windows_pending(&self) -> usize {
        self.windows
            .iter()
            .filter(|ws| ws.status != WindowStatus::Resolved)
            .count()
    }

    /// Look up a window by its id.
    #[must_use]
    pub fn get_window(&self, window_id: u64) -> Option<&WindowState> {
        self.windows
            .iter()
            .find(|ws| ws.window.window_id == window_id)
    }

    /// Look up a window mutably by its id.
    pub fn get_window_mut(&mut self, window_id: u64) -> Option<&mut WindowState> {
        self.windows
            .iter_mut()
            .find(|ws| ws.window.window_id == window_id)
    }

    /// Merge segments from all resolved windows into a single sorted,
    /// deduplicated list.
    ///
    /// For each resolved window, the quality result is preferred; if absent the
    /// fast result's segments are used. Segments are sorted by `start_sec` and
    /// overlapping segments (within 0.1 s) are deduplicated by keeping the one
    /// with higher confidence.
    #[must_use]
    pub fn merge_segments(&self) -> Vec<TranscriptionSegment> {
        let mut all_segments: Vec<TranscriptionSegment> = Vec::new();

        for ws in &self.windows {
            if ws.status != WindowStatus::Resolved {
                continue;
            }
            if let Some(ref quality) = ws.quality_result {
                all_segments.extend(quality.clone());
            } else if let Some(ref fast) = ws.fast_result {
                all_segments.extend(fast.segments.clone());
            }
        }

        // Sort by start_sec (None sorts before Some).
        all_segments.sort_by(|a, b| {
            a.start_sec
                .unwrap_or(0.0)
                .partial_cmp(&b.start_sec.unwrap_or(0.0))
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Deduplicate overlapping segments (within 0.1 sec tolerance).
        let mut deduped: Vec<TranscriptionSegment> = Vec::new();
        for seg in all_segments {
            let dominated = if let Some(last) = deduped.last() {
                let last_start = last.start_sec.unwrap_or(0.0);
                let seg_start = seg.start_sec.unwrap_or(0.0);
                let last_end = last.end_sec.unwrap_or(0.0);
                let seg_end = seg.end_sec.unwrap_or(0.0);
                (seg_start - last_start).abs() < 0.1 && (seg_end - last_end).abs() < 0.1
            } else {
                false
            };

            if dominated {
                // Keep the one with higher confidence.
                let last = deduped.last().expect("checked above");
                let last_conf = last.confidence.unwrap_or(0.0);
                let seg_conf = seg.confidence.unwrap_or(0.0);
                if seg_conf > last_conf {
                    let len = deduped.len();
                    deduped[len - 1] = seg;
                }
            } else {
                deduped.push(seg);
            }
        }

        deduped
    }
}

// ---------------------------------------------------------------------------
// bd-qlt.5: CorrectionTracker — retraction/correction lifecycle state machine
// ---------------------------------------------------------------------------

/// Tolerance thresholds that determine when a fast partial should be corrected.
#[derive(Debug, Clone)]
pub struct CorrectionTolerance {
    /// Maximum acceptable approximate WER before triggering a correction.
    pub max_wer: f64,
    /// Maximum acceptable confidence delta before triggering a correction.
    pub max_confidence_delta: f64,
    /// Maximum acceptable edit distance before triggering a correction.
    pub max_edit_distance: usize,
    /// When `true`, always emit a correction regardless of drift metrics.
    pub always_correct: bool,
}

impl Default for CorrectionTolerance {
    fn default() -> Self {
        Self {
            max_wer: 0.1,
            max_confidence_delta: 0.15,
            max_edit_distance: 50,
            always_correct: false,
        }
    }
}

/// Aggregate statistics tracked by [`CorrectionTracker`].
pub struct CorrectionStats {
    pub windows_processed: u64,
    pub corrections_emitted: u64,
    pub confirmations_emitted: u64,
    pub total_fast_latency_ms: u64,
    pub total_quality_latency_ms: u64,
    pub cumulative_wer: f64,
    pub max_observed_wer: f64,
}

/// The decision produced after comparing a fast partial against quality output.
pub enum CorrectionDecision {
    /// The quality model agreed with the fast partial.
    Confirm { seq: u64, drift: CorrectionDrift },
    /// The quality model disagreed; a correction has been issued.
    Correct { correction: CorrectionEvent },
}

/// Tracks the lifecycle of partials through the correction pipeline,
/// deciding whether each partial should be confirmed or corrected based
/// on configurable tolerance thresholds.
pub struct CorrectionTracker {
    tolerance: CorrectionTolerance,
    partials: HashMap<u64, PartialTranscript>,
    corrections: Vec<CorrectionEvent>,
    stats: CorrectionStats,
    next_correction_id: u64,
}

impl CorrectionTracker {
    /// Create a new `CorrectionTracker` with the given tolerance thresholds.
    #[must_use]
    pub fn new(tolerance: CorrectionTolerance) -> Self {
        Self {
            tolerance,
            partials: HashMap::new(),
            corrections: Vec::new(),
            stats: CorrectionStats {
                windows_processed: 0,
                corrections_emitted: 0,
                confirmations_emitted: 0,
                total_fast_latency_ms: 0,
                total_quality_latency_ms: 0,
                cumulative_wer: 0.0,
                max_observed_wer: 0.0,
            },
            next_correction_id: 0,
        }
    }

    /// Register a partial transcript produced by the fast model.
    ///
    /// Returns the `seq` number of the registered partial.
    pub fn register_partial(&mut self, partial: PartialTranscript) -> u64 {
        let seq = partial.seq;
        self.stats.total_fast_latency_ms += partial.latency_ms;
        self.partials.insert(seq, partial);
        seq
    }

    /// Submit quality model output for comparison against a previously
    /// registered partial.
    ///
    /// Returns a [`CorrectionDecision`] indicating whether the partial was
    /// confirmed or corrected. Returns `Err` if no partial is found for
    /// `window_id`.
    pub fn submit_quality_result(
        &mut self,
        window_id: u64,
        quality_model_id: &str,
        quality_segments: Vec<TranscriptionSegment>,
        quality_latency_ms: u64,
    ) -> FwResult<CorrectionDecision> {
        // Find the partial matching this window_id.
        let seq = self
            .partials
            .values()
            .find(|p| p.window_id == window_id)
            .map(|p| p.seq)
            .ok_or_else(|| {
                FwError::InvalidRequest(format!("no partial registered for window_id {window_id}"))
            })?;

        let drift = {
            let partial = self.partials.get(&seq).expect("just found above");
            CorrectionDrift::compute(&partial.segments, &quality_segments)
        };

        // Update stats.
        self.stats.windows_processed += 1;
        self.stats.total_quality_latency_ms += quality_latency_ms;
        self.stats.cumulative_wer += drift.wer_approx;
        if drift.wer_approx > self.stats.max_observed_wer {
            self.stats.max_observed_wer = drift.wer_approx;
        }

        let needs_correction = self.tolerance.always_correct
            || drift.wer_approx > self.tolerance.max_wer
            || drift.confidence_delta > self.tolerance.max_confidence_delta
            || drift.text_edit_distance > self.tolerance.max_edit_distance;

        if needs_correction {
            // Retract the partial.
            let partial = self.partials.get_mut(&seq).expect("just found above");
            partial.retract();
            let fast_segments = partial.segments.clone();

            let correction_id = self.next_correction_id;
            self.next_correction_id += 1;

            let correction = CorrectionEvent::new(
                correction_id,
                seq,
                window_id,
                quality_model_id.to_owned(),
                quality_segments,
                quality_latency_ms,
                chrono::Utc::now().to_rfc3339(),
                &fast_segments,
            );

            self.corrections.push(correction.clone());
            self.stats.corrections_emitted += 1;

            Ok(CorrectionDecision::Correct { correction })
        } else {
            // Confirm the partial.
            let partial = self.partials.get_mut(&seq).expect("just found above");
            partial.confirm();
            self.stats.confirmations_emitted += 1;

            Ok(CorrectionDecision::Confirm { seq, drift })
        }
    }

    /// Reference to aggregate statistics.
    #[must_use]
    pub fn stats(&self) -> &CorrectionStats {
        &self.stats
    }

    /// Fraction of windows that required correction (0.0 if none processed).
    #[must_use]
    pub fn correction_rate(&self) -> f64 {
        let total = self.stats.corrections_emitted + self.stats.confirmations_emitted;
        if total == 0 {
            0.0
        } else {
            self.stats.corrections_emitted as f64 / total as f64
        }
    }

    /// Mean approximate WER across all processed windows (0.0 if none).
    #[must_use]
    pub fn mean_wer(&self) -> f64 {
        if self.stats.windows_processed == 0 {
            0.0
        } else {
            self.stats.cumulative_wer / self.stats.windows_processed as f64
        }
    }

    /// All correction events emitted so far.
    #[must_use]
    pub fn corrections(&self) -> &[CorrectionEvent] {
        &self.corrections
    }

    /// Look up a registered partial by sequence number.
    #[must_use]
    pub fn get_partial(&self, seq: u64) -> Option<&PartialTranscript> {
        self.partials.get(&seq)
    }

    /// Returns `true` when every registered partial has been confirmed or
    /// retracted.
    #[must_use]
    pub fn all_resolved(&self) -> bool {
        self.partials
            .values()
            .all(|p| p.status == PartialStatus::Confirmed || p.status == PartialStatus::Retracted)
    }
}

// ---------------------------------------------------------------------------
// bd-qlt.7: SpeculationWindowController — alien-artifact adaptive sizing
// ---------------------------------------------------------------------------

/// Snapshot of the controller's observable state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ControllerState {
    pub correction_rate: f64,
    pub mean_wer: f64,
    pub window_count: u64,
    pub current_window_ms: u64,
}

/// Beta distribution posterior for Bayesian correction rate estimation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BetaPosterior {
    /// Corrections observed + prior.
    pub alpha: f64,
    /// Confirmations observed + prior.
    pub beta: f64,
}

impl BetaPosterior {
    #[must_use]
    pub fn new(alpha: f64, beta: f64) -> Self {
        Self { alpha, beta }
    }

    /// Prior: Beta(2,2) — weakly informative, centered at 0.5.
    #[must_use]
    pub fn weakly_informative() -> Self {
        Self::new(2.0, 2.0)
    }

    /// Observe a correction (alpha += 1).
    pub fn observe_correction(&mut self) {
        self.alpha += 1.0;
    }

    /// Observe a confirmation (beta += 1).
    pub fn observe_confirmation(&mut self) {
        self.beta += 1.0;
    }

    /// Mean of the posterior distribution.
    #[must_use]
    pub fn mean(&self) -> f64 {
        self.alpha / (self.alpha + self.beta)
    }

    /// Variance of the posterior distribution.
    #[must_use]
    pub fn variance(&self) -> f64 {
        let total = self.alpha + self.beta;
        (self.alpha * self.beta) / (total * total * (total + 1.0))
    }
}

/// Action the controller can take.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ControllerAction {
    Shrink(u64),
    Hold,
    Grow(u64),
}

/// Calibration tracker using Brier score over a rolling window.
#[derive(Debug, Clone)]
pub struct CalibrationTracker {
    predictions: Vec<f64>,
    outcomes: Vec<bool>,
    window_size: usize,
}

impl CalibrationTracker {
    #[must_use]
    pub fn new(window_size: usize) -> Self {
        Self {
            predictions: Vec::new(),
            outcomes: Vec::new(),
            window_size,
        }
    }

    pub fn record(&mut self, predicted: f64, actual_correction: bool) {
        self.predictions.push(predicted);
        self.outcomes.push(actual_correction);
        if self.predictions.len() > self.window_size {
            self.predictions.remove(0);
            self.outcomes.remove(0);
        }
    }

    /// Brier score: mean((predicted - actual)^2). Lower is better.
    #[must_use]
    pub fn brier_score(&self) -> f64 {
        if self.predictions.is_empty() {
            return 0.0;
        }
        let sum: f64 = self
            .predictions
            .iter()
            .zip(&self.outcomes)
            .map(|(p, &o)| {
                let actual = if o { 1.0 } else { 0.0 };
                (p - actual).powi(2)
            })
            .sum();
        sum / self.predictions.len() as f64
    }

    #[must_use]
    pub fn sample_count(&self) -> usize {
        self.predictions.len()
    }
}

/// Evidence ledger entry for each controller decision.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeculationControllerEntry {
    pub decision_id: u64,
    pub window_id: u64,
    pub state_snapshot: ControllerState,
    pub action_taken: ControllerAction,
    pub predicted_correction_rate: f64,
    pub actual_correction_rate: f64,
    pub brier_score: f64,
    pub confidence: f64,
    pub fallback_active: bool,
    pub fallback_reason: Option<String>,
}

/// Adaptive controller that tunes speculation window size based on
/// correction evidence. Implements the full alien-artifact engineering
/// contract (AGENTS.md).
pub struct SpeculationWindowController {
    initial_window_ms: u64,
    current_window_ms: u64,
    min_window_ms: u64,
    max_window_ms: u64,
    step_ms: u64,
    state: ControllerState,
    posterior: BetaPosterior,
    calibration: CalibrationTracker,
    evidence: Vec<SpeculationControllerEntry>,
    fallback_active: bool,
    fallback_reason: Option<String>,
    next_decision_id: u64,
    consecutive_zero_corrections: u64,
}

impl SpeculationWindowController {
    const BRIER_FALLBACK_THRESHOLD: f64 = 0.25;
    const HIGH_CORRECTION_RATE: f64 = 0.25;
    const LOW_CORRECTION_RATE: f64 = 0.0625;
    const HIGH_WER_THRESHOLD: f64 = 0.125;
    const MIN_WINDOWS_FOR_ADAPT: u64 = 5;
    const FULL_CONFIDENCE_WINDOWS: f64 = 20.0;
    const RUNAWAY_CORRECTION_RATE: f64 = 0.75;
    const ZERO_CORRECTION_FORCE_SHRINK: u64 = 20;

    #[must_use]
    pub fn new(initial_window_ms: u64, min_ms: u64, max_ms: u64, step_ms: u64) -> Self {
        Self {
            initial_window_ms,
            current_window_ms: initial_window_ms,
            min_window_ms: min_ms,
            max_window_ms: max_ms,
            step_ms,
            state: ControllerState {
                correction_rate: 0.0,
                mean_wer: 0.0,
                window_count: 0,
                current_window_ms: initial_window_ms,
            },
            posterior: BetaPosterior::weakly_informative(),
            calibration: CalibrationTracker::new(20),
            evidence: Vec::new(),
            fallback_active: false,
            fallback_reason: None,
            next_decision_id: 0,
            consecutive_zero_corrections: 0,
        }
    }

    /// Observe a correction decision and update internal state.
    pub fn observe(&mut self, decision: &CorrectionDecision, drift: &CorrectionDrift) {
        let is_correction = matches!(decision, CorrectionDecision::Correct { .. });

        if is_correction {
            self.posterior.observe_correction();
            self.consecutive_zero_corrections = 0;
        } else {
            self.posterior.observe_confirmation();
            self.consecutive_zero_corrections += 1;
        }

        self.state.window_count += 1;

        let total = self.posterior.alpha + self.posterior.beta - 4.0;
        let corrections = self.posterior.alpha - 2.0;
        if total > 0.0 {
            self.state.correction_rate = corrections / total;
        }

        self.state.mean_wer = if is_correction {
            let prev_total = self.state.mean_wer * (self.state.window_count - 1) as f64;
            (prev_total + drift.wer_approx) / self.state.window_count as f64
        } else {
            self.state.mean_wer * (self.state.window_count - 1) as f64
                / self.state.window_count as f64
        };
        self.state.current_window_ms = self.current_window_ms;

        let predicted = self.posterior.mean();
        self.calibration.record(predicted, is_correction);
    }

    /// Recommend an action based on current state.
    #[must_use]
    pub fn recommend(&self) -> ControllerAction {
        if self.state.window_count < Self::MIN_WINDOWS_FOR_ADAPT {
            return ControllerAction::Hold;
        }

        if self.calibration.brier_score() > Self::BRIER_FALLBACK_THRESHOLD
            && self.calibration.sample_count() >= 10
        {
            return ControllerAction::Hold;
        }

        if self.state.correction_rate > Self::RUNAWAY_CORRECTION_RATE
            && self.current_window_ms < self.max_window_ms
        {
            return ControllerAction::Grow(self.step_ms);
        }

        if self.consecutive_zero_corrections >= Self::ZERO_CORRECTION_FORCE_SHRINK
            && self.current_window_ms > self.min_window_ms
        {
            return ControllerAction::Shrink(self.step_ms);
        }

        let confidence = self.confidence();
        if confidence < 0.5 {
            return ControllerAction::Hold;
        }

        if self.state.correction_rate > Self::HIGH_CORRECTION_RATE
            && self.state.mean_wer > Self::HIGH_WER_THRESHOLD
            && self.current_window_ms < self.max_window_ms
        {
            return ControllerAction::Grow(self.step_ms);
        }

        if self.state.correction_rate < Self::LOW_CORRECTION_RATE
            && self.state.window_count > 10
            && self.current_window_ms > self.min_window_ms
        {
            return ControllerAction::Shrink(self.step_ms);
        }

        ControllerAction::Hold
    }

    /// Apply the recommended action and return the new window size.
    pub fn apply(&mut self) -> u64 {
        let action = self.recommend();

        let brier = self.calibration.brier_score();
        if brier > Self::BRIER_FALLBACK_THRESHOLD && self.calibration.sample_count() >= 10 {
            self.fallback_active = true;
            self.fallback_reason = Some(format!("Brier score {brier:.3} > threshold"));
            self.current_window_ms = self.initial_window_ms;
        } else if self.state.correction_rate > Self::RUNAWAY_CORRECTION_RATE {
            self.fallback_active = true;
            self.fallback_reason = Some("correction rate > 75%".to_owned());
            if let ControllerAction::Grow(delta) = &action {
                self.current_window_ms =
                    (self.current_window_ms + delta).min(self.max_window_ms);
            }
        } else {
            self.fallback_active = false;
            self.fallback_reason = None;
            match &action {
                ControllerAction::Shrink(delta) => {
                    self.current_window_ms = self
                        .current_window_ms
                        .saturating_sub(*delta)
                        .max(self.min_window_ms);
                }
                ControllerAction::Hold => {}
                ControllerAction::Grow(delta) => {
                    self.current_window_ms =
                        (self.current_window_ms + delta).min(self.max_window_ms);
                }
            }
        }

        let entry = SpeculationControllerEntry {
            decision_id: self.next_decision_id,
            window_id: self.state.window_count,
            state_snapshot: self.state.clone(),
            action_taken: action,
            predicted_correction_rate: self.posterior.mean(),
            actual_correction_rate: self.state.correction_rate,
            brier_score: brier,
            confidence: self.confidence(),
            fallback_active: self.fallback_active,
            fallback_reason: self.fallback_reason.clone(),
        };
        self.next_decision_id += 1;
        self.evidence.push(entry);

        self.state.current_window_ms = self.current_window_ms;
        self.current_window_ms
    }

    #[must_use]
    pub fn is_fallback_active(&self) -> bool {
        self.fallback_active
    }

    #[must_use]
    pub fn evidence(&self) -> &[SpeculationControllerEntry] {
        &self.evidence
    }

    #[must_use]
    pub fn current_window_ms(&self) -> u64 {
        self.current_window_ms
    }

    /// Confidence level: min(window_count / 20, 1.0).
    #[must_use]
    pub fn confidence(&self) -> f64 {
        (self.state.window_count as f64 / Self::FULL_CONFIDENCE_WINDOWS).min(1.0)
    }

    #[must_use]
    pub fn state(&self) -> &ControllerState {
        &self.state
    }

    #[must_use]
    pub fn posterior(&self) -> &BetaPosterior {
        &self.posterior
    }
}
