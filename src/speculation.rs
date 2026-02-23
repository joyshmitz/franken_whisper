//! Core speculative streaming types for dual-model transcription.
//!
//! The speculation module implements the "speculate then verify" pattern:
//! a fast model emits [`PartialTranscript`]s in real time, and a slower
//! quality model either confirms them or triggers a [`CorrectionEvent`].

use serde::{Deserialize, Serialize};

use crate::error::{FwError, FwResult};
use crate::model::TranscriptionSegment;
use std::collections::{HashMap, VecDeque};

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
    /// Number of windows confirmed without correction.
    pub confirmations_emitted: u64,
    /// `corrections_emitted / windows_processed` (0.0 when no windows).
    pub correction_rate: f64,
    /// Mean latency of the fast model across all partials (ms).
    pub mean_fast_latency_ms: f64,
    /// Mean latency of the quality model across all processed windows (ms).
    pub mean_quality_latency_ms: f64,
    /// Current window size being used (ms).
    pub current_window_size_ms: u64,
    /// Mean approximate WER across all processed windows.
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

    fn create_window(
        &mut self,
        audio_position_ms: u64,
        end_ms: u64,
        audio_hash: &str,
    ) -> SpeculationWindow {
        let id = self.next_window_id;
        self.next_window_id += 1;

        let window = SpeculationWindow::new(
            id,
            self.run_id.clone(),
            audio_position_ms,
            end_ms,
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

    /// Create the next speculation window starting at `audio_position_ms`.
    pub fn next_window(&mut self, audio_position_ms: u64, audio_hash: &str) -> SpeculationWindow {
        let end_ms = audio_position_ms.saturating_add(self.window_size_ms);
        self.create_window(audio_position_ms, end_ms, audio_hash)
    }

    /// Create a window bounded by `max_end_ms` (useful for final-window truncation).
    ///
    /// Returns `None` when `audio_position_ms >= max_end_ms`, preventing
    /// zero-length windows from being created.
    pub fn next_window_bounded(
        &mut self,
        audio_position_ms: u64,
        max_end_ms: u64,
        audio_hash: &str,
    ) -> Option<SpeculationWindow> {
        if audio_position_ms >= max_end_ms {
            return None;
        }
        let natural_end = audio_position_ms.saturating_add(self.window_size_ms);
        let end_ms = natural_end.min(max_end_ms);
        if end_ms <= audio_position_ms {
            return None;
        }
        Some(self.create_window(audio_position_ms, end_ms, audio_hash))
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
    predictions: VecDeque<f64>,
    outcomes: VecDeque<bool>,
    window_size: usize,
}

impl CalibrationTracker {
    #[must_use]
    pub fn new(window_size: usize) -> Self {
        Self {
            predictions: VecDeque::new(),
            outcomes: VecDeque::new(),
            window_size,
        }
    }

    pub fn record(&mut self, predicted: f64, actual_correction: bool) {
        self.predictions.push_back(predicted);
        self.outcomes.push_back(actual_correction);
        if self.predictions.len() > self.window_size {
            self.predictions.pop_front();
            self.outcomes.pop_front();
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
                self.current_window_ms = (self.current_window_ms + delta).min(self.max_window_ms);
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

// ---------------------------------------------------------------------------
// bd-qlt.8: CorrectionEvidenceLedger — per-window evidence trail
// ---------------------------------------------------------------------------

/// A single evidence entry recording one speculation window decision.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorrectionEvidenceEntry {
    pub entry_id: u64,
    pub window_id: u64,
    pub run_id: String,
    pub timestamp_rfc3339: String,
    pub fast_model_id: String,
    pub fast_latency_ms: u64,
    pub fast_confidence_mean: f64,
    pub fast_segment_count: usize,
    pub quality_model_id: String,
    pub quality_latency_ms: u64,
    pub quality_confidence_mean: f64,
    pub quality_segment_count: usize,
    pub drift: CorrectionDrift,
    pub decision: String,
    pub window_size_ms: u64,
    pub correction_rate_at_decision: f64,
    pub controller_confidence: f64,
    pub fallback_active: bool,
    pub fallback_reason: Option<String>,
}

/// Bounded evidence ledger recording every speculation decision for
/// post-hoc analysis, adaptive tuning, and debugging.
pub struct CorrectionEvidenceLedger {
    entries: VecDeque<CorrectionEvidenceEntry>,
    capacity: usize,
    total_recorded: u64,
}

impl CorrectionEvidenceLedger {
    /// Create a new ledger with the given capacity (default: 500).
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        Self {
            entries: VecDeque::with_capacity(capacity),
            capacity,
            total_recorded: 0,
        }
    }

    /// Record an entry. Evicts the oldest if at capacity.
    pub fn record(&mut self, entry: CorrectionEvidenceEntry) {
        if self.capacity == 0 {
            self.total_recorded += 1;
            return;
        }
        if self.entries.len() >= self.capacity {
            self.entries.pop_front();
        }
        self.entries.push_back(entry);
        self.total_recorded += 1;
    }

    /// Total entries ever recorded (including evicted).
    #[must_use]
    pub fn total_recorded(&self) -> u64 {
        self.total_recorded
    }

    /// Current entries in the ledger.
    #[must_use]
    pub fn entries(&self) -> &VecDeque<CorrectionEvidenceEntry> {
        &self.entries
    }

    /// Export all entries as a JSON Value for `PipelineCx::record_evidence()`.
    #[must_use]
    pub fn to_evidence_json(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "correction_evidence_ledger",
            "total_recorded": self.total_recorded,
            "retained": self.entries.len(),
            "capacity": self.capacity,
            "diagnostics": self.diagnostics(),
            "entries": self.entries.iter().collect::<Vec<_>>(),
        })
    }

    /// Summary diagnostics as JSON.
    #[must_use]
    pub fn diagnostics(&self) -> serde_json::Value {
        serde_json::json!({
            "correction_rate": self.correction_rate(),
            "mean_fast_latency_ms": self.mean_fast_latency(),
            "mean_quality_latency_ms": self.mean_quality_latency(),
            "mean_wer": self.mean_wer(),
            "latency_savings_pct": self.latency_savings_pct(),
        })
    }

    /// Fraction of entries that were corrections.
    #[must_use]
    pub fn correction_rate(&self) -> f64 {
        if self.entries.is_empty() {
            return 0.0;
        }
        let corrections = self
            .entries
            .iter()
            .filter(|e| is_correction_decision(&e.decision))
            .count();
        corrections as f64 / self.entries.len() as f64
    }

    /// Mean fast model latency across entries.
    #[must_use]
    pub fn mean_fast_latency(&self) -> f64 {
        if self.entries.is_empty() {
            return 0.0;
        }
        let sum: u64 = self.entries.iter().map(|e| e.fast_latency_ms).sum();
        sum as f64 / self.entries.len() as f64
    }

    /// Mean quality model latency across entries.
    #[must_use]
    pub fn mean_quality_latency(&self) -> f64 {
        if self.entries.is_empty() {
            return 0.0;
        }
        let sum: u64 = self.entries.iter().map(|e| e.quality_latency_ms).sum();
        sum as f64 / self.entries.len() as f64
    }

    /// Mean WER across entries.
    #[must_use]
    pub fn mean_wer(&self) -> f64 {
        if self.entries.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.entries.iter().map(|e| e.drift.wer_approx).sum();
        sum / self.entries.len() as f64
    }

    /// Percentage of latency savings from showing fast results immediately.
    ///
    /// `(mean_quality - mean_fast) / mean_quality * 100`
    #[must_use]
    pub fn latency_savings_pct(&self) -> f64 {
        let q = self.mean_quality_latency();
        if q == 0.0 {
            return 0.0;
        }
        let f = self.mean_fast_latency();
        (q - f) / q * 100.0
    }

    /// Window size over time as `(window_id, window_size_ms)` pairs.
    #[must_use]
    pub fn window_size_trend(&self) -> Vec<(u64, u64)> {
        self.entries
            .iter()
            .map(|e| (e.window_id, e.window_size_ms))
            .collect()
    }
}

fn is_correction_decision(decision: &str) -> bool {
    matches!(
        decision.trim().to_ascii_lowercase().as_str(),
        "correct" | "corrected" | "correction"
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::TranscriptionSegment;

    fn seg(text: &str, confidence: Option<f64>) -> TranscriptionSegment {
        TranscriptionSegment {
            text: text.to_owned(),
            start_sec: None,
            end_sec: None,
            confidence,
            speaker: None,
        }
    }

    #[test]
    fn speculation_window_duration_and_contains_boundary() {
        let w = SpeculationWindow::new(0, "r".into(), 1000, 4000, 500, "h".into());
        assert_eq!(w.duration_ms(), 3000);
        // contains_ms: start inclusive, end exclusive.
        assert!(w.contains_ms(1000));
        assert!(w.contains_ms(3999));
        assert!(!w.contains_ms(4000));
        assert!(!w.contains_ms(999));
        // Degenerate: start == end → duration 0, contains nothing.
        let w2 = SpeculationWindow::new(1, "r".into(), 5000, 5000, 0, "h".into());
        assert_eq!(w2.duration_ms(), 0);
        assert!(!w2.contains_ms(5000));
        // Saturating sub when end < start (shouldn't happen, but safe).
        let w3 = SpeculationWindow::new(2, "r".into(), 100, 50, 0, "h".into());
        assert_eq!(w3.duration_ms(), 0);
    }

    #[test]
    fn correction_drift_identical_empty_and_divergent() {
        // Identical segments → wer 0, edit distance 0.
        let segs = vec![seg("hello world", Some(0.9))];
        let drift = CorrectionDrift::compute(&segs, &segs);
        assert_eq!(drift.wer_approx, 0.0);
        assert_eq!(drift.text_edit_distance, 0);
        assert_eq!(drift.segment_count_delta, 0);
        assert!(drift.confidence_delta < f64::EPSILON);

        // Empty vs empty → all zeros.
        let empty_drift = CorrectionDrift::compute(&[], &[]);
        assert_eq!(empty_drift.text_edit_distance, 0);
        assert_eq!(empty_drift.segment_count_delta, 0);

        // Divergent segments → positive WER.
        let fast = vec![seg("the cat sat", Some(0.8))];
        let quality = vec![seg("a dog ran", Some(0.95))];
        let d = CorrectionDrift::compute(&fast, &quality);
        assert!(d.wer_approx > 0.0);
        assert!(d.text_edit_distance > 0);
        assert!((d.confidence_delta - 0.15).abs() < f64::EPSILON);
    }

    #[test]
    fn beta_posterior_prior_updates_and_variance() {
        let p = BetaPosterior::weakly_informative();
        assert!((p.mean() - 0.5).abs() < f64::EPSILON);
        let var_initial = p.variance();

        // Corrections shift mean up (towards 1).
        let mut p2 = BetaPosterior::weakly_informative();
        for _ in 0..10 {
            p2.observe_correction();
        }
        assert!(p2.mean() > 0.5);
        assert!(p2.variance() < var_initial);

        // Confirmations shift mean down (towards 0).
        let mut p3 = BetaPosterior::weakly_informative();
        for _ in 0..10 {
            p3.observe_confirmation();
        }
        assert!(p3.mean() < 0.5);
    }

    #[test]
    fn calibration_tracker_brier_score_and_eviction() {
        // Perfect predictions: predicted=1.0 when correction, 0.0 when confirm → Brier=0.
        let mut ct = CalibrationTracker::new(10);
        ct.record(1.0, true);
        ct.record(0.0, false);
        assert!(ct.brier_score() < f64::EPSILON);
        assert_eq!(ct.sample_count(), 2);

        // Worst predictions: predicted=0.0 when correction, 1.0 when confirm → Brier=1.
        let mut ct2 = CalibrationTracker::new(10);
        ct2.record(0.0, true);
        ct2.record(1.0, false);
        assert!((ct2.brier_score() - 1.0).abs() < f64::EPSILON);

        // Eviction: window_size=3, add 5 entries, only 3 remain.
        let mut ct3 = CalibrationTracker::new(3);
        for i in 0..5 {
            ct3.record(0.5, i % 2 == 0);
        }
        assert_eq!(ct3.sample_count(), 3);

        // Empty tracker → Brier 0.
        let ct4 = CalibrationTracker::new(5);
        assert_eq!(ct4.brier_score(), 0.0);
    }

    #[test]
    fn evidence_ledger_capacity_zero_eviction_and_diagnostics() {
        // Capacity 0: records total but stores nothing.
        let mut ledger = CorrectionEvidenceLedger::new(0);
        let entry = CorrectionEvidenceEntry {
            entry_id: 0,
            window_id: 0,
            run_id: "r".into(),
            timestamp_rfc3339: "2025-01-01T00:00:00Z".into(),
            fast_model_id: "tiny".into(),
            fast_latency_ms: 100,
            fast_confidence_mean: 0.8,
            fast_segment_count: 1,
            quality_model_id: "large".into(),
            quality_latency_ms: 500,
            quality_confidence_mean: 0.95,
            quality_segment_count: 1,
            drift: CorrectionDrift {
                wer_approx: 0.1,
                confidence_delta: 0.15,
                segment_count_delta: 0,
                text_edit_distance: 5,
            },
            decision: "confirm".into(),
            window_size_ms: 3000,
            correction_rate_at_decision: 0.0,
            controller_confidence: 0.5,
            fallback_active: false,
            fallback_reason: None,
        };
        ledger.record(entry.clone());
        assert_eq!(ledger.total_recorded(), 1);
        assert!(ledger.entries().is_empty());

        // Normal capacity with eviction.
        let mut ledger2 = CorrectionEvidenceLedger::new(2);
        let mut e = entry;
        for i in 0..3 {
            e.entry_id = i;
            e.fast_latency_ms = 100;
            e.quality_latency_ms = 500;
            e.decision = if i == 1 {
                "correct".into()
            } else {
                "confirm".into()
            };
            ledger2.record(e.clone());
        }
        assert_eq!(ledger2.total_recorded(), 3);
        assert_eq!(ledger2.entries().len(), 2);
        // Oldest (entry_id=0) was evicted; entry_id=1,2 remain.
        assert_eq!(ledger2.entries()[0].entry_id, 1);

        // Diagnostics: 1 correction out of 2 retained = 50%.
        assert!((ledger2.correction_rate() - 0.5).abs() < f64::EPSILON);
        assert!((ledger2.mean_fast_latency() - 100.0).abs() < f64::EPSILON);
        assert!((ledger2.mean_quality_latency() - 500.0).abs() < f64::EPSILON);
        // Latency savings: (500-100)/500*100 = 80%.
        assert!((ledger2.latency_savings_pct() - 80.0).abs() < f64::EPSILON);
    }

    // ── Task #206 — speculation pass 2 edge-case tests ───────────────

    #[test]
    fn window_manager_next_window_bounded_none_and_clamped() {
        let mut mgr = WindowManager::new("run-bounded", 5000, 500);

        // audio_position >= max_end → None.
        assert!(mgr.next_window_bounded(10_000, 10_000, "h").is_none());
        assert!(mgr.next_window_bounded(10_001, 10_000, "h").is_none());

        // natural_end (3000 + 5000 = 8000) > max_end (6000) → clamped to 6000.
        let w = mgr.next_window_bounded(3000, 6000, "h").unwrap();
        assert_eq!(w.end_ms, 6000);
        assert_eq!(w.duration_ms(), 3000); // 6000 - 3000

        // Happy path: natural_end fits within max_end.
        let w2 = mgr.next_window_bounded(0, 100_000, "h").unwrap();
        assert_eq!(w2.duration_ms(), 5000);
    }

    #[test]
    fn window_manager_set_window_size_clamps() {
        let mut mgr = WindowManager::new("run-clamp", 5000, 500);
        // min_window_ms = 1000, max_window_ms = 30000.

        // Below min → clamped to 1000.
        mgr.set_window_size(0);
        assert_eq!(mgr.current_window_size(), 1000);

        // Above max → clamped to 30000.
        mgr.set_window_size(u64::MAX);
        assert_eq!(mgr.current_window_size(), 30_000);

        // Within range → passes through.
        mgr.set_window_size(15_000);
        assert_eq!(mgr.current_window_size(), 15_000);
    }

    #[test]
    fn correction_tracker_unregistered_window_returns_error() {
        let mut tracker = CorrectionTracker::new(CorrectionTolerance::default());

        // submit_quality_result with no registered partial → Err.
        let result = tracker.submit_quality_result(
            999, // no partial for this window_id
            "quality-model",
            vec![seg("hello", Some(0.9))],
            100,
        );
        let err = match result {
            Err(e) => e,
            Ok(_) => panic!("expected error for unregistered window_id"),
        };
        let msg = err.to_string();
        assert!(
            msg.contains("no partial registered"),
            "error should mention missing partial: {msg}"
        );
    }

    #[test]
    fn correction_tracker_all_resolved_transitions() {
        let mut tracker = CorrectionTracker::new(CorrectionTolerance::default());

        // No partials → all_resolved() is vacuously true.
        assert!(tracker.all_resolved());

        // Register a partial → not all resolved (Pending).
        let partial = PartialTranscript::new(
            0,
            0,
            "fast".to_owned(),
            vec![seg("hello", Some(0.9))],
            50,
            "2026-01-01T00:00:00Z".to_owned(),
        );
        tracker.register_partial(partial);
        assert!(!tracker.all_resolved(), "pending partial → not resolved");

        // Submit matching quality result → Confirm → all resolved.
        let decision = tracker
            .submit_quality_result(0, "quality", vec![seg("hello", Some(0.9))], 100)
            .unwrap();
        let is_confirm = matches!(decision, CorrectionDecision::Confirm { .. });
        assert!(is_confirm, "expected Confirm decision");
        assert!(tracker.all_resolved());
    }

    #[test]
    fn is_correction_decision_all_variants_and_normalization() {
        assert!(is_correction_decision("correct"));
        assert!(is_correction_decision("corrected"));
        assert!(is_correction_decision("correction"));
        // Normalization: trim + lowercase.
        assert!(is_correction_decision(" CORRECTED "));
        assert!(is_correction_decision("\tCorrection\n"));
        // Non-corrections.
        assert!(!is_correction_decision("confirm"));
        assert!(!is_correction_decision("confirmed"));
        assert!(!is_correction_decision(""));
        assert!(!is_correction_decision("   "));
    }
}
