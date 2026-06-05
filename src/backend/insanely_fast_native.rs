//! Real native insanely-fast engine (bd-s8w8).
//!
//! This module is the in-process, pure-Rust insanely-fast-whisper engine. Like
//! [`whisper_cpp_native`](super::whisper_cpp_native) it runs **genuine** ASR
//! inference through [`crate::native_engine`] (real ggml weights, real log-mel
//! frontend, real encoder/decoder forward passes) — there are **no canned
//! phrases, no mock segmentation, and no subprocess execution**. The former
//! "pilot" that fabricated deterministic phrases from audio-energy regions is
//! gone.
//!
//! ## The insanely-fast identity: throughput via parallel windows
//!
//! [`whisper_cpp_native`](super::whisper_cpp_native) calls
//! [`decode::transcribe_samples`] once over the whole clip; that function
//! windows the audio into 30 s chunks and decodes them **sequentially**,
//! carrying whisper's seek-continuation and a rolling text prompt across window
//! boundaries.
//!
//! This engine trades that continuity for **throughput**: it splits the audio
//! into hard 30 s windows up front and decodes **multiple windows in parallel**
//! across a worker pool, then merges the per-window transcripts in window order
//! (offsetting each window's timestamps by `window_idx * 30 s`). This mirrors
//! how the real `insanely-fast-whisper` batches chunked audio and merges the
//! results.
//!
//! ### Honest tradeoff
//!
//! Hard 30 s boundaries lose whisper's seek-continuation: a word straddling a
//! boundary may be clipped or duplicated, and each window decodes with no prior
//! text context (no rolling prompt). That is the documented behavior of chunked
//! batched inference (`insanely-fast-whisper` chunks + merges identically). When
//! cross-window continuity matters more than throughput, use the sequential
//! [`whisper_cpp_native`](super::whisper_cpp_native) engine instead.
//!
//! ### Determinism
//!
//! Output is **identical regardless of worker count**. Each window is decoded
//! independently (greedy / temperature-0, shared read-only weights) and the
//! windows are merged in deterministic window order, so the worker count only
//! affects wall-clock time, never the transcript or timestamps. The gated tests
//! assert 2-worker output equals 1-worker output exactly.
//!
//! ## Model resolution, silence pre-gate, availability, word timestamps
//!
//! These policies are identical to
//! [`whisper_cpp_native`](super::whisper_cpp_native): the model is resolved from
//! `request.model` then `$FRANKEN_WHISPER_NATIVE_DEFAULT_MODEL`; a cheap energy
//! pre-gate returns an empty result for pure silence without loading any
//! weights; availability is the same honest default-or-scan model-file check;
//! and word-level timestamps reuse that module's interpolated splitting helpers.

use std::path::Path;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use serde_json::{Value, json};

use crate::error::{FwError, FwResult};
use crate::model::{BackendKind, TranscribeRequest, TranscriptionResult, TranscriptionSegment};
use crate::native_engine::mel::N_SAMPLES_30S;
use crate::native_engine::{self, NativeWhisperModel, decode};

use super::native_audio::analyze_wav;
use super::whisper_cpp_native::{WordTimestampMode, build_segments, word_timestamp_mode};

/// Stable schema tag for the honest native raw-output metadata (shared with the
/// whisper.cpp native engine so the evidence ledger has one schema).
const SCHEMA_VERSION: &str = "native-v2";

/// Default `batch_size` (max concurrent windows) when the request does not set
/// one — matches the historical insanely-fast default.
const DEFAULT_BATCH_SIZE: usize = 24;

/// One 30 s window in seconds, used to offset each window's timestamps back into
/// whole-clip time when merging.
const WINDOW_SECONDS: f64 = 30.0;

/// Honestly report whether the native insanely-fast engine can run.
///
/// Shares [`whisper_cpp_native::is_available`](super::whisper_cpp_native::is_available)'s
/// exact policy: both engines run the **same** [`crate::native_engine`] over the
/// **same** ggml model files, so they must agree on availability. Reports `true`
/// only when a usable model header exists (the configured default resolves, or
/// any `ggml-*.bin` with a valid header sits in a search dir). Never panics or
/// performs network access.
#[must_use]
pub fn is_available() -> bool {
    super::whisper_cpp_native::is_available()
}

/// Whether a HuggingFace token is present for the given request (diarization
/// gate, unchanged from the bridge contract).
pub(crate) fn hf_token_present_for_request(request: &TranscribeRequest) -> bool {
    super::insanely_fast::hf_token_present_for_request(request)
}

/// Resolve the effective model spec for a request, or a [`FwError`] explaining
/// how to provision one. Identical precedence to the whisper.cpp native engine:
/// `request.model` then `$FRANKEN_WHISPER_NATIVE_DEFAULT_MODEL`, else an
/// actionable [`FwError::BackendUnavailable`] naming the env var and reusing the
/// resolver's search-dir listing.
fn effective_model_spec(request: &TranscribeRequest) -> FwResult<String> {
    if let Some(model) = request.model.clone().filter(|m| !m.is_empty()) {
        return Ok(model);
    }
    if let Some(spec) = native_engine::default_model_spec() {
        return Ok(spec);
    }
    let dirs_hint = native_engine::resolve_model("default")
        .err()
        .map(|e| e.to_string())
        .unwrap_or_default();
    Err(FwError::BackendUnavailable(format!(
        "native insanely-fast engine has no model: pass --model, or set \
         $FRANKEN_WHISPER_NATIVE_DEFAULT_MODEL to a model short-name or path. \
         {dirs_hint}"
    )))
}

/// Split `samples` into fixed 30 s windows (`N_SAMPLES_30S` each). The last
/// window keeps the remainder (it may be shorter than 30 s). A clip shorter than
/// one window yields a single window. Empty input yields no windows.
fn split_windows(samples: &[f32]) -> Vec<&[f32]> {
    if samples.is_empty() {
        return Vec::new();
    }
    samples.chunks(N_SAMPLES_30S).collect()
}

/// Worker-pool sizing: pure, tested. Returns `(n_workers, threads_per_worker)`.
///
/// - `n_workers = min(n_windows, batch_size, max(1, total_threads / 2))` — never
///   more workers than windows, never more than the `batch_size` concurrency
///   cap, and never more than half the thread budget (each worker wants ≥ 2
///   intra-op threads where possible).
/// - `threads_per_worker = max(1, total_threads / n_workers)` — the intra-op
///   thread budget split evenly across workers, floored at 1.
///
/// Every branch is guarded so a zero or absurd input never yields a zero worker
/// count or a zero thread count (which would deadlock or divide-by-zero).
pub(crate) fn plan_workers(
    n_windows: usize,
    batch_size: Option<usize>,
    total_threads: usize,
) -> (usize, usize) {
    let batch = batch_size.unwrap_or(DEFAULT_BATCH_SIZE).max(1);
    let total_threads = total_threads.max(1);
    let half = (total_threads / 2).max(1);
    let n_workers = n_windows.max(1).min(batch).min(half);
    let n_workers = n_workers.max(1);
    let threads_per_worker = (total_threads / n_workers).max(1);
    (n_workers, threads_per_worker)
}

/// Resolve the total intra-op thread budget for a request (request override,
/// else the engine default), mirroring the whisper.cpp native engine.
fn total_threads_for(request: &TranscribeRequest) -> usize {
    request
        .backend_params
        .threads
        .map_or_else(native_engine::default_threads, |t| {
            usize::try_from(t).unwrap_or_else(|_| native_engine::default_threads())
        })
}

/// Build the per-worker [`decode::DecodeParams`] for a request, with the
/// caller-chosen `threads_per_worker` (the rest of the params are window-shape
/// independent).
fn decode_params(request: &TranscribeRequest, threads_per_worker: usize) -> decode::DecodeParams {
    decode::DecodeParams {
        language: request.language.clone(),
        translate: request.translate,
        timestamps: !request.backend_params.no_timestamps,
        n_threads: threads_per_worker,
        max_text_ctx: None,
        ..decode::DecodeParams::default()
    }
}

/// One window's decode result, tagged with its window index for ordered merge.
struct WindowResult {
    index: usize,
    output: decode::DecodeOutput,
}

/// Run real native insanely-fast (parallel-window) inference over
/// `normalized_wav` (guaranteed 16 kHz mono PCM16 by the pipeline) and return a
/// [`TranscriptionResult`].
///
/// See the module docs for the parallel-window strategy, determinism guarantee,
/// and shared model-resolution / silence-pre-gate / word-timestamp policies.
///
/// # Errors
///
/// - [`FwError::BackendUnavailable`] when no model can be resolved (or when
///   diarization is requested without a HuggingFace token).
/// - [`FwError::Io`] / [`FwError::InvalidRequest`] when the WAV cannot be read.
/// - [`FwError::Cancelled`] when the cancellation token's deadline expires
///   (propagated into every worker; the first error wins).
/// - Whatever model-load or decode errors the native engine surfaces.
pub fn run(
    request: &TranscribeRequest,
    normalized_wav: &Path,
    _work_dir: &Path,
    _timeout: Duration,
    token: Option<&crate::orchestrator::CancellationToken>,
) -> FwResult<TranscriptionResult> {
    if request.diarize && !hf_token_present_for_request(request) {
        return Err(FwError::BackendUnavailable(
            "diarization requires HF token (`--hf-token` or env `FRANKEN_WHISPER_HF_TOKEN` / `HF_TOKEN`)"
                .to_owned(),
        ));
    }
    if let Some(tok) = token {
        tok.checkpoint()?;
    }

    // Resolve the model spec up front so an unavailability error is reported
    // before any expensive work.
    let spec = effective_model_spec(request)?;

    // Silence pre-gate: cheap energy analysis avoids a multi-GB model load on a
    // pure-silence clip (shared policy with whisper.cpp native).
    let analysis = analyze_wav(normalized_wav, request.backend_params.duration_ms).ok();
    if let Some(analysis) = analysis.as_ref()
        && analysis.active_regions.is_empty()
    {
        return Ok(silence_result(request, &spec, analysis.duration_ms));
    }

    if let Some(tok) = token {
        tok.checkpoint()?;
    }

    // Resolve + load the model (cached, reference-counted, read-only weights).
    let model_path = native_engine::resolve_model(&spec)
        .map_err(|e| FwError::BackendUnavailable(e.to_string()))?;
    let model = NativeWhisperModel::load(&model_path)?;

    let samples = read_normalized_wav(normalized_wav)?;

    if let Some(tok) = token {
        tok.checkpoint()?;
    }

    // Split into 30 s windows and size the worker pool.
    let windows = split_windows(&samples);
    let batch_size = request
        .backend_params
        .batch_size
        .and_then(|v| usize::try_from(v).ok());
    let total_threads = total_threads_for(request);
    let (n_workers, threads_per_worker) = plan_workers(windows.len(), batch_size, total_threads);

    let params = decode_params(request, threads_per_worker);

    // Decode every window (parallel across workers); merge in window order.
    let window_results = decode_windows_parallel(&model, &windows, &params, n_workers, token)?;

    let merged = merge_windows(&window_results);

    let word_mode = word_timestamp_mode(request.backend_params.word_timestamps.as_ref());
    let segments = build_segments(
        &merged.segments,
        word_mode,
        request.backend_params.split_on_word,
        request.backend_params.no_timestamps,
        token,
    )?;
    let transcript = super::transcript_from_segments(&segments);
    let language = merged.language.or_else(|| request.language.clone());

    let raw_output = raw_output_json(
        &spec,
        &model_path,
        model.version_tag(),
        windows.len(),
        n_workers,
        threads_per_worker,
        &merged.windows,
        word_mode,
        request.backend_params.split_on_word,
        false,
    );

    Ok(TranscriptionResult {
        backend: BackendKind::InsanelyFast,
        transcript,
        language,
        segments,
        acceleration: None,
        raw_output,
        artifact_paths: Vec::new(),
    })
}

/// Decode every window, distributing them round-robin across `n_workers`
/// scoped threads, each running [`NativeWhisperModel::transcribe`] on the
/// **shared** `Arc<NativeWhisperModel>` (weights are read-only). Returns the
/// per-window outputs in arbitrary order (the caller sorts by index).
///
/// Cancellation + first-error: a shared [`AtomicBool`] short-circuits remaining
/// windows the moment any window errors (or the token expires); the first error
/// observed is returned. Per-window independence + the shared-weights model make
/// the merged result identical regardless of `n_workers`.
fn decode_windows_parallel(
    model: &NativeWhisperModel,
    windows: &[&[f32]],
    params: &decode::DecodeParams,
    n_workers: usize,
    token: Option<&crate::orchestrator::CancellationToken>,
) -> FwResult<Vec<WindowResult>> {
    let stop = AtomicBool::new(false);
    let first_error: Mutex<Option<FwError>> = Mutex::new(None);
    let results: Mutex<Vec<WindowResult>> = Mutex::new(Vec::with_capacity(windows.len()));

    std::thread::scope(|scope| {
        for worker_id in 0..n_workers {
            let stop = &stop;
            let first_error = &first_error;
            let results = &results;
            scope.spawn(move || {
                // A worker's checkpoint: honor the orchestrator token AND the
                // shared stop flag so a sibling's error/cancellation halts every
                // worker promptly.
                let checkpoint = || -> FwResult<()> {
                    if stop.load(Ordering::Relaxed) {
                        return Err(FwError::Cancelled(
                            "insanely-fast worker stopped".to_owned(),
                        ));
                    }
                    token.map_or(Ok(()), crate::orchestrator::CancellationToken::checkpoint)
                };
                // Round-robin window assignment: worker w handles windows
                // w, w + n_workers, w + 2*n_workers, ...
                let mut idx = worker_id;
                while idx < windows.len() {
                    if stop.load(Ordering::Relaxed) {
                        return;
                    }
                    match model.transcribe(windows[idx], params, &checkpoint) {
                        Ok(output) => {
                            results
                                .lock()
                                .unwrap_or_else(|e| e.into_inner())
                                .push(WindowResult { index: idx, output });
                        }
                        Err(err) => {
                            // First error wins; signal every worker to stop.
                            let mut slot = first_error.lock().unwrap_or_else(|e| e.into_inner());
                            if slot.is_none() {
                                *slot = Some(err);
                            }
                            stop.store(true, Ordering::Relaxed);
                            return;
                        }
                    }
                    idx += n_workers;
                }
            });
        }
    });

    if let Some(err) = first_error.into_inner().unwrap_or_else(|e| e.into_inner()) {
        return Err(err);
    }

    let mut results = results.into_inner().unwrap_or_else(|e| e.into_inner());
    results.sort_by_key(|r| r.index);
    Ok(results)
}

/// The deterministic merge of per-window outputs into a whole-clip result.
struct MergedOutput {
    segments: Vec<TranscriptionSegment>,
    windows: Vec<decode::WindowStats>,
    language: Option<String>,
}

/// Merge per-window decode outputs in window order. Each window's
/// segment/window-stat timestamps are offset by `window_idx * 30 s` to map them
/// back into whole-clip time. The detected language (if any) is taken from the
/// first window that reports one.
///
/// This is the determinism linchpin: because windows are independent and merged
/// strictly by index, the merged result does not depend on which worker decoded
/// which window or in what order they finished.
fn merge_windows(results: &[WindowResult]) -> MergedOutput {
    let mut segments = Vec::new();
    let mut windows = Vec::new();
    let mut language = None;

    for result in results {
        let offset = result.index as f64 * WINDOW_SECONDS;
        if language.is_none() {
            language = result.output.language.clone();
        }
        for seg in &result.output.segments {
            segments.push(TranscriptionSegment {
                start_sec: seg.start_sec.map(|s| s + offset),
                end_sec: seg.end_sec.map(|s| s + offset),
                text: seg.text.clone(),
                speaker: seg.speaker.clone(),
                confidence: seg.confidence,
            });
        }
        for win in &result.output.windows {
            windows.push(decode::WindowStats {
                avg_logprob: win.avg_logprob,
                no_speech_prob: win.no_speech_prob,
                tokens: win.tokens,
                window_offset_sec: win.window_offset_sec + offset,
            });
        }
    }

    MergedOutput {
        segments,
        windows,
        language,
    }
}

/// Read a normalized 16 kHz mono PCM16 WAV into f32 mono samples (shares the
/// engine's RIFF reader so production and gated e2e tests use one decoder).
fn read_normalized_wav(path: &Path) -> FwResult<Vec<f32>> {
    let bytes = std::fs::read(path)?;
    decode::read_wav_16k_mono(&bytes)
}

/// The honest raw-output metadata JSON for a real-inference parallel-window run.
#[allow(clippy::too_many_arguments)]
fn raw_output_json(
    spec: &str,
    model_path: &Path,
    version_tag: String,
    parallel_windows: usize,
    workers: usize,
    threads_per_worker: usize,
    windows: &[decode::WindowStats],
    word_mode: WordTimestampMode,
    split_on_word: bool,
    silence: bool,
) -> Value {
    let word_timestamps = if word_mode != WordTimestampMode::None || split_on_word {
        "interpolated"
    } else {
        "none"
    };
    let windows_json: Vec<Value> = windows
        .iter()
        .map(|w| {
            json!({
                "window_offset_sec": w.window_offset_sec,
                "tokens": w.tokens,
                "avg_logprob": w.avg_logprob,
                "no_speech_prob": w.no_speech_prob,
            })
        })
        .collect();
    json!({
        "engine": "insanely-fast-native",
        "schema_version": SCHEMA_VERSION,
        "in_process": true,
        "implementation": "real-inference",
        "silence": silence,
        "parallel_windows": parallel_windows,
        "workers": workers,
        "threads_per_worker": threads_per_worker,
        "model": spec,
        "model_path": model_path.display().to_string(),
        "model_version_tag": version_tag,
        "windows": windows_json,
        "word_timestamps": word_timestamps,
    })
}

/// Build the empty-but-valid result for a pure-silence clip, taken **without
/// loading the model** (the energy pre-gate already proved there is nothing to
/// transcribe).
fn silence_result(
    request: &TranscribeRequest,
    spec: &str,
    duration_ms: u64,
) -> TranscriptionResult {
    TranscriptionResult {
        backend: BackendKind::InsanelyFast,
        transcript: String::new(),
        language: request.language.clone(),
        segments: Vec::new(),
        acceleration: None,
        raw_output: json!({
            "engine": "insanely-fast-native",
            "schema_version": SCHEMA_VERSION,
            "in_process": true,
            "implementation": "real-inference",
            "silence": true,
            "parallel_windows": 0,
            "workers": 0,
            "threads_per_worker": 0,
            "model": spec,
            "model_loaded": false,
            "duration_ms": duration_ms,
            "windows": [],
            "word_timestamps": "none",
        }),
        artifact_paths: Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use std::path::{Path, PathBuf};
    use std::time::Duration;

    use crate::backend::Engine;
    use crate::model::{
        BackendKind, BackendParams, InputSource, TranscribeRequest, TranscriptionSegment,
    };
    use crate::native_engine::mel::SAMPLE_RATE;
    use crate::orchestrator::CancellationToken;

    use super::*;

    fn request() -> TranscribeRequest {
        TranscribeRequest {
            input: InputSource::File {
                path: PathBuf::from("input.wav"),
            },
            backend: BackendKind::InsanelyFast,
            model: Some("tiny.en".to_owned()),
            language: Some("en".to_owned()),
            translate: false,
            diarize: false,
            persist: false,
            db_path: PathBuf::from("state.sqlite3"),
            timeout_ms: None,
            backend_params: BackendParams::default(),
        }
    }

    fn write_pcm16_mono_wav(path: &Path, sample_rate: u32, samples: &[i16]) {
        let data_len = (samples.len() * 2) as u32;
        let mut bytes = Vec::with_capacity(44 + data_len as usize);
        bytes.extend_from_slice(b"RIFF");
        bytes.extend_from_slice(&(36u32 + data_len).to_le_bytes());
        bytes.extend_from_slice(b"WAVE");
        bytes.extend_from_slice(b"fmt ");
        bytes.extend_from_slice(&16u32.to_le_bytes());
        bytes.extend_from_slice(&1u16.to_le_bytes());
        bytes.extend_from_slice(&1u16.to_le_bytes());
        bytes.extend_from_slice(&sample_rate.to_le_bytes());
        bytes.extend_from_slice(&(sample_rate * 2).to_le_bytes());
        bytes.extend_from_slice(&2u16.to_le_bytes());
        bytes.extend_from_slice(&16u16.to_le_bytes());
        bytes.extend_from_slice(b"data");
        bytes.extend_from_slice(&data_len.to_le_bytes());
        for sample in samples {
            bytes.extend_from_slice(&sample.to_le_bytes());
        }
        std::fs::write(path, bytes).expect("write wav");
    }

    // ── Engine-trait shape ────────────────────────────────────────────────

    #[test]
    fn native_engine_name_follows_naming_convention() {
        let engine = super::super::InsanelyFastNativeEngine;
        assert_eq!(engine.name(), "insanely-fast-native");
    }

    #[test]
    fn native_engine_kind_matches_bridge_adapter() {
        let native = super::super::InsanelyFastNativeEngine;
        let bridge = super::super::InsanelyFastEngine;
        assert_eq!(native.kind(), bridge.kind());
        assert_eq!(native.kind(), BackendKind::InsanelyFast);
    }

    #[test]
    fn native_engine_name_distinct_from_bridge() {
        let native = super::super::InsanelyFastNativeEngine;
        let bridge = super::super::InsanelyFastEngine;
        assert_ne!(native.name(), bridge.name());
        assert!(native.name().contains("native"));
    }

    #[test]
    fn availability_agrees_with_whisper_cpp_native() {
        // Both engines run the same native engine over the same model files, so
        // their availability must be identical.
        assert_eq!(
            is_available(),
            super::super::whisper_cpp_native::is_available()
        );
    }

    // ── plan_workers pure-function matrix ─────────────────────────────────

    #[test]
    fn plan_workers_typical_split() {
        // 16 threads, plenty of windows, default batch: workers = min(8, 24, 8) = 8;
        // threads_per_worker = 16 / 8 = 2.
        let (w, t) = plan_workers(10, None, 16);
        assert_eq!(w, 8);
        assert_eq!(t, 2);
    }

    #[test]
    fn plan_workers_one_window() {
        // A single window => exactly one worker gets the whole thread budget.
        let (w, t) = plan_workers(1, None, 16);
        assert_eq!(w, 1);
        assert_eq!(t, 16);
    }

    #[test]
    fn plan_workers_threads_fewer_than_workers() {
        // 3 threads, many windows: half = max(3/2,1) = 1 => 1 worker, 3 threads.
        let (w, t) = plan_workers(50, None, 3);
        assert_eq!(w, 1);
        assert_eq!(t, 3);
    }

    #[test]
    fn plan_workers_batch_size_one_serializes() {
        // batch_size = 1 caps concurrency to one worker regardless of windows.
        let (w, t) = plan_workers(20, Some(1), 32);
        assert_eq!(w, 1);
        assert_eq!(t, 32);
    }

    #[test]
    fn plan_workers_zero_threads_guarded() {
        // Zero threads must never yield a zero worker or zero thread count.
        let (w, t) = plan_workers(8, None, 0);
        assert!(w >= 1, "workers floored at 1, got {w}");
        assert!(t >= 1, "threads_per_worker floored at 1, got {t}");
    }

    #[test]
    fn plan_workers_zero_windows_guarded() {
        // Defensive: zero windows still floors workers at 1 (run() never calls
        // with zero, but the pure fn must not divide by zero).
        let (w, t) = plan_workers(0, None, 8);
        assert!(w >= 1);
        assert!(t >= 1);
    }

    #[test]
    fn plan_workers_batch_caps_below_threads() {
        // batch_size limits workers even when threads/windows would allow more.
        let (w, t) = plan_workers(100, Some(4), 64);
        assert_eq!(w, 4);
        assert_eq!(t, 16);
    }

    // ── split_windows ─────────────────────────────────────────────────────

    #[test]
    fn split_windows_empty_is_none() {
        assert!(split_windows(&[]).is_empty());
    }

    #[test]
    fn split_windows_short_clip_single_window() {
        let samples = vec![0.1f32; SAMPLE_RATE]; // 1 s < 30 s.
        let w = split_windows(&samples);
        assert_eq!(w.len(), 1);
        assert_eq!(w[0].len(), SAMPLE_RATE);
    }

    #[test]
    fn split_windows_last_keeps_remainder() {
        // 30 s + 5 s => two windows: full + remainder.
        let samples = vec![0.1f32; N_SAMPLES_30S + 5 * SAMPLE_RATE];
        let w = split_windows(&samples);
        assert_eq!(w.len(), 2);
        assert_eq!(w[0].len(), N_SAMPLES_30S);
        assert_eq!(w[1].len(), 5 * SAMPLE_RATE);
    }

    // ── merge_windows offsets timestamps by window index ──────────────────

    fn out_with_segment(start: f64, end: f64, text: &str, offset_sec: f64) -> decode::DecodeOutput {
        decode::DecodeOutput {
            segments: vec![TranscriptionSegment {
                start_sec: Some(start),
                end_sec: Some(end),
                text: text.to_owned(),
                speaker: None,
                confidence: Some(0.9),
            }],
            language: Some("en".to_owned()),
            windows: vec![decode::WindowStats {
                avg_logprob: -0.2,
                no_speech_prob: 0.01,
                tokens: 3,
                window_offset_sec: offset_sec,
            }],
            word_timings: None,
        }
    }

    #[test]
    fn merge_windows_offsets_by_30s_per_index() {
        let results = vec![
            WindowResult {
                index: 0,
                output: out_with_segment(0.0, 5.0, "first", 0.0),
            },
            WindowResult {
                index: 1,
                output: out_with_segment(1.0, 4.0, "second", 0.0),
            },
        ];
        let merged = merge_windows(&results);
        assert_eq!(merged.segments.len(), 2);
        // Window 1's segment is offset by 30 s.
        assert_eq!(merged.segments[0].start_sec, Some(0.0));
        assert_eq!(merged.segments[1].start_sec, Some(31.0));
        assert_eq!(merged.segments[1].end_sec, Some(34.0));
        // Window stats offset too.
        assert_eq!(merged.windows[1].window_offset_sec, 30.0);
        assert_eq!(merged.language, Some("en".to_owned()));
    }

    // ── Model resolution / availability ───────────────────────────────────

    #[test]
    fn run_without_model_or_default_is_backend_unavailable() {
        let mut req = request();
        req.model = None;
        if native_engine::default_model_spec().is_some() {
            return;
        }
        let dir = tempfile::tempdir().expect("tempdir");
        let wav = dir.path().join("a.wav");
        let mut samples = vec![0i16; 1_600];
        samples.extend((0..16_000).map(|i| if i % 2 == 0 { 9_000i16 } else { -9_000 }));
        write_pcm16_mono_wav(&wav, 16_000, &samples);

        let err = run(&req, &wav, dir.path(), Duration::from_secs(1), None)
            .expect_err("no model => unavailable");
        assert!(matches!(err, FwError::BackendUnavailable(_)));
        let msg = err.to_string();
        assert!(
            msg.contains("FRANKEN_WHISPER_NATIVE_DEFAULT_MODEL"),
            "message names the env var: {msg}"
        );
    }

    #[test]
    fn run_nonexistent_model_spec_is_backend_unavailable_with_dirs() {
        let mut req = request();
        req.model = Some("definitely-not-a-real-model-zzz".to_owned());
        let dir = tempfile::tempdir().expect("tempdir");
        let wav = dir.path().join("a.wav");
        let mut samples = vec![0i16; 1_600];
        samples.extend((0..16_000).map(|i| if i % 2 == 0 { 9_000i16 } else { -9_000 }));
        write_pcm16_mono_wav(&wav, 16_000, &samples);

        let err = run(&req, &wav, dir.path(), Duration::from_secs(1), None)
            .expect_err("missing model => unavailable");
        assert!(matches!(err, FwError::BackendUnavailable(_)));
        let msg = err.to_string();
        assert!(
            msg.contains("ggml-definitely-not-a-real-model-zzz.bin"),
            "message names the searched filename: {msg}"
        );
    }

    // ── Diarization gate ──────────────────────────────────────────────────

    #[test]
    fn run_diarize_without_token_fails() {
        let dir = tempfile::tempdir().expect("tempdir");
        let wav = dir.path().join("tone.wav");
        let mut samples = vec![0i16; 1_600];
        samples.extend((0..16_000).map(|i| if i % 2 == 0 { 9_000i16 } else { -9_000 }));
        write_pcm16_mono_wav(&wav, 16_000, &samples);

        let mut req = request();
        req.diarize = true;
        let result = run(&req, &wav, dir.path(), Duration::from_secs(1), None);
        assert!(result.is_err(), "diarize without HF token should fail");
    }

    // ── Silence pre-gate ──────────────────────────────────────────────────

    #[test]
    fn run_pure_silence_returns_empty_without_loading_model() {
        let dir = tempfile::tempdir().expect("tempdir");
        let wav = dir.path().join("silence.wav");
        write_pcm16_mono_wav(&wav, 16_000, &vec![0i16; 16_000]);

        let req = request();
        let start = std::time::Instant::now();
        let result = run(&req, &wav, dir.path(), Duration::from_secs(5), None)
            .expect("silence run should succeed");
        let elapsed = start.elapsed();

        assert!(result.segments.is_empty(), "silence => no segments");
        assert!(result.transcript.is_empty(), "silence => empty transcript");
        assert_eq!(result.raw_output["silence"].as_bool(), Some(true));
        assert_eq!(result.raw_output["model_loaded"].as_bool(), Some(false));
        assert_eq!(
            result.raw_output["engine"].as_str(),
            Some("insanely-fast-native")
        );
        assert_eq!(
            result.raw_output["schema_version"].as_str(),
            Some(SCHEMA_VERSION)
        );
        assert!(
            elapsed < Duration::from_secs(2),
            "silence pre-gate should return fast, took {elapsed:?}"
        );
    }

    // ── Cancellation ──────────────────────────────────────────────────────

    #[test]
    fn run_expired_token_is_cancelled_quickly() {
        let dir = tempfile::tempdir().expect("tempdir");
        let wav = dir.path().join("tone.wav");
        let mut samples = vec![0i16; 1_600];
        samples.extend((0..16_000).map(|i| if i % 2 == 0 { 9_000i16 } else { -9_000 }));
        write_pcm16_mono_wav(&wav, 16_000, &samples);

        let req = request();
        let token = CancellationToken::with_deadline_from_now(Duration::from_millis(0));
        std::thread::sleep(Duration::from_millis(5));

        let start = std::time::Instant::now();
        let result = run(&req, &wav, dir.path(), Duration::from_secs(1), Some(&token));
        assert!(result.is_err(), "expired token must cancel");
        assert!(
            matches!(result.unwrap_err(), FwError::Cancelled(_)),
            "expected FW-CANCELLED"
        );
        assert!(
            start.elapsed() < Duration::from_secs(2),
            "cancellation should be fast"
        );
    }

    // ── raw_output schema ─────────────────────────────────────────────────

    #[test]
    fn raw_output_carries_parallel_window_metadata() {
        let json = raw_output_json(
            "tiny.en",
            Path::new("/models/ggml-tiny.en.bin"),
            "fw-native-v1+sha256:abc".to_owned(),
            2,
            2,
            4,
            &[],
            WordTimestampMode::Word,
            false,
            false,
        );
        assert_eq!(json["engine"].as_str(), Some("insanely-fast-native"));
        assert_eq!(json["schema_version"].as_str(), Some(SCHEMA_VERSION));
        assert_eq!(json["implementation"].as_str(), Some("real-inference"));
        assert_eq!(json["in_process"].as_bool(), Some(true));
        assert_eq!(json["parallel_windows"].as_u64(), Some(2));
        assert_eq!(json["workers"].as_u64(), Some(2));
        assert_eq!(json["threads_per_worker"].as_u64(), Some(4));
        assert_eq!(json["word_timestamps"].as_str(), Some("interpolated"));
    }

    // ── Gated end-to-end against the real tiny.en model + jfk.wav ─────────

    /// The exact reference transcript from
    /// `tests/fixtures/native/jfk_tiny_reference.json` (joined, trimmed).
    const JFK_REFERENCE: &str = "And so my fellow Americans ask not what your country can do for \
        you ask what you can do for your country.";

    fn tiny_en_available() -> bool {
        native_engine::find_model_file("tiny.en").is_some()
    }

    fn load_jfk_samples() -> Option<Vec<f32>> {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/native/jfk.wav");
        let bytes = std::fs::read(path).ok()?;
        decode::read_wav_16k_mono(&bytes).ok()
    }

    fn load_tiny_en() -> Option<std::sync::Arc<NativeWhisperModel>> {
        let path = native_engine::find_model_file("tiny.en")?;
        NativeWhisperModel::load(&path).ok()
    }

    #[test]
    fn gated_single_window_matches_whisper_cpp_native() {
        if !tiny_en_available() {
            eprintln!("SKIP gated_single_window: tiny.en model missing");
            return;
        }
        let wav = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/native/jfk.wav");
        let mut req = request();
        req.model = Some("tiny.en".to_owned());
        req.language = None;

        let engine = super::super::InsanelyFastNativeEngine;
        let result = crate::backend::Engine::run(
            &engine,
            &req,
            Path::new(wav),
            Path::new("."),
            Duration::from_secs(120),
            None,
        )
        .expect("e2e run");

        // jfk.wav is ~11 s => a single 30 s window. The single-window path runs
        // the SAME native engine over the SAME samples as whisper.cpp native, so
        // the transcript must match the shared reference exactly.
        assert_eq!(result.transcript.trim(), JFK_REFERENCE);
        assert_eq!(
            result.raw_output["engine"].as_str(),
            Some("insanely-fast-native")
        );
        assert_eq!(result.raw_output["parallel_windows"].as_u64(), Some(1));
        assert_eq!(result.raw_output["silence"].as_bool(), Some(false));
        assert!(result.segments.len() >= 2, "expected >= 2 segments");
    }

    /// Decode jfk-concatenated-3x (~33 s => 2 windows) with an explicit worker
    /// count and return the engine-level (offset, merged) outputs as plain text +
    /// segment bounds for determinism comparison.
    fn decode_long_jfk_with_workers(
        model: &NativeWhisperModel,
        samples: &[f32],
        batch_size: Option<usize>,
        total_threads: usize,
    ) -> (String, Vec<(f64, f64)>, usize) {
        let windows = split_windows(samples);
        let (n_workers, tpw) = plan_workers(windows.len(), batch_size, total_threads);
        let params = decode::DecodeParams {
            language: None,
            translate: false,
            timestamps: true,
            n_threads: tpw,
            max_text_ctx: None,
            ..decode::DecodeParams::default()
        };
        let results =
            decode_windows_parallel(model, &windows, &params, n_workers, None).expect("decode");
        let merged = merge_windows(&results);
        let text: String = merged
            .segments
            .iter()
            .map(|s| s.text.trim())
            .collect::<Vec<_>>()
            .join(" ");
        let bounds: Vec<(f64, f64)> = merged
            .segments
            .iter()
            .map(|s| (s.start_sec.unwrap_or(0.0), s.end_sec.unwrap_or(0.0)))
            .collect();
        (text, bounds, windows.len())
    }

    #[test]
    fn gated_determinism_two_workers_equals_one() {
        let Some(model) = load_tiny_en() else {
            eprintln!("SKIP gated_determinism: tiny.en model missing");
            return;
        };
        let Some(samples) = load_jfk_samples() else {
            eprintln!("SKIP gated_determinism: jfk.wav missing");
            return;
        };
        // Concatenate jfk 3x (~33 s) => 2 windows.
        let mut long = Vec::with_capacity(samples.len() * 3);
        for _ in 0..3 {
            long.extend_from_slice(&samples);
        }

        // Force 1 worker (batch_size 1) vs >= 2 workers (batch_size 8, threads 8).
        let (text_seq, bounds_seq, n_windows_seq) =
            decode_long_jfk_with_workers(&model, &long, Some(1), 8);
        let (text_par, bounds_par, n_windows_par) =
            decode_long_jfk_with_workers(&model, &long, Some(8), 8);

        assert!(n_windows_seq >= 2, "expected >= 2 windows for ~33s audio");
        assert_eq!(n_windows_seq, n_windows_par);

        // CRITICAL: parallel output must be byte-identical to sequential.
        assert_eq!(
            text_par, text_seq,
            "2-worker transcript must equal 1-worker transcript exactly"
        );
        assert_eq!(
            bounds_par, bounds_seq,
            "2-worker timestamps must equal 1-worker timestamps exactly"
        );

        // Timestamps monotonic non-decreasing across the window boundary.
        let mut prev_end = -1.0f64;
        for (start, end) in &bounds_seq {
            assert!(
                *start + 1e-6 >= prev_end - 1e-6,
                "segment start {start} must not precede previous end {prev_end}"
            );
            assert!(*end + 1e-6 >= *start, "segment end {end} >= start {start}");
            prev_end = *end;
        }

        // Segments span both windows: at least one segment starts at/after 30 s.
        assert!(
            bounds_seq.iter().any(|(s, _)| *s >= 30.0 - 1e-6),
            "expected segments in the second (>=30s) window: {bounds_seq:?}"
        );

        // The repeated sentence's signature word should recur.
        let lc = text_seq.to_lowercase();
        assert!(
            lc.matches("country").count() >= 2,
            "expected the repeated sentence at least twice: {text_seq}"
        );
    }
}
