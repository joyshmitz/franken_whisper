//! Greedy decode loop, timestamp rules, and 30-second windowing — the heart of
//! the native whisper engine.
//!
//! This is a faithful port of the **greedy / temperature-0 path** of
//! whisper.cpp's `whisper_full_with_state()` (see `src/whisper.cpp`). It strings
//! together the sibling modules — [`ggml`](super::ggml),
//! [`mel`](super::mel), [`tokenizer`](super::tokenizer),
//! [`encoder`](super::encoder), and [`decoder`](super::decoder) — into the
//! single user-facing [`transcribe_samples`] entry point.
//!
//! # Ported pieces (with cited whisper.cpp line numbers)
//!
//! - **Logit filter suite** ([`process_logits`]) — a line-for-line port of
//!   `whisper_process_logits()` (whisper.cpp 6178-6396), applied IN ORDER:
//!   blank suppression at the first step (6217-6222), `<|notimestamps|>` and
//!   control/task/lang/`sot`/`nosp`/`prev` suppression (6226-6260), optional
//!   non-speech suppression (6279-6296), timestamp-pairing (6298-6317),
//!   `max_initial_ts` clamp (6319-6328), timestamp monotonicity (6330-6338),
//!   and the log-space sum-of-timestamp-probs vs max-text-prob forcing rule
//!   (6340-6369). Argmax over the resulting logits follows
//!   `whisper_sample_token(best=true)` (6468-6525).
//! - **no_speech_prob** captured from the FIRST forward's softmax at
//!   `token_nosp`, BEFORE any logit filtering (whisper.cpp 7172-7182).
//! - **avg_logprob** accumulated as `sum(plog over result_len) / result_len`
//!   (whisper.cpp 6602-6617), where `plog` is the chosen token's log-softmax.
//! - **Seek / window advance**: `seek += seek_delta`, with `seek_delta` driven
//!   by the last in-window timestamp token (`2*(tid - token_beg)` centiseconds,
//!   whisper.cpp 7362-7375), the single-timestamp-ending whole-chunk skip
//!   (7753-7760), and the full-chunk advance when no timestamps closed a
//!   segment.
//! - **Segment building** from timestamp pairs, including the final open-tail
//!   segment, ported from whisper.cpp 7624-7730.
//! - **Failed-window / no-speech heuristic**: `no_speech_prob > 0.6 &&
//!   avg_logprob < -1.0 ⇒ silence`, emit nothing, advance the full window
//!   (whisper.cpp 7606-7607, defaults at 5973/5978/5979).
//! - **No-timestamps mode**: one segment spanning the window (whisper.cpp
//!   7402-7405 `single_segment || no_timestamps`).
//! - **Language detection**: for multilingual models with no language given,
//!   one `[sot]` forward on the first window then argmax over language-token
//!   logits, cached for later windows (port of
//!   `whisper_lang_auto_detect_with_state`, whisper.cpp 4035-4108).
//! - **Previous-context prompt**: `[sot_prev, ...]` prepended from the prior
//!   window's text tokens, capped at `n_text_ctx/2` (whisper.cpp 6927,
//!   7106-7133, 7611-7622).
//!
//! # Units
//!
//! Internally all audio offsets are in **centiseconds** (1 cs = 10 ms),
//! matching whisper.cpp's `seek` / `seek_delta` integer units (a timestamp
//! token step of `0.02 s` is `2 cs`). They are converted to floating-point
//! seconds only when a
//! [`TranscriptionSegment`](crate::model::TranscriptionSegment) is emitted.

#![allow(clippy::module_name_repetitions)]

use crate::error::{FwError, FwResult};
use crate::model::TranscriptionSegment;

use super::decoder::{self, DecoderState, DecoderWeights};
use super::dtw::{self, WordTiming};
use super::encoder::{self, EncoderWeights};
use super::ggml::GgmlModel;
use super::mel::{self, FRAMES_PER_CHUNK, SAMPLE_RATE};
use super::tokenizer::{LANGUAGES, Tokenizer};
use super::{Mel, MelFilterbank, WhisperHParams};

/// Length of one 30-second window in centiseconds (`30 s * 100 cs/s`).
/// whisper.cpp's `WHISPER_CHUNK_SIZE` is `30`; offsets there are scaled by
/// `*100` to centiseconds (e.g. `100*WHISPER_CHUNK_SIZE`, whisper.cpp 7404).
const CHUNK_CS: i64 = 3000;

/// Minimum residual centiseconds to consider the window "ended"
/// (whisper.cpp `delta_min = 10`, line 6865).
const DELTA_MIN: i64 = 10;

/// Default no-speech probability threshold (whisper.cpp 5979).
const NO_SPEECH_THRESHOLD: f64 = 0.6;

/// Default average-logprob threshold (whisper.cpp 5978).
const LOGPROB_THRESHOLD: f64 = -1.0;

/// Default maximum initial timestamp, in seconds (whisper.cpp 5973).
const MAX_INITIAL_TS_SEC: f32 = 1.0;

/// Finite sentinel for `avg_logprob` on an empty-result window (fix #9). A true
/// `f64::NEG_INFINITY` serializes to JSON `null` (serde_json has no infinity
/// representation), making `windows[].avg_logprob` non-numeric. `-999.0` is far
/// below any real average log-probability and below [`LOGPROB_THRESHOLD`], so it
/// keeps the no-speech/failed-window gate behavior identical while remaining a
/// finite, serializable number.
const EMPTY_WINDOW_AVG_LOGPROB: f64 = -999.0;

// ---------------------------------------------------------------------------
// Public model bundle + parameters + output (the bd-hsbx interface contract)
// ---------------------------------------------------------------------------

/// A fully-loaded whisper model: hyper-parameters, mel filterbank, tokenizer,
/// and the encoder / decoder transformer weights, ready for
/// [`transcribe_samples`].
#[derive(Debug)]
pub struct LoadedModel {
    pub hparams: WhisperHParams,
    pub filters: MelFilterbank,
    pub tokenizer: Tokenizer,
    pub encoder: EncoderWeights,
    pub decoder: DecoderWeights,
}

impl LoadedModel {
    /// Build a [`LoadedModel`] from a parsed ggml model file, loading the
    /// encoder and decoder weights and constructing the tokenizer from the
    /// embedded vocabulary.
    ///
    /// # Errors
    /// Propagates [`EncoderWeights::from_ggml`] / [`DecoderWeights::from_ggml`]
    /// shape-validation errors.
    pub fn from_ggml(model: GgmlModel) -> FwResult<Self> {
        let hparams = model.hparams;
        let filters = model.filters.clone();
        let tokenizer = Tokenizer::from_vocab(&hparams, model.vocab_tokens.clone());
        let encoder = EncoderWeights::from_ggml(&model)?;
        let decoder = DecoderWeights::from_ggml(&model)?;
        Ok(Self {
            hparams,
            filters,
            tokenizer,
            encoder,
            decoder,
        })
    }
}

/// Decoding parameters for [`transcribe_samples`] (greedy, temperature 0).
#[derive(Debug, Clone, Default)]
pub struct DecodeParams {
    /// Source language code (e.g. `"en"`). `None` triggers auto-detection on
    /// multilingual models; ignored by English-only models.
    pub language: Option<String>,
    /// Translate to English instead of transcribing in the source language.
    pub translate: bool,
    /// Emit timestamp tokens and split the transcript into timed segments.
    /// When `false`, each window yields a single segment spanning the window.
    pub timestamps: bool,
    /// Thread-count hint passed through to the encoder/decoder (the FrankenTorch
    /// kernels manage their own pool; this is informational).
    pub n_threads: usize,
    /// Optional cap on decoded tokens per window. `None` uses whisper's
    /// `n_text_ctx/2 - 4` default.
    pub max_text_ctx: Option<usize>,
    /// When `true`, record cross-attention weights of the model's alignment
    /// heads during decode and compute real **word-level timestamps** via DTW
    /// (bd-rjsx). Defaults to `false`; the recording cost (heads × tokens ×
    /// 1500 f32 per window) is only paid when this is set. The resulting
    /// per-segment word timings are returned in [`DecodeOutput::word_timings`].
    pub word_timestamps: bool,
    /// Optional model short-name hint (e.g. `"tiny.en"`) used to disambiguate
    /// alignment-head presets that share `(n_text_layer, n_text_state)` (the
    /// large-v1/v2/v3 family). Ignored unless `word_timestamps` is set.
    pub model_hint: Option<String>,
}

/// Per-window quality-control statistics, surfaced for the evidence ledger.
#[derive(Debug, Clone, PartialEq)]
pub struct WindowStats {
    /// Mean token log-probability over the window's result tokens.
    pub avg_logprob: f64,
    /// No-speech probability captured from the first forward's softmax.
    pub no_speech_prob: f64,
    /// Number of result tokens decoded in this window.
    pub tokens: usize,
    /// Window start offset in seconds.
    pub window_offset_sec: f64,
}

/// Result of [`transcribe_samples`]: timed segments, detected/used language,
/// and per-window QC statistics.
#[derive(Debug, Clone)]
pub struct DecodeOutput {
    pub segments: Vec<TranscriptionSegment>,
    pub language: Option<String>,
    pub windows: Vec<WindowStats>,
    /// Per-segment word timings, aligned 1:1 with `segments`, populated only
    /// when [`DecodeParams::word_timestamps`] was set (else `None`). Each inner
    /// `Vec<WordTiming>` is the DTW-aligned words of the corresponding segment,
    /// in order; an empty inner vec means that segment produced no timed words
    /// (e.g. a no-speech window). See bd-rjsx.
    pub word_timings: Option<Vec<Vec<WordTiming>>>,
}

// ---------------------------------------------------------------------------
// Logit filtering (port of whisper_process_logits, whisper.cpp 6178-6396)
// ---------------------------------------------------------------------------

/// Configuration the logit filter needs that does not change per step.
struct FilterConfig {
    /// Suppress the leading blank (`" "`) + `eot` on the first step
    /// (whisper.cpp 6217-6222). `space_token` is the id of the `" "` token, if
    /// present in the vocab.
    suppress_blank: bool,
    space_token: Option<i32>,
    /// Suppress non-speech tokens (whisper.cpp 6279-6296). Off by default, like
    /// whisper.cpp (`suppress_nst = false`, line 5970).
    suppress_nst: bool,
    /// `no_timestamps` mode masks every timestamp token (whisper.cpp 6227-6231).
    no_timestamps: bool,
    /// `tid0` for the `max_initial_ts` clamp: the maximum number of timestamp
    /// steps allowed on the initial step (whisper.cpp 6321-6327). `None`
    /// disables the clamp.
    max_initial_tid: Option<i32>,
    /// Per-window token budget for the `max_tokens` EOT-forcing filter (fix #6 —
    /// whisper.cpp 6234-6238). When timestamps are enabled and the running
    /// in-window token count reaches this budget, every text token below `eot`
    /// is masked, forcing a timestamp/eot to close the window. `None` (or `0`)
    /// disables the filter (matching upstream's `params.max_tokens > 0` guard).
    /// Inert in no-timestamps mode (the `!params.no_timestamps` guard).
    max_tokens: Option<usize>,
}

/// Apply whisper's logit-filter suite IN ORDER and return the (mutated) logits
/// plus the log-softmax `logprobs`. `prev_tokens` is the decoded text so far
/// (excluding the prompt); `seek_delta_cs` is the running window shift in
/// centiseconds (drives the monotonicity floor).
///
/// Port of `whisper_process_logits` (whisper.cpp 6178-6396); see the inline
/// comments for the matching upstream line ranges.
fn process_logits(
    tk: &Tokenizer,
    cfg: &FilterConfig,
    mut logits: Vec<f32>,
    prev_tokens: &[i32],
    has_ts: bool,
    seek_delta_cs: i64,
    tokens_in_window: usize,
) -> (Vec<f32>, Vec<f32>) {
    let n = logits.len();
    let beg = tk.timestamp_begin;
    let is_initial = prev_tokens.is_empty();

    let set = |logits: &mut [f32], id: i32| {
        if let Ok(i) = usize::try_from(id)
            && i < logits.len()
        {
            logits[i] = f32::NEG_INFINITY;
        }
    };

    // suppress blank (whisper.cpp 6217-6222): only on the very first step.
    if cfg.suppress_blank && is_initial {
        set(&mut logits, tk.eot);
        if let Some(sp) = cfg.space_token {
            set(&mut logits, sp);
        }
    }

    // suppress <|notimestamps|>; in no_timestamps mode mask all timestamps too
    // (whisper.cpp 6226-6231).
    set(&mut logits, tk.no_timestamps);
    if cfg.no_timestamps {
        for i in beg..(n as i32) {
            set(&mut logits, i);
        }
    }

    // max_tokens EOT-forcing filter (fix #6 — whisper.cpp 6234-6238): when
    // timestamps are enabled, the window is not a single segment, and the
    // running token count has reached the budget, mask every text token below
    // `eot` so the next step must emit a timestamp/eot and close the window.
    if !cfg.no_timestamps
        && let Some(max_tokens) = cfg.max_tokens
        && max_tokens > 0
        && tokens_in_window >= max_tokens
    {
        for i in 0..tk.eot {
            set(&mut logits, i);
        }
    }

    // suppress sot, nosp, solm, task tokens, prev (whisper.cpp 6241-6260).
    set(&mut logits, tk.sot);
    set(&mut logits, tk.no_speech);
    set(&mut logits, tk.solm);
    set(&mut logits, tk.translate);
    set(&mut logits, tk.transcribe);
    set(&mut logits, tk.sot_prev);

    // suppress language tokens (whisper.cpp 6254-6257).
    for (_, lang_id, _) in LANGUAGES {
        // language token for id n is sot+1+n (whisper.cpp whisper_token_lang).
        set(&mut logits, tk.sot + 1 + *lang_id);
    }

    // suppress non-speech tokens (whisper.cpp 6279-6296), opt-in.
    if cfg.suppress_nst {
        for &id in tk.non_speech_tokens() {
            set(&mut logits, id);
        }
    }

    // timestamps appear in pairs except directly before EOT (whisper.cpp
    // 6298-6317).
    let last_was_ts = prev_tokens.last().is_some_and(|&t| t >= beg);
    let penult_was_ts = prev_tokens.len() < 2 || prev_tokens[prev_tokens.len() - 2] >= beg;
    if last_was_ts {
        if penult_was_ts {
            // two timestamps back-to-back: forbid another timestamp.
            for i in beg..(n as i32) {
                set(&mut logits, i);
            }
        } else {
            // one timestamp open: force a timestamp or EOT (mask all text).
            for i in 0..tk.eot {
                set(&mut logits, i);
            }
        }
    }

    // initial timestamp cannot exceed max_initial_ts (whisper.cpp 6319-6328).
    if is_initial && let Some(tid0) = cfg.max_initial_tid {
        for i in (beg + tid0 + 1)..(n as i32) {
            set(&mut logits, i);
        }
    }

    // condition timestamp tokens to be increasing (whisper.cpp 6330-6338).
    if has_ts {
        let tid0 = (seek_delta_cs / 2) as i32; // centiseconds -> ts steps.
        for i in beg..(beg + tid0).min(n as i32) {
            set(&mut logits, i);
        }
    }

    // log-softmax over the (filtered) logits (whisper.cpp 6138-6158, 6341).
    let mut logprobs = compute_logprobs(&logits);

    // sum-of-timestamp-probs vs max-text-prob forcing rule (whisper.cpp
    // 6343-6369), implemented in log space exactly as upstream.
    {
        let beg_u = beg.max(0) as usize;
        // logsumexp over the timestamp logprobs.
        let mut ts_logprob = f32::NEG_INFINITY;
        if beg_u < logprobs.len() {
            let logprob_max = logprobs[beg_u..]
                .iter()
                .copied()
                .fold(f32::NEG_INFINITY, f32::max);
            if logprob_max.is_finite() {
                let mut logsumexp = 0.0f32;
                for &lp in &logprobs[beg_u..] {
                    if lp > f32::NEG_INFINITY {
                        logsumexp += (lp - logprob_max).exp();
                    }
                }
                if logsumexp > 0.0 {
                    ts_logprob = logsumexp.ln() + logprob_max;
                }
            }
        }
        let max_text_logprob = logprobs[..beg_u]
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);

        if ts_logprob > max_text_logprob {
            // force a timestamp: mask all text logits/logprobs.
            for i in 0..beg_u {
                logits[i] = f32::NEG_INFINITY;
                logprobs[i] = f32::NEG_INFINITY;
            }
        }
    }

    (logits, logprobs)
}

/// Numerically-stable log-softmax (whisper.cpp `whisper_compute_logprobs`,
/// lines 6138-6158). `-inf` logits map to `-inf` logprobs.
fn compute_logprobs(logits: &[f32]) -> Vec<f32> {
    let logit_max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut logsumexp = 0.0f32;
    for &l in logits {
        if l > f32::NEG_INFINITY {
            logsumexp += (l - logit_max).exp();
        }
    }
    let logsumexp = logsumexp.ln() + logit_max;
    logits
        .iter()
        .map(|&l| {
            if l > f32::NEG_INFINITY {
                l - logsumexp
            } else {
                f32::NEG_INFINITY
            }
        })
        .collect()
}

/// Argmax over `logits`, returning `(id, logprob_of_id)`. Mirrors
/// `whisper_sample_token(best=true)` (whisper.cpp 6503-6510): the chosen id is
/// the argmax of the (post-filter) probabilities — equivalently logits — and
/// `plog` is its log-softmax value.
fn argmax(logits: &[f32], logprobs: &[f32]) -> (i32, f32) {
    let mut best_i = 0usize;
    let mut best = f32::NEG_INFINITY;
    for (i, &l) in logits.iter().enumerate() {
        if l > best {
            best = l;
            best_i = i;
        }
    }
    (
        i32::try_from(best_i).unwrap_or(0),
        logprobs.get(best_i).copied().unwrap_or(0.0),
    )
}

// ---------------------------------------------------------------------------
// Segment building (port of whisper.cpp 7624-7730)
// ---------------------------------------------------------------------------

/// Build [`TranscriptionSegment`]s from a decoded token stream.
///
/// Port of the result-emission loop in whisper.cpp (7624-7730). `tokens` is the
/// full decoded sequence (text + timestamp tokens, no prompt); `seek_cs` is the
/// window start in centiseconds; `seek_delta_cs` is the window's final shift
/// (used to close the open-tail segment, whisper.cpp 7697). `plogs` is the chosen
/// token's `plog` per decoded token (same length as `tokens`), used for the
/// per-segment confidence.
///
/// In `single_segment` / no-timestamps mode (`split == false`) a single segment
/// spanning `[seek_cs, seek_cs + seek_delta_cs]` is produced (whisper.cpp
/// 7402-7405, 7645 guard).
///
/// All emitted segment bounds are clamped to `[seek_cs, seek_end_cs]` (fix #1):
/// a timestamp token can point into the zero-padded tail of the final window
/// (worst on a hard-cut last clip), which would otherwise yield an `end_sec`
/// past the real clip duration. `seek_end_cs` is the real (unpadded) audio
/// length in centiseconds — whisper.cpp's `n_len_org` (6859-6860).
fn build_segments(
    tk: &Tokenizer,
    tokens: &[i32],
    plogs: &[f32],
    seek_cs: i64,
    seek_delta_cs: i64,
    seek_end_cs: i64,
    split: bool,
) -> Vec<TranscriptionSegment> {
    let beg = tk.timestamp_begin;
    let mut segments = Vec::new();

    // Clamp a window-relative + seek-offset centisecond bound to the real audio
    // length, never below the window start (fix #1).
    let clamp = |t: i64| t.clamp(seek_cs, seek_end_cs.max(seek_cs));

    if !split {
        // Single segment spanning the whole window (whisper.cpp 7402-7405).
        let text = tk.decode(tokens).trim().to_string();
        if !text.is_empty() {
            segments.push(make_segment(
                clamp(seek_cs),
                clamp(seek_cs + seek_delta_cs),
                text,
                text_confidence(tk, tokens, plogs),
            ));
        }
        return segments;
    }

    // Timestamp-paired emission (whisper.cpp 7624-7694).
    // t0 starts at the first token's implied timestamp (whisper.cpp 7626);
    // in greedy the first decoded token is normally the opening <|0.00|> ts.
    let mut i0 = 0usize;
    let mut t0 = clamp(seek_cs + ts_offset_cs(tokens.first().copied(), beg));
    let mut i = 0usize;

    while i < tokens.len() {
        let tok = tokens[i];
        // A timestamp token strictly greater than `beg` closes a segment
        // (whisper.cpp 7645: `id > token_beg`). The bare `beg` (<|0.00|>) opens.
        if tok > beg {
            let t1 = clamp(seek_cs + 2 * i64::from(tok - beg));
            let text = tk.decode(&tokens[i0..=i]).trim().to_string();
            if !text.is_empty() {
                let conf = text_confidence(
                    tk,
                    tokens.get(i0..=i).unwrap_or(&[]),
                    plogs.get(i0..=i).unwrap_or(&[]),
                );
                segments.push(make_segment(t0, t1, text, conf));
            }
            t0 = t1;
            // Skip a run of consecutive timestamp tokens (whisper.cpp 7684-7690).
            while i + 1 < tokens.len() && tokens[i + 1] > beg {
                i += 1;
                t0 = clamp(seek_cs + 2 * i64::from(tokens[i] - beg));
            }
            i0 = i + 1;
        }
        i += 1;
    }

    // Open-tail segment: text after the last timestamp pair (whisper.cpp
    // 7696-7714). Closed at `seek + seek_delta`.
    if i0 < tokens.len() {
        let text = tk.decode(&tokens[i0..]).trim().to_string();
        if !text.is_empty() {
            let t1 = clamp(seek_cs + seek_delta_cs);
            let conf = text_confidence(
                tk,
                tokens.get(i0..).unwrap_or(&[]),
                plogs.get(i0..).unwrap_or(&[]),
            );
            segments.push(make_segment(t0, t1, text, conf));
        }
    }

    segments
}

/// Whether the prompt context should be cleared before decoding a window with a
/// very short audio tail (fix #2 — port of whisper.cpp 7046-7051):
///
/// ```text
/// if (seek > seek_start && seek + 500 >= seek_end) {
///     prompt_past0.clear();
///     prompt_past1.clear();
/// }
/// ```
///
/// On a non-first window (`seek_cs > 0`, our `seek_start` is always 0) whose
/// remaining audio is under 5 s (`seek_cs + 500 >= seek_end_cs`, 500 cs = 5 s),
/// upstream drops the carried prompt because a short tail "tends to confuse the
/// decoder and often make it repeat or hallucinate stuff". Extracted as a pure
/// predicate so it can be unit-tested without a model.
fn should_clear_short_tail_prompt(seek_cs: i64, seek_end_cs: i64) -> bool {
    seek_cs > 0 && seek_cs + 500 >= seek_end_cs
}

/// Timestamp offset (centiseconds) implied by a (possibly text) token id, used
/// for the opening `t0`. For a timestamp token it is `2*(id - beg)`; otherwise
/// `0` (the opening `<|0.00|>` is what whisper expects first).
fn ts_offset_cs(tok: Option<i32>, beg: i32) -> i64 {
    match tok {
        Some(id) if id >= beg => 2 * i64::from(id - beg),
        _ => 0,
    }
}

/// Per-segment confidence: `exp(mean token logprob)` clamped to `[0, 1]`.
/// Superseded in production by [`text_confidence`] (fix #8, which excludes
/// timestamp tokens); retained for the clamp/monotonicity unit test.
#[cfg(test)]
fn confidence(plogs: &[f32]) -> Option<f64> {
    if plogs.is_empty() {
        return None;
    }
    let mean = plogs.iter().map(|&p| f64::from(p)).sum::<f64>() / plogs.len() as f64;
    Some(mean.exp().clamp(0.0, 1.0))
}

/// Per-segment **text** confidence (fix #8): `exp(mean text-token logprob)`
/// clamped to `[0, 1]`, averaging only over the segment's *text* tokens —
/// excluding the leading/closing timestamp tokens (and any special tokens). The
/// metric documents itself as text confidence, so the closing `<|t|>` token's
/// logprob (which can be high-confidence and unrelated to the words) must not
/// dilute it. `tokens` and `plogs` are the segment's token ids and their chosen
/// logprobs, 1:1. If a segment has no text tokens, returns `None`.
fn text_confidence(tk: &Tokenizer, tokens: &[i32], plogs: &[f32]) -> Option<f64> {
    let mut sum = 0.0f64;
    let mut count = 0usize;
    for (i, &tok) in tokens.iter().enumerate() {
        // A text token is anything below the timestamp range that is not a
        // special control token.
        if tok < tk.timestamp_begin
            && !tk.is_special(tok)
            && let Some(&p) = plogs.get(i)
        {
            sum += f64::from(p);
            count += 1;
        }
    }
    if count == 0 {
        return None;
    }
    let mean = sum / count as f64;
    Some(mean.exp().clamp(0.0, 1.0))
}

/// Construct a [`TranscriptionSegment`] from centisecond bounds + text.
fn make_segment(
    t0_cs: i64,
    t1_cs: i64,
    text: String,
    confidence: Option<f64>,
) -> TranscriptionSegment {
    TranscriptionSegment {
        start_sec: Some(t0_cs as f64 / 100.0),
        end_sec: Some(t1_cs as f64 / 100.0),
        text,
        speaker: None,
        confidence,
    }
}

// ---------------------------------------------------------------------------
// Top-level transcription (port of whisper_full_with_state greedy path)
// ---------------------------------------------------------------------------

/// Transcribe 16 kHz mono PCM `samples` with the greedy / temperature-0 path of
/// whisper, returning timed segments + per-window QC statistics.
///
/// `checkpoint` is invoked between **every** decoder step (and between encoder
/// layers, via the underlying forward passes) so a caller can cancel a long
/// transcription at token granularity — the project's cancellation contract.
///
/// # Errors
/// - [`FwError::InvalidRequest`] for empty input or a model/shape mismatch
///   surfaced by the encoder/decoder.
/// - Whatever `checkpoint` returns (e.g. [`FwError::Cancelled`]), promptly.
pub fn transcribe_samples(
    m: &LoadedModel,
    samples_16k_mono: &[f32],
    params: &DecodeParams,
    checkpoint: &dyn Fn() -> FwResult<()>,
) -> FwResult<DecodeOutput> {
    if samples_16k_mono.is_empty() {
        return Err(FwError::InvalidRequest(
            "transcribe_samples: empty audio".into(),
        ));
    }

    let tk = &m.tokenizer;

    // Full-audio log-mel spectrogram (whisper computes once, then windows it).
    let full_mel = mel::log_mel(samples_16k_mono, &m.filters, params.n_threads)?;

    // Window bounds in centiseconds: seek runs [0, seek_end) where seek_end is
    // the **original** (unpadded) audio length — whisper.cpp's `n_len_org`
    // (whisper.cpp 6859-6860). `log_mel` trails a full 30 s of silence padding,
    // so `full_mel.n_frames` is NOT the right bound. Upstream computes the real
    // length as the mel `n_len_org` (whisper.cpp 3208):
    //   n_len_org = 1 + (n_samples + stage_2_pad - frame_size) / frame_step
    // with `stage_2_pad = 200`, `frame_size = 400`, `frame_step = 160`. One mel
    // frame = 10 ms = 1 cs (fix #7). Guard underflow for `n_samples < 200` (the
    // numerator `n_samples - 200` saturates to 0).
    let seek_end_cs = {
        let n_samples = samples_16k_mono.len() as i64;
        const STAGE_2_PAD: i64 = 200;
        const FRAME_SIZE: i64 = 400;
        const FRAME_STEP: i64 = 160;
        // n_samples + 200 - 400 = n_samples - 200, saturating at 0.
        let numer = (n_samples + STAGE_2_PAD - FRAME_SIZE).max(0);
        (1 + numer / FRAME_STEP).max(DELTA_MIN)
    };

    // Resolve language: explicit, else auto-detect once (multilingual only).
    let mut used_language = resolve_language(m, params, &full_mel, checkpoint)?;

    // The `" "` token id, for blank suppression (whisper.cpp 6220).
    let space_token = (0..tk.vocab_size()).find(|&id| tk.token_bytes(id) == Some(b" ".as_slice()));

    // max_initial_ts clamp tid0 (whisper.cpp 6321-6323): precision = 30 / n_audio_ctx.
    let max_initial_tid = if MAX_INITIAL_TS_SEC > 0.0 {
        let precision = 30.0f32 / m.hparams.n_audio_ctx.max(1) as f32;
        Some((MAX_INITIAL_TS_SEC / precision).round() as i32)
    } else {
        None
    };

    // Per-window decode budget (whisper.cpp 7205: n_text_ctx/2 - 4), with the
    // optional caller override.
    let n_text_ctx = m.decoder.n_text_ctx();
    let default_budget = (n_text_ctx / 2).saturating_sub(4).max(1);
    let max_tokens = params
        .max_text_ctx
        .unwrap_or(default_budget)
        .min(default_budget);

    let cfg = FilterConfig {
        suppress_blank: true,
        space_token,
        suppress_nst: false, // whisper.cpp default (5970).
        no_timestamps: !params.timestamps,
        max_initial_tid,
        // EOT-forcing budget (fix #6): only meaningful with timestamps on.
        max_tokens: Some(max_tokens),
    };
    // Prompt context cap (whisper.cpp 6927): n_text_ctx/2.
    let max_prompt_ctx = n_text_ctx / 2;

    let mut segments: Vec<TranscriptionSegment> = Vec::new();
    let mut windows: Vec<WindowStats> = Vec::new();
    // Per-segment DTW word timings, accumulated 1:1 with `segments` when
    // `params.word_timestamps` is set (bd-rjsx).
    let mut word_timings: Vec<Vec<WordTiming>> = Vec::new();
    // Alignment heads for this model, resolved once (DTW word timestamps only).
    let align_heads = if params.word_timestamps {
        dtw::alignment_heads(&m.hparams, params.model_hint.as_deref())
    } else {
        Vec::new()
    };
    // Rolling text context from prior windows (whisper.cpp prompt_past1).
    let mut prompt_past: Vec<i32> = Vec::new();

    let mut seek_cs: i64 = 0;
    while seek_cs + DELTA_MIN < seek_end_cs {
        checkpoint()?;

        // Encode this window's 3000-frame mel chunk.
        let frame_offset = usize::try_from(seek_cs).unwrap_or(0);
        let mel_window = mel::chunk_frames(&full_mel, frame_offset, FRAMES_PER_CHUNK);
        let enc = encoder::forward(&m.encoder, &mel_window, params.n_threads, checkpoint)?;
        let mut st = DecoderState::new(&m.decoder, &enc)?;

        // Short-tail prompt clearing (fix #2 — whisper.cpp 7046-7051): a
        // non-first window with under 5 s of audio left drops the carried prompt
        // to avoid repetition/hallucination on the tail.
        if should_clear_short_tail_prompt(seek_cs, seek_end_cs) {
            prompt_past.clear();
        }

        // Build the prompt: [sot_prev, ...past...] + sot_sequence (whisper.cpp
        // 7106-7133). prompt_init is the sot sequence for this language/task.
        let sot_seq = tk.sot_sequence(
            used_language.as_deref(),
            params.translate,
            params.timestamps,
        );
        let mut prompt: Vec<i32> = Vec::new();
        if !prompt_past.is_empty() && max_prompt_ctx > 1 {
            prompt.push(tk.sot_prev);
            let take = prompt_past.len().min(max_prompt_ctx.saturating_sub(1));
            prompt.extend_from_slice(&prompt_past[prompt_past.len() - take..]);
        }
        prompt.extend_from_slice(&sot_seq);

        // Prefill the prompt; the first forward's softmax gives no_speech_prob
        // (whisper.cpp 7165-7182). Compute it BEFORE filtering.
        let prefill_logits = decoder::forward_step(&m.decoder, &mut st, &prompt, checkpoint)?;
        let no_speech_prob = {
            let lp = compute_logprobs(&prefill_logits);
            usize::try_from(tk.no_speech)
                .ok()
                .and_then(|i| lp.get(i).copied())
                .map_or(0.0, |x| f64::from(x.exp()))
        };

        // Greedy decode loop.
        let mut decoded: Vec<i32> = Vec::new();
        let mut plogs: Vec<f32> = Vec::new();
        let mut has_ts = false;
        let mut seek_delta_cs = CHUNK_CS; // default: advance full window.
        let mut result_len = 0usize;
        let mut step_logits = prefill_logits;

        for i in 0..max_tokens {
            let (filtered, logprobs) = process_logits(
                tk,
                &cfg,
                step_logits,
                &decoded,
                has_ts,
                seek_delta_cs,
                decoded.len(),
            );
            let (tok, plog) = argmax(&filtered, &logprobs);
            decoded.push(tok);
            plogs.push(plog);

            // Update sliding window from a timestamp token (whisper.cpp 7362-7375).
            if tok > tk.timestamp_begin {
                let new_delta = 2 * i64::from(tok - tk.timestamp_begin);
                if has_ts && seek_delta_cs > new_delta && result_len < i {
                    // Going back in time: bail out of this window (whisper.cpp 7366-7369).
                    break;
                }
                seek_delta_cs = new_delta;
                result_len = i + 1;
                has_ts = true;
            }

            // End of segment (whisper.cpp 7387-7410).
            let reached_end = has_ts && seek_cs + seek_delta_cs + DELTA_MIN >= seek_end_cs;
            if tok == tk.eot || reached_end {
                if result_len == 0 && params.timestamps {
                    if reached_end {
                        result_len = i + 1;
                    } else {
                        // Decoder failed with no timestamps closed.
                        break;
                    }
                }
                if !params.timestamps {
                    result_len = i + 1;
                    seek_delta_cs = CHUNK_CS;
                }
                break;
            }

            // Cancellation between every decoder step (the project contract).
            checkpoint()?;

            // Forward the just-chosen token to get the next logits.
            step_logits = decoder::forward_step(&m.decoder, &mut st, &[tok], checkpoint)?;
        }

        // avg_logprob over result tokens (whisper.cpp 6602-6617).
        let result_plogs: &[f32] = if result_len > 0 && result_len <= plogs.len() {
            &plogs[..result_len]
        } else {
            &plogs[..]
        };
        let avg_logprob = if result_plogs.is_empty() {
            EMPTY_WINDOW_AVG_LOGPROB
        } else {
            result_plogs.iter().map(|&p| f64::from(p)).sum::<f64>() / result_plogs.len() as f64
        };

        // no-speech / failed-window gate (whisper.cpp 7606-7607): treat as
        // silence, emit nothing, advance the full window.
        let is_no_speech = no_speech_prob > NO_SPEECH_THRESHOLD && avg_logprob < LOGPROB_THRESHOLD;

        // single-timestamp-ending: skip the rest of the chunk (whisper.cpp
        // 7753-7760). Ordering fix #5: upstream emits segments (7624-7730) and
        // records DTW word timings with the ORIGINAL `seek_delta`, then applies
        // this whole-chunk skip ONLY to the seek advance (7753-7760). So we keep
        // `seek_delta_cs` untouched for build_segments/window_word_timings below
        // and compute a separate `seek_advance_cs` for the window step.
        let single_ts_ending = decoded.len() > 1
            && decoded[decoded.len() - 2] < tk.timestamp_begin
            && decoded[decoded.len() - 1] > tk.timestamp_begin;
        let seek_advance_cs = if single_ts_ending {
            (seek_end_cs - seek_cs).min(CHUNK_CS)
        } else {
            seek_delta_cs
        };

        windows.push(WindowStats {
            avg_logprob,
            no_speech_prob,
            tokens: result_len,
            window_offset_sec: seek_cs as f64 / 100.0,
        });

        if !is_no_speech && !decoded.is_empty() {
            // Use only the result_len tokens for emission (drop a trailing eot).
            let take = result_len.min(decoded.len());
            let result_tokens = &decoded[..take];
            let result_token_plogs = &plogs[..take];
            let win_segments = build_segments(
                tk,
                result_tokens,
                result_token_plogs,
                seek_cs,
                seek_delta_cs,
                seek_end_cs,
                params.timestamps,
            );

            // DTW word timestamps (bd-rjsx): record cross-attention over this
            // window's result tokens and align them to audio frames. Computed
            // before `win_segments` is moved so we stay 1:1 with it. When
            // requested we always push one (possibly empty) word vec per emitted
            // segment so `word_timings` stays aligned with `segments`.
            if params.word_timestamps {
                let win_words = if align_heads.is_empty() {
                    vec![Vec::new(); win_segments.len()]
                } else {
                    window_word_timings(
                        m,
                        &mut st,
                        used_language.as_deref(),
                        params,
                        &align_heads,
                        result_tokens,
                        &win_segments,
                        seek_cs,
                        seek_delta_cs,
                        checkpoint,
                    )?
                };
                word_timings.extend(win_words);
            }

            segments.extend(win_segments);

            // Update rolling context: the decoded text tokens (whisper.cpp
            // 7617-7622), capped to the prompt budget.
            prompt_past.clear();
            for &t in result_tokens {
                if !tk.is_special(t) {
                    prompt_past.push(t);
                }
            }
            if prompt_past.len() > max_prompt_ctx {
                let drop = prompt_past.len() - max_prompt_ctx;
                prompt_past.drain(0..drop);
            }
        } else if is_no_speech {
            prompt_past.clear();
        }

        // Advance the window (whisper.cpp 7763) with the (possibly chunk-skip
        // adjusted) advance, NOT the emission delta (fix #5).
        seek_cs += seek_advance_cs.max(DELTA_MIN);
    }

    // English-only models never report a language; multilingual report the used
    // (possibly auto-detected) one.
    if !tk.is_multilingual() {
        used_language = None;
    }

    Ok(DecodeOutput {
        segments,
        language: used_language,
        windows,
        word_timings: if params.word_timestamps {
            Some(word_timings)
        } else {
            None
        },
    })
}

/// Compute DTW word timings for one window, returning per-segment word lists
/// aligned 1:1 with `win_segments` (bd-rjsx).
///
/// Port of whisper.cpp `whisper_exp_compute_token_level_timestamps_dtw`
/// (8837-8990) driving: a single batched decoder forward over
/// `sot + [lang] + not + <text tokens> + eot` with cross-attention recording
/// on, then [`dtw::token_timestamps`] over the alignment heads, then a
/// token→word grouping that follows the same timestamp-token segmentation as
/// [`build_segments`].
///
/// The decoder state `st` is reset before the recording pass (the precomputed
/// cross K/V are retained by [`DecoderState::reset`]); the greedy decode that
/// produced `result_tokens` is already complete, so reusing `st` is safe.
#[allow(clippy::too_many_arguments)]
fn window_word_timings(
    m: &LoadedModel,
    st: &mut DecoderState,
    language: Option<&str>,
    params: &DecodeParams,
    align_heads: &[(usize, usize)],
    result_tokens: &[i32],
    win_segments: &[TranscriptionSegment],
    seek_cs: i64,
    seek_delta_cs: i64,
    checkpoint: &dyn Fn() -> FwResult<()>,
) -> FwResult<Vec<Vec<WordTiming>>> {
    let tk = &m.tokenizer;

    // The window's text tokens, in order (drop timestamp/special tokens).
    let text_tokens: Vec<i32> = result_tokens
        .iter()
        .copied()
        .filter(|&t| !tk.is_special(t))
        .collect();
    if text_tokens.is_empty() {
        return Ok(vec![Vec::new(); win_segments.len()]);
    }

    // Alignment token sequence: sot + [lang] + not + text + eot (whisper.cpp
    // 8866-8882). The `no_timestamps` token is always present in this pass.
    let mut prompt = vec![tk.sot];
    if tk.is_multilingual() {
        let lang = language.unwrap_or("en");
        let lang_tok = tk
            .language_token(lang)
            .or_else(|| tk.language_token("en"))
            .unwrap_or(tk.sot + 1);
        prompt.push(lang_tok);
    }
    prompt.push(tk.no_timestamps);
    let sot_len = prompt.len();
    prompt.extend_from_slice(&text_tokens);
    prompt.push(tk.eot);

    // Single batched forward with cross-attention recording.
    st.reset();
    let prev_record = st.record_cross_attn;
    st.record_cross_attn = true;
    let _ = decoder::forward_step(&m.decoder, st, &prompt, checkpoint)?;
    let attn = st.cross_attn_weights().to_vec();
    st.record_cross_attn = prev_record;

    // Audio length for this window in encoder frames (whisper.cpp
    // `n_frames = min(min(3000, seek_delta), seek_end - seek)`, then
    // `n_audio_tokens = n_frames / 2`). `seek_delta_cs` is in centiseconds
    // (1 cs = 1 mel frame = 10 ms); two mel frames per encoder frame.
    let n_audio_frames = (seek_delta_cs.clamp(0, CHUNK_CS) / 2) as usize;

    // Per-text-token END times (window-relative seconds), with normalization +
    // DTW already restricted to the text rows (fix #3) and using upstream's
    // END-boundary convention (fix #4). `first_text_row = sot_len`,
    // `n_text_rows = text_tokens.len()` (the trailing eot row is excluded).
    let text_ends = dtw::token_timestamps(
        &attn,
        m.hparams.n_text_head.max(0) as usize,
        align_heads,
        sot_len,
        text_tokens.len(),
        n_audio_frames,
        dtw::DEFAULT_MEDFILT_WIDTH,
    );
    if text_ends.is_empty() {
        return Ok(vec![Vec::new(); win_segments.len()]);
    }

    // Reconcile END boundaries → token START times for word grouping (fix #4):
    // a token's start is the previous token's END boundary; the first token
    // starts at the window start (0, window-relative). Add the window seek
    // offset (DTW times are relative to the window start).
    let seek_sec = seek_cs as f64 / 100.0;
    let text_starts: Vec<f32> = (0..text_tokens.len())
        .map(|i| {
            let t = if i == 0 {
                0.0
            } else {
                text_ends.get(i - 1).copied().unwrap_or(0.0)
            };
            (f64::from(t) + seek_sec) as f32
        })
        .collect();

    // Partition the text tokens by segment, mirroring `build_segments`'s
    // timestamp-token splitting, then group each segment's text tokens into
    // words. We walk `result_tokens` to find segment breaks (timestamp tokens
    // strictly greater than `timestamp_begin` close a segment) while advancing a
    // cursor into `text_tokens`/`text_starts`.
    let mut per_segment: Vec<Vec<WordTiming>> = Vec::with_capacity(win_segments.len());
    let mut text_cursor = 0usize;
    let mut seg_idx = 0usize;

    // Helper: bytes for a text token id.
    let token_byte =
        |id: i32| -> Vec<u8> { tk.token_bytes(id).map(<[u8]>::to_vec).unwrap_or_default() };

    if !params.timestamps {
        // Single segment spanning the window: all text tokens are one group.
        let bytes: Vec<Vec<u8>> = text_tokens.iter().map(|&t| token_byte(t)).collect();
        let slices: Vec<&[u8]> = bytes.iter().map(Vec::as_slice).collect();
        let seg_end = win_segments
            .first()
            .and_then(|s| s.end_sec)
            .unwrap_or(seek_sec + seek_delta_cs as f64 / 100.0);
        let words = dtw::group_tokens_into_words(&slices, &text_starts, seg_end);
        per_segment.push(words);
        while per_segment.len() < win_segments.len() {
            per_segment.push(Vec::new());
        }
        return Ok(per_segment);
    }

    // Timestamped mode: re-walk `result_tokens` to recover per-segment text-token
    // spans, the same way `build_segments` does (whisper.cpp 7624-7714).
    let beg = tk.timestamp_begin;
    let mut span_start = text_cursor; // first text-token index of current segment
    let mut i = 0usize;
    while i < result_tokens.len() {
        let tok = result_tokens[i];
        if tok > beg {
            // Close the current segment at this timestamp token.
            emit_segment_words(
                &text_tokens,
                &text_starts,
                span_start,
                text_cursor,
                win_segments,
                &mut seg_idx,
                &mut per_segment,
                seek_sec,
                seek_delta_cs,
                &token_byte,
            );
            // Skip a run of consecutive timestamp tokens.
            while i + 1 < result_tokens.len() && result_tokens[i + 1] > beg {
                i += 1;
            }
            span_start = text_cursor;
        } else if !tk.is_special(tok) {
            text_cursor += 1;
        }
        i += 1;
    }
    // Open-tail segment: any text tokens after the last timestamp pair.
    if span_start < text_cursor {
        emit_segment_words(
            &text_tokens,
            &text_starts,
            span_start,
            text_cursor,
            win_segments,
            &mut seg_idx,
            &mut per_segment,
            seek_sec,
            seek_delta_cs,
            &token_byte,
        );
    }

    // Pad to match `win_segments` length (defensive: a segment whose text was
    // empty/whitespace was dropped by `build_segments`).
    while per_segment.len() < win_segments.len() {
        per_segment.push(Vec::new());
    }
    per_segment.truncate(win_segments.len());
    Ok(per_segment)
}

/// Emit the words for one segment's text-token span `[start, end)` into
/// `per_segment`, advancing `seg_idx`. Skips empty spans (which `build_segments`
/// also drops, so they must not consume a `win_segments` slot).
#[allow(clippy::too_many_arguments)]
fn emit_segment_words(
    text_tokens: &[i32],
    text_starts: &[f32],
    start: usize,
    end: usize,
    win_segments: &[TranscriptionSegment],
    seg_idx: &mut usize,
    per_segment: &mut Vec<Vec<WordTiming>>,
    seek_sec: f64,
    seek_delta_cs: i64,
    token_byte: &dyn Fn(i32) -> Vec<u8>,
) {
    if start >= end {
        return;
    }
    // Skip whitespace-only spans the same way `build_segments` drops them.
    // Build the span text by concatenating all token BYTES first, then a SINGLE
    // `from_utf8_lossy` (fix #10) — matching `Tokenizer::decode`, which joins
    // bytes before the lossy conversion. Per-token lossy decoding could split a
    // multi-byte UTF-8 character across two BPE tokens into replacement
    // characters, making this emptiness gate diverge from `build_segments`'s.
    let span_bytes: Vec<Vec<u8>> = text_tokens[start..end]
        .iter()
        .map(|&t| token_byte(t))
        .collect();
    let mut joined: Vec<u8> = Vec::new();
    for b in &span_bytes {
        joined.extend_from_slice(b);
    }
    let span_text = String::from_utf8_lossy(&joined);
    if span_text.trim().is_empty() {
        return;
    }

    // Segment end: the next segment's start_sec, else this segment's end_sec,
    // else the window end.
    let seg_end = win_segments
        .get(*seg_idx)
        .and_then(|s| s.end_sec)
        .unwrap_or(seek_sec + seek_delta_cs as f64 / 100.0);

    let slices: Vec<&[u8]> = span_bytes.iter().map(Vec::as_slice).collect();
    let words = dtw::group_tokens_into_words(&slices, &text_starts[start..end], seg_end);
    per_segment.push(words);
    *seg_idx += 1;
}

/// Resolve the working language: an explicit code is echoed; a multilingual
/// model with no language auto-detects on the first window (whisper.cpp
/// 4035-4108); English-only models use `"en"` here (the caller squashes the
/// reported language to `None` at the end).
fn resolve_language(
    m: &LoadedModel,
    params: &DecodeParams,
    full_mel: &Mel,
    checkpoint: &dyn Fn() -> FwResult<()>,
) -> FwResult<Option<String>> {
    if let Some(lang) = &params.language {
        return Ok(Some(lang.clone()));
    }
    if !m.tokenizer.is_multilingual() {
        // English-only model: language is implicitly English, no detection.
        return Ok(Some("en".to_string()));
    }
    // Auto-detect: encode the first window, forward [sot], argmax over the
    // language-token logits (whisper.cpp 4053-4107).
    let mel_window = mel::chunk_frames(full_mel, 0, FRAMES_PER_CHUNK);
    let enc = encoder::forward(&m.encoder, &mel_window, params.n_threads, checkpoint)?;
    let mut st = DecoderState::new(&m.decoder, &enc)?;
    let logits = decoder::forward_step(&m.decoder, &mut st, &[m.tokenizer.sot], checkpoint)?;

    let mut best_code = "en";
    let mut best_logit = f32::NEG_INFINITY;
    for (code, lang_id, _) in LANGUAGES {
        if *lang_id >= m.tokenizer.num_languages() {
            continue;
        }
        let tok = m.tokenizer.sot + 1 + *lang_id;
        if let Ok(idx) = usize::try_from(tok)
            && let Some(&l) = logits.get(idx)
            && l > best_logit
        {
            best_logit = l;
            best_code = code;
        }
    }
    Ok(Some(best_code.to_string()))
}

/// Decode a standard 16-bit PCM WAV into 16 kHz mono `f32` samples in `[-1, 1]`.
///
/// A minimal RIFF chunk-walker: validates the `RIFF`/`WAVE` magic, reads the
/// `fmt ` chunk (must be PCM, 16-bit), skips any intervening chunks (e.g.
/// `LIST`), and reads the `data` chunk. Multi-channel audio is downmixed to
/// mono by averaging; a non-16 kHz rate is rejected (callers normalize first).
///
/// This is a test/utility helper kept here so the gated e2e test is
/// self-contained; production input normalization lives in `crate::audio`.
#[allow(dead_code)]
pub(crate) fn read_wav_16k_mono(bytes: &[u8]) -> FwResult<Vec<f32>> {
    let rd_u32 = |b: &[u8], o: usize| -> Option<u32> {
        b.get(o..o + 4)
            .map(|s| u32::from_le_bytes([s[0], s[1], s[2], s[3]]))
    };
    let rd_u16 = |b: &[u8], o: usize| -> Option<u16> {
        b.get(o..o + 2).map(|s| u16::from_le_bytes([s[0], s[1]]))
    };
    let bad = |s: &str| FwError::InvalidRequest(format!("read_wav: {s}"));

    if bytes.len() < 12 || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return Err(bad("not a RIFF/WAVE file"));
    }
    let mut pos = 12usize;
    let mut channels = 0u16;
    let mut sample_rate = 0u32;
    let mut bits = 0u16;
    let mut data: Option<&[u8]> = None;
    while pos + 8 <= bytes.len() {
        let id = &bytes[pos..pos + 4];
        let size = rd_u32(bytes, pos + 4).ok_or_else(|| bad("truncated chunk header"))? as usize;
        let body_start = pos + 8;
        let body_end = body_start.saturating_add(size).min(bytes.len());
        match id {
            b"fmt " => {
                let body = &bytes[body_start..body_end];
                let fmt = rd_u16(body, 0).ok_or_else(|| bad("bad fmt"))?;
                if fmt != 1 {
                    return Err(bad("only PCM (fmt=1) supported"));
                }
                channels = rd_u16(body, 2).ok_or_else(|| bad("bad channels"))?;
                sample_rate = rd_u32(body, 4).ok_or_else(|| bad("bad rate"))?;
                bits = rd_u16(body, 14).ok_or_else(|| bad("bad bits"))?;
            }
            b"data" => {
                data = Some(&bytes[body_start..body_end]);
            }
            _ => {}
        }
        // Chunks are word-aligned (pad byte if odd size).
        pos = body_end + (size & 1);
    }
    if bits != 16 {
        return Err(bad("only 16-bit PCM supported"));
    }
    if sample_rate != SAMPLE_RATE as u32 {
        return Err(bad("expected 16 kHz audio"));
    }
    let channels = usize::from(channels.max(1));
    let data = data.ok_or_else(|| bad("no data chunk"))?;
    let n_frames = data.len() / (2 * channels);
    let mut out = Vec::with_capacity(n_frames);
    for f in 0..n_frames {
        let mut acc = 0i32;
        for c in 0..channels {
            let o = (f * channels + c) * 2;
            let s = i16::from_le_bytes([data[o], data[o + 1]]);
            acc += i32::from(s);
        }
        out.push((acc as f32 / channels as f32) / 32768.0);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Synthetic tokenizer for hermetic logit-filter / segment tests.
    // -----------------------------------------------------------------------

    fn hp(n_vocab: i32) -> WhisperHParams {
        WhisperHParams {
            n_vocab,
            n_audio_ctx: 1500,
            n_audio_state: 384,
            n_audio_head: 6,
            n_audio_layer: 4,
            n_text_ctx: 448,
            n_text_state: 384,
            n_text_head: 6,
            n_text_layer: 4,
            n_mels: 80,
            ftype: 1,
        }
    }

    /// English-only synthetic vocab (51864) with known special ids:
    /// eot=50256, sot=50257, ..., no_timestamps=50362, timestamp_begin=50363.
    /// Recognizable text / non-speech tokens are placed at small ids.
    fn synth_tokenizer() -> Tokenizer {
        let n_vocab = 51864i32;
        let mut v: Vec<Vec<u8>> = (0..n_vocab).map(|_| vec![b'.']).collect();
        v[1] = b" ".to_vec(); // the blank token
        v[2] = b"hello".to_vec();
        v[3] = b" world".to_vec();
        v[4] = b"(".to_vec(); // non-speech symbol
        v[5] = b" -".to_vec(); // non-speech special hyphen
        Tokenizer::from_vocab(&hp(n_vocab), v)
    }

    fn base_cfg(tk: &Tokenizer) -> FilterConfig {
        let space = (0..tk.vocab_size()).find(|&id| tk.token_bytes(id) == Some(b" ".as_slice()));
        FilterConfig {
            suppress_blank: true,
            space_token: space,
            suppress_nst: false,
            no_timestamps: false,
            max_initial_tid: None,
            max_tokens: None,
        }
    }

    fn zeros(tk: &Tokenizer) -> Vec<f32> {
        vec![0.0f32; tk.vocab_size() as usize]
    }

    /// Logits where the text region strongly dominates the timestamp region, so
    /// the sum-of-timestamp-probs forcing rule (whisper.cpp 6343-6369) does NOT
    /// fire — isolating whichever individual suppression rule is under test.
    /// (With flat/zero logits the ~1500 timestamp tokens logsumexp to a large
    /// value and the forcing rule masks all text, which is correct but masks the
    /// rule we want to observe.)
    fn text_dominant(tk: &Tokenizer) -> Vec<f32> {
        let mut v = vec![-30.0f32; tk.vocab_size() as usize];
        // Raise the low text region (everything below the specials) to dominate.
        for x in v.iter_mut().take(tk.eot as usize) {
            *x = 5.0;
        }
        v
    }

    fn is_suppressed(logits: &[f32], id: i32) -> bool {
        logits[id as usize] == f32::NEG_INFINITY
    }

    // ----- Rule 1: blank + eot suppression at step 0 only -----

    #[test]
    fn blank_and_eot_suppressed_at_step0() {
        let tk = synth_tokenizer();
        let cfg = base_cfg(&tk);
        let (logits, _) = process_logits(&tk, &cfg, zeros(&tk), &[], false, CHUNK_CS, 0);
        assert!(is_suppressed(&logits, tk.eot), "eot suppressed at step 0");
        assert!(is_suppressed(&logits, 1), "blank ' ' suppressed at step 0");

        // After one token, blank/eot are NOT suppressed by the blank rule.
        // (Use text-dominant logits so the timestamp-forcing rule doesn't mask
        // text/eot — see `text_dominant`.)
        let (logits2, _) = process_logits(&tk, &cfg, text_dominant(&tk), &[2], false, CHUNK_CS, 0);
        assert!(!is_suppressed(&logits2, tk.eot), "eot allowed after step 0");
        assert!(!is_suppressed(&logits2, 1), "blank allowed after step 0");
    }

    // ----- Rule 2: control / task / lang / sot / nosp / prev suppression -----

    #[test]
    fn control_tokens_always_suppressed() {
        let tk = synth_tokenizer();
        let cfg = base_cfg(&tk);
        let (logits, _) = process_logits(&tk, &cfg, zeros(&tk), &[2], false, CHUNK_CS, 0);
        for id in [tk.sot, tk.no_speech, tk.solm, tk.sot_prev, tk.no_timestamps] {
            assert!(is_suppressed(&logits, id), "control {id} suppressed");
        }
        // For English-only models translate/transcribe ids exist (50357/50358).
        assert!(is_suppressed(&logits, tk.translate));
        assert!(is_suppressed(&logits, tk.transcribe));
    }

    // ----- Rule 2b: non-speech suppression (opt-in) -----

    #[test]
    fn non_speech_suppressed_when_enabled() {
        let tk = synth_tokenizer();
        let mut cfg = base_cfg(&tk);
        // Off by default: "(" (id 4) and " -" (id 5) are NOT suppressed.
        // Text-dominant logits keep the forcing rule from masking text.
        let (off, _) = process_logits(&tk, &cfg, text_dominant(&tk), &[2], false, CHUNK_CS, 0);
        assert!(!is_suppressed(&off, 4), "non-speech allowed when off");
        // On: they are suppressed.
        cfg.suppress_nst = true;
        let (on, _) = process_logits(&tk, &cfg, text_dominant(&tk), &[2], false, CHUNK_CS, 0);
        assert!(is_suppressed(&on, 4), "( suppressed when nst on");
        assert!(is_suppressed(&on, 5), "' -' suppressed when nst on");
    }

    // ----- Rule 3: timestamp pairing, both branches -----

    #[test]
    fn timestamp_pairing_two_ts_back_to_back_forbids_timestamp() {
        // last two tokens both timestamps => all timestamps suppressed, text ok.
        let tk = synth_tokenizer();
        let cfg = base_cfg(&tk);
        let prev = [tk.timestamp_begin + 5, tk.timestamp_begin + 10];
        let (logits, _) = process_logits(&tk, &cfg, zeros(&tk), &prev, true, 20, 0);
        assert!(
            is_suppressed(&logits, tk.timestamp_begin + 50),
            "timestamp suppressed when two ts precede"
        );
        assert!(!is_suppressed(&logits, 2), "text token allowed");
    }

    #[test]
    fn timestamp_pairing_one_ts_open_forces_timestamp_or_eot() {
        // last token is a timestamp, penultimate is text => the pairing rule
        // masks all text in [0, eot) (whisper.cpp 6312-6314), leaving only eot
        // and timestamps selectable. To observe that eot survives (rather than
        // being re-masked by the downstream timestamp-forcing rule), give eot a
        // dominant logit so max_text_logprob (which includes eot, since
        // eot < token_beg) beats the timestamp logsumexp.
        let tk = synth_tokenizer();
        let cfg = base_cfg(&tk);
        let prev = [2i32, tk.timestamp_begin + 10];
        let mut logits = vec![-5.0f32; tk.vocab_size() as usize];
        logits[tk.eot as usize] = 20.0; // eot clearly dominant
        let (logits, _) = process_logits(&tk, &cfg, logits, &prev, true, 20, 0);
        assert!(is_suppressed(&logits, 2), "text masked when one ts open");
        assert!(
            !is_suppressed(&logits, tk.eot),
            "eot survives (pair-before-eot allowed)"
        );
    }

    // ----- Rule 4: timestamp monotonicity -----

    #[test]
    fn timestamp_monotonicity_masks_earlier_timestamps() {
        let tk = synth_tokenizer();
        let cfg = base_cfg(&tk);
        // has_ts with seek_delta=100cs => tid0 = 50; timestamps below beg+50 masked.
        let (logits, _) = process_logits(&tk, &cfg, zeros(&tk), &[2], true, 100, 0);
        assert!(
            is_suppressed(&logits, tk.timestamp_begin + 10),
            "earlier timestamp masked"
        );
        assert!(
            !is_suppressed(&logits, tk.timestamp_begin + 60),
            "later timestamp allowed"
        );
    }

    // ----- Rule 5: max_initial_ts clamp -----

    #[test]
    fn max_initial_ts_clamps_first_step() {
        let tk = synth_tokenizer();
        let mut cfg = base_cfg(&tk);
        // precision = 30/1500 = 0.02s; max_initial_ts=1.0s => tid0 = 50.
        cfg.max_initial_tid = Some(50);
        let (logits, _) = process_logits(&tk, &cfg, zeros(&tk), &[], false, CHUNK_CS, 0);
        // timestamps beyond beg+50 masked on the initial step.
        assert!(
            is_suppressed(&logits, tk.timestamp_begin + 51),
            "initial timestamp > max clamped"
        );
        assert!(
            !is_suppressed(&logits, tk.timestamp_begin + 50),
            "initial timestamp at max allowed"
        );
    }

    // ----- Rule 6: logsumexp timestamp-forcing -----

    #[test]
    fn logsumexp_forces_timestamp_when_ts_mass_exceeds_text() {
        let tk = synth_tokenizer();
        let cfg = base_cfg(&tk);
        let mut logits = vec![0.0f32; tk.vocab_size() as usize];
        // Make text uniformly low, then spread a large amount of mass across many
        // timestamp tokens so their logsumexp exceeds any single text logit.
        for l in &mut logits {
            *l = -10.0;
        }
        logits[2] = 1.0; // one text token with a modest logit
        let beg = tk.timestamp_begin as usize;
        for i in beg..(beg + 200).min(logits.len()) {
            logits[i] = 0.5;
        }
        let (out, _) = process_logits(&tk, &cfg, logits, &[2], false, 0, 0);
        // All text logits (below beg) must be masked: a timestamp is forced.
        assert!(is_suppressed(&out, 2), "text masked: timestamp forced");
        assert!(is_suppressed(&out, 0), "all text masked");
        // A timestamp remains selectable.
        assert!(!is_suppressed(&out, beg as i32 + 1));
    }

    #[test]
    fn logsumexp_keeps_text_when_text_dominates() {
        let tk = synth_tokenizer();
        let cfg = base_cfg(&tk);
        let mut logits = vec![-20.0f32; tk.vocab_size() as usize];
        logits[2] = 10.0; // one very strong text token
        let beg = tk.timestamp_begin as usize;
        for l in &mut logits[beg..beg + 3] {
            *l = -5.0;
        }
        let (out, lp) = process_logits(&tk, &cfg, logits, &[2], false, 0, 0);
        assert!(!is_suppressed(&out, 2), "strong text not masked");
        let (tok, _) = argmax(&out, &lp);
        assert_eq!(tok, 2, "argmax selects the strong text token");
    }

    // ----- Rule 7: no_timestamps mode masks every timestamp -----

    #[test]
    fn no_timestamps_mode_masks_all_timestamps() {
        let tk = synth_tokenizer();
        let mut cfg = base_cfg(&tk);
        cfg.no_timestamps = true;
        let (logits, _) = process_logits(&tk, &cfg, zeros(&tk), &[2], false, CHUNK_CS, 0);
        assert!(is_suppressed(&logits, tk.timestamp_begin));
        assert!(is_suppressed(&logits, tk.timestamp_begin + 100));
        assert!(!is_suppressed(&logits, 2), "text still allowed");
    }

    // ----- Rule 8: max_tokens EOT-forcing filter (fix #6) -----

    #[test]
    fn max_tokens_forces_eot_when_budget_reached() {
        // Fix #6 (whisper.cpp 6234-6238): with timestamps on, once the running
        // token count reaches the budget, all text (< eot) is masked, leaving
        // only eot/timestamps selectable.
        let tk = synth_tokenizer();
        let mut cfg = base_cfg(&tk);
        cfg.max_tokens = Some(4);
        // Below budget: text still allowed (use text-dominant logits so the
        // forcing rule doesn't mask text).
        let (under, _) = process_logits(&tk, &cfg, text_dominant(&tk), &[2], false, CHUNK_CS, 3);
        assert!(!is_suppressed(&under, 2), "text allowed below budget");
        // At budget: all text below eot masked.
        let (at, _) = process_logits(&tk, &cfg, text_dominant(&tk), &[2], false, CHUNK_CS, 4);
        assert!(is_suppressed(&at, 2), "text masked at budget");
        assert!(is_suppressed(&at, 0), "all text masked at budget");
        // A timestamp remains selectable.
        assert!(!is_suppressed(&at, tk.timestamp_begin + 1));
    }

    #[test]
    fn max_tokens_filter_inert_in_no_timestamps_mode() {
        // The EOT-force is guarded by `!no_timestamps` upstream.
        let tk = synth_tokenizer();
        let mut cfg = base_cfg(&tk);
        cfg.no_timestamps = true;
        cfg.max_tokens = Some(2);
        let (out, _) = process_logits(&tk, &cfg, text_dominant(&tk), &[2], false, CHUNK_CS, 5);
        assert!(
            !is_suppressed(&out, 2),
            "text not masked in no-timestamps mode"
        );
    }

    #[test]
    fn max_tokens_filter_disabled_when_none_or_zero() {
        let tk = synth_tokenizer();
        let mut cfg = base_cfg(&tk);
        cfg.max_tokens = None;
        let (out, _) = process_logits(&tk, &cfg, text_dominant(&tk), &[2], false, CHUNK_CS, 99);
        assert!(!is_suppressed(&out, 2), "text allowed when budget is None");
        cfg.max_tokens = Some(0);
        let (out0, _) = process_logits(&tk, &cfg, text_dominant(&tk), &[2], false, CHUNK_CS, 99);
        assert!(!is_suppressed(&out0, 2), "text allowed when budget is 0");
    }

    // -----------------------------------------------------------------------
    // Segment building.
    // -----------------------------------------------------------------------

    #[test]
    fn segments_from_timestamp_pairs() {
        let tk = synth_tokenizer();
        let beg = tk.timestamp_begin;
        // <|0.00|> hello world <|3.00|>  => one segment [0.00, 3.00].
        // 3.00s = 150 steps.
        let tokens = vec![beg, 2, 3, beg + 150];
        let plogs = vec![-0.1f32; tokens.len()];
        let segs = build_segments(&tk, &tokens, &plogs, 0, 3000, i64::MAX, true);
        assert_eq!(segs.len(), 1);
        assert_eq!(segs[0].text, "hello world");
        assert!((segs[0].start_sec.unwrap() - 0.0).abs() < 1e-9);
        assert!((segs[0].end_sec.unwrap() - 3.0).abs() < 1e-9);
        assert!(segs[0].confidence.unwrap() > 0.0);
    }

    #[test]
    fn segment_confidence_excludes_closing_timestamp_token() {
        // Fix #8: confidence averages only the segment's TEXT tokens (hello,
        // world), not the leading/closing timestamp tokens. Give the timestamp
        // tokens a very negative plog and the text tokens 0.0; the confidence
        // must reflect only the text (exp(0)=1), proving the ts plogs are
        // excluded. Observed delta vs the old all-token average: previously the
        // closing/opening ts plogs (-5.0 each) dragged the mean to ~-2.5 →
        // conf≈0.08; now text-only → conf=1.0.
        let tk = synth_tokenizer();
        let beg = tk.timestamp_begin;
        let tokens = vec![beg, 2, 3, beg + 150];
        // ts plogs very negative, text plogs perfect (0.0).
        let plogs = vec![-5.0f32, 0.0, 0.0, -5.0f32];
        let segs = build_segments(&tk, &tokens, &plogs, 0, 3000, i64::MAX, true);
        assert_eq!(segs.len(), 1);
        let conf = segs[0].confidence.unwrap();
        assert!(
            (conf - 1.0).abs() < 1e-9,
            "text-only confidence should be exp(0)=1, got {conf}"
        );
    }

    #[test]
    fn text_confidence_none_for_timestamp_only_span() {
        let tk = synth_tokenizer();
        let beg = tk.timestamp_begin;
        // A span with no text tokens → None.
        assert!(text_confidence(&tk, &[beg, beg + 10], &[-0.1, -0.1]).is_none());
    }

    #[test]
    fn segments_two_pairs() {
        let tk = synth_tokenizer();
        let beg = tk.timestamp_begin;
        // <|0|> hello <|1|> world <|2|>
        let tokens = vec![beg, 2, beg + 50, 3, beg + 100];
        let plogs = vec![-0.2f32; tokens.len()];
        let segs = build_segments(&tk, &tokens, &plogs, 0, 2000, i64::MAX, true);
        assert_eq!(segs.len(), 2);
        assert_eq!(segs[0].text, "hello");
        assert!((segs[0].end_sec.unwrap() - 1.0).abs() < 1e-9);
        assert_eq!(segs[1].text, "world");
        assert!((segs[1].start_sec.unwrap() - 1.0).abs() < 1e-9);
        assert!((segs[1].end_sec.unwrap() - 2.0).abs() < 1e-9);
    }

    #[test]
    fn segments_open_tail() {
        let tk = synth_tokenizer();
        let beg = tk.timestamp_begin;
        // <|0|> hello <|1|> world   (no closing timestamp) => open tail closed at
        // seek + seek_delta.
        let tokens = vec![beg, 2, beg + 50, 3];
        let plogs = vec![-0.2f32; tokens.len()];
        let segs = build_segments(&tk, &tokens, &plogs, 0, 1500, i64::MAX, true);
        assert_eq!(segs.len(), 2);
        assert_eq!(segs[1].text, "world");
        // open tail end = seek_cs(0) + seek_delta(1500cs) = 15.0s.
        assert!((segs[1].end_sec.unwrap() - 15.0).abs() < 1e-9);
    }

    #[test]
    fn segments_single_no_timestamps() {
        let tk = synth_tokenizer();
        // No-timestamps mode: text tokens only, one segment spanning the window.
        let tokens = vec![2i32, 3];
        let plogs = vec![-0.3f32; 2];
        let segs = build_segments(&tk, &tokens, &plogs, 500, 3000, i64::MAX, false);
        assert_eq!(segs.len(), 1);
        assert_eq!(segs[0].text, "hello world");
        assert!((segs[0].start_sec.unwrap() - 5.0).abs() < 1e-9);
        assert!((segs[0].end_sec.unwrap() - 35.0).abs() < 1e-9);
    }

    #[test]
    fn segments_with_window_offset() {
        let tk = synth_tokenizer();
        let beg = tk.timestamp_begin;
        // Window starting at 30s (seek_cs=3000): <|0|> hello <|1|>.
        let tokens = vec![beg, 2, beg + 50];
        let plogs = vec![-0.1f32; tokens.len()];
        let segs = build_segments(&tk, &tokens, &plogs, 3000, 3000, i64::MAX, true);
        assert_eq!(segs.len(), 1);
        assert!((segs[0].start_sec.unwrap() - 30.0).abs() < 1e-9);
        assert!((segs[0].end_sec.unwrap() - 31.0).abs() < 1e-9);
    }

    #[test]
    fn segments_clamped_to_real_audio_length() {
        // Fix #1: a closing timestamp token pointing past the real (unpadded)
        // audio length must NOT yield an end_sec beyond the clip duration.
        // Synthetic last window: <|0.00|> hello world <|10.00|> but the real
        // audio is only 6.00 s (seek_end_cs = 600). The segment end must clamp
        // to 6.00 s.
        let tk = synth_tokenizer();
        let beg = tk.timestamp_begin;
        // <|0.00|> hello world <|10.00|> ; 10.00 s = 500 steps.
        let tokens = vec![beg, 2, 3, beg + 500];
        let plogs = vec![-0.1f32; tokens.len()];
        // Window at seek 0, full 30 s delta, but real audio only 6.00 s.
        let segs = build_segments(&tk, &tokens, &plogs, 0, 3000, 600, true);
        assert_eq!(segs.len(), 1);
        assert_eq!(segs[0].text, "hello world");
        assert!((segs[0].start_sec.unwrap() - 0.0).abs() < 1e-9);
        // End clamped from 10.00 s to the real audio length 6.00 s.
        assert!(
            (segs[0].end_sec.unwrap() - 6.0).abs() < 1e-9,
            "end {} clamped to real length 6.0",
            segs[0].end_sec.unwrap()
        );

        // Open-tail clamp: text after the last timestamp pair, seek_delta beyond
        // real length, must also clamp.
        let tokens2 = vec![beg, 2, beg + 200, 3]; // <|0|> hello <|4|> world (open)
        let plogs2 = vec![-0.1f32; tokens2.len()];
        let segs2 = build_segments(&tk, &tokens2, &plogs2, 0, 3000, 600, true);
        assert_eq!(segs2.len(), 2);
        // open tail would close at seek_delta=30 s, clamped to 6.0 s.
        assert!((segs2[1].end_sec.unwrap() - 6.0).abs() < 1e-9);
    }

    #[test]
    fn short_tail_prompt_clearing_predicate() {
        // Fix #2 (whisper.cpp 7046-7051): clear on a non-first window whose
        // remaining audio is < 5 s; never on the first window (seek == 0).
        // seek_end = 1000 cs (10 s).
        // First window: never clears.
        assert!(!should_clear_short_tail_prompt(0, 1000));
        // Non-first window, > 5 s left: no clear (seek 400 -> 6 s left).
        assert!(!should_clear_short_tail_prompt(400, 1000));
        // Non-first window, exactly 5 s left: clears (boundary `>=`).
        assert!(should_clear_short_tail_prompt(500, 1000));
        // Non-first window, < 5 s left: clears.
        assert!(should_clear_short_tail_prompt(600, 1000));
        // Last partial window near the very end.
        assert!(should_clear_short_tail_prompt(900, 1000));
    }

    #[test]
    fn confidence_is_clamped_and_monotone() {
        // mean logprob 0 => exp(0)=1 (clamped); very negative => ~0.
        assert_eq!(confidence(&[0.0, 0.0]), Some(1.0));
        let c = confidence(&[-2.0, -2.0]).unwrap();
        assert!(c > 0.0 && c < 1.0);
        assert!(confidence(&[]).is_none());
    }

    // -----------------------------------------------------------------------
    // WAV reader.
    // -----------------------------------------------------------------------

    #[test]
    fn wav_reader_round_trips_a_synthetic_clip() {
        // Build a 16kHz mono 16-bit WAV with a tiny ramp and read it back.
        let samples_i16: Vec<i16> = (0..8).map(|i| (i * 1000) as i16).collect();
        let data_bytes: Vec<u8> = samples_i16.iter().flat_map(|s| s.to_le_bytes()).collect();
        let mut wav = Vec::new();
        wav.extend_from_slice(b"RIFF");
        wav.extend_from_slice(&(36 + data_bytes.len() as u32).to_le_bytes());
        wav.extend_from_slice(b"WAVE");
        wav.extend_from_slice(b"fmt ");
        wav.extend_from_slice(&16u32.to_le_bytes());
        wav.extend_from_slice(&1u16.to_le_bytes()); // PCM
        wav.extend_from_slice(&1u16.to_le_bytes()); // mono
        wav.extend_from_slice(&16000u32.to_le_bytes());
        wav.extend_from_slice(&32000u32.to_le_bytes()); // byte rate
        wav.extend_from_slice(&2u16.to_le_bytes()); // block align
        wav.extend_from_slice(&16u16.to_le_bytes()); // bits
        wav.extend_from_slice(b"data");
        wav.extend_from_slice(&(data_bytes.len() as u32).to_le_bytes());
        wav.extend_from_slice(&data_bytes);

        let out = read_wav_16k_mono(&wav).unwrap();
        assert_eq!(out.len(), 8);
        assert!((out[0] - 0.0).abs() < 1e-9);
        assert!((out[1] - 1000.0 / 32768.0).abs() < 1e-6);
    }

    // -----------------------------------------------------------------------
    // Gated end-to-end tests against the real tiny.en model + jfk.wav.
    // -----------------------------------------------------------------------

    /// The reference transcript whisper-cli produced (see
    /// `tests/fixtures/native/jfk_tiny_reference.json`), trimmed.
    const JFK_REFERENCE: &str = "And so my fellow Americans ask not what your country can do for you ask what you can do for your country.";

    fn load_jfk_samples() -> Option<Vec<f32>> {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/native/jfk.wav");
        let bytes = std::fs::read(path).ok()?;
        read_wav_16k_mono(&bytes).ok()
    }

    fn load_tiny_en() -> Option<LoadedModel> {
        let path = super::super::find_model_file("tiny.en")?;
        let model = GgmlModel::load(&path).ok()?;
        LoadedModel::from_ggml(model).ok()
    }

    fn noop() -> FwResult<()> {
        Ok(())
    }

    fn e2e_params() -> DecodeParams {
        DecodeParams {
            language: None,
            translate: false,
            timestamps: true,
            n_threads: 4,
            max_text_ctx: None,
            ..DecodeParams::default()
        }
    }

    #[test]
    fn gated_e2e_jfk_tiny_en_matches_reference() {
        let (Some(model), Some(samples)) = (load_tiny_en(), load_jfk_samples()) else {
            eprintln!("SKIP gated_e2e_jfk_tiny_en: tiny.en model or jfk.wav missing");
            return;
        };
        let params = e2e_params();
        let t = std::time::Instant::now();
        let out = transcribe_samples(&model, &samples, &params, &noop).expect("transcribe");
        let elapsed = t.elapsed();
        eprintln!(
            "e2e jfk (11s): {elapsed:?} for {} samples, {} segments",
            samples.len(),
            out.segments.len()
        );

        let joined: String = out
            .segments
            .iter()
            .map(|s| s.text.trim())
            .collect::<Vec<_>>()
            .join(" ");
        eprintln!("PRODUCED:  {joined}");
        eprintln!("REFERENCE: {JFK_REFERENCE}");
        assert_eq!(
            joined.trim(),
            JFK_REFERENCE,
            "greedy temp-0 transcript must match whisper-cli reference EXACTLY"
        );

        // Windows stats populated.
        assert!(!out.windows.is_empty(), "window stats populated");
        assert!(out.windows[0].tokens > 0);
        // English-only model reports no language.
        assert_eq!(out.language, None);

        // Segment timestamps within 0.3s of the reference fixture.
        // Reference: [0.00, 7.96] and [7.96, 10.76].
        assert!(out.segments.len() >= 2, "expected >= 2 segments");
        let s0_end = out.segments[0].end_sec.unwrap();
        assert!(
            (s0_end - 7.96).abs() < 0.3,
            "segment 0 end {s0_end} within 0.3s of 7.96"
        );
        let last_end = out.segments.last().unwrap().end_sec.unwrap();
        assert!(
            (last_end - 10.76).abs() < 0.3,
            "last segment end {last_end} within 0.3s of 10.76"
        );
    }

    #[test]
    fn gated_e2e_deterministic_across_runs() {
        let (Some(model), Some(samples)) = (load_tiny_en(), load_jfk_samples()) else {
            eprintln!("SKIP gated_e2e_deterministic: tiny.en model or jfk.wav missing");
            return;
        };
        let params = e2e_params();
        let a = transcribe_samples(&model, &samples, &params, &noop).expect("run a");
        let b = transcribe_samples(&model, &samples, &params, &noop).expect("run b");
        let ja: String = a.segments.iter().map(|s| s.text.clone()).collect();
        let jb: String = b.segments.iter().map(|s| s.text.clone()).collect();
        assert_eq!(ja, jb, "greedy temp-0 must be deterministic across runs");
    }

    #[test]
    fn gated_e2e_multi_window_monotonic_timestamps() {
        let (Some(model), Some(samples)) = (load_tiny_en(), load_jfk_samples()) else {
            eprintln!("SKIP gated_e2e_multi_window: tiny.en model or jfk.wav missing");
            return;
        };
        // Concatenate jfk 3x (~33s) to force more than one 30s window.
        let mut long = Vec::with_capacity(samples.len() * 3);
        for _ in 0..3 {
            long.extend_from_slice(&samples);
        }
        let params = e2e_params();
        let out = transcribe_samples(&model, &long, &params, &noop).expect("transcribe long");
        eprintln!(
            "multi-window: {} windows, {} segments",
            out.windows.len(),
            out.segments.len()
        );
        assert!(out.windows.len() >= 2, "expected > 1 window for ~33s audio");

        // Timestamps monotonic non-decreasing across the whole transcript,
        // including the window boundary.
        let mut prev_end = -1.0f64;
        for seg in &out.segments {
            let start = seg.start_sec.unwrap();
            let end = seg.end_sec.unwrap();
            assert!(
                start + 1e-6 >= prev_end - 1e-6,
                "segment start {start} must not precede previous end {prev_end}"
            );
            assert!(end + 1e-6 >= start, "segment end {end} >= start {start}");
            prev_end = end;
        }

        // The sentence's signature word "country" should appear at least twice
        // (once per repeated clip, conservatively).
        let joined: String = out.segments.iter().map(|s| s.text.to_lowercase()).collect();
        let occurrences = joined.matches("country").count();
        assert!(
            occurrences >= 2,
            "expected the repeated sentence at least twice, got {occurrences} 'country' hits in: {joined}"
        );
    }

    /// Gated end-to-end DTW word-timestamp check (bd-rjsx).
    ///
    /// Verified reference (whisper-cli `-m ggml-tiny.en.bin -f jfk.wav -ml 1
    /// --no-prints` and `-dtw tiny.en`, run 2026-06-04): the JFK clip contains
    /// the word "ask" twice — first occurrence starts at **3.29 s**, second at
    /// **7.96 s**. The bead's sanity band [7.0, 9.5] s references the *second*
    /// "ask"; that band is the hard requirement and our native DTW lands the
    /// second "ask" at **≈8.66 s** (observed 2026-06-04), inside it. For the
    /// first "ask", our native engine's DTW lands it at **≈3.88 s** — within
    /// ~0.6 s of whisper-cli's 3.29 s reference, the expected small drift
    /// between our pure-Rust forward pass and whisper.cpp's; we bound it with a
    /// ±0.75 s band around the reference.
    #[test]
    fn gated_e2e_dtw_word_timestamps_jfk_tiny_en() {
        let (Some(model), Some(samples)) = (load_tiny_en(), load_jfk_samples()) else {
            eprintln!("SKIP gated_e2e_dtw_word_timestamps: tiny.en model or jfk.wav missing");
            return;
        };
        let params = DecodeParams {
            language: None,
            translate: false,
            timestamps: true,
            n_threads: 4,
            max_text_ctx: None,
            word_timestamps: true,
            model_hint: Some("tiny.en".to_owned()),
        };
        let out = transcribe_samples(&model, &samples, &params, &noop).unwrap();

        let word_timings = out
            .word_timings
            .as_ref()
            .expect("word_timings present when requested");
        assert_eq!(
            word_timings.len(),
            out.segments.len(),
            "word_timings 1:1 with segments"
        );

        // Flatten to a single ordered word list.
        let mut words: Vec<&WordTiming> = word_timings.iter().flatten().collect();
        words.sort_by(|a, b| a.start_sec.partial_cmp(&b.start_sec).unwrap());

        assert!(!words.is_empty(), "DTW produced no words");

        // Word count == whitespace word count of the transcript.
        let transcript_words = JFK_REFERENCE.split_whitespace().count();
        let emitted_words: usize = word_timings.iter().map(Vec::len).sum();
        assert_eq!(
            emitted_words, transcript_words,
            "word count {emitted_words} != transcript word count {transcript_words}"
        );

        // Strictly monotonic, non-overlapping within the global ordering.
        for w in words.windows(2) {
            assert!(
                w[0].end_sec <= w[1].start_sec + 1e-6,
                "overlap: {:?} then {:?}",
                w[0],
                w[1]
            );
            assert!(w[0].start_sec <= w[0].end_sec, "reversed word: {:?}", w[0]);
        }

        // Find the two "ask" occurrences (normalize punctuation/case).
        let asks: Vec<f64> = words
            .iter()
            .filter(|w| {
                w.text
                    .trim_matches(|c: char| !c.is_alphanumeric())
                    .eq_ignore_ascii_case("ask")
            })
            .map(|w| w.start_sec)
            .collect();
        assert!(
            asks.len() >= 2,
            "expected two 'ask' occurrences, got {asks:?}"
        );

        // First "ask" ≈ 3.29 s whisper-cli reference (native observed ≈3.88 s);
        // ±0.75 s band covers the cross-implementation drift.
        assert!(
            (asks[0] - 3.29).abs() <= 0.75,
            "first 'ask' start {} not within 3.29 ± 0.75 s",
            asks[0]
        );
        // Second "ask" — the bead's hard requirement: inside [7.0, 9.5] s.
        // whisper-cli reference 7.96 s; our native DTW observed ≈8.66 s
        // (2026-06-04), both comfortably inside the band.
        assert!(
            (7.0..=9.5).contains(&asks[1]),
            "second 'ask' {} outside the bead's sanity band [7.0, 9.5]",
            asks[1]
        );
    }

    /// DTW word timestamps are deterministic across runs (bd-rjsx).
    #[test]
    fn gated_e2e_dtw_word_timestamps_deterministic() {
        let (Some(model), Some(samples)) = (load_tiny_en(), load_jfk_samples()) else {
            eprintln!("SKIP gated_e2e_dtw_deterministic: tiny.en model or jfk.wav missing");
            return;
        };
        let params = DecodeParams {
            language: None,
            translate: false,
            timestamps: true,
            n_threads: 4,
            max_text_ctx: None,
            word_timestamps: true,
            model_hint: Some("tiny.en".to_owned()),
        };
        let a = transcribe_samples(&model, &samples, &params, &noop).unwrap();
        let b = transcribe_samples(&model, &samples, &params, &noop).unwrap();
        assert_eq!(a.word_timings, b.word_timings);
    }
}
