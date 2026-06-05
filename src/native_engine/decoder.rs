//! Whisper text decoder forward pass with an incremental KV cache.
//!
//! This module ports whisper.cpp's `whisper_build_graph_decoder()` (see
//! `src/whisper.cpp`, the `cur = d_te[token] + d_pe[pos]` token/position
//! embedding through the final `ggml_mul_mat(d_te, cur)` logits product). It
//! computes, for a batch of one or more newly-decoded tokens appended to a
//! running self-attention KV cache, the next-token logits for the **last**
//! position only.
//!
//! # Block structure (per decoder layer)
//!
//! Mirroring whisper exactly:
//!
//! ```text
//! x = token_emb[token] + pos_emb[cache_len + i]            (input embedding)
//! for each layer:
//!     x = x + self_out( self_attn( attn_ln(x) ) )          (causal, KV cache)
//!     x = x + cross_out( cross_attn( cross_attn_ln(x) ) )  (encoder K/V, no mask)
//!     x = x + mlp( mlp_ln(x) )
//! x = decoder.ln(x)
//! logits = x[last] @ token_embedding^T                     (weight tying)
//! ```
//!
//! Self-attention: query `w+b`, key `w` (**no bias**), value `w+b`, out
//! `w+b`; Q and K are each scaled by `d_head^-0.25` (handled inside
//! [`nn::attention`]); causal over the whole cache. Cross-attention: query
//! `w+b`, key `w` (no bias), value `w+b`, out `w+b`; the cross K/V are
//! **precomputed once per audio window** from the encoder output and reused
//! across every decode step. Cross-attention is **unmasked**.
//!
//! # Logits product / weight tying — memory rationale
//!
//! The output projection ties weights with the token embedding:
//! `logits = x @ token_embedding^T`. The embedding is stored naturally as
//! `[n_vocab, n_state]` (its ggml on-disk logical shape). Pre-transposing it
//! to `[n_state, n_vocab]` for a `matmul_bias`-style `x @ w_t` would store a
//! **second** full copy of the largest tensor in the model
//! (`51864 * 384 * 4 B ≈ 80 MB` for tiny, `≈ 265 MB` for large) — pure
//! waste. Instead we keep the single `[n_vocab, n_state]` copy and compute
//! the last row's logits as
//! `logits^T = token_emb @ x_last^T` → `[n_vocab, 1]` via one
//! [`nn::matmul`], then read off the column. This is exactly whisper.cpp's
//! `ggml_mul_mat(d_te, cur)` (`d_te` is `[n_state, n_vocab]` in ggml's
//! column-fastest convention, i.e. our `[n_vocab, n_state]`), and we only
//! ever materialize logits for the **last** position, not the whole batch.
//!
//! # Cross-attention weight recording (DTW word timestamps)
//!
//! Word-level timestamps (bead bd-rjsx) align text to audio via DTW over the
//! softmaxed cross-attention weights of selected heads. [`nn::attention`]
//! does not surface those intermediate weights, so the cross path is
//! implemented **locally** in this file (`q·k^T → softmax → @v`) rather than
//! by calling [`nn::attention`]: that lets us capture the per-head
//! `[tokens, enc_frames]` softmax matrices when recording is enabled. The
//! recording is opt-in ([`DecoderState::record_cross_attn`]) and off by
//! default, so the steady-state decode path pays nothing.

#![allow(clippy::module_name_repetitions)]

use super::nn::{self, KvCache};
use super::{Mat, WhisperHParams};
use crate::error::{FwError, FwResult};

/// Layer-norm epsilon used throughout whisper (`hparams.eps = 1e-5f`).
const LN_EPS: f32 = 1e-5;

/// Pre-transposed linear weight `[in, out]` plus its optional bias `[out]`.
///
/// Whisper linear layers are `y = x @ W^T + b` with `W` shaped `[out, in]`;
/// we transpose to `[in, out]` at load time so every forward matmul is a
/// contiguous `[m, in] x [in, out]` (see [`nn::matmul_bias`]).
#[derive(Debug, Clone)]
struct Linear {
    /// Pre-transposed weight, shape `[in, out]`.
    w_t: Mat,
    /// Optional bias, length `out`.
    bias: Option<Vec<f32>>,
}

impl Linear {
    /// Apply `x @ w_t (+ bias)`.
    fn forward(&self, x: &Mat) -> FwResult<Mat> {
        nn::matmul_bias(x, &self.w_t, self.bias.as_deref())
    }
}

/// Affine layer-norm parameters (`weight`, `bias`), each length `n_state`.
#[derive(Debug, Clone)]
struct LayerNorm {
    w: Vec<f32>,
    b: Vec<f32>,
}

impl LayerNorm {
    /// In-place normalize each row of `x` then apply the affine transform.
    fn apply(&self, x: &mut Mat) {
        nn::layer_norm(x, &self.w, &self.b, LN_EPS);
    }
}

/// All weights of a single decoder block.
#[derive(Debug, Clone)]
struct DecoderLayer {
    attn_ln: LayerNorm,
    attn_q: Linear,
    /// Self-attention key projection — **no bias** in whisper.
    attn_k: Linear,
    attn_v: Linear,
    attn_out: Linear,

    cross_attn_ln: LayerNorm,
    cross_attn_q: Linear,
    /// Cross-attention key projection — **no bias** in whisper.
    cross_attn_k: Linear,
    cross_attn_v: Linear,
    cross_attn_out: Linear,

    mlp_ln: LayerNorm,
    mlp_0: Linear,
    mlp_2: Linear,
}

/// Fully-loaded decoder weights, ready for [`forward_step`].
#[derive(Debug, Clone)]
pub struct DecoderWeights {
    /// Token embedding, kept in its natural `[n_vocab, n_state]` orientation
    /// and reused (transposed implicitly) for the tied output projection. See
    /// the module-level "Logits product" note for the memory rationale.
    token_embedding: Mat,
    /// Learned positional embedding `[n_text_ctx, n_state]`.
    positional_embedding: Mat,
    layers: Vec<DecoderLayer>,
    /// Final pre-logits layer norm (`decoder.ln`).
    ln: LayerNorm,

    n_state: usize,
    n_head: usize,
    n_vocab: usize,
    n_text_ctx: usize,
}

/// Pull a tensor and assert its logical shape, returning the flat f32 values.
fn tensor_shaped(
    model: &crate::native_engine::ggml::GgmlModel,
    name: &str,
    expected: &[usize],
) -> FwResult<Vec<f32>> {
    let (shape, values) = model.tensor_f32(name)?;
    if shape != expected {
        return Err(FwError::InvalidRequest(format!(
            "decoder tensor '{name}' shape {shape:?} != expected {expected:?}"
        )));
    }
    Ok(values)
}

/// Load a `[rows, cols]` matrix tensor with shape validation.
fn load_mat(
    model: &crate::native_engine::ggml::GgmlModel,
    name: &str,
    rows: usize,
    cols: usize,
) -> FwResult<Mat> {
    let values = tensor_shaped(model, name, &[rows, cols])?;
    Ok(Mat::from_vec(rows, cols, values))
}

/// Load a 1-D vector tensor (e.g. a bias) with length validation.
fn load_vec(
    model: &crate::native_engine::ggml::GgmlModel,
    name: &str,
    len: usize,
) -> FwResult<Vec<f32>> {
    tensor_shaped(model, name, &[len])
}

/// Transpose a `[rows, cols]` matrix into a fresh `[cols, rows]` matrix.
fn transpose(m: &Mat) -> Mat {
    Mat::from_vec(m.cols, m.rows, nn::transpose_parallel(&m.data, m.rows, m.cols))
}

/// Load a whisper linear layer (`[out, in]` weight) pre-transposed to
/// `[in, out]`, plus an optional `[out]` bias.
fn load_linear(
    model: &crate::native_engine::ggml::GgmlModel,
    weight_name: &str,
    bias_name: Option<&str>,
    out_dim: usize,
    in_dim: usize,
) -> FwResult<Linear> {
    let w = load_mat(model, weight_name, out_dim, in_dim)?; // [out, in]
    let w_t = transpose(&w); // [in, out]
    let bias = match bias_name {
        Some(name) => Some(load_vec(model, name, out_dim)?),
        None => None,
    };
    Ok(Linear { w_t, bias })
}

/// Load a layer-norm (`weight`, `bias`), each length `n_state`.
fn load_layer_norm(
    model: &crate::native_engine::ggml::GgmlModel,
    prefix: &str,
    n_state: usize,
) -> FwResult<LayerNorm> {
    let w = load_vec(model, &format!("{prefix}.weight"), n_state)?;
    let b = load_vec(model, &format!("{prefix}.bias"), n_state)?;
    Ok(LayerNorm { w, b })
}

impl DecoderWeights {
    /// Load and pre-transpose every decoder weight from a parsed ggml model.
    ///
    /// All linear weights are transposed to `[in, out]`; the token embedding
    /// is kept `[n_vocab, n_state]`. Shapes are validated against the model's
    /// hyper-parameters, so a malformed/mismatched file is rejected here
    /// rather than corrupting the forward pass.
    ///
    /// # Errors
    /// [`FwError::InvalidRequest`] if any expected tensor is missing or has an
    /// unexpected shape; propagates [`ggml::GgmlModel::tensor_f32`] decode
    /// errors.
    pub fn from_ggml(model: &crate::native_engine::ggml::GgmlModel) -> FwResult<Self> {
        let hp: WhisperHParams = model.hparams;
        let n_state = usize::try_from(hp.n_text_state)
            .map_err(|_| FwError::InvalidRequest("negative n_text_state".into()))?;
        let n_head = usize::try_from(hp.n_text_head)
            .map_err(|_| FwError::InvalidRequest("negative n_text_head".into()))?;
        let n_layer = usize::try_from(hp.n_text_layer)
            .map_err(|_| FwError::InvalidRequest("negative n_text_layer".into()))?;
        let n_vocab = usize::try_from(hp.n_vocab)
            .map_err(|_| FwError::InvalidRequest("negative n_vocab".into()))?;
        let n_text_ctx = usize::try_from(hp.n_text_ctx)
            .map_err(|_| FwError::InvalidRequest("negative n_text_ctx".into()))?;
        if n_head == 0 || !n_state.is_multiple_of(n_head) {
            return Err(FwError::InvalidRequest(format!(
                "n_text_head {n_head} must divide n_text_state {n_state}"
            )));
        }

        let token_embedding = load_mat(model, "decoder.token_embedding.weight", n_vocab, n_state)?;
        let positional_embedding =
            load_mat(model, "decoder.positional_embedding", n_text_ctx, n_state)?;
        let ln = load_layer_norm(model, "decoder.ln", n_state)?;

        let mlp_hidden = 4 * n_state; // whisper MLP expansion factor is 4.
        let mut layers = Vec::with_capacity(n_layer);
        for i in 0..n_layer {
            let p = format!("decoder.blocks.{i}");
            let layer = DecoderLayer {
                attn_ln: load_layer_norm(model, &format!("{p}.attn_ln"), n_state)?,
                attn_q: load_linear(
                    model,
                    &format!("{p}.attn.query.weight"),
                    Some(&format!("{p}.attn.query.bias")),
                    n_state,
                    n_state,
                )?,
                attn_k: load_linear(
                    model,
                    &format!("{p}.attn.key.weight"),
                    None, // key has no bias
                    n_state,
                    n_state,
                )?,
                attn_v: load_linear(
                    model,
                    &format!("{p}.attn.value.weight"),
                    Some(&format!("{p}.attn.value.bias")),
                    n_state,
                    n_state,
                )?,
                attn_out: load_linear(
                    model,
                    &format!("{p}.attn.out.weight"),
                    Some(&format!("{p}.attn.out.bias")),
                    n_state,
                    n_state,
                )?,
                cross_attn_ln: load_layer_norm(model, &format!("{p}.cross_attn_ln"), n_state)?,
                cross_attn_q: load_linear(
                    model,
                    &format!("{p}.cross_attn.query.weight"),
                    Some(&format!("{p}.cross_attn.query.bias")),
                    n_state,
                    n_state,
                )?,
                cross_attn_k: load_linear(
                    model,
                    &format!("{p}.cross_attn.key.weight"),
                    None, // key has no bias
                    n_state,
                    n_state,
                )?,
                cross_attn_v: load_linear(
                    model,
                    &format!("{p}.cross_attn.value.weight"),
                    Some(&format!("{p}.cross_attn.value.bias")),
                    n_state,
                    n_state,
                )?,
                cross_attn_out: load_linear(
                    model,
                    &format!("{p}.cross_attn.out.weight"),
                    Some(&format!("{p}.cross_attn.out.bias")),
                    n_state,
                    n_state,
                )?,
                mlp_ln: load_layer_norm(model, &format!("{p}.mlp_ln"), n_state)?,
                mlp_0: load_linear(
                    model,
                    &format!("{p}.mlp.0.weight"),
                    Some(&format!("{p}.mlp.0.bias")),
                    mlp_hidden,
                    n_state,
                )?,
                mlp_2: load_linear(
                    model,
                    &format!("{p}.mlp.2.weight"),
                    Some(&format!("{p}.mlp.2.bias")),
                    n_state,
                    mlp_hidden,
                )?,
            };
            layers.push(layer);
        }

        Ok(Self {
            token_embedding,
            positional_embedding,
            layers,
            ln,
            n_state,
            n_head,
            n_vocab,
            n_text_ctx,
        })
    }

    /// Number of decoder layers.
    #[must_use]
    pub fn n_layer(&self) -> usize {
        self.layers.len()
    }

    /// Number of attention heads.
    #[must_use]
    pub fn n_head(&self) -> usize {
        self.n_head
    }

    /// Hidden width.
    #[must_use]
    pub fn n_state(&self) -> usize {
        self.n_state
    }

    /// Vocabulary size (logits length).
    #[must_use]
    pub fn n_vocab(&self) -> usize {
        self.n_vocab
    }

    /// Maximum decode context length (KV-cache capacity).
    #[must_use]
    pub fn n_text_ctx(&self) -> usize {
        self.n_text_ctx
    }
}

/// Per-window decoder state: self-attention KV cache, precomputed cross K/V,
/// and an optional cross-attention weight recording buffer.
///
/// One [`DecoderState`] is built per audio window from that window's encoder
/// output (which fixes the cross-attention keys/values for the whole window)
/// and is then advanced token-by-token via [`forward_step`].
#[derive(Debug)]
pub struct DecoderState {
    /// One self-attention KV cache per layer.
    kv: Vec<KvCache>,
    /// Precomputed cross-attention keys per layer, `[enc_frames, n_state]`,
    /// already scaled by `d_head^-0.25` (so the local cross path scales only
    /// the query — see [`forward_step`]).
    cross_k: Vec<Mat>,
    /// Precomputed cross-attention values per layer, `[enc_frames, n_state]`.
    cross_v: Vec<Mat>,
    /// Number of tokens currently in the self-attention cache.
    len: usize,
    /// Number of encoder frames (cross-attention key/value length).
    enc_frames: usize,
    n_state: usize,
    n_head: usize,
    /// When set, [`forward_step`] records the softmaxed cross-attention
    /// weights of the most recent call into [`Self::cross_attn_weights`].
    pub record_cross_attn: bool,
    /// Recorded cross-attention weights from the last [`forward_step`], one
    /// `[tokens, enc_frames]` matrix per `(layer, head)` in
    /// `layer * n_head + head` order. Empty unless [`Self::record_cross_attn`]
    /// was set for that call.
    cross_attn_weights: Vec<Mat>,
}

impl DecoderState {
    /// Build per-window state, precomputing cross-attention K/V for every
    /// layer from `encoder_out` (`[enc_frames, n_state]`).
    ///
    /// The cross keys are scaled by `d_head^-0.25` here (matching whisper's
    /// `whisper_build_graph_cross`, which scales `Kcross` once at precompute
    /// time); the query is scaled by the same factor at decode time, so the
    /// `q·k` product carries the full `d_head^-0.5` factor.
    ///
    /// # Errors
    /// [`FwError::InvalidRequest`] if `encoder_out` width != `n_state`;
    /// propagates projection matmul errors.
    pub fn new(w: &DecoderWeights, encoder_out: &Mat) -> FwResult<Self> {
        if encoder_out.cols != w.n_state {
            return Err(FwError::InvalidRequest(format!(
                "encoder_out width {} != decoder n_state {}",
                encoder_out.cols, w.n_state
            )));
        }
        let enc_frames = encoder_out.rows;
        let d_head = w.n_state / w.n_head;
        let k_scale = (d_head as f32).powf(-0.25);

        let mut cross_k = Vec::with_capacity(w.layers.len());
        let mut cross_v = Vec::with_capacity(w.layers.len());
        for layer in &w.layers {
            // Kcross = (encoder_out @ Wk^T) * d_head^-0.25  (no bias).
            let mut k = layer.cross_attn_k.forward(encoder_out)?;
            for val in &mut k.data {
                *val *= k_scale;
            }
            // Vcross = encoder_out @ Wv^T + bv.
            let v = layer.cross_attn_v.forward(encoder_out)?;
            cross_k.push(k);
            cross_v.push(v);
        }

        let kv = (0..w.layers.len())
            .map(|_| KvCache::new(w.n_text_ctx, w.n_state))
            .collect();

        Ok(Self {
            kv,
            cross_k,
            cross_v,
            len: 0,
            enc_frames,
            n_state: w.n_state,
            n_head: w.n_head,
            record_cross_attn: false,
            cross_attn_weights: Vec::new(),
        })
    }

    /// Reset the self-attention cache (and recorded weights) for reuse within
    /// the same window. The precomputed cross K/V are retained.
    pub fn reset(&mut self) {
        for c in &mut self.kv {
            c.reset();
        }
        self.len = 0;
        self.cross_attn_weights.clear();
    }

    /// Number of tokens currently in the self-attention cache.
    #[must_use]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether no tokens have been decoded yet.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Number of encoder frames the cross-attention attends over.
    #[must_use]
    pub fn enc_frames(&self) -> usize {
        self.enc_frames
    }

    /// Recorded cross-attention weights from the most recent [`forward_step`]
    /// (when [`Self::record_cross_attn`] was set): one `[tokens, enc_frames]`
    /// matrix per `(layer, head)` in `layer * n_head + head` order.
    #[must_use]
    pub fn cross_attn_weights(&self) -> &[Mat] {
        &self.cross_attn_weights
    }
}

/// Build the per-step input embedding `token_emb[token] + pos_emb[cache_len+i]`
/// for every token in the batch, returning a `[tokens, n_state]` matrix.
fn embed_tokens(w: &DecoderWeights, tokens: &[i32], cache_len: usize) -> FwResult<Mat> {
    let n_state = w.n_state;
    let mut x = vec![0.0f32; tokens.len() * n_state];
    for (i, &tok) in tokens.iter().enumerate() {
        let tok_idx = usize::try_from(tok)
            .map_err(|_| FwError::InvalidRequest(format!("negative token id {tok}")))?;
        if tok_idx >= w.n_vocab {
            return Err(FwError::InvalidRequest(format!(
                "token id {tok_idx} >= n_vocab {}",
                w.n_vocab
            )));
        }
        let pos = cache_len + i;
        if pos >= w.n_text_ctx {
            return Err(FwError::InvalidRequest(format!(
                "decode position {pos} >= n_text_ctx {}",
                w.n_text_ctx
            )));
        }
        let dst = &mut x[i * n_state..(i + 1) * n_state];
        let te = w.token_embedding.row(tok_idx);
        let pe = w.positional_embedding.row(pos);
        for ((d, &t), &p) in dst.iter_mut().zip(te).zip(pe) {
            *d = t + p;
        }
    }
    Ok(Mat::from_vec(tokens.len(), n_state, x))
}

/// Local multi-head cross-attention with optional weight capture.
///
/// `q` is `[tokens, n_state]`; `cross_k`/`cross_v` are
/// `[enc_frames, n_state]` (the keys already pre-scaled by `d_head^-0.25`).
/// The query is scaled by `d_head^-0.25` so the `q·k` scores carry the full
/// `d_head^-0.5` factor — identical to [`nn::attention`]'s convention but
/// implemented here so the softmaxed `[tokens, enc_frames]` per-head weights
/// can be recorded. There is **no mask** (cross-attention is bidirectional
/// over the encoder frames).
///
/// When `record` is set, `recorded` is populated with one
/// `[tokens, enc_frames]` matrix per head (in head order).
fn cross_attention(
    q: &Mat,
    cross_k: &Mat,
    cross_v: &Mat,
    n_head: usize,
    record: bool,
    recorded: &mut Vec<Mat>,
) -> FwResult<Mat> {
    let n_state = q.cols;
    if !n_state.is_multiple_of(n_head) || n_head == 0 {
        return Err(FwError::InvalidRequest(format!(
            "cross_attention: n_head {n_head} must divide n_state {n_state}"
        )));
    }
    let tq = q.rows;
    let tk = cross_k.rows;
    let d_head = n_state / n_head;
    let q_scale = (d_head as f32).powf(-0.25);

    let mut out = vec![0.0f32; tq * n_state];
    for h in 0..n_head {
        let base = h * d_head;

        // Scaled query head [tq, d_head].
        let mut qh = vec![0.0f32; tq * d_head];
        for i in 0..tq {
            let src = &q.row(i)[base..base + d_head];
            let dst = &mut qh[i * d_head..(i + 1) * d_head];
            for (d, &s) in dst.iter_mut().zip(src) {
                *d = s * q_scale;
            }
        }
        let qh = Mat::from_vec(tq, d_head, qh);

        // Key head transposed [d_head, tk] for a contiguous matmul.
        let mut kh_t = vec![0.0f32; d_head * tk];
        for j in 0..tk {
            let src = &cross_k.row(j)[base..base + d_head];
            for (d, &s) in src.iter().enumerate() {
                kh_t[d * tk + j] = s;
            }
        }
        let kh_t = Mat::from_vec(d_head, tk, kh_t);

        // scores = qh @ kh^T -> [tq, tk], softmax per query row (no mask).
        let mut scores = nn::matmul(&qh, &kh_t)?;
        nn::softmax_rows(&mut scores);

        if record {
            recorded.push(scores.clone());
        }

        // Value head [tk, d_head]; out_h = scores @ vh.
        let mut vh = vec![0.0f32; tk * d_head];
        for j in 0..tk {
            let src = &cross_v.row(j)[base..base + d_head];
            vh[j * d_head..(j + 1) * d_head].copy_from_slice(src);
        }
        let vh = Mat::from_vec(tk, d_head, vh);
        let out_h = nn::matmul(&scores, &vh)?; // [tq, d_head]

        for i in 0..tq {
            let src = &out_h.data[i * d_head..(i + 1) * d_head];
            out[i * n_state + base..i * n_state + base + d_head].copy_from_slice(src);
        }
    }
    Ok(Mat::from_vec(tq, n_state, out))
}

/// Run one decoder forward step over `tokens`, returning the next-token
/// logits for the **last** token only (length `n_vocab`).
///
/// `tokens` is the newly-decoded batch: typically a single token for an
/// incremental step, or the whole prompt for the initial prefill. Each
/// token's self-attention key/value is appended to `st`'s per-layer KV cache,
/// and positional embeddings are indexed at `st.len() + i`. The checkpoint
/// closure is invoked **between** layers so a caller can cancel a long
/// decode; [`nn`] kernels themselves are uncancellable (see that module's
/// cancellation contract).
///
/// When [`DecoderState::record_cross_attn`] is set, the softmaxed
/// cross-attention weights of this call are captured into
/// [`DecoderState::cross_attn_weights`] (one `[tokens, enc_frames]` per
/// `(layer, head)`), for downstream DTW word-timestamp alignment.
///
/// # Errors
/// [`FwError::InvalidRequest`] on an empty batch, an out-of-range token id,
/// a decode position past `n_text_ctx`, or a width mismatch; propagates the
/// checkpoint closure's error and any kernel error.
pub fn forward_step(
    w: &DecoderWeights,
    st: &mut DecoderState,
    tokens: &[i32],
    checkpoint: &dyn Fn() -> FwResult<()>,
) -> FwResult<Vec<f32>> {
    if tokens.is_empty() {
        return Err(FwError::InvalidRequest(
            "forward_step: empty token batch".into(),
        ));
    }
    if st.n_state != w.n_state || st.n_head != w.n_head {
        return Err(FwError::InvalidRequest(
            "forward_step: state/weights shape mismatch".into(),
        ));
    }

    let cache_len = st.len;
    let mut x = embed_tokens(w, tokens, cache_len)?;

    if st.record_cross_attn {
        st.cross_attn_weights.clear();
    }

    for (li, layer) in w.layers.iter().enumerate() {
        // ── self-attention (causal, over the KV cache) ──
        let mut h = x.clone();
        layer.attn_ln.apply(&mut h);
        let q = layer.attn_q.forward(&h)?;
        let k = layer.attn_k.forward(&h)?; // no bias
        let v = layer.attn_v.forward(&h)?;
        let attn = nn::attention_with_cache(&q, &k, &v, w.n_head, &mut st.kv[li])?;
        let attn_out = layer.attn_out.forward(&attn)?;
        add_into(&mut x, &attn_out);

        // ── cross-attention (encoder K/V, no mask, optional recording) ──
        let mut hc = x.clone();
        layer.cross_attn_ln.apply(&mut hc);
        let qc = layer.cross_attn_q.forward(&hc)?;
        let cross = cross_attention(
            &qc,
            &st.cross_k[li],
            &st.cross_v[li],
            w.n_head,
            st.record_cross_attn,
            &mut st.cross_attn_weights,
        )?;
        let cross_out = layer.cross_attn_out.forward(&cross)?;
        add_into(&mut x, &cross_out);

        // ── MLP ──
        let mut hm = x.clone();
        layer.mlp_ln.apply(&mut hm);
        let mut ff = layer.mlp_0.forward(&hm)?;
        nn::gelu(&mut ff);
        let ff = layer.mlp_2.forward(&ff)?;
        add_into(&mut x, &ff);

        // Per-layer cancellation point (nn kernels are uncancellable).
        if li + 1 < w.layers.len() {
            checkpoint()?;
        }
    }

    // Advance the logical cache length once (all layers appended in lockstep).
    st.len += tokens.len();

    // Final layer norm, then logits for the last position only.
    w.ln.apply(&mut x);
    let last = x.rows - 1;
    let x_last = Mat::from_vec(1, w.n_state, x.row(last).to_vec());
    logits_last(w, &x_last)
}

/// Tied output projection for a single position.
///
/// Computes `logits^T = token_emb @ x_last^T` → `[n_vocab, 1]` from the
/// `[n_vocab, n_state]` embedding and the `[1, n_state]` last hidden row,
/// avoiding a second transposed embedding copy (see the module-level memory
/// note). Returns the flat `[n_vocab]` logits.
fn logits_last(w: &DecoderWeights, x_last: &Mat) -> FwResult<Vec<f32>> {
    // x_last^T is [n_state, 1].
    let x_t = Mat::from_vec(w.n_state, 1, x_last.data.clone());
    let logits = nn::matmul(&w.token_embedding, &x_t)?; // [n_vocab, 1]
    Ok(logits.data)
}

/// In-place `dst += src` (same shape assumed; residual add).
fn add_into(dst: &mut Mat, src: &Mat) {
    for (d, &s) in dst.data.iter_mut().zip(src.data.iter()) {
        *d += s;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::native_engine::find_model_file;
    use crate::native_engine::ggml::GgmlModel;

    /// Deterministic LCG -> f32 in [-1, 1), matching nn.rs's generator style.
    struct Lcg(u64);
    impl Lcg {
        fn new(seed: u64) -> Self {
            Self(seed)
        }
        fn next_u32(&mut self) -> u32 {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (self.0 >> 32) as u32
        }
        fn next_f32(&mut self) -> f32 {
            (self.next_u32() as f32 / u32::MAX as f32) * 2.0 - 1.0
        }
        fn mat(&mut self, rows: usize, cols: usize) -> Mat {
            let data = (0..rows * cols).map(|_| self.next_f32() * 0.2).collect();
            Mat::from_vec(rows, cols, data)
        }
        fn vec(&mut self, n: usize) -> Vec<f32> {
            (0..n).map(|_| self.next_f32() * 0.2).collect()
        }
    }

    /// Synthetic decoder hyper-parameters for hermetic tests.
    const N_STATE: usize = 8;
    const N_HEAD: usize = 2;
    const N_LAYER: usize = 2;
    const N_VOCAB: usize = 16;
    const N_CTX: usize = 32;

    /// Build a small but structurally complete [`DecoderWeights`] in memory.
    fn synthetic_weights(seed: u64) -> DecoderWeights {
        let mut rng = Lcg::new(seed);
        let lin = |rng: &mut Lcg, out: usize, inp: usize, bias: bool| {
            // Store pre-transposed [in, out] directly.
            Linear {
                w_t: rng.mat(inp, out),
                bias: if bias { Some(rng.vec(out)) } else { None },
            }
        };
        let ln = |rng: &mut Lcg| LayerNorm {
            w: {
                let mut v = rng.vec(N_STATE);
                for x in &mut v {
                    *x += 1.0; // keep scales near 1 for sane activations
                }
                v
            },
            b: rng.vec(N_STATE),
        };
        let mlp_hidden = 4 * N_STATE;
        let mut layers = Vec::new();
        for _ in 0..N_LAYER {
            layers.push(DecoderLayer {
                attn_ln: ln(&mut rng),
                attn_q: lin(&mut rng, N_STATE, N_STATE, true),
                attn_k: lin(&mut rng, N_STATE, N_STATE, false),
                attn_v: lin(&mut rng, N_STATE, N_STATE, true),
                attn_out: lin(&mut rng, N_STATE, N_STATE, true),
                cross_attn_ln: ln(&mut rng),
                cross_attn_q: lin(&mut rng, N_STATE, N_STATE, true),
                cross_attn_k: lin(&mut rng, N_STATE, N_STATE, false),
                cross_attn_v: lin(&mut rng, N_STATE, N_STATE, true),
                cross_attn_out: lin(&mut rng, N_STATE, N_STATE, true),
                mlp_ln: ln(&mut rng),
                mlp_0: lin(&mut rng, mlp_hidden, N_STATE, true),
                mlp_2: lin(&mut rng, N_STATE, mlp_hidden, true),
            });
        }
        DecoderWeights {
            token_embedding: rng.mat(N_VOCAB, N_STATE),
            positional_embedding: rng.mat(N_CTX, N_STATE),
            layers,
            ln: ln(&mut rng),
            n_state: N_STATE,
            n_head: N_HEAD,
            n_vocab: N_VOCAB,
            n_text_ctx: N_CTX,
        }
    }

    fn noop_checkpoint() -> FwResult<()> {
        Ok(())
    }

    #[test]
    fn forward_step_shapes() {
        let w = synthetic_weights(1);
        let mut rng = Lcg::new(99);
        let enc = rng.mat(6, N_STATE);
        let mut st = DecoderState::new(&w, &enc).unwrap();
        assert_eq!(st.enc_frames(), 6);
        assert!(st.is_empty());

        let logits = forward_step(&w, &mut st, &[1], &noop_checkpoint).unwrap();
        assert_eq!(logits.len(), N_VOCAB);
        assert!(logits.iter().all(|v| v.is_finite()));
        assert_eq!(st.len(), 1);

        let logits2 = forward_step(&w, &mut st, &[2], &noop_checkpoint).unwrap();
        assert_eq!(logits2.len(), N_VOCAB);
        assert_eq!(st.len(), 2);
    }

    #[test]
    fn incremental_equals_batch_for_last_logits() {
        let w = synthetic_weights(2);
        let mut rng = Lcg::new(7);
        let enc = rng.mat(6, N_STATE);
        let (a, b) = (3i32, 5i32);

        // Incremental: feed [a] then [b]; keep the b-logits.
        let mut st_inc = DecoderState::new(&w, &enc).unwrap();
        let _ = forward_step(&w, &mut st_inc, &[a], &noop_checkpoint).unwrap();
        let inc_b = forward_step(&w, &mut st_inc, &[b], &noop_checkpoint).unwrap();

        // Batch: feed [a, b] in one shot; logits are for the last token (b).
        let mut st_batch = DecoderState::new(&w, &enc).unwrap();
        let batch_b = forward_step(&w, &mut st_batch, &[a, b], &noop_checkpoint).unwrap();

        let max = inc_b
            .iter()
            .zip(&batch_b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max);
        assert!(max < 1e-4, "incremental vs batch last-logit diff {max}");
    }

    #[test]
    fn causality_earlier_logits_independent_of_future() {
        // Batch [a, b1] vs [a, b2]: the logits *for position a* (recomputed by
        // taking the first row) must be identical regardless of the later
        // token, since self-attention is causal. We verify by comparing the
        // a-logits obtained from two separate single-token incremental decodes
        // that share the same prefix but diverge afterward.
        let w = synthetic_weights(3);
        let mut rng = Lcg::new(11);
        let enc = rng.mat(6, N_STATE);
        let a = 4i32;

        let mut st1 = DecoderState::new(&w, &enc).unwrap();
        let a_logits_1 = forward_step(&w, &mut st1, &[a], &noop_checkpoint).unwrap();
        // Alter a LATER token in st1.
        let _ = forward_step(&w, &mut st1, &[7], &noop_checkpoint).unwrap();

        let mut st2 = DecoderState::new(&w, &enc).unwrap();
        let a_logits_2 = forward_step(&w, &mut st2, &[a], &noop_checkpoint).unwrap();
        let _ = forward_step(&w, &mut st2, &[9], &noop_checkpoint).unwrap();

        // The a-logits captured *before* the divergent token are identical.
        for (x, y) in a_logits_1.iter().zip(&a_logits_2) {
            assert!((x - y).abs() < 1e-6, "a-logits changed: {x} vs {y}");
        }
    }

    #[test]
    fn cross_attn_dependence_on_encoder_out() {
        let w = synthetic_weights(4);
        let mut rng = Lcg::new(13);
        let enc_a = rng.mat(6, N_STATE);
        let mut enc_b = enc_a.clone();
        for v in &mut enc_b.data {
            *v += 0.5; // perturb the encoder output
        }

        let mut st_a = DecoderState::new(&w, &enc_a).unwrap();
        let la = forward_step(&w, &mut st_a, &[1], &noop_checkpoint).unwrap();

        let mut st_b = DecoderState::new(&w, &enc_b).unwrap();
        let lb = forward_step(&w, &mut st_b, &[1], &noop_checkpoint).unwrap();

        let changed = la.iter().zip(&lb).any(|(x, y)| (x - y).abs() > 1e-4);
        assert!(changed, "logits should depend on encoder_out");
    }

    #[test]
    fn cross_attn_recording_shape_and_normalization() {
        let w = synthetic_weights(5);
        let mut rng = Lcg::new(17);
        let enc_frames = 6;
        let enc = rng.mat(enc_frames, N_STATE);
        let mut st = DecoderState::new(&w, &enc).unwrap();
        st.record_cross_attn = true;

        let n_tokens = 3;
        let toks = [1i32, 2, 3];
        let _ = forward_step(&w, &mut st, &toks, &noop_checkpoint).unwrap();

        let rec = st.cross_attn_weights();
        // One [tokens, enc_frames] matrix per (layer, head).
        assert_eq!(rec.len(), N_LAYER * N_HEAD);
        for m in rec {
            assert_eq!(m.rows, n_tokens);
            assert_eq!(m.cols, enc_frames);
            // Each query row is a softmax distribution: sums to ~1.
            for r in 0..m.rows {
                let s: f32 = m.row(r).iter().sum();
                assert!((s - 1.0).abs() < 1e-4, "cross-attn row sum {s} != 1");
                assert!(m.row(r).iter().all(|&v| v >= 0.0));
            }
        }
    }

    #[test]
    fn recording_off_by_default() {
        let w = synthetic_weights(6);
        let mut rng = Lcg::new(19);
        let enc = rng.mat(6, N_STATE);
        let mut st = DecoderState::new(&w, &enc).unwrap();
        let _ = forward_step(&w, &mut st, &[1, 2], &noop_checkpoint).unwrap();
        assert!(st.cross_attn_weights().is_empty());
    }

    #[test]
    fn reset_clears_cache() {
        let w = synthetic_weights(7);
        let mut rng = Lcg::new(23);
        let enc = rng.mat(6, N_STATE);
        let mut st = DecoderState::new(&w, &enc).unwrap();
        let l1 = forward_step(&w, &mut st, &[3], &noop_checkpoint).unwrap();
        let _ = forward_step(&w, &mut st, &[4], &noop_checkpoint).unwrap();
        assert_eq!(st.len(), 2);
        st.reset();
        assert!(st.is_empty());
        // After reset, decoding the same first token reproduces its logits.
        let l1b = forward_step(&w, &mut st, &[3], &noop_checkpoint).unwrap();
        for (x, y) in l1.iter().zip(&l1b) {
            assert!((x - y).abs() < 1e-6, "reset should restore initial state");
        }
    }

    #[test]
    fn checkpoint_cancellation_propagates() {
        let w = synthetic_weights(8);
        let mut rng = Lcg::new(29);
        let enc = rng.mat(6, N_STATE);
        let mut st = DecoderState::new(&w, &enc).unwrap();
        let cancel = || -> FwResult<()> { Err(FwError::Cancelled("stop".into())) };
        let err = forward_step(&w, &mut st, &[1], &cancel);
        // With N_LAYER=2, the checkpoint fires once (between layer 0 and 1).
        assert!(matches!(err, Err(FwError::Cancelled(_))));
    }

    #[test]
    fn rejects_bad_inputs() {
        let w = synthetic_weights(9);
        let mut rng = Lcg::new(31);
        let enc = rng.mat(6, N_STATE);
        let mut st = DecoderState::new(&w, &enc).unwrap();
        // Empty batch.
        assert!(forward_step(&w, &mut st, &[], &noop_checkpoint).is_err());
        // Out-of-range token id.
        assert!(forward_step(&w, &mut st, &[N_VOCAB as i32], &noop_checkpoint).is_err());
        // Negative token id.
        assert!(forward_step(&w, &mut st, &[-1], &noop_checkpoint).is_err());
        // Encoder width mismatch.
        let bad_enc = Mat::zeros(6, N_STATE + 1);
        assert!(DecoderState::new(&w, &bad_enc).is_err());
    }

    /// Gated: validate `from_ggml` against the real tiny.en model and run a
    /// zero-encoder forward, asserting a finite, valid-id logits vector.
    #[test]
    fn real_tiny_en_from_ggml_and_forward() {
        let Some(path) = find_model_file("tiny.en") else {
            eprintln!("SKIP real_tiny_en_from_ggml_and_forward: ggml-tiny.en.bin not found");
            return;
        };
        let model = GgmlModel::load(&path).expect("load tiny.en");
        let w = DecoderWeights::from_ggml(&model).expect("from_ggml");
        assert_eq!(w.n_vocab(), 51864);
        assert_eq!(w.n_state(), 384);
        assert_eq!(w.n_head(), 6);
        assert_eq!(w.n_layer(), 4);
        assert_eq!(w.n_text_ctx(), 448);

        // Zero encoder output [1500, 384]; build state and forward the SOT
        // token (50257 is whisper's <|startoftranscript|> for tiny.en).
        let enc = Mat::zeros(1500, 384);
        let mut st = DecoderState::new(&w, &enc).expect("decoder state");
        let sot = 50257i32;
        let logits = forward_step(&w, &mut st, &[sot], &noop_checkpoint).expect("forward");
        assert_eq!(logits.len(), 51864);
        assert!(
            logits.iter().all(|v| v.is_finite()),
            "logits must be finite"
        );
        let argmax = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap();
        assert!(argmax < 51864, "argmax {argmax} must be a valid id");
    }
}
