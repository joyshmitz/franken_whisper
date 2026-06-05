//! Inference micro-op kernels facade + KV-cache multi-head attention.
//!
//! This module is the numerical heart of the native engine: a thin,
//! row-major-`Mat`-centric set of building blocks that the encoder and
//! decoder compose into transformer forward passes.
//!
//! # Kernel integration choices (frankentorch `ft-kernel-cpu`)
//!
//! The single big win from FrankenTorch is its rayon-parallel,
//! `matrixmultiply`-backed sgemm. We delegate **all** matrix multiplies to
//! [`ft_kernel_cpu::matmul_tensor_contiguous_f32`], constructing the
//! required [`ft_core::TensorMeta`] for a contiguous 2-D f32 CPU tensor via
//! [`TensorMeta::from_shape`]`(vec![rows, cols], DType::F32, Device::Cpu)`
//! (this is exactly how `ft-dispatch` builds contiguous metas — no custom
//! strides or storage offset are needed for our row-major `Mat`).
//!
//! For the small, per-row activation ops (`layer_norm`, `softmax_rows`,
//! `gelu`) we implement locally rather than routing through the ft kernels.
//! Rationale, documented per-op below: the ft entry points (e.g.
//! [`ft_kernel_cpu::softmax_dim_tensor_contiguous_f32`]) operate over a
//! generic strided `(outer, dim, inner)` decomposition and return a fresh
//! `Vec`, whereas our hot path wants in-place row updates with f64
//! accumulation (layer_norm) and exact whisper.cpp tanh-GELU semantics
//! (the ft `gelu_value_f32` is the *erf* form — wrong for whisper). Keeping
//! these local avoids an allocation + copy round-trip and a semantic
//! mismatch, while the asymptotically dominant matmuls still get the
//! parallel kernel.
//!
//! # Cancellation contract
//!
//! This module is intentionally **pure**: no function here takes a
//! cancellation / checkpoint closure and none can be cancelled mid-call.
//! The project's `&dyn Fn() -> FwResult<()>` checkpoint contract is honored
//! by *callers* (encoder/decoder), which invoke the checkpoint **between**
//! layer calls — every individual op here is bounded and fast enough that
//! per-op cancellation would only add noise. Keeping nn.rs pure also makes
//! every function trivially testable and free of hidden control flow.
//!
//! # Scaling convention (attention)
//!
//! [`attention`] follows the openai/whisper convention of scaling **both**
//! Q and K by `d_head^-0.25` before the QK^T product, which is numerically
//! equivalent to whisper.cpp's single `1/sqrt(d_head)` factor on the QK
//! scores: `(q·d^-0.25)·(k·d^-0.25) = q·k·d^-0.5 = q·k / sqrt(d)`. See the
//! [`attention`] docs for the whisper.cpp citation.

#![allow(clippy::module_name_repetitions)]

use ft_core::{DType, Device, TensorMeta};

use super::Mat;
use crate::error::{FwError, FwResult};

/// Build a contiguous 2-D f32 CPU `TensorMeta` for a `[rows, cols]` tensor.
///
/// Mirrors how `ft-dispatch` constructs metas for a plain contiguous
/// tensor: `from_shape` fills in row-major strides and zero storage offset,
/// which is exactly the layout of our row-major [`Mat`].
fn meta_2d(rows: usize, cols: usize) -> TensorMeta {
    TensorMeta::from_shape(vec![rows, cols], DType::F32, Device::Cpu)
}

/// Map a FrankenTorch `KernelError` into [`FwError`].
///
/// Kernel failures here are almost always shape/contract violations from
/// our own callers (mismatched dimensions), so [`FwError::InvalidRequest`]
/// is the right bucket; the kernel's `Display` carries the specifics.
fn kernel_err(e: ft_kernel_cpu::KernelError) -> FwError {
    FwError::InvalidRequest(format!("ft-kernel-cpu: {e}"))
}

/// `[m,k] x [k,n] -> [m,n]`, delegating to FrankenTorch's parallel sgemm.
///
/// # Errors
/// Returns [`FwError::InvalidRequest`] if the inner dimensions disagree
/// (`a.cols != b.rows`) or the kernel rejects the shapes.
pub fn matmul(a: &Mat, b: &Mat) -> FwResult<Mat> {
    if a.cols != b.rows {
        return Err(FwError::InvalidRequest(format!(
            "matmul inner dim mismatch: [{},{}] x [{},{}]",
            a.rows, a.cols, b.rows, b.cols
        )));
    }
    let (m, k, n) = (a.rows, a.cols, b.cols);
    let lhs_meta = meta_2d(m, k);
    let rhs_meta = meta_2d(k, n);
    let data = ft_kernel_cpu::matmul_tensor_contiguous_f32(&a.data, &b.data, &lhs_meta, &rhs_meta)
        .map_err(kernel_err)?;
    Ok(Mat::from_vec(m, n, data))
}

/// Affine projection `x @ w_t (+ bias)`.
///
/// Whisper linear layers are `y = x @ W^T + b` with `W` shaped
/// `[out, in]`. To keep every matmul a **contiguous** `[m,k] x [k,n]`, the
/// model loader pre-transposes `W` to `w_t` of shape `[in, out]` once at
/// load time, so this function is a plain `x @ w_t` plus a broadcast bias
/// add over rows. `bias` (when present) must have length `w_t.cols`
/// (= `out`).
///
/// # Errors
/// [`FwError::InvalidRequest`] on a dimension mismatch (`x.cols != w_t.rows`
/// or `bias.len() != w_t.cols`) or kernel rejection.
pub fn matmul_bias(x: &Mat, w_t: &Mat, bias: Option<&[f32]>) -> FwResult<Mat> {
    let mut out = matmul(x, w_t)?;
    if let Some(b) = bias {
        if b.len() != out.cols {
            return Err(FwError::InvalidRequest(format!(
                "matmul_bias: bias len {} != out cols {}",
                b.len(),
                out.cols
            )));
        }
        let cols = out.cols;
        for row in out.data.chunks_mut(cols) {
            for (v, &bv) in row.iter_mut().zip(b.iter()) {
                *v += bv;
            }
        }
    }
    Ok(out)
}

/// In-place per-row layer normalization with affine scale/shift.
///
/// For each row: subtract the row mean, divide by `sqrt(var + eps)`, then
/// apply `w * x + b` element-wise. Mean and variance use **f64**
/// accumulation for numerical stability (whisper hidden dims of 384..1280
/// make naive f32 sums lossy). Rows are independent.
///
/// Implemented locally (not via an ft kernel) because we want the in-place,
/// fused mean/var/affine pass over each row rather than an allocating
/// reduction; see module docs.
///
/// `w` and `b` must each have length `x.cols`; on mismatch this is a no-op
/// guard (callers pass model-shaped weights, so a mismatch is a load bug).
pub fn layer_norm(x: &mut Mat, w: &[f32], b: &[f32], eps: f32) {
    let cols = x.cols;
    if cols == 0 || w.len() != cols || b.len() != cols {
        return;
    }
    let eps = f64::from(eps);
    for row in x.data.chunks_mut(cols) {
        let mut sum = 0.0f64;
        for &v in row.iter() {
            sum += f64::from(v);
        }
        let mean = sum / cols as f64;
        let mut var = 0.0f64;
        for &v in row.iter() {
            let d = f64::from(v) - mean;
            var += d * d;
        }
        var /= cols as f64;
        let inv_std = 1.0 / (var + eps).sqrt();
        for ((v, &wi), &bi) in row.iter_mut().zip(w.iter()).zip(b.iter()) {
            let normed = (f64::from(*v) - mean) * inv_std;
            *v = (normed * f64::from(wi) + f64::from(bi)) as f32;
        }
    }
}

/// whisper.cpp coefficient `sqrt(2/pi)` (`SQRT_2_OVER_PI` in ggml `vec.h`).
const GELU_SQRT_2_OVER_PI: f32 = 0.797_884_6;
/// whisper.cpp `GELU_COEF_A`.
const GELU_COEF_A: f32 = 0.044_715;

/// In-place exact whisper.cpp tanh-approximation GELU.
///
/// Matches ggml `ggml_gelu_f32` in
/// `ggml/src/ggml-cpu/vec.h`:
/// `0.5*x*(1 + tanh(sqrt(2/pi) * x * (1 + 0.044715*x*x)))`,
/// which is the standard `0.5x(1 + tanh(sqrt(2/pi)(x + 0.044715 x^3)))`
/// (note `x*(1 + a*x*x) == x + a*x^3`). We deliberately use this tanh form
/// rather than ft's `gelu_value_f32`, which is the *erf* GELU and would
/// diverge from whisper's activations.
pub fn gelu(x: &mut Mat) {
    for v in &mut x.data {
        let x = *v;
        *v = 0.5 * x * (1.0 + (GELU_SQRT_2_OVER_PI * x * (1.0 + GELU_COEF_A * x * x)).tanh());
    }
}

/// In-place numerically-stable per-row softmax (max-subtract).
///
/// Each row is softmaxed independently: subtract the row max before
/// exponentiating (so large logits like `1e30` don't overflow to `inf`),
/// then normalize by the row sum. Implemented locally to operate in place
/// over `Mat` rows; see module docs.
pub fn softmax_rows(x: &mut Mat) {
    let cols = x.cols;
    if cols == 0 {
        return;
    }
    for row in x.data.chunks_mut(cols) {
        let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        if !max.is_finite() {
            // All -inf (e.g. fully masked row): leave as-is to avoid NaNs.
            continue;
        }
        let mut sum = 0.0f32;
        for v in row.iter_mut() {
            let e = (*v - max).exp();
            *v = e;
            sum += e;
        }
        if sum > 0.0 {
            let inv = 1.0 / sum;
            for v in row.iter_mut() {
                *v *= inv;
            }
        }
    }
}

/// 1-D convolution via im2col + sgemm.
///
/// - `x` is `[T, Cin]` (time-major, channel-minor — whisper's mel/conv
///   activation layout).
/// - `w` is the flat `[Cout, Cin, K]` weight (row-major: index
///   `co*Cin*K + ci*K + kk`).
/// - `bias` has length `Cout`.
/// - Output is `[T_out, Cout]` with `T_out = (T + 2*pad - K)/stride + 1`.
///
/// We build the im2col matrix `[T_out, Cin*K]` (each output position's
/// receptive field flattened in `(ci, kk)` order), reshape the weights to
/// `[Cin*K, Cout]` (transposing `[Cout, Cin*K]`), and a single
/// [`matmul`] yields `[T_out, Cout]`; the bias is then broadcast-added.
///
/// # Errors
/// [`FwError::InvalidRequest`] if `x.cols != cin`, `w.len() != cout*cin*k`,
/// `bias.len() != cout`, `stride == 0`, or the padded input is shorter than
/// the kernel (empty output).
#[allow(clippy::too_many_arguments)]
pub fn conv1d(
    x: &Mat,
    w: &[f32],
    cout: usize,
    cin: usize,
    k: usize,
    bias: &[f32],
    stride: usize,
    pad: usize,
) -> FwResult<Mat> {
    if x.cols != cin {
        return Err(FwError::InvalidRequest(format!(
            "conv1d: x.cols {} != cin {cin}",
            x.cols
        )));
    }
    if w.len() != cout * cin * k {
        return Err(FwError::InvalidRequest(format!(
            "conv1d: weight len {} != cout*cin*k = {}",
            w.len(),
            cout * cin * k
        )));
    }
    if bias.len() != cout {
        return Err(FwError::InvalidRequest(format!(
            "conv1d: bias len {} != cout {cout}",
            bias.len()
        )));
    }
    if stride == 0 {
        return Err(FwError::InvalidRequest("conv1d: stride must be > 0".into()));
    }
    let t_in = x.rows;
    let padded = t_in + 2 * pad;
    if padded < k {
        return Err(FwError::InvalidRequest(format!(
            "conv1d: padded length {padded} < kernel {k}"
        )));
    }
    let t_out = (padded - k) / stride + 1;

    // im2col: [T_out, Cin*K], column index = ci*K + kk.
    let patch = cin * k;
    let mut cols = vec![0.0f32; t_out * patch];
    for o in 0..t_out {
        let start = o * stride; // position in the padded input
        let row = &mut cols[o * patch..(o + 1) * patch];
        for kk in 0..k {
            let p = start + kk; // padded index
            // map padded index back to real input index
            if p < pad {
                continue; // left zero-pad
            }
            let ti = p - pad;
            if ti >= t_in {
                continue; // right zero-pad
            }
            let src = x.row(ti); // [Cin]
            for ci in 0..cin {
                row[ci * k + kk] = src[ci];
            }
        }
    }
    let im2col = Mat::from_vec(t_out, patch, cols);

    // Reshape weights [Cout, Cin*K] -> w_t [Cin*K, Cout] (transpose).
    let mut w_t = vec![0.0f32; patch * cout];
    for co in 0..cout {
        for j in 0..patch {
            w_t[j * cout + co] = w[co * patch + j];
        }
    }
    let w_t = Mat::from_vec(patch, cout, w_t);

    // [T_out, Cin*K] x [Cin*K, Cout] -> [T_out, Cout], then add bias.
    matmul_bias(&im2col, &w_t, Some(bias))
}

/// Incremental key/value cache for autoregressive self-attention.
///
/// Stores keys and values as contiguous `[capacity_tokens, n_state]` row-
/// major buffers; [`KvCache::append`] copies new per-token rows in and
/// advances `len`. [`KvCache::keys`] / [`KvCache::values`] expose the
/// populated prefix as a `[len, n_state]` [`Mat`] for [`attention`].
#[derive(Debug, Clone)]
pub struct KvCache {
    k: Vec<f32>,
    v: Vec<f32>,
    len: usize,
    capacity_tokens: usize,
    n_state: usize,
}

impl KvCache {
    /// Allocate a cache for up to `capacity_tokens` tokens of width
    /// `n_state`.
    #[must_use]
    pub fn new(capacity_tokens: usize, n_state: usize) -> Self {
        Self {
            k: vec![0.0; capacity_tokens * n_state],
            v: vec![0.0; capacity_tokens * n_state],
            len: 0,
            capacity_tokens,
            n_state,
        }
    }

    /// Number of tokens currently cached.
    #[must_use]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the cache is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Per-token width.
    #[must_use]
    pub fn n_state(&self) -> usize {
        self.n_state
    }

    /// Clear the cache (retains the allocation).
    pub fn reset(&mut self) {
        self.len = 0;
    }

    /// Append `k`/`v` rows (`[t, n_state]` each) to the cache.
    ///
    /// # Errors
    /// [`FwError::InvalidRequest`] if widths disagree with `n_state` or the
    /// append would exceed `capacity_tokens`.
    pub fn append(&mut self, k: &Mat, v: &Mat) -> FwResult<()> {
        if k.cols != self.n_state || v.cols != self.n_state {
            return Err(FwError::InvalidRequest(format!(
                "KvCache::append width mismatch: n_state={}, k.cols={}, v.cols={}",
                self.n_state, k.cols, v.cols
            )));
        }
        if k.rows != v.rows {
            return Err(FwError::InvalidRequest(format!(
                "KvCache::append row mismatch: k.rows={}, v.rows={}",
                k.rows, v.rows
            )));
        }
        let t = k.rows;
        if self.len + t > self.capacity_tokens {
            return Err(FwError::InvalidRequest(format!(
                "KvCache::append overflow: len {} + {t} > capacity {}",
                self.len, self.capacity_tokens
            )));
        }
        let off = self.len * self.n_state;
        let span = t * self.n_state;
        self.k[off..off + span].copy_from_slice(&k.data);
        self.v[off..off + span].copy_from_slice(&v.data);
        self.len += t;
        Ok(())
    }

    /// View of the cached keys as a `[len, n_state]` matrix.
    #[must_use]
    pub fn keys(&self) -> Mat {
        Mat::from_vec(
            self.len,
            self.n_state,
            self.k[..self.len * self.n_state].to_vec(),
        )
    }

    /// View of the cached values as a `[len, n_state]` matrix.
    #[must_use]
    pub fn values(&self) -> Mat {
        Mat::from_vec(
            self.len,
            self.n_state,
            self.v[..self.len * self.n_state].to_vec(),
        )
    }
}

/// Multi-head scaled-dot-product attention.
///
/// - `q` is `[Tq, n_state]`, `k`/`v` are `[Tk, n_state]`.
/// - `n_head` must divide `n_state`; per-head width is `d_head =
///   n_state / n_head`.
/// - `causal_offset`: `None` for full (cross / bidirectional) attention;
///   `Some(cache_len)` for causal self-attention where query position `i`
///   attends to all keys with index `<= cache_len + i`. (For a fresh decode
///   over the whole prompt `cache_len = 0`; for an incremental single-token
///   step `cache_len = past_len` and `Tq = 1`.)
///
/// Heads are split along the state dimension, each head's `q`/`k` rows are
/// scaled by `d_head^-0.25`, then per-head `qk^T` scores are masked (causal,
/// if requested), softmaxed per query row, and multiplied by `v`; the head
/// outputs are concatenated back to `[Tq, n_state]`.
///
/// # Scaling convention
/// Scaling both Q and K by `d_head^-0.25` reproduces the openai/whisper
/// scaling and is numerically equal to whisper.cpp's single
/// `KQscale = 1/sqrt(d_head)` applied to the QK scores
/// (`whisper.cpp` decoder path scales Qcur and Kcur each by
/// `pow(n_state_head, -0.25)` — lines ~2506/2550/2557 of `src/whisper.cpp`;
/// the encoder uses the algebraically identical single `1/sqrtf(d)` factor
/// at line ~2069). The identity is
/// `(q·d^-0.25)·(k·d^-0.25) = q·k·d^-0.5 = q·k / sqrt(d)`.
///
/// # Errors
/// [`FwError::InvalidRequest`] if `n_head == 0`, `n_state % n_head != 0`,
/// the q/k/v widths disagree, or `k.rows != v.rows`.
pub fn attention(
    q: &Mat,
    k: &Mat,
    v: &Mat,
    n_head: usize,
    causal_offset: Option<usize>,
) -> FwResult<Mat> {
    let n_state = q.cols;
    if n_head == 0 || !n_state.is_multiple_of(n_head) {
        return Err(FwError::InvalidRequest(format!(
            "attention: n_head {n_head} must divide n_state {n_state}"
        )));
    }
    if k.cols != n_state || v.cols != n_state {
        return Err(FwError::InvalidRequest(format!(
            "attention: width mismatch q={n_state} k={} v={}",
            k.cols, v.cols
        )));
    }
    if k.rows != v.rows {
        return Err(FwError::InvalidRequest(format!(
            "attention: k.rows {} != v.rows {}",
            k.rows, v.rows
        )));
    }
    let tq = q.rows;
    let tk = k.rows;
    let d_head = n_state / n_head;
    if d_head == 0 {
        return Err(FwError::InvalidRequest("attention: d_head == 0".into()));
    }
    let scale = (d_head as f32).powf(-0.25);
    let cache_len = causal_offset.unwrap_or(0);

    let mut out = vec![0.0f32; tq * n_state];

    // Process each head independently (heads are embarrassingly parallel,
    // but keep it serial-over-heads + parallel-inside-matmul so we don't
    // oversubscribe rayon; the per-head qk^T and @v are the costly parts
    // and already go through the parallel sgemm).
    for h in 0..n_head {
        let base = h * d_head;

        // Gather this head's scaled Q [Tq, d_head] and K [Tk, d_head].
        let mut qh = vec![0.0f32; tq * d_head];
        for i in 0..tq {
            let src = &q.row(i)[base..base + d_head];
            let dst = &mut qh[i * d_head..(i + 1) * d_head];
            for (d, &s) in dst.iter_mut().zip(src) {
                *d = s * scale;
            }
        }
        let mut kh = vec![0.0f32; tk * d_head];
        for j in 0..tk {
            let src = &k.row(j)[base..base + d_head];
            let dst = &mut kh[j * d_head..(j + 1) * d_head];
            for (d, &s) in dst.iter_mut().zip(src) {
                *d = s * scale;
            }
        }
        let qh = Mat::from_vec(tq, d_head, qh);
        let kh = Mat::from_vec(tk, d_head, kh);

        // scores = qh @ kh^T -> [Tq, Tk]. kh^T is [d_head, Tk]; build it
        // explicitly so the matmul stays contiguous.
        let mut kh_t = vec![0.0f32; d_head * tk];
        for j in 0..tk {
            for d in 0..d_head {
                kh_t[d * tk + j] = kh.data[j * d_head + d];
            }
        }
        let kh_t = Mat::from_vec(d_head, tk, kh_t);
        let mut scores = matmul(&qh, &kh_t)?; // [Tq, Tk]

        // Causal mask: query i attends to keys <= cache_len + i.
        if causal_offset.is_some() {
            for i in 0..tq {
                let limit = cache_len + i;
                let row = &mut scores.data[i * tk..(i + 1) * tk];
                for (j, s) in row.iter_mut().enumerate() {
                    if j > limit {
                        *s = f32::NEG_INFINITY;
                    }
                }
            }
        }

        softmax_rows(&mut scores); // [Tq, Tk]

        // Gather this head's V [Tk, d_head] (unscaled), out_h = scores @ V.
        let mut vh = vec![0.0f32; tk * d_head];
        for j in 0..tk {
            let src = &v.row(j)[base..base + d_head];
            let dst = &mut vh[j * d_head..(j + 1) * d_head];
            dst.copy_from_slice(src);
        }
        let vh = Mat::from_vec(tk, d_head, vh);
        let out_h = matmul(&scores, &vh)?; // [Tq, d_head]

        // Scatter head output back into the merged [Tq, n_state].
        for i in 0..tq {
            let src = &out_h.data[i * d_head..(i + 1) * d_head];
            out[i * n_state + base..i * n_state + base + d_head].copy_from_slice(src);
        }
    }

    Ok(Mat::from_vec(tq, n_state, out))
}

/// Incremental self-attention that extends a [`KvCache`].
///
/// Appends the step's `k_new`/`v_new` (`[Tq, n_state]`) to `cache`, then
/// runs causal [`attention`] of `q` against the *entire* cached K/V with
/// `causal_offset = past_len` (the cache length before this append). For a
/// single-token decode step `Tq == 1` and every cached key is visible; for a
/// multi-token prompt prefill each query `i` still only sees keys up to
/// `past_len + i`.
///
/// # Errors
/// Propagates [`KvCache::append`] and [`attention`] errors.
pub fn attention_with_cache(
    q: &Mat,
    k_new: &Mat,
    v_new: &Mat,
    n_head: usize,
    cache: &mut KvCache,
) -> FwResult<Mat> {
    let past_len = cache.len();
    cache.append(k_new, v_new)?;
    let k_all = cache.keys();
    let v_all = cache.values();
    attention(q, &k_all, &v_all, n_head, Some(past_len))
}

/// Cache-blocked, multi-threaded out-of-place transpose: `data` viewed as
/// row-major `[rows, cols]` becomes row-major `[cols, rows]`.
///
/// Used at model-load time to pre-transpose every linear weight (ggml stores
/// PyTorch's `[out, in]`; the inference matmuls want `[in, out]`). The naive
/// column-strided serial loop dominated `model_weights` time on large models
/// (hotspot #5, tests/artifacts/perf/20260605T0218Z): ~3 GB of strided writes.
/// 64x64 tiles keep both source reads and destination writes inside cache
/// lines; independent row-bands fan out across threads.
///
/// Isomorphism: a pure permutation — every output element is the same
/// `data[r * cols + c]` the serial loop wrote, so results are bit-identical.
pub(crate) fn transpose_parallel(data: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    debug_assert_eq!(data.len(), rows * cols, "transpose shape/data mismatch");
    const TILE: usize = 64;
    const PAR_THRESHOLD: usize = 1 << 20;
    let mut out = vec![0.0f32; rows * cols];

    let tile_band = |band_rows: std::ops::Range<usize>, out: &mut [f32]| {
        // `out` here is the FULL output buffer for serial mode, or a row-band
        // is not separable for transpose outputs (writes scatter across all
        // of `out`), so parallel mode splits by output row bands (i.e. source
        // column bands) instead — see below.
        for r0 in band_rows.clone().step_by(TILE) {
            let r1 = (r0 + TILE).min(band_rows.end);
            for c0 in (0..cols).step_by(TILE) {
                let c1 = (c0 + TILE).min(cols);
                for r in r0..r1 {
                    for c in c0..c1 {
                        out[c * rows + r] = data[r * cols + c];
                    }
                }
            }
        }
    };

    let workers = std::thread::available_parallelism()
        .map(std::num::NonZero::get)
        .unwrap_or(1)
        .min(8);
    if rows * cols < PAR_THRESHOLD || workers < 2 {
        tile_band(0..rows, &mut out);
        return out;
    }

    // Parallel split: each worker owns a contiguous band of OUTPUT rows
    // (= source columns c in [c0, c1)), so output slices are disjoint.
    let band = cols.div_ceil(workers);
    std::thread::scope(|s| {
        for (w, out_band) in out.chunks_mut(band * rows).enumerate() {
            let c_start = w * band;
            s.spawn(move || {
                let c_end = (c_start + band).min(cols);
                for c0 in (c_start..c_end).step_by(TILE) {
                    let c1 = (c0 + TILE).min(c_end);
                    for r0 in (0..rows).step_by(TILE) {
                        let r1 = (r0 + TILE).min(rows);
                        for c in c0..c1 {
                            let dst_row = c - c_start;
                            for r in r0..r1 {
                                out_band[dst_row * rows + r] = data[r * cols + c];
                            }
                        }
                    }
                }
            });
        }
    });
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic LCG (Numerical Recipes constants) -> f32 in [-1, 1).
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
            let data = (0..rows * cols).map(|_| self.next_f32()).collect();
            Mat::from_vec(rows, cols, data)
        }
    }

    fn naive_matmul(a: &Mat, b: &Mat) -> Mat {
        let (m, k, n) = (a.rows, a.cols, b.cols);
        let mut out = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                // Accumulate the reference dot product in f64: this is the
                // *true* value the f32 sgemm approximates. Summing the
                // reference in f32 would itself drift (a different, worse,
                // accumulation order than matrixmultiply's blocked kernel),
                // so a near-zero element where ±1 terms cancel would show a
                // large spurious relative error against an equally-noisy
                // reference. f64 here makes the reference the gold standard.
                let mut acc = 0.0f64;
                for p in 0..k {
                    acc += f64::from(a.data[i * k + p]) * f64::from(b.data[p * n + j]);
                }
                out[i * n + j] = acc as f32;
            }
        }
        Mat::from_vec(m, n, out)
    }

    /// Combined absolute + relative error: an element passes if it is within
    /// `1e-5` relative OR within an absolute floor scaled to the magnitude of
    /// the computation. For dot products of `k` values in `[-1, 1)` the f32
    /// rounding noise grows like `~sqrt(k) * eps`, so we scale the absolute
    /// floor by `sqrt(k)`; this judges near-zero (cancelling) elements by
    /// absolute error rather than a meaningless relative one.
    fn max_rel_err_k(a: &[f32], b: &[f32], k: usize) -> f32 {
        let abs_floor = 1e-5 * (k.max(1) as f32).sqrt();
        a.iter()
            .zip(b)
            .map(|(&x, &y)| {
                let abs = (x - y).abs();
                let denom = x.abs().max(y.abs()).max(1e-6);
                // Pass if within abs floor; otherwise report the relative error.
                if abs <= abs_floor { 0.0 } else { abs / denom }
            })
            .fold(0.0f32, f32::max)
    }

    fn max_rel_err(a: &[f32], b: &[f32]) -> f32 {
        max_rel_err_k(a, b, 1)
    }

    #[test]
    fn matmul_matches_naive_various_shapes() {
        let mut rng = Lcg::new(1);
        // Includes the decoder-step [1,k]x[k,n] shape and an encoder-sized one.
        let shapes = [(1, 384, 384), (1500, 384, 384), (7, 13, 5), (32, 64, 48)];
        for (m, k, n) in shapes {
            let a = rng.mat(m, k);
            let b = rng.mat(k, n);
            let got = matmul(&a, &b).unwrap();
            let want = naive_matmul(&a, &b);
            assert_eq!(got.rows, m);
            assert_eq!(got.cols, n);
            assert!(
                max_rel_err_k(&got.data, &want.data, k) < 1e-5,
                "shape {m}x{k}x{n} rel err too high"
            );
        }
    }

    #[test]
    fn matmul_inner_dim_mismatch_errors() {
        let a = Mat::zeros(2, 3);
        let b = Mat::zeros(4, 5);
        assert!(matmul(&a, &b).is_err());
    }

    #[test]
    fn matmul_bias_matches_manual() {
        let mut rng = Lcg::new(2);
        let x = rng.mat(4, 6); // [m=4, in=6]
        let w_t = rng.mat(6, 3); // [in=6, out=3]
        let bias = [0.5f32, -0.25, 1.0];
        let got = matmul_bias(&x, &w_t, Some(&bias)).unwrap();
        let base = naive_matmul(&x, &w_t);
        for i in 0..4 {
            for (j, b) in bias.iter().enumerate() {
                let expected = base.data[i * 3 + j] + b;
                assert!((got.data[i * 3 + j] - expected).abs() < 1e-5);
            }
        }
        // No-bias path == plain matmul.
        let no_bias = matmul_bias(&x, &w_t, None).unwrap();
        assert!(max_rel_err(&no_bias.data, &base.data) < 1e-5);
    }

    #[test]
    fn matmul_bias_wrong_bias_len_errors() {
        let x = Mat::zeros(2, 3);
        let w_t = Mat::zeros(3, 4);
        assert!(matmul_bias(&x, &w_t, Some(&[1.0, 2.0])).is_err());
    }

    #[test]
    fn layer_norm_known_small_case() {
        // Row [1,2,3,4]: mean=2.5, var=1.25, std=sqrt(1.25).
        let mut x = Mat::from_vec(1, 4, vec![1.0, 2.0, 3.0, 4.0]);
        let w = [1.0f32; 4];
        let b = [0.0f32; 4];
        layer_norm(&mut x, &w, &b, 0.0);
        let std = 1.25f32.sqrt();
        let expected = [
            (1.0 - 2.5) / std,
            (2.0 - 2.5) / std,
            (3.0 - 2.5) / std,
            (4.0 - 2.5) / std,
        ];
        for (g, e) in x.data.iter().zip(expected) {
            assert!((g - e).abs() < 1e-5, "got {g}, want {e}");
        }
    }

    #[test]
    fn layer_norm_property_zero_mean_unit_var() {
        let mut rng = Lcg::new(3);
        let mut x = rng.mat(5, 64);
        let w = vec![1.0f32; 64];
        let b = vec![0.0f32; 64];
        layer_norm(&mut x, &w, &b, 1e-5);
        for r in 0..5 {
            let row = x.row(r);
            let mean: f64 = row.iter().map(|&v| f64::from(v)).sum::<f64>() / 64.0;
            let var: f64 = row
                .iter()
                .map(|&v| (f64::from(v) - mean).powi(2))
                .sum::<f64>()
                / 64.0;
            assert!(mean.abs() < 1e-4, "row {r} mean {mean}");
            assert!((var - 1.0).abs() < 1e-2, "row {r} var {var}");
        }
    }

    #[test]
    fn layer_norm_affine_applied() {
        let mut x = Mat::from_vec(1, 4, vec![1.0, 2.0, 3.0, 4.0]);
        let w = [2.0f32, 2.0, 2.0, 2.0];
        let b = [1.0f32, 1.0, 1.0, 1.0];
        layer_norm(&mut x, &w, &b, 0.0);
        let std = 1.25f32.sqrt();
        let expected = (1.0 - 2.5) / std * 2.0 + 1.0;
        assert!((x.data[0] - expected).abs() < 1e-5);
    }

    #[test]
    fn gelu_known_values() {
        let mut x = Mat::from_vec(1, 3, vec![0.0, 1.0, -1.0]);
        gelu(&mut x);
        // Compute expected from the tanh-approx formula directly.
        let f = |v: f32| {
            0.5 * v * (1.0 + (GELU_SQRT_2_OVER_PI * v * (1.0 + GELU_COEF_A * v * v)).tanh())
        };
        assert!((x.data[0] - 0.0).abs() < 1e-6, "gelu(0)=0");
        assert!((x.data[1] - f(1.0)).abs() < 1e-6);
        assert!((x.data[2] - f(-1.0)).abs() < 1e-6);
        // Spec reference magnitudes.
        assert!(
            (x.data[1] - 0.8412).abs() < 1e-3,
            "gelu(1)~0.8412, got {}",
            x.data[1]
        );
        assert!(
            (x.data[2] - (-0.1588)).abs() < 1e-3,
            "gelu(-1)~-0.1588, got {}",
            x.data[2]
        );
    }

    #[test]
    fn softmax_rows_sums_to_one() {
        let mut rng = Lcg::new(4);
        let mut x = rng.mat(6, 11);
        softmax_rows(&mut x);
        for r in 0..6 {
            let s: f32 = x.row(r).iter().sum();
            assert!((s - 1.0).abs() < 1e-5, "row {r} sum {s}");
            assert!(x.row(r).iter().all(|&v| v >= 0.0));
        }
    }

    #[test]
    fn softmax_rows_max_stability() {
        // Large values must not overflow to NaN/inf.
        let mut x = Mat::from_vec(1, 3, vec![1e30, 1e30 + 1.0, 0.0]);
        softmax_rows(&mut x);
        let s: f32 = x.row(0).iter().sum();
        assert!((s - 1.0).abs() < 1e-5, "sum {s}");
        assert!(x.data.iter().all(|v| v.is_finite()));
    }

    #[allow(clippy::too_many_arguments)]
    fn naive_conv1d(
        x: &Mat,
        w: &[f32],
        cout: usize,
        cin: usize,
        k: usize,
        bias: &[f32],
        stride: usize,
        pad: usize,
    ) -> Mat {
        let t_in = x.rows;
        let t_out = (t_in + 2 * pad - k) / stride + 1;
        let mut out = vec![0.0f32; t_out * cout];
        for o in 0..t_out {
            for co in 0..cout {
                // f64 reference accumulation; see `naive_matmul`.
                let mut acc = f64::from(bias[co]);
                for kk in 0..k {
                    let p = o * stride + kk;
                    if p < pad {
                        continue;
                    }
                    let ti = p - pad;
                    if ti >= t_in {
                        continue;
                    }
                    for ci in 0..cin {
                        acc += f64::from(w[co * cin * k + ci * k + kk])
                            * f64::from(x.data[ti * cin + ci]);
                    }
                }
                out[o * cout + co] = acc as f32;
            }
        }
        Mat::from_vec(t_out, cout, out)
    }

    #[test]
    fn conv1d_matches_naive() {
        let mut rng = Lcg::new(5);
        // (t_in, cin, cout, k, stride, pad)
        let cases = [
            (10, 3, 4, 3, 1, 1),
            (10, 3, 4, 3, 2, 1),
            (16, 5, 2, 3, 1, 1),
            (8, 2, 6, 5, 2, 1),
        ];
        for (t_in, cin, cout, k, stride, pad) in cases {
            let x = rng.mat(t_in, cin);
            let w: Vec<f32> = (0..cout * cin * k).map(|_| rng.next_f32()).collect();
            let bias: Vec<f32> = (0..cout).map(|_| rng.next_f32()).collect();
            let got = conv1d(&x, &w, cout, cin, k, &bias, stride, pad).unwrap();
            let want = naive_conv1d(&x, &w, cout, cin, k, &bias, stride, pad);
            assert_eq!(got.rows, want.rows, "t_out mismatch");
            assert_eq!(got.cols, cout);
            assert!(
                max_rel_err_k(&got.data, &want.data, cin * k) < 1e-5,
                "conv case {t_in},{cin},{cout},{k},{stride},{pad}"
            );
        }
    }

    /// Reference single-head attention (no cache, optional causal).
    fn naive_attention_single_head(q: &Mat, k: &Mat, v: &Mat, causal: bool) -> Mat {
        let d = q.cols;
        let tq = q.rows;
        let tk = k.rows;
        let scale = (d as f32).powf(-0.25);
        let mut out = vec![0.0f32; tq * d];
        for i in 0..tq {
            // scores
            let mut scores = vec![0.0f32; tk];
            for (j, score) in scores.iter_mut().enumerate() {
                let mut acc = 0.0f32;
                for c in 0..d {
                    acc += (q.data[i * d + c] * scale) * (k.data[j * d + c] * scale);
                }
                *score = if causal && j > i {
                    f32::NEG_INFINITY
                } else {
                    acc
                };
            }
            // softmax
            let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for s in &mut scores {
                *s = (*s - max).exp();
                sum += *s;
            }
            for s in &mut scores {
                *s /= sum;
            }
            // weighted sum of v
            for c in 0..d {
                let mut acc = 0.0f32;
                for (j, &sj) in scores.iter().enumerate() {
                    acc += sj * v.data[j * d + c];
                }
                out[i * d + c] = acc;
            }
        }
        Mat::from_vec(tq, d, out)
    }

    #[test]
    fn attention_matches_single_head_reference() {
        let mut rng = Lcg::new(6);
        let q = rng.mat(5, 8);
        let k = rng.mat(7, 8);
        let v = rng.mat(7, 8);
        let got = attention(&q, &k, &v, 1, None).unwrap();
        let want = naive_attention_single_head(&q, &k, &v, false);
        assert!(max_rel_err(&got.data, &want.data) < 1e-4);
    }

    #[test]
    fn attention_multi_head_merge() {
        // With n_head heads each head is an independent single-head attention
        // over its slice; concatenating reference per-head results must match.
        let mut rng = Lcg::new(7);
        let n_head = 4;
        let d_head = 6;
        let n_state = n_head * d_head;
        let tq = 3;
        let tk = 5;
        let q = rng.mat(tq, n_state);
        let k = rng.mat(tk, n_state);
        let v = rng.mat(tk, n_state);
        let got = attention(&q, &k, &v, n_head, None).unwrap();

        let mut want = vec![0.0f32; tq * n_state];
        for h in 0..n_head {
            let base = h * d_head;
            let slice = |m: &Mat, rows: usize| {
                let mut out = vec![0.0f32; rows * d_head];
                for r in 0..rows {
                    out[r * d_head..(r + 1) * d_head]
                        .copy_from_slice(&m.row(r)[base..base + d_head]);
                }
                Mat::from_vec(rows, d_head, out)
            };
            let qh = slice(&q, tq);
            let kh = slice(&k, tk);
            let vh = slice(&v, tk);
            let oh = naive_attention_single_head(&qh, &kh, &vh, false);
            for r in 0..tq {
                want[r * n_state + base..r * n_state + base + d_head].copy_from_slice(oh.row(r));
            }
        }
        assert!(max_rel_err(&got.data, &want).is_finite());
        assert!(max_rel_err(&got.data, &want) < 1e-4);
    }

    #[test]
    fn attention_causal_mask_property() {
        // Changing future keys/values must not change earlier query outputs.
        let mut rng = Lcg::new(8);
        let n_head = 2;
        let n_state = 8;
        let tq = 4;
        let q = rng.mat(tq, n_state);
        let mut k = rng.mat(tq, n_state);
        let mut v = rng.mat(tq, n_state);
        let out_a = attention(&q, &k, &v, n_head, Some(0)).unwrap();

        // Perturb the LAST key/value row (a "future" token for queries 0..2).
        for c in 0..n_state {
            k.data[(tq - 1) * n_state + c] += 5.0;
            v.data[(tq - 1) * n_state + c] += 5.0;
        }
        let out_b = attention(&q, &k, &v, n_head, Some(0)).unwrap();

        // Rows 0..tq-1 (which cannot attend to the last key) are unchanged.
        for i in 0..tq - 1 {
            for c in 0..n_state {
                assert!(
                    (out_a.data[i * n_state + c] - out_b.data[i * n_state + c]).abs() < 1e-6,
                    "row {i} changed under future-key perturbation"
                );
            }
        }
        // The last row DOES change (it attends to the perturbed key).
        let last_changed = (0..n_state).any(|c| {
            (out_a.data[(tq - 1) * n_state + c] - out_b.data[(tq - 1) * n_state + c]).abs() > 1e-4
        });
        assert!(
            last_changed,
            "last query row should react to its own key change"
        );
    }

    #[test]
    fn kv_cache_incremental_equals_full_recompute() {
        let mut rng = Lcg::new(9);
        let n_head = 3;
        let d_head = 5;
        let n_state = n_head * d_head;
        let n_tokens = 5;

        // Full per-token q/k/v (each token contributes one row).
        let q_all = rng.mat(n_tokens, n_state);
        let k_all = rng.mat(n_tokens, n_state);
        let v_all = rng.mat(n_tokens, n_state);

        // Full recompute: causal attention over all tokens at once.
        let full = attention(&q_all, &k_all, &v_all, n_head, Some(0)).unwrap();

        // Incremental: feed one token at a time through a KvCache.
        let mut cache = KvCache::new(n_tokens, n_state);
        let mut inc = vec![0.0f32; n_tokens * n_state];
        for t in 0..n_tokens {
            let qi = Mat::from_vec(1, n_state, q_all.row(t).to_vec());
            let ki = Mat::from_vec(1, n_state, k_all.row(t).to_vec());
            let vi = Mat::from_vec(1, n_state, v_all.row(t).to_vec());
            let step = attention_with_cache(&qi, &ki, &vi, n_head, &mut cache).unwrap();
            inc[t * n_state..(t + 1) * n_state].copy_from_slice(&step.data);
        }
        assert_eq!(cache.len(), n_tokens);
        assert!(
            max_rel_err(&inc, &full.data) < 1e-4,
            "incremental != full recompute"
        );
    }

    #[test]
    fn kv_cache_append_overflow_and_reset() {
        let mut cache = KvCache::new(2, 4);
        let row = Mat::from_vec(1, 4, vec![1.0, 2.0, 3.0, 4.0]);
        cache.append(&row, &row).unwrap();
        cache.append(&row, &row).unwrap();
        assert_eq!(cache.len(), 2);
        assert!(cache.append(&row, &row).is_err(), "third append overflows");
        cache.reset();
        assert!(cache.is_empty());
        cache.append(&row, &row).unwrap();
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn attention_rejects_bad_head_count() {
        let q = Mat::zeros(2, 6);
        let k = Mat::zeros(2, 6);
        let v = Mat::zeros(2, 6);
        assert!(attention(&q, &k, &v, 0, None).is_err());
        assert!(
            attention(&q, &k, &v, 4, None).is_err(),
            "4 does not divide 6"
        );
    }
}
