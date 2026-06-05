//! Log-mel spectrogram frontend (pure Rust, exact whisper.cpp semantics).
//!
//! This is a faithful port of whisper.cpp's `log_mel_spectrogram` /
//! `whisper_pcm_to_mel` path (see `whisper.cpp` `src/whisper.cpp`
//! `log_mel_spectrogram`, `log_mel_spectrogram_worker_thread`, `fft`, `dft`,
//! and `fill_hann_window`). Every constant and every numeric behavior that
//! could plausibly diverge from whisper.cpp is documented inline so the output
//! is bit-for-bit comparable (within f32 rounding) to the reference encoder
//! input.
//!
//! Pipeline (per whisper.cpp):
//! 1. Pad the PCM: a leading *reflective* pad of `N_FFT/2` samples, the audio
//!    itself, then a trailing pad of one full 30 s chunk of zeros plus another
//!    `N_FFT/2` zeros. (The 30 s tail guarantees at least 3000 frames exist.)
//! 2. Slide a length-`N_FFT` Hann (periodic) window with hop `HOP`.
//! 3. FFT each windowed frame; take the **power** spectrum `re^2 + im^2` over
//!    the one-sided `N_FFT/2 + 1 = 201` bins.
//! 4. Project through the model's `n_mel x 201` filterbank, then
//!    `log10(max(e, 1e-10))`.
//! 5. Globally clamp to `max - 8.0` and normalize `(x + 4) / 4`.
//!
//! The FFT is whisper.cpp's exact recursive structure: radix-2 split when the
//! length is even, falling back to a naive O(N^2) DFT when the length is odd.
//! For `N = 400 = 2^4 * 5^2` the recursion is `400 -> 200 -> 100 -> 50 -> 25`,
//! and `25` (odd) is computed by the naive DFT. This is numerically a correct
//! discrete Fourier transform; it is validated against an independent naive DFT
//! reference in the tests.

use std::f64::consts::PI;

use crate::error::{FwError, FwResult};

use super::{Mel, MelFilterbank};

/// Audio sample rate whisper expects, in Hz (`WHISPER_SAMPLE_RATE`).
pub const SAMPLE_RATE: usize = 16_000;
/// FFT / analysis window length in samples (`WHISPER_N_FFT`).
pub const N_FFT: usize = 400;
/// Hop / frame step in samples (`WHISPER_HOP_LENGTH`).
pub const HOP: usize = 160;
/// One full 30 s chunk of audio in samples (`WHISPER_SAMPLE_RATE * 30`).
pub const N_SAMPLES_30S: usize = SAMPLE_RATE * 30;
/// Mel frames produced per 30 s chunk (`N_SAMPLES_30S / HOP`).
pub const FRAMES_PER_CHUNK: usize = N_SAMPLES_30S / HOP; // 3000
/// One-sided FFT bin count: `N_FFT/2 + 1`, i.e. bin_0 .. bin_nyquist.
pub const N_FREQ_BINS: usize = N_FFT / 2 + 1; // 201

/// The exact value every "silence" mel sample collapses to.
///
/// Derivation (all bins zero ⇒ every pre-normalization sample equals
/// `log10(1e-10) = -10`): the global max is also `-10`, so the clamp floor is
/// `-10 - 8 = -18`, but no sample is below `-10`, so the clamp is a no-op and
/// every sample normalizes to `(-10 + 4) / 4 = -1.5`. This is the value
/// `chunk_frames` pads with when slicing past the end of a real mel, matching
/// whisper.cpp's `n_len`-padding behavior (frames beyond the audio are all
/// `log10(1e-10)` pre-clamp, hence `-1.5` post-normalize whenever the loudest
/// real frame is at most 8 dB above the noise floor — which, after the global
/// clamp, every silence frame always is).
pub const SILENCE_FLOOR: f32 = -1.5;

/// Compute the periodic Hann window of length `N_FFT`.
///
/// whisper.cpp uses `periodic = true`, i.e. divisor `length` (not `length-1`):
/// `w[i] = 0.5 * (1 - cos(2*pi*i / length))`. We compute in f64 then narrow to
/// f32 to mirror `cosf` precision behavior closely without the platform `cosf`
/// dependency.
fn hann_window() -> [f32; N_FFT] {
    let mut w = [0.0f32; N_FFT];
    for (i, wi) in w.iter_mut().enumerate() {
        // periodic ⇒ offset 0 ⇒ divisor == length.
        *wi = (0.5 * (1.0 - ((2.0 * PI * i as f64) / N_FFT as f64).cos())) as f32;
    }
    w
}

/// The length-`N_FFT` periodic Hann window, computed once and cached for the
/// life of the process.
///
/// The window is a pure function of `N_FFT`, but [`log_mel`] used to recompute
/// it (400 `cos` calls) on every call. whisper.cpp likewise caches its window
/// globally. We memoize behind a [`OnceLock`](std::sync::OnceLock) so the cosine
/// loop runs at most once regardless of how many clips are processed.
fn cached_hann_window() -> &'static [f32; N_FFT] {
    static HANN: std::sync::OnceLock<[f32; N_FFT]> = std::sync::OnceLock::new();
    HANN.get_or_init(hann_window)
}

/// Naive O(N^2) DFT of a real input. `out` is interleaved complex,
/// length `2*N`: `out[2k] = Re`, `out[2k+1] = Im`. Mirrors whisper.cpp `dft`.
fn dft(input: &[f32], out: &mut [f32]) {
    let n = input.len();
    for k in 0..n {
        let mut re = 0.0f32;
        let mut im = 0.0f32;
        for (j, &x) in input.iter().enumerate() {
            // t = 2*pi*k*j / N
            let theta = (2.0 * PI * (k * j) as f64) / n as f64;
            re += x * theta.cos() as f32;
            im -= x * theta.sin() as f32;
        }
        out[2 * k] = re;
        out[2 * k + 1] = im;
    }
}

/// Recursive Cooley-Tukey FFT of a real input, exactly mirroring whisper.cpp's
/// `fft`: radix-2 decimation-in-time when `N` is even, falling back to the
/// naive DFT when `N` is odd (the prime-factor base case). `out` is interleaved
/// complex of length `2*N`. `scratch` provides the deinterleaved even/odd
/// buffers and child outputs so we avoid per-call heap allocation in the hot
/// loop.
fn fft(input: &[f32], out: &mut [f32]) {
    let n = input.len();
    if n == 1 {
        out[0] = input[0];
        out[1] = 0.0;
        return;
    }
    if n % 2 == 1 {
        dft(input, out);
        return;
    }

    let half = n / 2;
    let mut even = vec![0.0f32; half];
    let mut odd = vec![0.0f32; half];
    for i in 0..half {
        even[i] = input[2 * i];
        odd[i] = input[2 * i + 1];
    }

    let mut even_fft = vec![0.0f32; 2 * half];
    let mut odd_fft = vec![0.0f32; 2 * half];
    fft(&even, &mut even_fft);
    fft(&odd, &mut odd_fft);

    for k in 0..half {
        // twiddle: t = 2*pi*k / N, factor = cos(t) - i*sin(t)
        let theta = (2.0 * PI * k as f64) / n as f64;
        let re = theta.cos() as f32;
        let im = -(theta.sin() as f32);

        let re_odd = odd_fft[2 * k];
        let im_odd = odd_fft[2 * k + 1];

        let e_re = even_fft[2 * k];
        let e_im = even_fft[2 * k + 1];

        out[2 * k] = e_re + re * re_odd - im * im_odd;
        out[2 * k + 1] = e_im + re * im_odd + im * re_odd;

        out[2 * (k + half)] = e_re - re * re_odd + im * im_odd;
        out[2 * (k + half) + 1] = e_im - re * im_odd - im * re_odd;
    }
}

/// Build the padded sample buffer exactly as whisper.cpp's
/// `log_mel_spectrogram` does:
/// - leading `N_FFT/2` samples: *reflective* pad — `reverse_copy(samples[1..1+pad])`,
///   i.e. samples 1..=pad reversed (NOT including sample 0);
/// - the audio itself;
/// - trailing: 30 s of zeros + another `N_FFT/2` zeros.
///
/// Returns `(padded, valid_len)` where `valid_len = n_samples + stage_2_pad` is
/// the prefix the worker actually FFTs (frames past it are pure noise floor and
/// whisper.cpp writes `log10(1e-10)` directly without an FFT).
fn build_padded(samples: &[f32]) -> (Vec<f32>, usize) {
    let n_samples = samples.len();
    let stage_1_pad = N_SAMPLES_30S;
    let stage_2_pad = N_FFT / 2; // 200

    let total = n_samples + stage_1_pad + 2 * stage_2_pad;
    let mut padded = vec![0.0f32; total];

    // audio at offset stage_2_pad
    padded[stage_2_pad..stage_2_pad + n_samples].copy_from_slice(samples);

    // leading reflective pad: reverse of samples[1..=stage_2_pad].
    // whisper.cpp: std::reverse_copy(samples + 1, samples + 1 + stage_2_pad, begin).
    // If the audio is shorter than the reflection width, only what exists is
    // reflected (the rest stays zero) — mirrors the C++ pointer arithmetic for
    // realistic (>= pad-length) inputs; we clamp for safety on tiny inputs.
    let refl = stage_2_pad.min(n_samples.saturating_sub(1));
    for i in 0..refl {
        // padded[i] = samples[stage_2_pad - i]  (reverse of indices 1..=refl)
        padded[i] = samples[stage_2_pad - i];
    }

    // trailing zeros already present from vec![0.0; total].
    let valid_len = n_samples + stage_2_pad;
    (padded, valid_len)
}

/// Number of mel frames whisper.cpp produces for a given raw sample count:
/// `n_len = (padded_len - N_FFT) / HOP`.
#[must_use]
pub fn n_frames_for(n_samples: usize) -> usize {
    let padded_len = n_samples + N_SAMPLES_30S + 2 * (N_FFT / 2);
    (padded_len - N_FFT) / HOP
}

/// Compute one frame's pre-clamp `log10` mel column into `out` (length
/// `n_mel`). `frame_idx` selects the sample offset (`frame_idx * HOP`);
/// `valid_len` is the FFT-able prefix length. Frames whose offset is past
/// `valid_len` are the pure noise floor (`log10(1e-10)`), matching
/// whisper.cpp's direct-write of `log10(1e-10)` without an FFT.
fn compute_frame_column(
    frame_idx: usize,
    padded: &[f32],
    valid_len: usize,
    hann: &[f32; N_FFT],
    filters: &MelFilterbank,
    out: &mut [f32],
) {
    let offset = frame_idx * HOP;
    let n_fft_bins = filters.n_fft_bins; // 201

    if offset >= valid_len {
        let floor = (1e-10f64).log10() as f32;
        out[..filters.n_mel].fill(floor);
        return;
    }

    // Windowed frame (zero-padded past valid_len).
    let mut fft_in = [0.0f32; N_FFT];
    let avail = N_FFT.min(valid_len - offset);
    for j in 0..avail {
        fft_in[j] = hann[j] * padded[offset + j];
    }

    let mut fft_out = vec![0.0f32; 2 * N_FFT];
    fft(&fft_in, &mut fft_out);

    // Power spectrum over one-sided bins: re^2 + im^2 (NOT sqrt magnitude).
    let mut power = [0.0f32; N_FREQ_BINS];
    for (j, p) in power.iter_mut().enumerate() {
        let re = fft_out[2 * j];
        let im = fft_out[2 * j + 1];
        *p = re * re + im * im;
    }

    // Mel projection + log10 floor. Accumulate in f64 like whisper.cpp.
    for (j, o) in out.iter_mut().enumerate().take(filters.n_mel) {
        let row = &filters.data[j * n_fft_bins..j * n_fft_bins + n_fft_bins];
        let mut sum = 0.0f64;
        for (k, &fk) in power.iter().enumerate() {
            sum += f64::from(fk) * f64::from(row[k]);
        }
        *o = sum.max(1e-10).log10() as f32;
    }
}

/// Compute the log-mel spectrogram for the full padded input, exactly mirroring
/// whisper.cpp's `log_mel_spectrogram` / `whisper_pcm_to_mel`.
///
/// `samples` is the raw 16 kHz mono f32 PCM. The returned [`Mel`] covers the
/// full padded signal (`n_frames = (padded_len - N_FFT) / HOP`); callers slice
/// 30 s windows (3000 frames) out of it with [`chunk_frames`]. Frames are
/// computed independently and split across `n_threads` worker threads via
/// `std::thread::scope` (no rayon dependency); the result is identical for any
/// thread count.
///
/// # Errors
/// Returns [`FwError::InvalidRequest`] if the filterbank's bin count does not
/// match the expected `N_FFT/2 + 1 = 201`, or if it has zero mel bins.
pub fn log_mel(samples: &[f32], filters: &MelFilterbank, n_threads: usize) -> FwResult<Mel> {
    if filters.n_fft_bins != N_FREQ_BINS {
        return Err(FwError::InvalidRequest(format!(
            "mel filterbank has {} bins, expected {N_FREQ_BINS} (N_FFT/2+1)",
            filters.n_fft_bins
        )));
    }
    if filters.n_mel == 0 {
        return Err(FwError::InvalidRequest(
            "mel filterbank has zero mel bins".to_string(),
        ));
    }

    let hann = cached_hann_window();
    let (padded, valid_len) = build_padded(samples);
    let n_frames = (padded.len() - N_FFT) / HOP;
    let n_mel = filters.n_mel;

    let mut data = vec![0.0f32; n_mel * n_frames];

    let n_threads = n_threads.max(1).min(n_frames.max(1));

    // The output is mel-major (data[mel * n_frames + frame]), so disjoint frame
    // ranges still map to interleaved backing indices and the slice can't be
    // split cleanly. Each worker instead computes its contiguous frame range
    // into a private frame-major buffer; we stitch those into the mel-major
    // global buffer single-threaded afterwards. The frame partition is disjoint
    // and deterministic, so the result is identical for any thread count.
    let frames_per_thread = n_frames.div_ceil(n_threads);

    let columns: Vec<(usize, usize, Vec<f32>)> = std::thread::scope(|scope| {
        let mut handles = Vec::with_capacity(n_threads);
        for t in 0..n_threads {
            let start = t * frames_per_thread;
            if start >= n_frames {
                break;
            }
            let end = (start + frames_per_thread).min(n_frames);
            let padded_ref = &padded;
            // `hann` is already a `&'static [f32; N_FFT]`; copy the reference
            // into the worker closure.
            let hann_ref = hann;
            let filters_ref = filters;
            handles.push(scope.spawn(move || {
                let len = end - start;
                // frame-major local buffer: local[frame_local * n_mel + mel]
                let mut local = vec![0.0f32; len * n_mel];
                for (fl, frame) in (start..end).enumerate() {
                    let col = &mut local[fl * n_mel..fl * n_mel + n_mel];
                    compute_frame_column(frame, padded_ref, valid_len, hann_ref, filters_ref, col);
                }
                (start, end, local)
            }));
        }
        handles
            .into_iter()
            .map(|h| match h.join() {
                Ok(v) => v,
                // A worker can only panic on an internal logic bug; re-raise it
                // on the owning thread rather than silently dropping frames.
                Err(payload) => std::panic::resume_unwind(payload),
            })
            .collect()
    });

    // Stitch frame-major locals into the mel-major global buffer.
    for (start, end, local) in columns {
        for (fl, frame) in (start..end).enumerate() {
            for m in 0..n_mel {
                data[m * n_frames + frame] = local[fl * n_mel + m];
            }
        }
    }

    // Global clamp + normalize: log_spec = max(log_spec, max - 8); (x + 4) / 4.
    let mut mmax = f32::NEG_INFINITY;
    for &v in &data {
        if v > mmax {
            mmax = v;
        }
    }
    let floor = mmax - 8.0;
    for v in &mut data {
        if *v < floor {
            *v = floor;
        }
        *v = (*v + 4.0) / 4.0;
    }

    Ok(Mel {
        n_mel,
        n_frames,
        data,
    })
}

/// Slice a window of `n_frames` frames from `mel` starting at `frame_offset`,
/// zero-padding (with the post-normalization [`SILENCE_FLOOR`]) past the end.
///
/// whisper.cpp pads the *encoder input* mel with the noise-floor value (the
/// normalized `log10(1e-10)` ⇒ `-1.5`), NOT raw zeros — because the entire mel
/// has already been clamped and normalized before it reaches the encoder. This
/// helper reproduces that: a window extending past the real data is filled with
/// [`SILENCE_FLOOR`].
#[must_use]
pub fn chunk_frames(mel: &Mel, frame_offset: usize, n_frames: usize) -> Mel {
    let mut data = vec![SILENCE_FLOOR; mel.n_mel * n_frames];
    for m in 0..mel.n_mel {
        for f in 0..n_frames {
            let src = frame_offset + f;
            if src < mel.n_frames {
                data[m * n_frames + f] = mel.data[m * mel.n_frames + src];
            }
        }
    }
    Mel {
        n_mel: mel.n_mel,
        n_frames,
        data,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic LCG (no `rand` dependency). glibc constants.
    struct Lcg(u64);
    impl Lcg {
        fn new(seed: u64) -> Self {
            Self(seed)
        }
        fn next_f32(&mut self) -> f32 {
            // x_{n+1} = (a*x + c) mod 2^31
            self.0 = self.0.wrapping_mul(1_103_515_245).wrapping_add(12_345) & 0x7fff_ffff;
            // map to [-1, 1)
            (self.0 as f32 / (1u32 << 30) as f32) - 1.0
        }
    }

    /// Independent naive DFT reference (computed fully in f64 then narrowed).
    fn naive_dft(input: &[f32]) -> Vec<(f64, f64)> {
        let n = input.len();
        let mut out = Vec::with_capacity(n);
        for k in 0..n {
            let mut re = 0.0f64;
            let mut im = 0.0f64;
            for (j, &x) in input.iter().enumerate() {
                let theta = (2.0 * PI * (k * j) as f64) / n as f64;
                re += f64::from(x) * theta.cos();
                im -= f64::from(x) * theta.sin();
            }
            out.push((re, im));
        }
        out
    }

    fn fft_to_complex(input: &[f32]) -> Vec<(f64, f64)> {
        let mut out = vec![0.0f32; 2 * input.len()];
        fft(input, &mut out);
        (0..input.len())
            .map(|k| (f64::from(out[2 * k]), f64::from(out[2 * k + 1])))
            .collect()
    }

    #[test]
    fn fft_matches_naive_dft() {
        for &n in &[400usize, 256, 200, 5] {
            let mut lcg = Lcg::new(0xDEAD_BEEF ^ n as u64);
            let input: Vec<f32> = (0..n).map(|_| lcg.next_f32()).collect();

            let got = fft_to_complex(&input);
            let want = naive_dft(&input);

            // reference energy scale for relative error
            let scale: f64 = want
                .iter()
                .map(|(re, im)| (re * re + im * im).sqrt())
                .fold(0.0f64, f64::max)
                .max(1e-9);

            for k in 0..n {
                let dre = (got[k].0 - want[k].0).abs();
                let dim = (got[k].1 - want[k].1).abs();
                let rel = (dre.max(dim)) / scale;
                assert!(
                    rel < 1e-4,
                    "N={n} k={k} rel={rel} got={:?} want={:?}",
                    got[k],
                    want[k]
                );
            }
        }
    }

    #[test]
    fn fft_parseval() {
        // sum |x|^2 == (1/N) sum |X|^2
        for &n in &[400usize, 200, 256] {
            let mut lcg = Lcg::new(0x1234_5678 ^ n as u64);
            let input: Vec<f32> = (0..n).map(|_| lcg.next_f32()).collect();
            let time_energy: f64 = input.iter().map(|&x| f64::from(x) * f64::from(x)).sum();
            let spec = fft_to_complex(&input);
            let freq_energy: f64 =
                spec.iter().map(|(re, im)| re * re + im * im).sum::<f64>() / n as f64;
            let rel = (time_energy - freq_energy).abs() / time_energy.max(1e-9);
            assert!(rel < 1e-4, "Parseval N={n} rel={rel}");
        }
    }

    fn dummy_filters(n_mel: usize) -> MelFilterbank {
        // Identity-ish: each mel bin sums a disjoint contiguous slice of freq
        // bins. Sufficient for silence/determinism tests (exact weights don't
        // matter — silence yields zero power regardless).
        let mut data = vec![0.0f32; n_mel * N_FREQ_BINS];
        for m in 0..n_mel {
            let bin = (m * N_FREQ_BINS / n_mel).min(N_FREQ_BINS - 1);
            data[m * N_FREQ_BINS + bin] = 1.0;
        }
        MelFilterbank {
            n_mel,
            n_fft_bins: N_FREQ_BINS,
            data,
        }
    }

    #[test]
    fn silence_collapses_to_floor() {
        let filters = dummy_filters(80);
        let samples = vec![0.0f32; SAMPLE_RATE]; // 1s of silence
        let mel = log_mel(&samples, &filters, 4).expect("log_mel");
        for &v in &mel.data {
            assert!(
                (v - SILENCE_FLOOR).abs() < 1e-6,
                "silence sample {v} != {SILENCE_FLOOR}"
            );
        }
        // Sanity: analytic floor derivation.
        let pre = (1e-10f64).log10() as f32; // -10
        assert_eq!((pre + 4.0) / 4.0, SILENCE_FLOOR);
    }

    #[test]
    fn determinism_across_thread_counts() {
        let filters = dummy_filters(80);
        let mut lcg = Lcg::new(0xABCD_1234);
        let samples: Vec<f32> = (0..SAMPLE_RATE / 2).map(|_| lcg.next_f32() * 0.3).collect();

        let a = log_mel(&samples, &filters, 1).expect("t1");
        let b = log_mel(&samples, &filters, 4).expect("t4");
        assert_eq!(a.n_frames, b.n_frames);
        assert_eq!(a.n_mel, b.n_mel);
        assert_eq!(a.data, b.data, "thread-count nondeterminism");
    }

    #[test]
    fn chunk_frames_pads_with_silence_floor() {
        let filters = dummy_filters(4);
        let samples = vec![0.0f32; 1000];
        let mel = log_mel(&samples, &filters, 1).expect("log_mel");
        // Slice well past the end.
        let big = mel.n_frames + 50;
        let win = chunk_frames(&mel, mel.n_frames - 10, big);
        assert_eq!(win.n_frames, big);
        // The trailing slots (past real data) must be SILENCE_FLOOR.
        for m in 0..win.n_mel {
            for f in 11..big {
                assert!(
                    (win.data[m * big + f] - SILENCE_FLOOR).abs() < 1e-6,
                    "pad slot ({m},{f}) not floor"
                );
            }
        }
    }

    /// Read the 80x201 filterbank directly from a ggml model file using the
    /// known header layout: magic(4) + 11 hparams i32 (44) + n_mel i32 +
    /// n_fft i32 + n_mel*n_fft f32 (little-endian). Avoids depending on the
    /// `ggml` module (which may not compile yet).
    fn read_filters_from_model(path: &std::path::Path) -> Option<MelFilterbank> {
        let bytes = std::fs::read(path).ok()?;
        let rd_i32 = |b: &[u8], off: usize| -> i32 {
            i32::from_le_bytes([b[off], b[off + 1], b[off + 2], b[off + 3]])
        };
        if bytes.len() < 4 + 44 + 8 {
            return None;
        }
        let magic = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        if magic != 0x6767_6d6c {
            return None;
        }
        let after_hparams = 4 + 44; // 48
        let n_mel = rd_i32(&bytes, after_hparams) as usize;
        let n_fft = rd_i32(&bytes, after_hparams + 4) as usize;
        let data_off = after_hparams + 8;
        let count = n_mel * n_fft;
        if bytes.len() < data_off + count * 4 {
            return None;
        }
        let mut data = Vec::with_capacity(count);
        for i in 0..count {
            let o = data_off + i * 4;
            data.push(f32::from_le_bytes([
                bytes[o],
                bytes[o + 1],
                bytes[o + 2],
                bytes[o + 3],
            ]));
        }
        Some(MelFilterbank {
            n_mel,
            n_fft_bins: n_fft,
            data,
        })
    }

    #[test]
    fn sine_440_argmax_matches_analytic_filter_projection() {
        let Some(path) = super::super::find_model_file("tiny.en") else {
            eprintln!("skipping: tiny.en model not found");
            return;
        };
        let Some(filters) = read_filters_from_model(&path) else {
            eprintln!("skipping: could not parse filterbank from {path:?}");
            return;
        };
        assert_eq!(filters.n_fft_bins, N_FREQ_BINS);

        // 440 Hz sine at 16 kHz, 1 second.
        let freq = 440.0f64;
        let samples: Vec<f32> = (0..SAMPLE_RATE)
            .map(|n| (2.0 * PI * freq * n as f64 / SAMPLE_RATE as f64).sin() as f32)
            .collect();

        let mel = log_mel(&samples, &filters, 4).expect("log_mel");

        // Empirical argmax over mel bins, averaged across a band of mid frames
        // (well inside the audio, away from edges and the 30s zero tail).
        let mid = (SAMPLE_RATE / HOP) / 2; // ~50
        let mut acc = vec![0.0f64; filters.n_mel];
        let mut cnt = 0;
        for f in mid.saturating_sub(5)..(mid + 5) {
            if f >= mel.n_frames {
                break;
            }
            for (m, a) in acc.iter_mut().enumerate() {
                *a += f64::from(mel.data[m * mel.n_frames + f]);
            }
            cnt += 1;
        }
        assert!(cnt > 0);
        let emp_argmax = acc
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap();

        // Analytic expectation: a pure 440 Hz tone concentrates power at the
        // nearest FFT bin = round(440 * N_FFT / SAMPLE_RATE) = round(11) = 11.
        // Project a unit-power spectrum at that bin through the filterbank and
        // take the strongest mel bin.
        let bin = (freq * N_FFT as f64 / SAMPLE_RATE as f64).round() as usize;
        let mut proj = vec![0.0f64; filters.n_mel];
        for (m, p) in proj.iter_mut().enumerate() {
            *p = f64::from(filters.data[m * filters.n_fft_bins + bin]);
        }
        let ana_argmax = proj
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap();

        assert_eq!(
            emp_argmax, ana_argmax,
            "empirical mel argmax {emp_argmax} != analytic {ana_argmax} for 440Hz"
        );
    }
}
