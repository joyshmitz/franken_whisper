//! whisper.cpp ggml `.bin` model-file parser (f32 / f16).
//!
//! This module ports the *exact* on-disk format read by whisper.cpp's
//! `whisper_model_load()` (see `src/whisper.cpp` in the upstream repo). The
//! goal is byte-for-byte fidelity so the rest of the native engine inherits
//! whisper's weights, mel filterbank, and byte-level-BPE vocab without any
//! re-derivation.
//!
//! # File layout
//!
//! All scalars are little-endian. The file is a flat stream:
//!
//! 1. `magic` — one `u32`, must equal `0x6767_6d6c` (`"ggml"`).
//! 2. `hparams` — eleven consecutive `i32` (see [`WhisperHParams`]).
//! 3. mel filterbank — `n_mel` (`i32`), `n_fft` (`i32`), then
//!    `n_mel * n_fft` `f32` weights, row-major (`data[mel * n_fft + bin]`).
//! 4. vocab — `n_vocab_in_file` (`i32`), then that many entries each of
//!    `len` (`u32`) followed by `len` raw token bytes (byte-level BPE,
//!    already applied — decoding is concatenation).
//! 5. tensor directory — repeated until EOF: `n_dims` (`i32`),
//!    `name_len` (`i32`), `ttype` (`i32`), then `n_dims` dimension `i32`s in
//!    **ggml order** (fastest axis first / reversed row-major), then
//!    `name_len` name bytes, then the tensor payload (`n_elements * bpe`
//!    bytes, `bpe` = 4 for f32, 2 for f16). **There is no padding/alignment
//!    between entries** — whisper.cpp reads each tensor's bytes immediately
//!    after its name and loops; EOF is detected by the read of the next
//!    `n_dims`/`name_len`/`ttype` triple coming up short. We assert the
//!    parser consumes the file exactly to EOF.
//!
//! # ftype
//!
//! `hparams.ftype` selects the storage type of the "big" tensors: `0` = f32,
//! `1` = f16. Quantized formats (any other value) are rejected with a
//! structured [`FwError::Unsupported`] for now (bead bd-frp7 epic scope).
//! Note: each tensor entry *also* carries its own per-tensor `ttype`, so a
//! single file mixes f32 (e.g. biases, conv weights) and f16 (matmul
//! weights); we honour the per-tensor type when dequantizing.
//!
//! # Memory
//!
//! The whole file is read into a single `Vec<u8>` blob and tensor entries
//! index into it by `(byte_offset, byte_len)`. Files run up to ~3 GB, which
//! is acceptable for now. TODO(bd-A14 memory bead): switch to a seek-based /
//! mmap-free streaming loader to avoid holding the entire blob resident.

use std::collections::HashMap;
use std::path::Path;

use ft_core::Float16;

use super::{GgmlDType, MelFilterbank, WhisperHParams};
use crate::error::{FwError, FwResult};

/// ggml file magic: ASCII `"ggml"` as a little-endian `u32`.
const GGML_MAGIC: u32 = 0x6767_6d6c;

/// Maximum number of tensor dimensions ggml encodes (matches upstream's
/// fixed `ne[4]`). Any `n_dims` outside `1..=GGML_MAX_DIMS` is malformed.
const GGML_MAX_DIMS: usize = 4;

/// A single tensor's location and metadata within the model [`GgmlModel::blob`].
///
/// `shape` is stored in **row-major (PyTorch) logical order**: the ggml file
/// stores dimensions reversed (fastest-moving axis `ne[0]` first), and we
/// reverse them on load so that, e.g., `decoder.token_embedding.weight`
/// reports `[n_vocab, n_state]` exactly as PyTorch would. The raw payload in
/// the blob is unchanged (still ggml/row-major-contiguous), so the flat
/// element order returned by [`GgmlModel::tensor_f32`] matches the reversed
/// shape directly.
#[derive(Debug, Clone)]
pub struct TensorEntry {
    /// Logical shape in row-major (PyTorch) order — the reverse of the
    /// `ne[]` order stored in the file.
    pub shape: Vec<usize>,
    /// Element storage type for this tensor (`F32` or `F16`).
    pub dtype: GgmlDType,
    /// Byte offset of the tensor payload within [`GgmlModel::blob`].
    byte_offset: usize,
    /// Byte length of the tensor payload within [`GgmlModel::blob`].
    byte_len: usize,
}

impl TensorEntry {
    /// Total number of elements (product of the logical shape).
    #[must_use]
    pub fn n_elements(&self) -> usize {
        self.shape.iter().product()
    }
}

/// A fully parsed whisper.cpp ggml model file.
///
/// Holds the header hyper-parameters, embedded mel filterbank, byte-level
/// vocab, a name→[`TensorEntry`] directory, and the raw file bytes the
/// directory indexes into. Construct via [`GgmlModel::load`].
#[derive(Debug)]
pub struct GgmlModel {
    /// Header hyper-parameters (the eleven `i32`s after the magic).
    pub hparams: WhisperHParams,
    /// Mel filterbank embedded in the file (`n_mel x n_fft_bins`).
    pub filters: MelFilterbank,
    /// Vocab tokens as raw bytes, indexed by token id. Length is the vocab
    /// count stored *in the file* (which may be smaller than
    /// `hparams.n_vocab`; the gap is special/extra tokens synthesized by id —
    /// see [`GgmlModel::n_extra_tokens`]).
    pub vocab_tokens: Vec<Vec<u8>>,
    /// Tensor directory: tensor name → location/metadata in [`Self::blob`].
    tensors: HashMap<String, TensorEntry>,
    /// The entire model file read into memory; tensor payloads are slices.
    blob: Vec<u8>,
}

impl GgmlModel {
    /// Parse a ggml `.bin` model file from `path`.
    ///
    /// Reads the whole file into memory, validates the magic, parses the
    /// header / filterbank / vocab / tensor directory, and asserts the parse
    /// consumes the file exactly (no trailing bytes).
    ///
    /// # Errors
    ///
    /// - [`FwError::Io`] if the file cannot be read.
    /// - [`FwError::InvalidRequest`] for a bad magic, a truncated/malformed
    ///   structure, or trailing bytes after the tensor directory.
    /// - [`FwError::Unsupported`] for a quantized `ftype`.
    pub fn load(path: &Path) -> FwResult<Self> {
        let blob = std::fs::read(path)?;
        Self::parse(blob)
    }

    /// Parse an in-memory ggml blob (used by [`Self::load`] and tests).
    fn parse(blob: Vec<u8>) -> FwResult<Self> {
        let mut cur = Cursor::new(&blob);

        let magic = cur.read_u32()?;
        if magic != GGML_MAGIC {
            return Err(FwError::InvalidRequest(format!(
                "bad ggml magic: got {magic:#010x}, expected {GGML_MAGIC:#010x}"
            )));
        }

        let hparams = WhisperHParams {
            n_vocab: cur.read_i32()?,
            n_audio_ctx: cur.read_i32()?,
            n_audio_state: cur.read_i32()?,
            n_audio_head: cur.read_i32()?,
            n_audio_layer: cur.read_i32()?,
            n_text_ctx: cur.read_i32()?,
            n_text_state: cur.read_i32()?,
            n_text_head: cur.read_i32()?,
            n_text_layer: cur.read_i32()?,
            n_mels: cur.read_i32()?,
            ftype: cur.read_i32()?,
        };

        // ftype gates the "big tensor" storage type. whisper.cpp strips a
        // quantization-version factor before mapping; for the formats we
        // support (f32 / f16) ftype is exactly 0 or 1. Anything else is a
        // quantized format we don't decode yet.
        if hparams.ftype != 0 && hparams.ftype != 1 {
            return Err(FwError::Unsupported(format!(
                "quantized ggml ftype {} is not supported (only ftype 0=f32, 1=f16)",
                hparams.ftype
            )));
        }

        // Mel filterbank.
        let n_mel = cur.read_i32()?;
        let n_fft = cur.read_i32()?;
        let n_mel = usize_from_i32(n_mel, "filters.n_mel")?;
        let n_fft_bins = usize_from_i32(n_fft, "filters.n_fft")?;
        let n_filter = n_mel
            .checked_mul(n_fft_bins)
            .ok_or_else(|| FwError::InvalidRequest("mel filterbank size overflow".to_owned()))?;
        let mut data = Vec::with_capacity(n_filter);
        for _ in 0..n_filter {
            data.push(cur.read_f32()?);
        }
        let filters = MelFilterbank {
            n_mel,
            n_fft_bins,
            data,
        };

        // Vocab — raw byte-level BPE tokens.
        let n_vocab_file = cur.read_i32()?;
        let n_vocab_file = usize_from_i32(n_vocab_file, "file vocab count")?;
        let mut vocab_tokens = Vec::with_capacity(n_vocab_file);
        for _ in 0..n_vocab_file {
            let len = cur.read_u32()? as usize;
            vocab_tokens.push(cur.read_bytes(len)?.to_vec());
        }

        // Tensor directory — loop until EOF.
        let mut tensors: HashMap<String, TensorEntry> = HashMap::new();
        loop {
            // whisper.cpp reads the next (n_dims, name_len, ttype) triple and
            // only *then* checks EOF; a clean end-of-directory is exactly the
            // point where there are no more bytes for that triple.
            if cur.at_end() {
                break;
            }
            let n_dims = cur.read_i32()?;
            let name_len = cur.read_i32()?;
            let ttype = cur.read_i32()?;

            let n_dims = usize_from_i32(n_dims, "tensor n_dims")?;
            if n_dims == 0 || n_dims > GGML_MAX_DIMS {
                return Err(FwError::InvalidRequest(format!(
                    "tensor n_dims {n_dims} out of range 1..={GGML_MAX_DIMS}"
                )));
            }
            let name_len = usize_from_i32(name_len, "tensor name length")?;

            let dtype = match ttype {
                0 => GgmlDType::F32,
                1 => GgmlDType::F16,
                other => {
                    return Err(FwError::Unsupported(format!(
                        "tensor element type {other} is not supported (only 0=f32, 1=f16)"
                    )));
                }
            };

            // Dimensions are stored in ggml order (ne[0] = fastest axis).
            let mut ne = Vec::with_capacity(n_dims);
            for _ in 0..n_dims {
                ne.push(usize_from_i32(cur.read_i32()?, "tensor dimension")?);
            }

            let name_bytes = cur.read_bytes(name_len)?;
            let name = String::from_utf8(name_bytes.to_vec()).map_err(|_| {
                FwError::InvalidRequest("tensor name is not valid UTF-8".to_owned())
            })?;

            // Reverse ggml dims → row-major (PyTorch) logical shape.
            let mut shape = ne;
            shape.reverse();

            let n_elements: usize = shape
                .iter()
                .copied()
                .try_fold(1usize, |acc, d| acc.checked_mul(d))
                .ok_or_else(|| {
                    FwError::InvalidRequest(format!("tensor '{name}' element count overflow"))
                })?;
            let bpe = match dtype {
                GgmlDType::F32 => 4usize,
                GgmlDType::F16 => 2usize,
            };
            let byte_len = n_elements.checked_mul(bpe).ok_or_else(|| {
                FwError::InvalidRequest(format!("tensor '{name}' byte length overflow"))
            })?;

            let byte_offset = cur.pos();
            cur.skip(byte_len)?;

            tensors.insert(
                name,
                TensorEntry {
                    shape,
                    dtype,
                    byte_offset,
                    byte_len,
                },
            );
        }

        if !cur.at_end() {
            return Err(FwError::InvalidRequest(format!(
                "trailing bytes after tensor directory: {} byte(s) unconsumed",
                blob.len() - cur.pos()
            )));
        }

        Ok(Self {
            hparams,
            filters,
            vocab_tokens,
            tensors,
            blob,
        })
    }

    /// Number of "extra"/special tokens synthesized by id, i.e. the gap
    /// between `hparams.n_vocab` and the vocab count stored in the file.
    ///
    /// whisper.cpp fills ids `[file_vocab, hparams.n_vocab)` with synthetic
    /// placeholder names (`[_EOT_]`, `[_SOT_]`, `[_LANG_xx]`, `[_TT_n]`,
    /// `[_extra_token_n]`, …). We don't need the *names* to decode real text
    /// (those ids never appear in transcribed text output), only to know how
    /// many ids exist beyond the file vocab; the tokenizer bead (bd-zpfy)
    /// derives the special-id *values* from `hparams`. tiny.en: file vocab
    /// 50257, `n_vocab` 51864 → 1607 extra. large-v3: 51866 vs 50257 → 1609.
    #[must_use]
    pub fn n_extra_tokens(&self) -> usize {
        (self.hparams.n_vocab.max(0) as usize).saturating_sub(self.vocab_tokens.len())
    }

    /// Tensor names in sorted order (stable iteration, e.g. for fixture dumps).
    pub fn tensor_names(&self) -> impl Iterator<Item = &str> {
        let mut names: Vec<&str> = self.tensors.keys().map(String::as_str).collect();
        names.sort_unstable();
        names.into_iter()
    }

    /// Look up a tensor entry by name.
    #[must_use]
    pub fn tensor(&self, name: &str) -> Option<&TensorEntry> {
        self.tensors.get(name)
    }

    /// Decode a tensor to `(logical_shape, f32_values)`.
    ///
    /// f16 tensors are dequantized to f32 in pure safe Rust. The returned
    /// values are in the tensor's flat (row-major contiguous) order, matching
    /// the reversed logical shape.
    ///
    /// # Errors
    ///
    /// - [`FwError::InvalidRequest`] if `name` is unknown or the stored byte
    ///   length is inconsistent with the shape/dtype (corruption).
    pub fn tensor_f32(&self, name: &str) -> FwResult<(Vec<usize>, Vec<f32>)> {
        let entry = self
            .tensors
            .get(name)
            .ok_or_else(|| FwError::InvalidRequest(format!("unknown tensor '{name}'")))?;
        let raw = self
            .blob
            .get(entry.byte_offset..entry.byte_offset + entry.byte_len)
            .ok_or_else(|| {
                FwError::InvalidRequest(format!("tensor '{name}' payload out of bounds"))
            })?;

        let n_elements = entry.n_elements();
        let values = match entry.dtype {
            GgmlDType::F32 => {
                if raw.len() != n_elements * 4 {
                    return Err(FwError::InvalidRequest(format!(
                        "tensor '{name}' f32 byte length {} != {} elements * 4",
                        raw.len(),
                        n_elements
                    )));
                }
                raw.chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect()
            }
            GgmlDType::F16 => {
                if raw.len() != n_elements * 2 {
                    return Err(FwError::InvalidRequest(format!(
                        "tensor '{name}' f16 byte length {} != {} elements * 2",
                        raw.len(),
                        n_elements
                    )));
                }
                raw.chunks_exact(2)
                    .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
                    .collect()
            }
        };

        Ok((entry.shape.clone(), values))
    }
}

/// Convert an IEEE-754 half-precision bit pattern to `f32`.
///
/// Delegates to `ft_core::Float16` (a re-export of the well-tested `half`
/// crate) per the bd-frp7 epic's "prefer half through ft-core" guidance.
/// Handles subnormals, infinities, and NaN correctly; see the unit tests for
/// the canonical bit-pattern matrix (`0x3C00`=1.0, `0x7C00`=+inf, …).
#[inline]
#[must_use]
fn f16_to_f32(bits: u16) -> f32 {
    Float16::from_bits(bits).to_f32()
}

/// Convert an `i32` count/dimension to `usize`, rejecting negatives.
fn usize_from_i32(value: i32, what: &str) -> FwResult<usize> {
    usize::try_from(value)
        .map_err(|_| FwError::InvalidRequest(format!("{what} is negative ({value})")))
}

/// Minimal little-endian byte cursor over a borrowed blob. Every read is
/// bounds-checked and surfaces a structured error on underflow instead of
/// panicking.
struct Cursor<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }

    fn pos(&self) -> usize {
        self.pos
    }

    fn at_end(&self) -> bool {
        self.pos >= self.buf.len()
    }

    fn read_bytes(&mut self, len: usize) -> FwResult<&'a [u8]> {
        let end = self
            .pos
            .checked_add(len)
            .ok_or_else(|| FwError::InvalidRequest("read length overflow".to_owned()))?;
        let slice = self.buf.get(self.pos..end).ok_or_else(|| {
            FwError::InvalidRequest(format!(
                "unexpected end of file: needed {len} byte(s) at offset {}, have {}",
                self.pos,
                self.buf.len()
            ))
        })?;
        self.pos = end;
        Ok(slice)
    }

    fn skip(&mut self, len: usize) -> FwResult<()> {
        self.read_bytes(len).map(|_| ())
    }

    fn read_u32(&mut self) -> FwResult<u32> {
        let b = self.read_bytes(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn read_i32(&mut self) -> FwResult<i32> {
        Ok(self.read_u32()? as i32)
    }

    fn read_f32(&mut self) -> FwResult<f32> {
        let b = self.read_bytes(4)?;
        Ok(f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::native_engine::find_model_file;

    /// Builder for a minimal but fully valid in-memory ggml blob, used for
    /// hermetic parser coverage that doesn't require a real model file.
    struct SyntheticModel {
        bytes: Vec<u8>,
    }

    impl SyntheticModel {
        /// A minimal model: tiny hparams, a 2x3 filterbank, a 3-token vocab,
        /// one 2x2 f32 tensor, and one 2x2 f16 tensor.
        fn minimal() -> Self {
            let mut b = SyntheticModel { bytes: Vec::new() };
            b.push_u32(GGML_MAGIC);
            // hparams: n_vocab .. ftype (11 i32). ftype=1 (f16).
            for v in [5i32, 1500, 384, 6, 4, 448, 384, 6, 4, 80, 1] {
                b.push_i32(v);
            }
            // filterbank 2x3.
            b.push_i32(2);
            b.push_i32(3);
            for v in [0.0f32, 0.1, 0.2, 0.3, 0.4, 0.5] {
                b.push_f32(v);
            }
            // vocab: 3 tokens.
            b.push_i32(3);
            for tok in [b"!".as_slice(), b"\"", b"ab"] {
                b.push_u32(tok.len() as u32);
                b.bytes.extend_from_slice(tok);
            }
            // tensor 1: f32, ggml dims [3, 2] -> logical shape [2, 3].
            b.push_tensor(
                "w_f32",
                0,
                &[3, 2],
                &Payload::F32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            );
            // tensor 2: f16, ggml dims [2, 2] -> logical shape [2, 2].
            b.push_tensor(
                "w_f16",
                1,
                &[2, 2],
                &Payload::F16(vec![1.0, 0.0, -2.0, 0.5]),
            );
            b
        }

        fn push_u32(&mut self, v: u32) {
            self.bytes.extend_from_slice(&v.to_le_bytes());
        }
        fn push_i32(&mut self, v: i32) {
            self.bytes.extend_from_slice(&v.to_le_bytes());
        }
        fn push_f32(&mut self, v: f32) {
            self.bytes.extend_from_slice(&v.to_le_bytes());
        }

        /// `ne` is in ggml order (fastest axis first).
        fn push_tensor(&mut self, name: &str, ttype: i32, ne: &[i32], payload: &Payload) {
            self.push_i32(ne.len() as i32);
            self.push_i32(name.len() as i32);
            self.push_i32(ttype);
            for &d in ne {
                self.push_i32(d);
            }
            self.bytes.extend_from_slice(name.as_bytes());
            match payload {
                Payload::F32(vals) => {
                    for &v in vals {
                        self.push_f32(v);
                    }
                }
                Payload::F16(vals) => {
                    for &v in vals {
                        self.bytes
                            .extend_from_slice(&Float16::from_f32(v).to_bits().to_le_bytes());
                    }
                }
            }
        }
    }

    enum Payload {
        F32(Vec<f32>),
        F16(Vec<f32>),
    }

    // ── f16 dequant unit tests over canonical bit patterns ──

    #[test]
    fn f16_known_bit_patterns() {
        assert_eq!(f16_to_f32(0x3C00), 1.0);
        assert_eq!(f16_to_f32(0x0000), 0.0);
        assert_eq!(f16_to_f32(0x8000), -0.0);
        assert!(f16_to_f32(0x8000).is_sign_negative());
        assert_eq!(f16_to_f32(0xC000), -2.0);
        // Smallest positive subnormal: 2^-24.
        let subnormal = f16_to_f32(0x0001);
        assert!(
            (subnormal - 2f32.powi(-24)).abs() < 1e-30,
            "got {subnormal}"
        );
        assert_eq!(f16_to_f32(0x7C00), f32::INFINITY);
        assert_eq!(f16_to_f32(0xFC00), f32::NEG_INFINITY);
        assert!(f16_to_f32(0x7E00).is_nan());
    }

    // ── synthetic-blob parser tests ──

    #[test]
    fn header_and_filterbank_roundtrip() {
        let model = GgmlModel::parse(SyntheticModel::minimal().bytes).expect("parse");
        assert_eq!(model.hparams.n_vocab, 5);
        assert_eq!(model.hparams.n_audio_state, 384);
        assert_eq!(model.hparams.ftype, 1);
        assert_eq!(model.filters.n_mel, 2);
        assert_eq!(model.filters.n_fft_bins, 3);
        assert_eq!(model.filters.data, vec![0.0, 0.1, 0.2, 0.3, 0.4, 0.5]);
    }

    #[test]
    fn vocab_bytes_preserved() {
        let model = GgmlModel::parse(SyntheticModel::minimal().bytes).expect("parse");
        assert_eq!(model.vocab_tokens.len(), 3);
        assert_eq!(model.vocab_tokens[0], b"!");
        assert_eq!(model.vocab_tokens[1], b"\"");
        assert_eq!(model.vocab_tokens[2], b"ab");
        // hparams n_vocab (5) > file vocab (3) => 2 extra/special tokens.
        assert_eq!(model.n_extra_tokens(), 2);
    }

    #[test]
    fn tensor_shape_is_reversed_from_ggml_order() {
        let model = GgmlModel::parse(SyntheticModel::minimal().bytes).expect("parse");
        // ggml ne = [3, 2] => logical row-major shape [2, 3].
        let entry = model.tensor("w_f32").expect("w_f32 present");
        assert_eq!(entry.shape, vec![2, 3]);
        assert_eq!(entry.dtype, GgmlDType::F32);
        assert_eq!(entry.n_elements(), 6);
    }

    #[test]
    fn tensor_f32_values_and_f16_dequant() {
        let model = GgmlModel::parse(SyntheticModel::minimal().bytes).expect("parse");
        let (shape, vals) = model.tensor_f32("w_f32").expect("decode w_f32");
        assert_eq!(shape, vec![2, 3]);
        assert_eq!(vals, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        let (shape, vals) = model.tensor_f32("w_f16").expect("decode w_f16");
        assert_eq!(shape, vec![2, 2]);
        assert_eq!(vals, vec![1.0, 0.0, -2.0, 0.5]);
    }

    #[test]
    fn tensor_names_are_sorted() {
        let model = GgmlModel::parse(SyntheticModel::minimal().bytes).expect("parse");
        let names: Vec<&str> = model.tensor_names().collect();
        assert_eq!(names, vec!["w_f16", "w_f32"]);
    }

    #[test]
    fn trailing_bytes_are_rejected() {
        // Append a well-formed (n_dims, name_len, ttype) triple describing a
        // zero-length tensor whose payload is empty: the directory loop sees
        // a non-EOF position, parses the entry, and then finds extra bytes
        // left over (the name `"x"`'s tensor having zero elements still leaves
        // the appended junk unconsumed) — exercising the explicit
        // trailing-bytes guard rather than a mid-read truncation.
        let mut bytes = SyntheticModel::minimal().bytes;
        // A complete extra header for a 1-D f32 tensor with dim 0 and a 1-byte
        // name, i.e. nelements==0 so byte_len==0, leaving the appended junk.
        bytes.extend_from_slice(&1i32.to_le_bytes()); // n_dims
        bytes.extend_from_slice(&1i32.to_le_bytes()); // name_len
        bytes.extend_from_slice(&0i32.to_le_bytes()); // ttype f32
        bytes.extend_from_slice(&0i32.to_le_bytes()); // ne[0] = 0
        bytes.extend_from_slice(b"z"); // name
        // Now genuine trailing junk that no tensor entry will consume.
        bytes.extend_from_slice(&[0xAA, 0xBB, 0xCC]);
        let err = GgmlModel::parse(bytes).expect_err("must reject trailing bytes");
        assert!(matches!(err, FwError::InvalidRequest(_)), "got {err:?}");
        // Either guard (explicit trailing-bytes check, or the short-read on the
        // next phantom header) is an acceptable rejection of a corrupt tail.
        let msg = err.to_string();
        assert!(
            msg.contains("trailing bytes") || msg.contains("unexpected end of file"),
            "{err}"
        );
    }

    #[test]
    fn bad_magic_is_rejected() {
        let mut bytes = SyntheticModel::minimal().bytes;
        bytes[0] = 0x00; // corrupt the magic
        let err = GgmlModel::parse(bytes).expect_err("must reject bad magic");
        assert!(matches!(err, FwError::InvalidRequest(_)), "got {err:?}");
        assert!(err.to_string().contains("magic"), "{err}");
    }

    #[test]
    fn unsupported_ftype_is_rejected() {
        let mut b = SyntheticModel { bytes: Vec::new() };
        b.push_u32(GGML_MAGIC);
        // ftype = 7 (a quantized format) in the 11th hparam slot.
        for v in [5i32, 1500, 384, 6, 4, 448, 384, 6, 4, 80, 7] {
            b.push_i32(v);
        }
        let err = GgmlModel::parse(b.bytes).expect_err("must reject quantized ftype");
        match err {
            FwError::Unsupported(msg) => {
                assert!(msg.contains('7'), "ftype value should be listed: {msg}");
            }
            other => panic!("expected Unsupported, got {other:?}"),
        }
    }

    #[test]
    fn truncated_header_is_rejected() {
        let mut bytes = SyntheticModel::minimal().bytes;
        bytes.truncate(10); // cut off mid-hparams
        let err = GgmlModel::parse(bytes).expect_err("must reject truncation");
        assert!(matches!(err, FwError::InvalidRequest(_)), "got {err:?}");
    }

    // ── gated tests against the real tiny.en model ──

    #[test]
    fn real_tiny_en_full_parse() {
        let Some(path) = find_model_file("tiny.en") else {
            eprintln!("SKIP real_tiny_en_full_parse: ggml-tiny.en.bin not found");
            return;
        };
        let model = GgmlModel::load(&path).expect("load tiny.en");

        // Exact hparams.
        assert_eq!(model.hparams.n_vocab, 51864);
        assert_eq!(model.hparams.n_audio_ctx, 1500);
        assert_eq!(model.hparams.n_audio_state, 384);
        assert_eq!(model.hparams.n_audio_head, 6);
        assert_eq!(model.hparams.n_audio_layer, 4);
        assert_eq!(model.hparams.n_text_ctx, 448);
        assert_eq!(model.hparams.n_text_state, 384);
        assert_eq!(model.hparams.n_text_layer, 4);
        assert_eq!(model.hparams.n_mels, 80);
        assert_eq!(model.hparams.ftype, 1);

        // Filterbank dims.
        assert_eq!(model.filters.n_mel, 80);
        assert_eq!(model.filters.n_fft_bins, 201);
        assert_eq!(model.filters.data.len(), 80 * 201);

        // File vocab (50257) < hparams n_vocab (51864): 1607 extra tokens.
        assert_eq!(model.vocab_tokens.len(), 50257);
        assert_eq!(model.n_extra_tokens(), 51864 - 50257);
        assert_eq!(model.vocab_tokens[0], b"!");
        assert_eq!(model.vocab_tokens[1], b"\"");
        assert_eq!(model.vocab_tokens[2], b"#");

        // Full-file consumption is enforced inside parse() (trailing-byte
        // check); reaching here proves the parser consumed exactly to EOF.

        // Known tensors and their logical (row-major) shapes.
        let conv1 = model.tensor("encoder.conv1.weight").expect("conv1");
        assert_eq!(conv1.shape, vec![384, 80, 3]);

        let tok_emb = model
            .tensor("decoder.token_embedding.weight")
            .expect("token_embedding");
        assert_eq!(tok_emb.shape, vec![51864, 384]);

        let pos_emb = model
            .tensor("encoder.positional_embedding")
            .expect("positional_embedding");
        assert_eq!(pos_emb.shape, vec![1500, 384]);

        // Spot-check that decoded data is finite.
        for name in [
            "encoder.conv1.weight",
            "encoder.positional_embedding",
            "decoder.token_embedding.weight",
        ] {
            let (_shape, vals) = model.tensor_f32(name).expect("decode");
            assert!(
                vals.iter().all(|v| v.is_finite()),
                "tensor {name} has non-finite values"
            );
            assert!(!vals.is_empty());
        }
    }

    #[test]
    fn real_large_v3_turbo_hparams() {
        let Some(path) = find_model_file("large-v3-turbo") else {
            eprintln!("SKIP real_large_v3_turbo_hparams: ggml-large-v3-turbo.bin not found");
            return;
        };
        let model = GgmlModel::load(&path).expect("load large-v3-turbo");
        assert_eq!(model.hparams.n_vocab, 51866);
        assert_eq!(model.hparams.n_audio_ctx, 1500);
        assert_eq!(model.hparams.n_audio_state, 1280);
        assert_eq!(model.hparams.n_audio_head, 20);
        assert_eq!(model.hparams.n_audio_layer, 32);
        assert_eq!(model.hparams.n_text_ctx, 448);
        assert_eq!(model.hparams.n_text_state, 1280);
        assert_eq!(model.hparams.n_text_head, 20);
        assert_eq!(model.hparams.n_text_layer, 4);
        assert_eq!(model.hparams.n_mels, 128);
        assert_eq!(model.hparams.ftype, 1);
        assert_eq!(model.filters.n_mel, 128);
        assert_eq!(model.filters.n_fft_bins, 201);
        // File stores 50257 tokens; hparams 51866 => 1609 extra.
        assert_eq!(model.vocab_tokens.len(), 50257);
        assert_eq!(model.n_extra_tokens(), 51866 - 50257);
        assert_eq!(model.vocab_tokens[0], b"!");
        assert_eq!(model.vocab_tokens[1], b"\"");
        assert_eq!(model.vocab_tokens[2], b"#");
    }
}
