# Legacy Project Analysis

> Self-contained analysis of the three legacy projects that inform franken_whisper's design.
> This document is the definitive reference. No need to consult original sources.

---

## Table of Contents

1. [Overview](#overview)
2. [legacy_whispercpp (C/C++)](#1-legacy_whispercpp-cc)
3. [legacy_insanely_fast_whisper (Python)](#2-legacy_insanely_fast_whisper-python)
4. [legacy_whisper_diarization (Python)](#3-legacy_whisper_diarization-python)
5. [Cross-Project Comparison Matrix](#cross-project-comparison-matrix)
6. [Synthesis Notes for franken_whisper](#synthesis-notes-for-franken_whisper)

---

## Overview

franken_whisper synthesizes the best **ideas** (not code) from three distinct speech-processing projects into a unified Rust-native architecture. Each legacy project occupies a different niche in the ASR (Automatic Speech Recognition) landscape:

- **whisper.cpp**: Local-first, CPU-optimized, real-time capable inference engine.
- **insanely-fast-whisper**: GPU-first, throughput-optimized, ergonomic batch transcription tool.
- **whisper-diarization**: Multi-stage pipeline for high-fidelity speaker-attributed transcription.

None of these projects alone covers the full problem space. Together, they define the behavioral oracle for what franken_whisper must achieve and surpass.

---

## 1. legacy_whispercpp (C/C++)

### Summary

whisper.cpp is a C/C++ port of OpenAI's Whisper model, built on the ggml tensor library. It prioritizes local execution, minimal dependencies, and broad hardware support. It is the reference implementation for fast CPU-based Whisper inference.

### Key Strengths

#### Fast Local CPU Inference with SIMD Optimizations
- Hand-tuned SIMD paths for ARM NEON, x86 AVX/AVX2/AVX-512, and WASM SIMD.
- ggml tensor library avoids heavyweight ML framework dependencies (no PyTorch, no TensorFlow).
- Inference runs entirely in-process with no Python interpreter overhead.
- Startup time is measured in milliseconds, not seconds.

#### Streaming and Real-Time Transcription
- Dedicated `whisper-stream` binary for microphone and live audio input.
- Sliding-window inference with configurable step size and keep-window overlap.
- Produces partial transcripts in real time, suitable for live captioning.
- No other legacy project offers streaming capability.

#### Silero VAD (Voice Activity Detection)
- Integrated Silero VAD model for pre-filtering silence and non-speech audio.
- Configurable parameters: detection threshold, minimum speech duration, minimum silence duration, speech padding, maximum speech duration.
- Reduces compute waste by skipping silent regions before they reach the Whisper model.
- VAD output can be used independently of transcription.

#### TinyDiarize for Lightweight Speaker Identification
- Built-in speaker turn detection via `[SPEAKER_TURN]` token injection.
- Lightweight: no external model downloads, no separate diarization pipeline.
- Limited to detecting speaker *changes* rather than identifying or clustering speakers.
- Enabled via `--tdrz` flag.

#### Word-Level Timestamps with Configurable Precision
- Token-level timestamp extraction from the Whisper decoder.
- Configurable maximum segment length (`-ml` flag) for fine-grained control.
- Split-on-word boundary mode (`-sow`) prevents mid-word segment breaks.
- Produces timestamps suitable for subtitle generation and karaoke-style output.

#### Quantized Models for Reduced Memory
- Integer quantization formats: Q4_0, Q4_1, Q5_0, Q5_1, Q8_0.
- Model size reductions of 2-4x with modest accuracy loss.
- Enables running large models on memory-constrained devices (embedded, mobile, older hardware).
- Quantization is applied at the ggml level, transparent to the inference code.

#### 60+ CLI Flags for Fine-Grained Control
- Decode parameters: beam size, best-of, temperature, entropy threshold, logprob threshold, no-speech threshold.
- Audio parameters: sample rate override, channel selection, offset/duration windowing.
- Output format flags: txt, vtt, srt, csv, json, json-full, lrc, karaoke.
- VAD parameters: threshold, min speech/silence duration, padding, max speech duration.
- Language and task flags: language auto-detect, translate mode, initial prompt, grammar constraints.
- Performance flags: thread count, processor count, GPU layer offload count.

#### Multi-Platform Hardware Acceleration
- **CPU**: ARM NEON, AVX/AVX2/AVX-512, WASM SIMD.
- **GPU/NPU**: CUDA/cuBLAS (NVIDIA), Metal (Apple), Core ML (Apple), Vulkan (cross-platform), OpenCL (cross-platform), OpenVINO (Intel), SYCL (Intel), CANN (Huawei Ascend).
- Each backend is a compile-time option with CMake flags.
- GPU layer offload is configurable (partial or full model offload).

#### Extremely Low Memory Footprint
- The tiny.en model runs in under 100MB of RAM.
- Quantized base models fit comfortably in 256MB.
- Suitable for Raspberry Pi, edge devices, and containerized microservices.
- No Python runtime, no JIT compilation, no garbage collector overhead.

#### Additional Notable Features
- HTTP server mode with multipart file upload support.
- Performance benchmarking tool (`whisper-bench`) for reproducible measurements.
- Confidence color-coding in terminal output (green/yellow/red by probability).
- FFmpeg integration guidance for input normalization (expects 16kHz mono WAV).
- Multiple output formats including LRC (lyrics) and karaoke-style colored text.

### Architectural Pattern

```
Audio File → FFmpeg (external) → 16kHz Mono PCM → ggml Whisper Model → Decode Loop → Output Formatter
                                                          ↑
                                                    Silero VAD (optional pre-filter)
```

Single-pass architecture. Audio goes in, segments come out. No multi-stage pipeline. VAD is a pre-filter, not a post-processing step. The entire flow runs in a single process with configurable threading.

### Performance Characteristics

- **CPU throughput**: ~1x real-time for base model on modern x86 with AVX2; ~10x real-time for tiny model.
- **GPU throughput**: Variable by backend; CUDA provides best results, Metal good for Apple Silicon.
- **Startup latency**: Sub-second model load for quantized models.
- **Memory**: 75MB (tiny.en Q5) to ~3GB (large-v3 f16), linear with model size.
- **Streaming latency**: Configurable via step size; typically 1-3 seconds for real-time use.

### Known Limitations

- **Memory safety**: C/C++ with manual memory management. Buffer overflows, use-after-free, and undefined behavior are possible. No formal memory safety guarantees.
- **Build complexity**: CMake-based build system with platform-specific flags for each hardware backend. Cross-compilation requires careful configuration. Dependency on system libraries (CUDA toolkit, Metal framework, etc.) varies by target.
- **No Python integration**: No built-in Python bindings. Third-party bindings exist (pywhispercpp, whispercpp-python) but are not part of the core project and may lag behind releases.
- **Limited diarization**: TinyDiarize detects speaker *turns* but does not identify, cluster, or label speakers. Insufficient for multi-speaker attribution in meetings or interviews.
- **No source separation**: Cannot isolate speech from background music, noise, or overlapping speakers.
- **No forced alignment**: Timestamps come directly from decoder attention, which can drift on long audio.
- **No punctuation restoration**: Output punctuation depends entirely on the Whisper model's predictions.

---

## 2. legacy_insanely_fast_whisper (Python)

### Summary

insanely-fast-whisper is a Python CLI tool built on HuggingFace Transformers that maximizes GPU transcription throughput. It leverages Flash Attention 2, batched decoding, and BetterTransformer optimizations to achieve dramatic speedups on GPU hardware. It is the reference implementation for throughput-optimized batch transcription.

### Key Strengths

#### Flash Attention 2 for ~15x GPU Speedup
- Implements Flash Attention 2 via the `flash_attn` package for memory-efficient attention computation.
- Reduces attention memory complexity from O(n^2) to O(n), enabling longer audio without OOM.
- Achieves approximately 15x wall-clock speedup on long audio files compared to standard attention.
- Enabled via `--flash` flag; requires compatible GPU (Ampere or newer NVIDIA architecture).

#### GPU Batching for High Throughput
- Audio is chunked (default `chunk_length_s=30`) and batched for parallel GPU inference.
- Configurable `batch_size` (default 24) tunes throughput vs. VRAM usage tradeoff.
- Multiple chunks are processed simultaneously, saturating GPU compute.
- Particularly effective for long-form audio (hours of content) where batch parallelism amortizes overhead.

#### Ergonomic CLI with Sensible Defaults
- Minimal required arguments: just `--file-name` to transcribe.
- Sensible defaults for model (`openai/whisper-large-v3`), task (`transcribe`), language (auto-detect).
- Output defaults to JSON with clean transcript structure.
- No build step: `pip install insanely-fast-whisper` and run.

#### pyannote Diarization Integration
- Speaker diarization via pyannote.audio v3.1 pipeline.
- Requires HuggingFace authentication token (`--hf-token`) and acceptance of pyannote model license.
- Supports speaker count constraints: `--num-speakers`, `--min-speakers`, `--max-speakers`.
- Diarization results are merged with transcription at the segment level.
- Produces speaker-labeled output segments.

#### Automatic Device Selection
- Detects available hardware: CUDA (NVIDIA GPU), MPS (Apple Silicon), CPU (fallback).
- `--device-id` allows explicit device selection or multi-GPU targeting.
- FP16 mixed precision enabled by default on GPU for 2x memory savings.
- BetterTransformer optimization applied automatically when available.

#### Simple Deployment
- Single `pip install` with well-defined PyPI dependencies.
- No CMake, no system library compilation, no platform-specific build flags.
- Works in Docker, Colab, cloud VMs with GPU, and local workstations.
- Distil-Whisper model support for smaller footprint with minor accuracy loss.

#### Timestamp Modes
- `--timestamp` flag with `chunk` (segment-level) or `word` (word-level) granularity.
- Word-level timestamps extracted from the Whisper model's token-level timing data.
- Chunk timestamps provide segment boundaries suitable for subtitle workflows.

### Architectural Pattern

```
Audio File (or URL) → HuggingFace Pipeline (AutomaticSpeechRecognitionPipeline)
                              ↓
                    Chunking (30s default) → Batched GPU Inference → Segment Assembly
                              ↓                                          ↓
                    [Optional] pyannote Diarization ──────────→ Speaker-Labeled Output
                              ↓
                         JSON Output File
```

Single-pipeline architecture using HuggingFace's `pipeline()` abstraction. The heavy lifting is delegated to Transformers internals with PyTorch as the compute backend. Diarization is an optional parallel pipeline whose results are merged post-hoc.

### Performance Characteristics

- **GPU throughput**: 150 minutes of audio transcribed in ~98 seconds on A100 (with Flash Attention 2 + batching). This is approximately 90x real-time.
- **CPU throughput**: Poor. Designed for GPU-first execution. CPU fallback is functional but extremely slow compared to whisper.cpp.
- **Startup latency**: 5-15 seconds for model loading (PyTorch + model weights download on first run).
- **Memory (GPU VRAM)**: 4-10GB depending on model size, batch size, and precision. Large-v3 with batch_size=24 needs ~8GB VRAM.
- **Memory (system RAM)**: 2-4GB for Python runtime + PyTorch + model weight staging.

### Known Limitations

- **Python GIL limits concurrency**: The Global Interpreter Lock prevents true multi-threaded execution. Parallelism comes from GPU kernel dispatch, not CPU threading.
- **Large dependency tree**: Requires PyTorch (~2GB), Transformers, accelerate, optimum, flash-attn, and optionally pyannote-audio. Total environment can exceed 5GB.
- **GPU-first design**: CPU performance is an afterthought. Without a capable GPU, insanely-fast-whisper is slower than whisper.cpp by a large margin.
- **No streaming support**: Batch-oriented design. Audio must be fully available before transcription begins. Not suitable for live/real-time use cases.
- **Memory hungry**: Large batch sizes and Flash Attention still require significant VRAM. Consumer GPUs (4-6GB) may need reduced batch sizes.
- **No source separation**: Cannot isolate speech from background noise or music.
- **No forced alignment**: Timestamps are from the Whisper decoder directly, no post-hoc correction.
- **No punctuation restoration**: Relies on the Whisper model's built-in punctuation predictions.
- **HuggingFace token requirement for diarization**: pyannote models require license acceptance and authentication, adding friction to automated deployments.

---

## 3. legacy_whisper_diarization (Python)

### Summary

whisper-diarization is a Python pipeline that chains multiple specialized models together to produce speaker-attributed, timestamp-corrected, punctuation-restored transcriptions. It is the reference implementation for high-fidelity multi-speaker transcription with post-processing.

### Key Strengths

#### Multi-Stage Pipeline Architecture
The pipeline chains six distinct processing stages, each contributing a specialized capability:

1. **Demucs Source Separation** (vocal isolation from music/noise)
2. **Faster-Whisper ASR** (speech-to-text transcription)
3. **CTC Forced Alignment** (timestamp correction via ctc-forced-aligner)
4. **NeMo TitaNet Speaker Embedding Extraction** (speaker vector computation)
5. **Speaker Clustering and Matching** (assigning speaker identities to segments)
6. **Multilingual Punctuation Restoration** (post-hoc punctuation insertion)

Each stage is independently valuable and could be used or replaced in isolation.

#### Demucs Source Separation (Vocal Isolation)
- Facebook/Meta's Demucs model separates audio into stems: vocals, drums, bass, other.
- The vocals stem is extracted and fed to the ASR stage, dramatically improving transcription accuracy in noisy environments.
- Particularly effective for: podcasts with background music, conference recordings with ambient noise, media content with soundtracks.
- Toggle: `--no-stem` to skip separation when audio is already clean.
- No other legacy project offers source separation capability.

#### CTC Forced Alignment for Precise Timestamps
- Uses `ctc-forced-aligner` to realign word-level timestamps against the audio signal.
- The Whisper decoder's attention-based timestamps can drift, especially on long audio or with repeated phrases. CTC alignment corrects this drift.
- Produces word-level timestamps that are more accurate than the Whisper decoder's native output.
- Enables precise speaker boundary detection by providing exact word timing.
- `--suppress-numerals` option replaces number words with spellings to improve alignment stability.

#### NeMo TitaNet Speaker Embeddings with Clustering
- NVIDIA NeMo's TitaNet model extracts speaker embedding vectors from audio segments.
- Embeddings are clustered (typically via spectral clustering or agglomerative clustering) to identify distinct speakers.
- Produces per-word speaker attribution, not just per-segment. This is significantly more granular than pyannote's segment-level diarization.
- Speaker count is determined automatically by the clustering algorithm, or can be constrained.
- MarbleNet VAD (from NeMo) is used to identify speech regions for embedding extraction.

#### Multilingual Punctuation Restoration
- Post-processing stage that inserts punctuation (periods, commas, question marks) into the raw transcript.
- Supports 12+ languages including English, German, French, Spanish, Italian, and others.
- Uses a dedicated punctuation restoration model (`deepmultilingualpunctuation`).
- Significantly improves readability of the final transcript, especially for languages where Whisper's built-in punctuation is weak.

#### Comprehensive Output Formats
- **SRT**: SubRip subtitle format with speaker labels (e.g., `[SPEAKER 01]: Hello, how are you?`).
- **TXT**: Plain text with speaker labels and timestamps.
- **Diarized transcript**: Full text with speaker attribution per utterance.
- SRT parsing is hardened to handle both colon-separated and dot-separated timestamp formats.
- Speaker label recognition supports multiple patterns: `SPEAKER`, `SPK`, `SPKR`, `S0`, etc.

#### Parallel Processing Mode
- `diarize_parallel.py` variant for systems with >10GB VRAM.
- Runs ASR and diarization stages concurrently, reducing wall-clock time.
- Memory-throughput tradeoff: uses more VRAM but completes faster.

### Architectural Pattern

```
Audio File
    ↓
[Optional] Demucs Source Separation → Isolated Vocals
    ↓
Faster-Whisper ASR → Raw Transcript + Coarse Timestamps
    ↓
CTC Forced Alignment (ctc-forced-aligner) → Corrected Word-Level Timestamps
    ↓
MarbleNet VAD → Speech Regions
    ↓
NeMo TitaNet → Speaker Embeddings per Region
    ↓
Spectral Clustering → Speaker Labels per Word
    ↓
Punctuation Restoration → Final Punctuated Transcript
    ↓
Output Formatter → SRT / TXT / Diarized Transcript
```

Multi-stage pipeline architecture. Each stage transforms the data and passes enriched results to the next. Stages are sequential (except in parallel mode) and each one depends on the previous stage's output. This architecture is the most complex of the three legacy projects but produces the highest-fidelity output.

### Performance Characteristics

- **Throughput**: Slower than both other projects due to multiple model passes. Typical: 0.3-0.5x real-time on a modern GPU (i.e., a 10-minute audio file takes 20-30 minutes).
- **Startup latency**: 30-60 seconds. Multiple models must be downloaded on first run and loaded into memory (Demucs, Faster-Whisper, CTC aligner, NeMo TitaNet, punctuation model).
- **Memory (GPU VRAM)**: 6-12GB depending on which stages are active. Demucs alone needs ~2GB; NeMo TitaNet needs ~2GB; Faster-Whisper needs ~2-6GB depending on model size.
- **Memory (system RAM)**: 4-8GB for Python runtime + multiple model weights.
- **Disk**: Multiple model downloads totaling 5-10GB on first run.

### Known Limitations

- **Heavy pipeline**: Requires downloading and loading 4-5 separate models. First run involves gigabytes of downloads.
- **Slow startup**: 30-60 seconds to load all models even from cache. Not suitable for interactive or real-time use.
- **Python only**: No native Rust integration path. Would need subprocess bridging or full reimplementation.
- **Complex dependency chain**: Demucs (Facebook Research), NeMo (NVIDIA), CTC aligner, Faster-Whisper, and the punctuation model each have their own dependency trees. Version conflicts are common.
- **Limited error handling**: Pipeline failures in intermediate stages can produce partial/corrupt output without clear error reporting.
- **No streaming**: All audio must be fully processed through each stage sequentially.
- **No quantized models**: Uses full-precision models only. No option for reduced-precision inference.
- **ffmpeg dependency**: Requires external ffmpeg for input format normalization.
- **No real-time VAD**: MarbleNet VAD is used for diarization region extraction, not for real-time speech detection.

---

## Cross-Project Comparison Matrix

| Feature | whisper.cpp | insanely-fast-whisper | whisper-diarization |
|---|:---:|:---:|:---:|
| **CPU inference** | Excellent | Poor | Moderate |
| **GPU inference** | Good | Excellent | Good |
| **Streaming** | Yes | No | No |
| **VAD** | Silero | No | MarbleNet (for diarization) |
| **Diarization** | Basic (TinyDiarize) | pyannote (segment-level) | NeMo TitaNet (word-level) |
| **Source separation** | No | No | Demucs |
| **Forced alignment** | No | No | CTC |
| **Punctuation restoration** | No | No | Yes (multilingual) |
| **Word timestamps** | Yes (decoder) | Yes (decoder) | Yes (CTC-corrected) |
| **Quantized models** | Yes (Q4/Q5/Q8) | No | No |
| **Memory efficiency** | Excellent | Poor | Moderate |
| **Install complexity** | High (CMake + platform deps) | Low (pip install) | High (many model deps) |
| **Startup latency** | Sub-second | 5-15 seconds | 30-60 seconds |
| **Language** | C/C++ | Python | Python |
| **Model backend** | ggml (native) | PyTorch/Transformers | Faster-Whisper + NeMo |
| **Output formats** | txt/vtt/srt/csv/json/lrc | JSON | SRT/TXT |
| **CLI surface** | 60+ flags | ~15 flags | ~10 flags |
| **Batch processing** | Sequential | GPU-batched | Sequential |
| **Flash Attention** | No | Yes (v2) | No |
| **Speaker count control** | No | Yes (num/min/max) | Auto (clustering) |
| **HTTP server** | Yes | No | No |
| **Confidence scores** | Yes (color-coded) | Yes | No |
| **Hardware backends** | 10+ (CUDA/Metal/Vulkan/...) | CUDA/MPS/CPU | CUDA/CPU |

---

## Synthesis Notes for franken_whisper

### What franken_whisper takes from each project

**From whisper.cpp (concepts, not code):**
- Local-first execution model with minimal startup latency.
- Streaming/real-time transcription as a first-class capability.
- VAD as a pre-filter to reduce unnecessary computation.
- Quantized/reduced-precision model support for memory-constrained environments.
- Rich CLI surface with fine-grained parameter control.
- Multi-platform hardware acceleration abstraction.
- Low memory footprint design principles.

**From insanely-fast-whisper (concepts, not code):**
- GPU batching strategy for throughput-critical workloads.
- Flash Attention integration for memory-efficient long-form processing.
- Ergonomic CLI defaults that "just work" for common cases.
- Automatic device selection (GPU detection and fallback).
- Speaker diarization integration with configurable speaker count constraints.

**From whisper-diarization (concepts, not code):**
- Multi-stage pipeline architecture where each stage is independently testable.
- Source separation as an optional pre-processing stage.
- CTC forced alignment for timestamp correction (superior to decoder-only timestamps).
- Speaker embedding extraction and clustering for true multi-speaker attribution.
- Multilingual punctuation restoration as a post-processing stage.
- Word-level speaker attribution (not just segment-level).

### Gaps that franken_whisper must fill (present in no legacy project)

- **Unified architecture**: One system that handles CPU-optimized, GPU-batched, and pipeline modes under a single API.
- **Agent-first output**: Stable machine-parseable NDJSON progress and result envelopes.
- **Durable local storage**: Run history, segments, and events persisted in frankensqlite.
- **Cancel-correct orchestration**: Deterministic cancellation semantics via asupersync.
- **Conformance verification**: Cross-engine compatibility testing with tolerance envelopes.
- **Replay determinism**: Persisted input/output hashes for drift detection across engine versions.
- **Memory safety**: Rust's ownership model eliminates the C/C++ memory safety concerns of whisper.cpp.
- **Unified dependency model**: Cargo-managed dependencies instead of CMake + pip + system libraries.

### Priority hierarchy for feature synthesis

1. **Essential (must have for v1)**: File transcription, word timestamps, multiple output formats, CLI with robot mode, durable storage, quality gates.
2. **High (needed for production use)**: Streaming, VAD pre-filter, GPU acceleration, diarization, source separation toggle.
3. **Medium (competitive advantage)**: Forced alignment, punctuation restoration, quantized model support, Flash Attention.
4. **Lower (beyond parity)**: HTTP server, TUI interface, adaptive backend routing, conformance harness.

---

*Document created as part of bead bd-1bk.1. This is the definitive legacy analysis reference for the franken_whisper project.*
