# franken_whisper

<div align="center">
  <img src="franken_whisper_illustration.webp" alt="franken_whisper - Agent-first Rust ASR orchestration stack">
</div>

<div align="center">

[![License: MIT+Rider](https://img.shields.io/badge/License-MIT%2BOpenAI%2FAnthropic%20Rider-blue.svg)](./LICENSE)
[![Rust Edition](https://img.shields.io/badge/Rust-2024_Edition-orange.svg)](https://doc.rust-lang.org/edition-guide/rust-2024/)
[![unsafe forbidden](https://img.shields.io/badge/unsafe-forbidden-success.svg)](https://github.com/rust-secure-code/safety-dance/)

</div>

**Agent-first Rust ASR orchestration stack with adaptive backend routing, real-time NDJSON streaming, and SQLite-backed persistence.**

---

## The Problem

Speech-to-text pipelines are fragmented. You need whisper.cpp for speed, insanely-fast-whisper for GPU batching, and whisper-diarization for speaker identification. Each has its own CLI, output format, error handling, and deployment story. Orchestrating them from scripts means parsing inconsistent stdout, handling timeouts manually, and losing run history.

Agent workflows need structured, streaming, machine-readable output, not human-oriented terminal decorations that break when piped.

## The Solution

franken_whisper is a single Rust binary that wraps all three backends behind a unified interface with:

- **Adaptive backend routing:** Bayesian decision contract selects the best engine per-request with explicit loss matrix, posterior calibration, and deterministic fallback
- **Real-time NDJSON streaming:** every pipeline stage emits sequenced, timestamped events with stable schema (v1.0.0) for agent consumption
- **Durable run history:** every transcription is persisted to SQLite with full event logs, replay envelopes, and JSONL export/import
- **Graceful cancellation:** Ctrl+C propagates through the entire pipeline via cancellation tokens with proper resource cleanup
- **TTY audio transport:** low-bandwidth audio relay over PTY links using mulaw+zlib+base64 NDJSON with handshake, integrity checks, and deterministic retransmission

## Why franken_whisper?

| Feature | whisper.cpp | insanely-fast-whisper | whisper-diarization | **franken_whisper** |
|---------|:-----------:|:---------------------:|:-------------------:|:-------------------:|
| Streaming output | partial | no | no | **NDJSON stage events** |
| Machine-readable errors | no | no | no | **structured error codes** |
| Adaptive backend selection | -- | -- | -- | **Bayesian routing** |
| Run persistence | no | no | no | **SQLite + JSONL** |
| Diarization | no | yes (HF token) | yes | **yes (any backend)** |
| GPU acceleration | CUDA/Metal | CUDA/MPS | CUDA | **frankentorch/frankenjax** |
| Cancellation support | SIGKILL | SIGKILL | SIGKILL | **graceful token-based** |
| TTY audio relay | no | no | no | **mulaw+zlib+b64 NDJSON** |
| Memory safety | C++ | Python | Python | **`#![forbid(unsafe_code)]`** |

---

## Quick Example

```bash
# Transcribe any audio file — MP3/FLAC/OGG/AAC decoded natively, no ffmpeg needed
cargo run -- transcribe --input meeting.mp3 --json

# Transcribe a video file — audio extracted automatically
cargo run -- transcribe --input presentation.mp4 --json

# Stream real-time pipeline events (agent mode)
cargo run -- robot run --input meeting.mp3 --backend auto

# Speculative streaming: fast partial results with quality corrections
cargo run -- robot run --input meeting.mp3 --speculative \
  --fast-model tiny.en --quality-model large-v3

# Transcribe with speaker diarization
cargo run -- transcribe --input meeting.mp3 --diarize --hf-token "$HF_TOKEN" --json

# TinyDiarize: whisper.cpp's built-in speaker-turn detection (no HF token needed)
cargo run -- transcribe --input meeting.mp3 --tiny-diarize --json

# Discover available backends and their capabilities
cargo run -- robot backends

# System health check (backends, ffmpeg, database, resources)
cargo run -- robot health

# Query run history
cargo run -- runs --limit 10 --format json

# Export runs to portable JSONL snapshot (full or incremental)
cargo run -- sync export-jsonl --output ./snapshot

# TTY audio: encode, transmit over lossy link, decode
cargo run -- tty-audio encode --input audio.wav > frames.ndjson
cat frames.ndjson | cargo run -- tty-audio decode --output restored.wav
```

---

## Design Philosophy

### Agent-First, Human-Optional

Every command produces structured NDJSON on stdout. Human-friendly output is the exception (plain `transcribe` mode), not the rule. The `robot` subcommand is the primary interface. It emits sequenced stage events with stable schema versioning so upstream agents can parse output without fragile regex.

### Deterministic by Default

Given identical inputs and parameters, franken_whisper produces identical outputs. The retransmit loop, replay envelopes, and conformance harness all enforce determinism. Random elements (UUIDs, timestamps) are isolated to metadata fields, never to computational outputs.

### Fail Loud, Recover Gracefully

Every error has a structured code (`FW-IO`, `FW-CMD-TIMEOUT`, `FW-BACKEND-UNAVAILABLE`, etc.) and propagates through the NDJSON event stream. Cancellation tokens allow in-flight work to checkpoint and clean up rather than being killed mid-write.

### Composition Over Configuration

The 10-stage pipeline (Ingest, Normalize, VAD, Separate, Backend, Accelerate, Align, Punctuate, Diarize, Persist) is composed dynamically per-request. Stages are skipped when unnecessary, budgeted independently, and profiled automatically.

### No Unsafe Code

The entire codebase uses `#![forbid(unsafe_code)]`. Memory safety is enforced at the compiler level, not by convention.

### Zero External Dependencies for Common Audio

franken_whisper can transcribe MP3, AAC, FLAC, WAV, OGG, and other common audio files without ffmpeg, Python, or any other runtime dependency beyond the backend engine itself. The built-in Rust audio decoder (symphonia) handles format detection, codec decoding, sample rate conversion, and channel mixing entirely in-process. ffmpeg is only invoked as a fallback for video files and exotic codecs — and even then, it's auto-provisioned if missing.

This matters because most "simple" transcription tools actually have deep dependency chains: ffmpeg for normalization, Python for wrappers, pip for packages, conda for environments. franken_whisper's audio path is a single statically-linked Rust binary.

---

## The Whisper Ecosystem Landscape

The whisper ecosystem has dozens of tools. Here's how they relate to each other and where franken_whisper fits:

```
                     ┌──────────────────────────────────────────────────────┐
                     │           INFERENCE ENGINES (run models)            │
                     │                                                      │
                     │  whisper.cpp (C++, CPU/Metal/CUDA, ~47k★)           │
                     │  faster-whisper (Python/CTranslate2, ~14k★)         │
                     │  OpenAI Whisper (Python/PyTorch, ~95k★)             │
                     └────────────────────────┬─────────────────────────────┘
                                              │
                     ┌────────────────────────▼─────────────────────────────┐
                     │      ENHANCED PIPELINES (add features on top)        │
                     │                                                      │
                     │  WhisperX (faster-whisper + wav2vec2 + pyannote)     │
                     │  whisper-diarization (Whisper + Demucs + TitaNet)    │
                     │  insanely-fast-whisper (HF Transformers, max GPU)    │
                     │  whisper-timestamped (DTW word timestamps)           │
                     └────────────────────────┬─────────────────────────────┘
                                              │
                     ┌────────────────────────▼─────────────────────────────┐
                     │   ORCHESTRATION (manage multiple engines/pipelines)   │
                     │                                                      │
                     │  ▸ franken_whisper ◂ (Rust, Bayesian routing,        │
                     │     10-stage pipeline, speculative streaming,         │
                     │     conformance validation, evidence-based decisions) │
                     └──────────────────────────────────────────────────────┘
```

Most tools in the ecosystem occupy one level. franken_whisper occupies the orchestration level: it wraps inference engines and enhanced pipelines behind a unified interface, then adds capabilities that none of them provide individually.

## Comparison vs Alternatives

### Orchestration & Architecture

| Capability | whisper.cpp | faster-whisper | WhisperX | WhisperLive | WhisperS2T | **franken_whisper** |
|------------|:-----------:|:--------------:|:--------:|:-----------:|:----------:|:-------------------:|
| Language | C++ | Python | Python | Python | Python | **Rust** |
| Multi-backend | -- | -- | -- | 3 backends | 4 backends | **3 backends + 3 native pilots** |
| Backend selection | -- | -- | -- | manual | manual | **Bayesian adaptive routing** |
| Pipeline stages | monolithic | monolithic | 3-stage | monolithic | monolithic | **10 composable stages** |
| Per-stage budgets | -- | -- | -- | -- | -- | **independent timeouts** |
| Speculative streaming | -- | -- | -- | single-model | -- | **dual-model fast+quality** |
| Conformance validation | -- | -- | -- | -- | -- | **cross-engine 50ms tolerance** |
| Native rollout governance | -- | -- | -- | -- | -- | **5-stage shadow→sole** |
| Memory safety | C++ | Python GC | Python GC | Python GC | Python GC | **`#![forbid(unsafe_code)]`** |

### Persistence & Observability

| Capability | whisper.cpp | faster-whisper | WhisperX | WhisperLive | **franken_whisper** |
|------------|:-----------:|:--------------:|:--------:|:-----------:|:-------------------:|
| Run history | none | none | none | none | **SQLite + JSONL export** |
| Decision audit trail | -- | -- | -- | -- | **200-entry evidence ledger** |
| Replay envelopes | -- | -- | -- | -- | **SHA-256 content hashing** |
| Replay packs | -- | -- | -- | -- | **4-artifact reproducibility bundle** |
| Structured errors | exit code | exceptions | exceptions | -- | **12 `FW-*` error codes** |
| NDJSON streaming | partial | -- | -- | WebSocket | **sequenced stage events** |
| Cancellation | SIGKILL | KeyboardInterrupt | -- | -- | **cooperative `CancellationToken`** |
| Resource cleanup | none guaranteed | GC | GC | GC | **RAII + bounded finalizers** |
| Latency profiling | -- | -- | -- | -- | **per-stage with tuning recs** |

### Audio & Format Support

| Capability | whisper.cpp | faster-whisper | WhisperX | **franken_whisper** |
|------------|:-----------:|:--------------:|:--------:|:-------------------:|
| Native audio decode | WAV only | -- (needs ffmpeg) | -- (needs ffmpeg) | **MP3/AAC/FLAC/WAV/OGG/ALAC (symphonia)** |
| ffmpeg required? | for non-WAV | yes | yes | **no (fallback only)** |
| Video audio extraction | -- | -- | -- | **automatic (`-vn` flag)** |
| TTY audio transport | -- | -- | -- | **mulaw+zlib+b64 NDJSON** |
| Microphone capture | -- | -- | -- | **platform-specific ffmpeg** |
| Auto-provision ffmpeg | -- | -- | -- | **downloads static binary if missing** |

### Commercial API Comparison

| Capability | Groq Whisper API | Deepgram Nova-3 | AssemblyAI | **franken_whisper** |
|------------|:----------------:|:---------------:|:----------:|:-------------------:|
| Runs locally | no | no | no | **yes** |
| Open source | no | no | no | **yes (MIT)** |
| Data leaves machine | yes | yes | yes | **never** |
| Cost per hour of audio | ~$0.04 | ~$0.75 | ~$0.65 | **$0 (your hardware)** |
| Inference speed | very fast | fast | moderate | **depends on backend** |
| Multi-model routing | -- | -- | -- | **Bayesian adaptive** |
| Diarization | limited | yes | yes | **yes (any backend)** |
| Custom pipeline stages | -- | -- | -- | **10 composable stages** |

---

## Installation

### From Source (Recommended)

```bash
git clone <repo-url>
cd franken_whisper

# minimal build
cargo build --release

# with TUI support
cargo build --release --features tui

# with GPU acceleration
cargo build --release --features gpu-frankentorch
cargo build --release --features gpu-frankenjax
```

The release profile is optimized for size (`opt-level = "z"`, LTO, single codegen unit, stripped symbols).

### Prerequisites

- **Rust nightly** (2024 edition)
- **ffmpeg** (optional), only needed for video files, exotic audio codecs symphonia can't decode, and live microphone capture; franken_whisper's built-in Rust audio decoder handles MP3, AAC, FLAC, WAV, OGG, and other common formats natively with zero external dependencies — ffmpeg is tried automatically as a fallback when the built-in decoder can't handle a format
- **Backend binaries** (at least one):
  - `whisper-cli` (from whisper.cpp); override: `FRANKEN_WHISPER_WHISPER_CPP_BIN`
  - `insanely-fast-whisper` (Python); override: `FRANKEN_WHISPER_INSANELY_FAST_BIN`
  - `python3` with `pyannote.audio` (for diarization backend); override: `FRANKEN_WHISPER_PYTHON_BIN`
- **HuggingFace token** (for diarization): `--hf-token` or `FRANKEN_WHISPER_HF_TOKEN` / `HF_TOKEN`

### Path Dependencies

franken_whisper depends on sibling projects via Cargo path dependencies:

```
../asupersync          # orchestration, cancellation, decision contracts
../frankensqlite       # SQLite persistence (fsqlite crate)
../frankentui          # TUI (optional, feature: tui)
../frankentorch        # GPU acceleration (optional, feature: gpu-frankentorch)
../frankenjax          # GPU acceleration (optional, feature: gpu-frankenjax)
```

---

## Quick Start

### 1. Basic Transcription

```bash
# plain text output
cargo run -- transcribe --input audio.mp3

# full JSON report (includes segments, timing, backend info)
cargo run -- transcribe --input audio.mp3 --json

# specific backend
cargo run -- transcribe --input audio.mp3 --backend whisper_cpp --json

# with language hint
cargo run -- transcribe --input audio.mp3 --language ja --json
```

### 2. Robot Mode (Agent Integration)

```bash
# real-time NDJSON event stream
cargo run -- robot run --input audio.mp3 --backend auto
```

Output (one JSON object per line):
```json
{"event":"run_start","schema_version":"1.0.0","request":{"input":"audio.mp3","backend":"auto"}}
{"event":"stage","schema_version":"1.0.0","run_id":"...","seq":1,"stage":"ingest","code":"ingest.start","message":"materializing input"}
{"event":"stage","schema_version":"1.0.0","run_id":"...","seq":2,"stage":"normalize","code":"normalize.ok","message":"audio normalized"}
{"event":"run_complete","schema_version":"1.0.0","run_id":"...","backend":"whisper_cpp","transcript":"Hello world..."}
```

### 3. Speaker Diarization

```bash
cargo run -- transcribe \
  --input meeting.mp3 \
  --diarize \
  --hf-token "$HF_TOKEN" \
  --min-speakers 2 \
  --max-speakers 5 \
  --json
```

### 4. Microphone Capture

```bash
# record 30 seconds from default mic
cargo run -- transcribe --mic --mic-seconds 30 --json

# specific device
cargo run -- transcribe --mic --mic-device "hw:0" --json
```

### 5. Stdin Input

```bash
# pipe audio bytes
cat audio.mp3 | cargo run -- transcribe --stdin --json
```

---

## Command Reference

### `transcribe`

Core transcription command. Runs the full pipeline: ingest, normalize, backend execution, optional acceleration, and persistence.

```bash
cargo run -- transcribe [OPTIONS]
```

**Input (mutually exclusive):**

| Flag | Description |
|------|-------------|
| `--input <PATH>` | Audio/video file path |
| `--stdin` | Read audio bytes from stdin |
| `--mic` | Capture from microphone via ffmpeg |

**Backend & Model:**

| Flag | Default | Description |
|------|---------|-------------|
| `--backend <KIND>` | `auto` | `auto`, `whisper_cpp`, `insanely_fast`, `whisper_diarization` |
| `--model <MODEL>` | backend-specific | Model name/path forwarded to backend |
| `--language <LANG>` | auto-detect | Language hint (ISO 639-1) |
| `--translate` | false | Translate to English |
| `--diarize` | false | Enable speaker diarization |

**Output:**

| Flag | Description |
|------|-------------|
| `--json` | Full JSON run report |
| `--output-txt` | Plain text (whisper.cpp) |
| `--output-vtt` | WebVTT subtitles |
| `--output-srt` | SRT subtitles |
| `--output-csv` | CSV |
| `--output-json-full` | Extended JSON with metadata |
| `--output-lrc` | LRC karaoke format |

**Storage:**

| Flag | Default | Description |
|------|---------|-------------|
| `--db <PATH>` | `.franken_whisper/storage.sqlite3` | SQLite database path |
| `--no-persist` | false | Skip persistence |

**Inference Tuning (whisper.cpp):**

| Flag | Default | Description |
|------|---------|-------------|
| `--threads <N>` | 4 | Computation threads |
| `--processors <N>` | 1 | Parallel processors |
| `--no-gpu` | false | Force CPU-only |
| `--beam-size <N>` | 5 | Beam search width |
| `--best-of <N>` | 5 | Sampling candidates |
| `--temperature <F>` | 0.0 | Sampling temperature |
| `--temperature-increment <F>` | -- | Temperature fallback increment |
| `--entropy-threshold <F>` | -- | Entropy threshold for fallback |
| `--logprob-threshold <F>` | -- | Log probability threshold |
| `--no-speech-threshold <F>` | -- | No-speech probability threshold |
| `--max-context <N>` | -- | Maximum context tokens from prior segment |
| `--max-segment-length <N>` | -- | Maximum segment length in characters |
| `--no-timestamps` | false | Suppress timestamps |
| `--detect-language-only` | false | Detect language and exit (no transcription) |
| `--split-on-word` | false | Split segments on word boundaries |
| `--no-fallback` | false | Disable temperature fallback |
| `--suppress-nst` | false | Suppress non-speech tokens |
| `--tiny-diarize` | false | Enable TinyDiarize (speaker-turn token injection) |
| `--prompt <TEXT>` | -- | Initial prompt to guide transcription style |
| `--carry-initial-prompt` | false | Carry prompt across segments |

**Audio Windowing (whisper.cpp):**

| Flag | Default | Description |
|------|---------|-------------|
| `--offset-ms <N>` | 0 | Start transcription at offset (ms) |
| `--duration-ms <N>` | -- | Transcribe only this duration (ms) |
| `--audio-ctx <N>` | -- | Audio context size (tokens) |
| `--word-threshold <F>` | -- | Word-level timestamp confidence threshold |
| `--suppress-regex <REGEX>` | -- | Suppress tokens matching regex |

**VAD (Voice Activity Detection):**

| Flag | Default | Description |
|------|---------|-------------|
| `--vad` | false | Enable Voice Activity Detection |
| `--vad-model <PATH>` | -- | Custom VAD model path |
| `--vad-threshold <F>` | -- | Speech detection threshold |
| `--vad-min-speech-ms <N>` | -- | Minimum speech duration (ms) |
| `--vad-min-silence-ms <N>` | -- | Minimum silence duration (ms) |
| `--vad-max-speech-s <F>` | -- | Maximum speech duration (seconds) |
| `--vad-speech-pad-ms <N>` | -- | Speech padding (ms) |
| `--vad-samples-overlap <N>` | -- | Sample overlap between windows |

**Batching (insanely-fast-whisper):**

| Flag | Default | Description |
|------|---------|-------------|
| `--batch-size <N>` | 24 | Parallel inference batch size |
| `--gpu-device <DEV>` | auto | GPU device (`0`, `cuda:0`, `mps`) |
| `--flash-attention` | false | Enable Flash Attention 2 |
| `--hf-token <TOKEN>` | env | HuggingFace token for diarization |
| `--timestamp-level` | `chunk` | `chunk` or `word` granularity |
| `--transcript-path <PATH>` | -- | Override transcript output path |

**Diarization:**

| Flag | Description |
|------|-------------|
| `--num-speakers <N>` | Exact speaker count |
| `--min-speakers <N>` | Minimum speakers |
| `--max-speakers <N>` | Maximum speakers |
| `--no-stem` | Disable vocal isolation (Demucs source separation) |
| `--suppress-numerals` | Spell out numbers for alignment stability |
| `--diarization-model <MODEL>` | Override whisper model for diarization stage |

**Speculative Streaming (Phase 5):**

| Flag | Default | Description |
|------|---------|-------------|
| `--speculative` | false | Enable dual-model speculative cancel-correct mode |
| `--fast-model <MODEL>` | -- | Fast model for low-latency partial transcripts |
| `--quality-model <MODEL>` | -- | Quality model for correction/verification |
| `--speculative-window-ms <N>` | 3000 | Sliding window size (ms) |
| `--speculative-overlap-ms <N>` | 500 | Window overlap (ms) |
| `--correction-tolerance-wer <F>` | -- | WER tolerance for confirmation vs. retraction |
| `--no-adaptive` | false | Disable adaptive window sizing |
| `--always-correct` | false | Force quality model on every window (evaluation mode) |

### `robot`

Agent-first interface with structured NDJSON output.

```bash
# streaming transcription with stage events
cargo run -- robot run [TRANSCRIBE_OPTIONS]

# emit JSON schema for all event types
cargo run -- robot schema

# discover backends and capabilities
cargo run -- robot backends

# system health diagnostics (backends, ffmpeg, database, resources)
cargo run -- robot health

# query routing decision history
cargo run -- robot routing-history [--run-id <ID>] [--limit 20]
```

**Robot Event Types:**

| Event | Description |
|-------|-------------|
| `run_start` | Request accepted, pipeline starting |
| `stage` | Pipeline stage progress (sequenced, timestamped) |
| `run_complete` | Transcription finished with full result |
| `run_error` | Pipeline failed with structured error code |
| `backends` | Backend discovery response with per-backend capabilities |
| `health.report` | System health diagnostics (backend/ffmpeg/DB/resource status) |
| `routing.history` | Decision routing history entries with posterior snapshots |
| `transcript.partial` | Speculative fast-model partial transcript (immediate) |
| `transcript.confirm` | Quality model confirms partial (drift within tolerance) |
| `transcript.retract` | Quality model retracts partial (drift exceeds tolerance) |
| `transcript.correct` | Quality model correction with corrected segments |
| `speculation.stats` | Aggregate speculation pipeline statistics |

**Stage Codes:**

Stages emit paired `*.start` / `*.ok` codes (or `*.error` on failure, `*.skip` when not needed):

`ingest.start`, `ingest.ok`, `normalize.start`, `normalize.ok`, `vad.start`, `vad.ok`, `separate.start`, `separate.ok`, `backend.start`, `backend.ok`, `backend.routing.decision_contract`, `accelerate.start`, `accelerate.ok`, `align.start`, `align.ok`, `punctuate.start`, `punctuate.ok`, `diarize.start`, `diarize.ok`, `persist.start`, `persist.ok`, `orchestration.latency_profile`

**Health Report:** The `robot health` command probes all subsystems and returns a structured diagnostic:

```json
{
  "event": "health.report",
  "backends": [{"name": "whisper.cpp", "available": true, "version": "1.7.2"}],
  "ffmpeg": {"available": true, "path": "/usr/bin/ffmpeg"},
  "database": {"path": ".franken_whisper/storage.sqlite3", "size_bytes": 12345},
  "overall_status": "ok"
}
```

### `runs`

Query persisted run history.

```bash
cargo run -- runs [--limit 20] [--format plain|json|ndjson] [--id <RUN_ID>]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--limit <N>` | 20 | Max recent runs |
| `--format` | `plain` | `plain` (table), `json` (pretty), `ndjson` (streaming) |
| `--id <UUID>` | -- | Fetch specific run details |

### `sync`

One-way JSONL snapshot export/import.

```bash
# export
cargo run -- sync export-jsonl --output ./snapshot [--db <PATH>]

# import
cargo run -- sync import-jsonl --input ./snapshot --conflict-policy reject|skip|overwrite|overwrite-strict
```

Export produces: `runs.jsonl`, `segments.jsonl`, `events.jsonl`, `manifest.json` (with SHA-256 checksums).

### `tty-audio`

Low-bandwidth audio transport over TTY/PTY links using the `mulaw+zlib+b64` NDJSON protocol.

```bash
# encode audio to NDJSON frames
cargo run -- tty-audio encode --input audio.wav [--chunk-ms 200]

# decode NDJSON frames to WAV
cat frames.ndjson | cargo run -- tty-audio decode --output restored.wav [--recovery fail_closed|skip_missing]

# generate retransmit plan from lossy stream
cat frames.ndjson | cargo run -- tty-audio retransmit-plan

# emit individual control frames
cargo run -- tty-audio control handshake
cargo run -- tty-audio control ack --up-to-seq 42
cargo run -- tty-audio control backpressure --remaining-capacity 64
cargo run -- tty-audio control retransmit-request --sequences 1,2,4
cargo run -- tty-audio control retransmit-response --sequences 1,2,4

# automated retransmit loop with strategy escalation
cat frames.ndjson | cargo run -- tty-audio control retransmit-loop --rounds 3

# convenience shorthands
cargo run -- tty-audio send-control handshake|eof|reset
cat frames.ndjson | cargo run -- tty-audio retransmit --rounds 3
```

**Recovery Strategies:**

The retransmit loop escalates recovery effort across rounds:

```
Simple (1 frame/round) -> Redundant (2 frames/round) -> Escalate (4 frames/round)
```

**Integrity Checks:**

Each frame carries optional CRC32 and SHA-256 hashes of raw (pre-compression) audio bytes. Mismatches cause frame drops (skip_missing) or stream failure (fail_closed).

See [`docs/tty-audio-protocol.md`](docs/tty-audio-protocol.md) for the full protocol specification.
For operator replay/framing guarantees, see
[`docs/tty-replay-guarantees.md`](docs/tty-replay-guarantees.md).

### `tui`

Interactive TUI for human operators (feature-gated, requires `--features tui`).

```bash
cargo run --features tui -- tui
```

**Features:**

- **Live transcription view:** Real-time segment display with auto-scroll behavior — scrolling up pauses auto-scroll, scrolling to bottom re-enables it
- **Speaker labels and timestamps:** Each segment displays start/end times, speaker identification, and confidence scores
- **Runs list:** Browse persisted run history with timing and backend info
- **Timeline view:** Visual timeline of pipeline stages with duration bars
- **Event detail panes:** Inspect individual NDJSON events with full payload
- **Segment retention:** Caps display at 10,000 segments with oldest-first drain to prevent memory issues on very long sessions
- **Keyboard navigation:** Focus cycling between panes, vim-style keybindings

Built on the [FrankenTUI](https://github.com/Dicklesworthstone/frankentui) framework. Note: the `tui` feature gate adds ~2.5 minutes to compilation due to the FrankenTUI dependency tree.

---

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `FRANKEN_WHISPER_WHISPER_CPP_BIN` | `whisper-cli` | whisper.cpp binary name/path |
| `FRANKEN_WHISPER_INSANELY_FAST_BIN` | `insanely-fast-whisper` | insanely-fast-whisper binary |
| `FRANKEN_WHISPER_PYTHON_BIN` | `python3` | Python interpreter for diarization |
| `FRANKEN_WHISPER_HF_TOKEN` | -- | HuggingFace token (preferred over `HF_TOKEN`) |
| `HF_TOKEN` | -- | HuggingFace token (fallback) |
| `FRANKEN_WHISPER_DIARIZATION_DEVICE` | -- | GPU device for diarization backend |
| `FRANKEN_WHISPER_STATE_DIR` | `.franken_whisper` | State directory root |
| `FRANKEN_WHISPER_DB` | `.franken_whisper/storage.sqlite3` | SQLite database path |
| `FRANKEN_WHISPER_FFMPEG_BIN` | auto | Explicit ffmpeg binary path override |
| `FRANKEN_WHISPER_FFPROBE_BIN` | auto | Explicit ffprobe binary path override |
| `FRANKEN_WHISPER_AUTO_PROVISION_FFMPEG` | `1` | Auto-provision local ffmpeg/ffprobe bundle when system binaries are missing (`0`/`false` disables) |
| `FRANKEN_WHISPER_FORCE_FFMPEG_NORMALIZE` | `0` | Force file normalization through ffmpeg even when the built-in Rust decoder can handle the format (`1`/`true` enables) |
| `FRANKEN_WHISPER_NATIVE_EXECUTION` | `0` | Enable native in-process engine dispatch (`1`/`true`) |
| `FRANKEN_WHISPER_BRIDGE_NATIVE_RECOVERY` | `1` | In bridge-only mode, allow recoverable bridge failures to fall back to native engines (`0`/`false` disables) |
| `FRANKEN_WHISPER_NATIVE_ROLLOUT_STAGE` | `primary` | Native engine rollout stage |
| `RUST_LOG` | -- | tracing filter (e.g. `franken_whisper=debug`) |

### Cargo Features

| Feature | Description |
|---------|-------------|
| `tui` | Enable interactive TUI via frankentui |
| `gpu-frankentorch` | Enable frankentorch GPU acceleration |
| `gpu-frankenjax` | Enable frankenjax GPU acceleration |

No features are enabled by default.

### Backend Routing

The `auto` backend uses adaptive Bayesian routing:

**Non-diarization priority:** `whisper_cpp` > `insanely_fast` > `whisper_diarization`

**Diarization priority:** `insanely_fast` > `whisper_diarization` > `whisper_cpp`

Each `auto` run emits a `backend.routing.decision_contract` stage event with explicit state/action/loss/posterior/calibration terms. The router falls back to deterministic static priority when calibration score drops below 0.3 or Brier score exceeds 0.35.

### Native Engine Rollout

Native Rust engine replacements follow a staged rollout:

| Stage | Behavior |
|-------|----------|
| `shadow` | Deterministic bridge execution only; native conformance validated out-of-band |
| `validated` | Deterministic bridge execution only with stricter conformance gating |
| `fallback` | Deterministic bridge execution only; fallback policy and evidence paths hardened |
| `primary` | Native preferred with deterministic bridge fallback (requires `FRANKEN_WHISPER_NATIVE_EXECUTION=1`) |
| `sole` | Native only (requires `FRANKEN_WHISPER_NATIVE_EXECUTION=1`) |

Routing and execution are jointly controlled by:
- `FRANKEN_WHISPER_NATIVE_ROLLOUT_STAGE` (stage policy)
- `FRANKEN_WHISPER_NATIVE_EXECUTION` (whether native dispatch is enabled at runtime)

`backend.ok` and `replay.envelope` stage payloads include explicit execution-path metadata:
`implementation`, `execution_mode`, `native_rollout_stage`, and `native_fallback_error`.

Conformance harness enforces 50ms canonical timestamp tolerance and emits `target/conformance/bridge_native_conformance_bundle.json`.

---

## Architecture

```
                           ┌─────────────────────────┐
                           │      CLI / Robot         │
                           │   (clap + NDJSON emit)   │
                           └────────────┬────────────┘
                                        │
                           ┌────────────▼────────────┐
                           │  FrankenWhisperEngine    │
                           │   (orchestrator.rs)      │
                           │                          │
                           │  10-Stage Pipeline:      │
                           │  1. Ingest               │
                           │  2. Normalize (Rust/ffmpeg)│
                           │  3. VAD                  │
                           │  4. Source Separate       │
                           │  5. Backend Execution    │
                           │  6. Accelerate (GPU)     │
                           │  7. Alignment            │
                           │  8. Punctuation          │
                           │  9. Diarization          │
                           │ 10. Persist              │
                           └────┬───────┬───────┬────┘
                                │       │       │
                  ┌─────────────▼─┐ ┌───▼───┐ ┌─▼──────────────┐
                  │   Backends    │ │ Accel │ │   Storage      │
                  │               │ │       │ │                │
                  │ whisper.cpp   │ │ frank │ │ fsqlite        │
                  │ insanely-fast │ │ torch │ │ (SQLite WAL)   │
                  │ whisper-diar  │ │ frank │ │                │
                  │ native pilots │ │  jax  │ │ JSONL export   │
                  └───────────────┘ └───────┘ └────────────────┘

  ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐
  │   TTY Audio      │    │   Conformance    │    │   Replay         │
  │                  │    │                  │    │                  │
  │ mulaw+zlib+b64   │    │ 50ms tolerance   │    │ SHA-256 content  │
  │ NDJSON transport │    │ cross-engine     │    │ hash envelopes   │
  │ handshake/retry  │    │ comparator       │    │ drift detection  │
  └──────────────────┘    └──────────────────┘    └──────────────────┘
```

### Data Flow

1. **Ingest:** Materialize input from file, stdin, or microphone capture
2. **Normalize:** Convert to 16kHz mono WAV via built-in Rust decoder (ffmpeg fallback for video/exotic formats)
3. **VAD:** (Optional) Voice Activity Detection to skip silence
4. **Source Separate:** (Optional) Vocal isolation for cleaner transcription
5. **Backend:** Dispatch to selected engine (adaptive routing or explicit)
6. **Accelerate:** (Optional) GPU confidence normalization via frankentorch/frankenjax
7. **Alignment:** (Optional) Forced alignment for word-level timestamps
8. **Punctuation:** (Optional) Punctuation restoration
9. **Diarization:** (Optional) Speaker identification and labeling
10. **Persist:** Write run report, segments, and events to SQLite

Each stage emits `*.start` and `*.ok` events to the NDJSON stream with timing, sequence numbers, and structured payloads.

---

## Technical Details

### Bayesian Backend Router

When `--backend auto` is selected, franken_whisper uses a formal Bayesian decision contract to choose the best engine for each request rather than trying backends in a fixed order.

**State Space (3 states):**
- `all_available`: all three backends found on PATH and responsive
- `partial_available`: 1-2 backends operational
- `none_available`: nothing usable

**Action Space (4 actions):**
- `try_whisper_cpp`, `try_insanely_fast`, `try_diarization` (reordered per-request based on `--diarize`)
- `fallback_error`: return structured error when nothing is available

**Loss Matrix:**

The router maintains a 3x4 loss matrix (states x actions). Each cell contains an expected cost computed from three weighted factors:

```
cost = (0.45 x latency_cost) + (0.35 x quality_cost) + (0.20 x failure_cost)
```

- **Latency cost** scales with audio duration (short/medium/long buckets) and backend latency proxy
- **Quality cost** depends on backend capability relative to the request (diarization support, GPU availability)
- **Failure cost** is `(1.0 - p_success) x 100`, where `p_success` comes from the Bayesian posterior

Availability penalties push costs sharply when backends are missing: +333 for partial availability, +1000 for none.

**Bayesian Posterior:**

Each backend starts with a Beta distribution prior reflecting expected reliability:

| Backend | Prior | Interpretation |
|---------|-------|----------------|
| whisper_cpp | Beta(7, 3) | Strong expectation of success |
| insanely_fast | Beta(6, 4) | Moderate expectation |
| whisper_diarization | Beta(5, 5) | Weakest prior (most uncertain) |

After each run, the posterior is updated with the observed outcome. Over time, backends that succeed frequently get stronger posteriors and lower costs.

**Calibration & Fallback:**

The router tracks a sliding window of 50 prediction-outcome pairs and computes a Brier score:

```
Brier = (1/N) x sum((predicted_i - actual_i)^2)
```

The adaptive router falls back to deterministic static priority when any of these hold:
- Fewer than 5 observations (insufficient data)
- Calibration score < 0.3 (posterior margin too narrow)
- Brier score > 0.35 (predictions don't match reality)

This guarantees the system never makes worse decisions than a simple priority list, even when the Bayesian model is poorly calibrated.

**Evidence Ledger:**

Every routing decision is recorded in a circular buffer (capacity: 200 entries) containing the decision ID, trace ID, observed state, chosen action, posterior snapshot, calibration metrics, and whether fallback was triggered. This ledger is queryable via `robot routing-history` and persisted in stage event payloads for post-hoc analysis.

### Pipeline Stage Budgets

Each pipeline stage runs under an independent millisecond budget enforced via `asupersync::time::timeout`. Default budgets:

| Stage | Budget | Rationale |
|-------|--------|-----------|
| Ingest | 15s | File I/O or mic capture |
| Normalize | 180s | Audio decode + resample (built-in or ffmpeg for large files) |
| VAD | 10s | Lightweight energy detection |
| Source Separate | 30s | Demucs-style vocal isolation |
| Backend | 900s (15 min) | Full inference (long audio on CPU) |
| Accelerate | 20s | GPU confidence normalization |
| Align | 30s | CTC forced alignment |
| Punctuate | 10s | Punctuation model inference |
| Diarize | 30s | Speaker clustering |
| Persist | 20s | SQLite transaction |
| Cleanup | 5s | Finalizer timeout |

Every budget is overridable via `FRANKEN_WHISPER_STAGE_BUDGET_<STAGE>_MS` environment variables.

**Automatic Latency Profiling:**

After each run, the orchestrator emits an `orchestration.latency_profile` stage event with per-stage timing decomposition:

- **queue_ms**: time waiting before stage starts
- **service_ms**: actual work time
- **external_process_ms**: subprocess wall time (ffmpeg, whisper-cli)
- **p50_ms / p95_ms / p99_ms**: quantile estimates from the current session

The profiler computes a utilization ratio (`service_ms / budget_ms`) and emits deterministic tuning recommendations:

| Utilization | Recommendation |
|-------------|----------------|
| <= 30% | `decrease_budget_candidate` |
| 30-90% | `keep_budget` |
| >= 90% | `increase_budget` (suggest 1.25x current) |

### Replay Envelopes & Drift Detection

Every completed run produces a `ReplayEnvelope` containing four SHA-256 hashes:

```
┌──────────────────────────────────────────────┐
│              ReplayEnvelope                  │
├──────────────────────────────────────────────┤
│ input_content_hash:  SHA-256(normalized WAV) │
│ backend_identity:    "whisper-cli-v1.7.2"    │
│ backend_version:     "1.7.2"                 │
│ output_payload_hash: SHA-256(raw backend JSON)│
└──────────────────────────────────────────────┘
```

Given identical input audio and the same backend version, the output hash should be identical. If it changes between runs, something drifted: a model update, a parameter change, or a non-determinism in the backend. The conformance harness uses replay comparison reports to flag regressions automatically:

- `input_hash_match`: did the input change?
- `backend_identity_match`: same binary?
- `backend_version_match`: same version?
- `output_hash_match`: same transcription?

All four must match for `within_tolerance()` to return true.

### Replay Packs

Beyond individual replay envelopes, franken_whisper can generate self-contained **replay packs** — a directory of four JSON artifacts that capture everything needed to reproduce and analyze a run:

```
replay_pack/
  env.json                # EnvSnapshot: OS, arch, backend identity/version, fw version
  manifest.json           # PackManifest: trace_id, run_id, timestamps, content hashes,
                          #   segment/event/evidence counts
  repro.lock              # ReproLock: routing evidence, replay envelope, request params
  tolerance_manifest.json # ToleranceManifest: schema version, timestamp tolerance,
                          #   text/speaker exactness, native rollout stage
```

Replay packs are designed for post-hoc analysis: "why did this run produce different output?" The `repro.lock` captures the exact routing decision and evidence that led to backend selection. The `tolerance_manifest.json` records the conformance thresholds that were active. Together with the replay envelope's content hashes, this provides a complete audit trail from input to output.

### Conformance Harness

The conformance module enforces cross-engine compatibility using a 50ms canonical timestamp tolerance (`CANONICAL_TIMESTAMP_TOLERANCE_SEC = 0.05`). This is the single source of truth for how much timing drift is acceptable between bridge adapters and native engines.

**Segment Comparison:**

The comparator aligns expected vs. observed segment lists and counts violations:

| Violation Type | Condition |
|----------------|-----------|
| Text mismatch | Segment text differs at same index |
| Speaker mismatch | Speaker label differs (optional check) |
| Timestamp violation | start/end differs by > 50ms |
| Length mismatch | Different segment counts |

A `SegmentComparisonReport` with zero violations and matching lengths passes the conformance gate.

**Overlap Detection:**

The `SegmentConformancePolicy` can optionally reject overlapping segments, where one segment's `end_sec` exceeds the next segment's `start_sec` beyond a configurable epsilon (default: 1 microsecond). This catches backends that produce garbled timeline output.

**WER Approximation:**

The conformance module includes a Levenshtein-based Word Error Rate calculator for comparing transcripts across backends or across fast/quality model outputs. Words are tokenized by whitespace and compared using edit distance, producing a `wer_approx` metric normalized to [0.0, 1.0].

**Segment Invariant Validation:**

Individual segments are validated against invariants before comparison:

- Timestamps must be finite (no NaN, no infinity)
- Start and end times must be non-negative
- Start must be ≤ end (no inverted segments)
- Confidence scores must be in [0.0, 1.0]
- Text must be non-empty

### Confidence Normalization (Acceleration)

The acceleration stage normalizes per-segment confidence scores into a proper probability distribution. This matters because raw backend confidences are often uncalibrated; whisper.cpp and insanely-fast-whisper use different scoring scales.

**Algorithm:**

1. Extract confidence values from all segments
2. Replace missing/invalid values (NaN, infinity) with a text-length-based baseline: `ln(1 + char_count) + 1.0`
3. Compute pre-mass: `sum(confidences)` before normalization
4. Apply softmax normalization (GPU path via frankentorch/frankenjax, or CPU fallback)
5. Compute post-mass: `sum(normalized)` (should equal 1.0)
6. Record both masses in the `AccelerationReport` for validation

The CPU fallback uses safe division: `value / sum` with a guard for near-zero sums (falls back to uniform `1/N`).

**Acceleration Paths:**

| Path | Trigger | Method |
|------|---------|--------|
| frankentorch | `--features gpu-frankentorch` | Tensor softmax via `ft_api::FrankenTorchSession` |
| frankenjax | `--features gpu-frankenjax` | JAX-based normalization via `fj_api` |
| CPU fallback | no GPU features | Safe division with NaN/inf guards |

### Audio Normalization Pipeline

Input audio is normalized to a standardized 16 kHz, mono, 16-bit PCM WAV before reaching any backend. The primary path uses a **built-in Rust audio decoder** (powered by symphonia) that requires zero external dependencies:

```
Input file (any format)
  │
  ├─► Built-in Rust decoder (PRIMARY)
  │     symphonia: MP3, AAC, FLAC, WAV, OGG, Vorbis, ALAC, PCM variants
  │     Custom resampler: sinc interpolation to 16 kHz
  │     Channel mixer: stereo/surround → mono via sample averaging
  │     Output: normalized_16k_mono.wav (PCM S16LE)
  │
  └─► ffmpeg subprocess (FALLBACK — only if built-in decoder fails)
        Triggered for: video files, exotic codecs (AC3, DTS, Opus-in-MKV, etc.)
        Args: -hide_banner -loglevel error -y -i <input> -vn -ar 16000 -ac 1 -c:a pcm_s16le <output>
        The -vn flag strips video tracks automatically (handles MP4, MKV, AVI, etc.)
```

The built-in path is fast — normalization of a 2-minute MP3 completes in ~260ms on a typical machine, with no subprocess spawning, no temporary file juggling, and no PATH dependency.

**Fallback chain when ffmpeg is needed:**

1. Explicit binary path (`FRANKEN_WHISPER_FFMPEG_BIN`)
2. System-installed `ffmpeg` on PATH
3. Auto-provisioned local binary (linux/x86_64 — downloaded to `.franken_whisper/ffmpeg/`)
4. If all fail: `FW-CMD-MISSING` error with actionable message

**Override behavior:**

Set `FRANKEN_WHISPER_FORCE_FFMPEG_NORMALIZE=1` to bypass the built-in decoder and always use ffmpeg. This is useful for debugging format-specific issues or when you need ffmpeg's specific audio filters.

**Microphone capture** still requires ffmpeg (live audio capture is inherently OS-dependent):

| OS | Format | Default Device |
|----|--------|----------------|
| Linux | `alsa` | `default` |
| macOS | `avfoundation` | `:0` |
| Windows | `dshow` | `audio=default` |

### Built-In Audio Decoder Internals

The built-in normalizer (`normalize_to_wav_with_builtin_decoder`) is a pure-Rust audio pipeline that produces whisper-compatible WAV without spawning any subprocess:

**Format Detection:** Symphonia's `get_probe().format()` uses file extension hints and magic-byte probing to identify the container format. Supported containers include MP3 (MPEG Layer III), MP4/M4A (AAC), FLAC, WAV/RIFF, OGG (Vorbis), and WavPack.

**Decoding Loop:**

```
for each packet in format_reader:
    decoded = codec_decoder.decode(packet)
    convert decoded samples to f32
    if multi-channel: average all channels → mono
    append to sample buffer
```

Sample conversion handles `i16`, `i32`, `f32`, and `f64` source formats. Multi-channel audio is mixed to mono by averaging corresponding samples across channels.

**Resampling:** A linear interpolation resampler converts from the source sample rate (commonly 44.1 kHz or 48 kHz) to whisper's required 16 kHz:

```
ratio = src_rate / dst_rate
for each output sample i:
    position = i * ratio
    left = input[floor(position)]
    right = input[ceil(position)]
    output[i] = left + frac(position) * (right - left)
```

This is computationally lightweight (no FFT, no filter bank) while being sufficient for speech — whisper models are robust to minor resampling artifacts. A 2-minute 48 kHz stereo MP3 normalizes in ~260ms.

**WAV Output:** The final mono f32 buffer is quantized to 16-bit signed PCM (`i16`) via clamp-and-round, then written as a standard RIFF WAV header + raw PCM data. The output is always `normalized_16k_mono.wav` in the work directory.

### Speculative Streaming Architecture

franken_whisper supports a "speculate then verify" dual-model streaming pattern for real-time transcription with quality corrections. This is the architecture behind the `transcript.partial`, `transcript.confirm`, and `transcript.retract` event types.

**Concept:** A fast model produces low-latency partial transcripts immediately, while a quality model runs in parallel on the same audio window. When the quality model finishes, it either confirms the fast model's output or retracts and replaces it.

```
Audio Stream
  │
  ├──► WindowManager (sliding windows with overlap)
  │       │
  │       ├──► Fast Model ──► PartialTranscript (status: Pending)
  │       │                        │
  │       │                        ▼ emit "transcript.partial" event
  │       │
  │       └──► Quality Model ──► CorrectionDrift analysis
  │                                  │
  │                                  ├─ drift below tolerance ──► "transcript.confirm"
  │                                  └─ drift above tolerance ──► "transcript.retract" + corrected text
  │
  └──► CorrectionTracker (adaptive thresholds)
```

**WindowManager:** Divides the audio stream into overlapping windows. Each window gets a unique `window_id`, an SHA-256 hash of its audio content, and slots for both the fast and quality model results. Windows are resolved once both models finish.

**CorrectionDrift Metrics:** When the quality model disagrees with the fast model, the system quantifies the disagreement:

- **wer_approx:** Approximate word error rate between fast and quality transcripts
- **confidence_delta:** Difference in mean segment confidence
- **segment_count_delta:** Whether the models produced different segment counts
- **text_edit_distance:** Levenshtein distance between transcript texts

**CorrectionTracker:** Maintains running statistics of drift metrics across windows. In adaptive mode, confirmation/retraction thresholds adjust based on the observed distribution — if the fast model is consistently accurate, the tolerance widens; if it frequently disagrees with the quality model, the tolerance tightens.

**ConcurrentTwoLaneExecutor:** Runs both models in parallel lanes with independent timeout budgets. Results are collected asynchronously, and the faster result (always the fast model by design) is emitted immediately while the quality result triggers correction logic when it arrives.

### Alien-Artifact Engineering Contracts

Every adaptive controller in franken_whisper follows a formal "alien-artifact engineering contract" — a design discipline that prevents adaptive systems from making unbounded bad decisions.

**The problem it solves:** Adaptive algorithms (Bayesian routers, auto-tuners, ML-based controllers) can behave unpredictably when their models are wrong. A Bayesian router with a bad prior will confidently make terrible decisions. An auto-tuner with noisy data will oscillate. The standard response is "just add more data" or "tune the hyperparameters," but for a CLI tool that runs on user machines, there's no ops team watching dashboards.

**The contract requires every adaptive controller to declare:**

| Component | Purpose | Example (Backend Router) |
|-----------|---------|--------------------------|
| **State space** | What does the controller observe? | 3 availability states (all/partial/none) |
| **Action space** | What can it decide? | 4 actions (try each backend + error) |
| **Loss matrix** | What's the cost of each state×action? | 3×4 matrix: latency(45%) + quality(35%) + failure(20%) |
| **Posterior terms** | How confident is the model? | Beta distribution per backend |
| **Calibration metric** | How accurate are predictions? | Brier score on 50-observation sliding window |
| **Deterministic fallback** | What happens when the model is wrong? | Static priority list |
| **Fallback trigger** | When does fallback activate? | Brier > 0.35 or calibration < 0.3 or < 5 observations |
| **Evidence ledger** | Audit trail of all decisions | Circular buffer of 200 `RoutingEvidenceLedgerEntry` records |

**Why this matters:** The contract guarantees bounded worst-case behavior. Even if the Bayesian model is perfectly miscalibrated, the system falls back to a simple priority list that always works. The evidence ledger makes every decision inspectable after the fact. The loss matrix makes the tradeoffs explicit rather than buried in code.

**Controllers using this contract:**
- Backend router (Bayesian backend selection)
- Adaptive bitrate controller (TTY audio link quality)
- Budget tuner (pipeline stage timeout recommendations)
- Correction tracker (speculation confirmation thresholds)

### Pipeline Composition & Stage Isolation

The 10-stage pipeline is not a hardcoded sequence — it's composed dynamically per-request based on the input source, backend capabilities, and user flags.

**PipelineCx (Pipeline Context):**

Every pipeline run creates a `PipelineCx` that carries shared state through all stages:

| Field | Type | Purpose |
|-------|------|---------|
| `trace_id` | `TraceId` | Unique identifier from `(timestamp_ms, random_u64)` |
| `deadline` | `Option<DateTime<Utc>>` | Absolute wall-clock deadline for the entire pipeline |
| `budget` | `Budget` | Remaining time budget (decremented by stage service times) |
| `evidence` | `Vec<Value>` | JSON evidence accumulator for post-hoc analysis |
| `finalizers` | `FinalizerRegistry` | Cleanup handlers run on pipeline exit (bounded to 5s) |

**CancellationToken (Copy + Send + Sync):**

A lightweight handle extracted from `PipelineCx` for passing into background threads and subprocess monitors:

```rust
struct CancellationToken {
    deadline: Option<DateTime<Utc>>,
}
```

The token's `checkpoint()` method checks two conditions: (1) has Ctrl+C been pressed (global `AtomicBool`), and (2) has the deadline passed. If either is true, it returns `Err(Cancelled)`. This is polled cooperatively — stages call `checkpoint()` at safe points (between loop iterations, before COMMIT, after subprocess completion).

**Stage Budget Isolation:**

Each stage has an independent timeout budget. A slow normalization stage cannot eat into the backend's time budget. Budgets are configured via environment variables (`FRANKEN_WHISPER_STAGE_BUDGET_<STAGE>_MS`) and profiled automatically. After each run, the orchestrator computes utilization ratios and emits tuning recommendations: `decrease_budget_candidate` (≤30% utilized), `keep_budget` (30-90%), or `increase_budget` (≥90%, suggests 1.25x current).

**Dynamic Stage Composition:**

Not every run executes all 10 stages. The pipeline skips stages that aren't needed:

| Condition | Skipped Stages |
|-----------|----------------|
| Input is already 16kHz mono WAV | Normalize (passthrough) |
| No `--diarize` flag | Diarize |
| No `--vad` flag | VAD |
| No GPU features compiled | Accelerate (CPU fallback used inline) |
| `--no-persist` flag | Persist |
| Backend doesn't support alignment | Align |

Skipped stages still emit `*.skip` events to the NDJSON stream so agents can distinguish "not needed" from "failed."

### Native Engine Rollout Governance

The transition from external bridge adapters (spawning `whisper-cli`, `python3`) to in-process native Rust engines follows a 5-stage rollout with conformance gating at each stage. This prevents a buggy native engine from silently degrading transcription quality.

**Rollout Stages:**

```
Shadow ──► Validated ──► Fallback ──► Primary ──► Sole
  │            │             │           │          │
  │            │             │           │          └─ Native only, bridge removed
  │            │             │           └─ Native preferred, bridge fallback on error
  │            │             └─ Bridge preferred, native fallback hardened
  │            └─ Bridge only, stricter conformance gating
  └─ Bridge only, native conformance validated out-of-band
```

**Conformance Gate:** At each stage transition, the conformance harness compares native vs. bridge output on a test corpus. The 50ms canonical timestamp tolerance (`CANONICAL_TIMESTAMP_TOLERANCE_SEC = 0.05`) is the single source of truth. A native engine that produces timestamps >50ms different from the bridge adapter for the same audio is blocked from promotion.

**Segment Validation Rules:**

The conformance policy validates individual segments against invariants:

- Timestamps must be finite (no NaN, no infinity)
- Start and end times must be non-negative
- Start must be ≤ end
- No overlapping segments (configurable epsilon: 1μs default)
- Confidence scores must be in [0.0, 1.0]
- Text must be non-empty

**Runtime Control:**

Two environment variables jointly control native engine behavior:

- `FRANKEN_WHISPER_NATIVE_ROLLOUT_STAGE`: which stage the deployment is at (shadow/validated/fallback/primary/sole)
- `FRANKEN_WHISPER_NATIVE_EXECUTION`: whether native dispatch is enabled at runtime (0/1)

Both must agree for native engines to actually execute. Setting `NATIVE_EXECUTION=1` with stage `shadow` has no effect — the stage gate prevents native execution regardless of the runtime flag.

**Execution Path Metadata:**

Every `backend.ok` and `replay.envelope` stage event includes explicit execution-path metadata: `implementation` (bridge or native), `execution_mode`, `native_rollout_stage`, and `native_fallback_error` (populated when native fails and bridge recovers).

### Storage Internals

The storage layer uses `fsqlite` (from the frankensqlite project) with three tables:

```sql
runs     (run_id PK, started_at, finished_at, backend, input_path,
          request_json, result_json, transcript, replay_json, ...)

segments (run_id FK, idx, start_sec, end_sec, speaker, text, confidence)

events   (run_id FK, seq, ts_rfc3339, stage, code, message, payload_json)
```

**Atomic Persistence with Retry:**

The `persist_report` function wraps all inserts (run + segments + events) in a single transaction. If SQLite returns "database is busy" (concurrent writer), it retries with exponential backoff:

- 8 retry attempts
- Backoff delay: `5ms x (attempt + 1)` (5ms, 10ms, 15ms, ... 40ms)
- Cancellation token checked before each COMMIT; if the pipeline deadline expired, the transaction rolls back cleanly

**Cancellation-Safe Writes:**

The token checkpoint pattern ensures no partial data reaches the database:

```
BEGIN TRANSACTION
  INSERT INTO runs ...
  INSERT INTO segments ... (N rows)
  INSERT INTO events ... (M rows)
  token.checkpoint()?  <-- rolls back if cancelled
COMMIT
```

### Sync Architecture

The sync module provides one-way JSONL snapshot export/import with distributed lock safety.

**Lock Protocol:**

Before any export or import, a JSON lock file is created at `{state_root}/locks/sync.lock`:

```json
{"pid": 12345, "created_at_rfc3339": "2026-02-22T12:00:00Z", "operation": "export"}
```

Stale lock detection checks two conditions:
1. Is the PID still alive? (reads `/proc/{pid}` on Linux)
2. Is the lock older than 5 minutes?

If either check fails, the lock is archived with a reason suffix and a new lock is acquired.

**Export Format:**

An export produces four files:

```
snapshot/
  runs.jsonl        # one JSON object per run
  segments.jsonl    # one JSON object per segment
  events.jsonl      # one JSON object per event
  manifest.json     # metadata + SHA-256 checksums
```

The manifest contains row counts and SHA-256 checksums of each JSONL file, enabling integrity verification on import.

**Incremental Export:**

Full exports re-dump the entire database. For large databases with many runs, incremental export is more efficient:

```bash
cargo run -- sync export-jsonl --output ./snapshot --incremental
```

Incremental mode uses a cursor file (`exports/cursor.json`) tracking the last export timestamp and run count. Only runs created after the cursor are exported. The incremental manifest (`IncrementalExportManifest`) includes the cursor state, row counts, and checksums for the incremental JSONL files.

**JSONL Compression:**

Sync supports optional gzip compression for JSONL files, reducing snapshot size for archival or transfer:

```
snapshot/
  runs.jsonl.gz          # gzip-compressed
  segments.jsonl.gz
  events.jsonl.gz
  manifest.json          # always uncompressed (small)
```

**Sync Validation:**

After import, `validate_sync()` compares the database state against the imported JSONL files, checking for row count mismatches and checksum mismatches. This provides end-to-end integrity verification.

**Conflict Policies:**

| Policy | Behavior on duplicate run_id |
|--------|------------------------------|
| `reject` | Fail the entire import |
| `skip` | Silently skip existing runs |
| `overwrite` | Replaces conflicting `runs` rows, but fails closed if child-row `UPDATE`/`DELETE` would be required |
| `overwrite-strict` | Verified strict replacement for imported runs, including child-row updates (delete+insert) and stale child-row pruning |

Runtime guidance: use `overwrite` for conservative fail-closed behavior and `overwrite-strict` when operator intent is to mutate child rows in-place for exact replacement of imported runs.

### TTY Audio: Adaptive Bitrate & FEC

The TTY audio module goes beyond simple encode/decode. The `AdaptiveBitrateController` monitors link quality in real time and adjusts compression dynamically:

| Frame Loss Rate | Link Quality | Compression | Critical Frame FEC |
|-----------------|--------------|-------------|-------------------|
| < 1% | High | zlib level 1 (fast) | 1x (no duplication) |
| 1% - 10% | Moderate | zlib level 6 (default) | 2x |
| > 10% | Poor | zlib level 9 (best) | 3x |

**Critical Frame FEC (Forward Error Correction):**

Control frames essential for protocol correctness (handshake, session_close, ack) are emitted multiple times based on the current link quality. Under 10% loss, every handshake frame is transmitted 3 times to ensure at least one copy arrives. This is a probabilistic reliability guarantee: with independent frame loss at rate `p`, the probability all `k` copies are lost is `p^k`.

**Link Quality Assessment:**

The controller maintains running `frames_sent` and `frames_lost` counters. After each delivery attempt, it recalculates:

```
frame_loss_rate = frames_lost / frames_sent
link_quality = 1.0 - frame_loss_rate
```

Quality transitions trigger compression level changes on subsequent frames, providing automatic adaptation without manual tuning.

**Transcript Streaming over TTY (Protocol v2):**

Beyond raw audio transport, the TTY protocol supports real-time transcript streaming via three control frame types:

| Frame Type | Direction | Purpose |
|------------|-----------|---------|
| `TranscriptPartial` | sender → receiver | Speculative partial transcript from fast model |
| `TranscriptRetract` | sender → receiver | Retract a previous partial (quality model disagrees) |
| `TranscriptCorrect` | sender → receiver | Send corrected transcript from quality model |

These frames carry `TranscriptSegmentCompact` payloads — a wire-efficient representation using single-letter field names (`s`/`e`/`t`/`sp`/`c` for start/end/text/speaker/confidence) to minimize bandwidth. This enables the speculative streaming pipeline to operate over TTY links where only text-based NDJSON can flow.

**Telemetry Counters:**

The decode path tracks comprehensive telemetry:

- `frames_decoded`: successful audio frame count
- `gaps`: sequence number discontinuities (with expected/actual pairs)
- `duplicates`: repeated sequence numbers
- `integrity_failures`: CRC32/SHA-256 mismatches
- `dropped_frames`: frames discarded due to policy

### Graceful Shutdown

franken_whisper's shutdown path is designed to never leave the system in an inconsistent state.

**Signal Flow:**

```
Ctrl+C
  │
  ▼
ctrlc handler
  │ sets AtomicBool (SeqCst)
  │ calls optional callback
  ▼
CancellationToken.checkpoint()
  │ returns Err(Cancelled) at next checkpoint
  ▼
Pipeline stage catches Cancelled
  │ rolls back any in-progress transaction
  │ cleans up temp files via finalizers
  ▼
FrankenWhisperEngine
  │ runs all registered finalizers (bounded to 5s)
  │ emits run_error event with FW-CANCELLED code
  ▼
CLI exits with code 130 (128 + SIGINT)
```

Stages don't catch signals directly. Instead, they poll `token.checkpoint()` at safe points: between loop iterations, before COMMIT, after subprocess completion. This cooperative cancellation model ensures:

1. No half-written SQLite rows (transactions roll back)
2. No orphaned ffmpeg/whisper-cli subprocesses (killed on token cancellation)
3. No leaked temp files (finalizers run within bounded timeout)
4. Deterministic exit code (130 for signal, vs. other codes for errors)

### Error Codes

| Code | Meaning |
|------|---------|
| `FW-IO` | I/O error (file not found, permission denied) |
| `FW-JSON` | JSON serialization/deserialization failure |
| `FW-CMD-MISSING` | Required external binary not found on PATH |
| `FW-CMD-FAILED` | Backend subprocess exited with non-zero status |
| `FW-CMD-TIMEOUT` | Backend subprocess exceeded timeout |
| `FW-BACKEND-UNAVAILABLE` | No suitable backend found for request |
| `FW-INVALID-REQUEST` | Malformed or contradictory request parameters |
| `FW-STORAGE` | SQLite persistence error |
| `FW-UNSUPPORTED` | Requested feature not available |
| `FW-MISSING-ARTIFACT` | Expected output file not produced by backend |
| `FW-CANCELLED` | Operation cancelled via token or Ctrl+C |
| `FW-STAGE-TIMEOUT` | Pipeline stage exceeded its budget |

---

## Codebase Statistics

| Metric | Value |
|--------|-------|
| Total source lines (src/) | 84,687 |
| Library tests (`#[test]` in `src/`) | 2,954 |
| Integration test files (`tests/*.rs`) | 23 |
| Benchmark suites | 5 (criterion) |
| Public modules | 18 |
| Error variants | 12 (each with structured code) |
| Backend engines | 6 (3 bridge + 3 native pilot) |
| Pipeline stages | 10 (composable, independently budgeted) |
| CLI subcommands | 6 (transcribe, robot, runs, sync, tty-audio, tui) |
| CLI flags (transcribe) | 70+ (inference, VAD, diarization, speculative, audio windowing) |
| Robot event types | 12 (run lifecycle, stage, speculation, health, routing) |
| TTY control frame types | 10 (handshake, ack, retransmit, backpressure, transcript streaming, session close) |
| TTY protocol versions | 2 (v1 audio, v2 transcript streaming) |
| Replay pack artifacts | 4 (env, manifest, repro.lock, tolerance_manifest) |
| Sync conflict policies | 4 (reject, skip, overwrite, overwrite-strict) |
| Native rollout stages | 5 (shadow, validated, fallback, primary, sole) |
| Conformance tolerance | 50ms canonical timestamp tolerance |
| Evidence ledger capacity | 200 entries (circular buffer) |
| Router history window | 50 outcome records per backend |
| Clippy enforcement | `#![forbid(unsafe_code)]` + `-D warnings` on all targets |
| Cargo features | 3 (tui, gpu-frankentorch, gpu-frankenjax) |
| Release optimizations | opt-level z, LTO, single codegen unit, panic=abort, stripped |

---

## Troubleshooting

### "FW-CMD-MISSING: whisper-cli not found"

No backend binary is on your PATH. Install at least one:

```bash
# whisper.cpp
brew install whisper-cpp   # macOS
# or build from source: https://github.com/ggerganov/whisper.cpp

# or override the binary name
export FRANKEN_WHISPER_WHISPER_CPP_BIN=/path/to/whisper-cli
```

### "FW-BACKEND-UNAVAILABLE: diarization requires HF token"

Diarization needs a HuggingFace API token for pyannote models:

```bash
export FRANKEN_WHISPER_HF_TOKEN="hf_your_token_here"
# or pass directly
cargo run -- transcribe --input audio.mp3 --diarize --hf-token "hf_..."
```

### "FW-CMD-TIMEOUT: backend exceeded timeout"

The backend took longer than the allowed duration:

```bash
# increase timeout (seconds)
cargo run -- transcribe --input long_audio.mp3 --timeout 600 --json
```

### Robot mode outputs nothing

Ensure you're using the `robot run` subcommand, not just `robot`:

```bash
cargo run -- robot run --input audio.mp3 --backend auto
```

### SQLite "database is locked"

Another franken_whisper process is writing. The storage layer retries with exponential backoff (5-40ms), but simultaneous heavy writes may conflict. Use `--no-persist` to skip persistence, or use separate `--db` paths.

### "safe legacy runs migration failed"

When opening older DBs missing `runs.replay_json` / `runs.acceleration_json`, franken_whisper now attempts a snapshot/rebuild/swap migration with rollback safety and integrity checks.

If that migration still fails (for example due severe on-disk corruption), recover via snapshot:

1. Preserve the original DB as immutable evidence.
2. Export from a known-good source snapshot (or recover from existing JSONL export).
3. Create a fresh target DB.
4. Import via:

```bash
cargo run -- sync import-jsonl --input ./snapshot --conflict-policy reject
```

For in-place strict replacement flows, use `--conflict-policy overwrite-strict`. For conservative recovery (or suspected corruption), prefer importing into a fresh target DB.

---

## Limitations

- **Backend binaries required.** franken_whisper orchestrates external ASR engines; it does not include inference runtimes. You need whisper.cpp, insanely-fast-whisper, or whisper-diarization installed.
- **ffmpeg only needed for video/exotic formats.** The built-in Rust decoder handles common audio formats (MP3, AAC, FLAC, WAV, OGG) natively. ffmpeg is used as an automatic fallback for video files and exotic codecs that symphonia doesn't support. Microphone capture still depends on ffmpeg availability.
- **Path dependencies.** The project depends on sibling Cargo workspace members (`asupersync`, `frankensqlite`, etc.) via relative paths. It is not published to crates.io as a standalone crate.
- **Native engines are pilots.** Native Rust engine implementations are deterministic conformance pilots. They can execute in-process when `FRANKEN_WHISPER_NATIVE_EXECUTION=1` and rollout stage is `primary|sole`; otherwise bridge adapters remain active.
- **No bidirectional sync.** JSONL export/import is one-way. There is no merge or conflict resolution beyond the explicit `--conflict-policy` flag.
- **Legacy schema migration is rollback-safe but not magic.** Legacy `runs` schemas are migrated through snapshot/rebuild/swap with integrity checks; severely corrupted DBs may still require JSONL restore into a fresh DB.
- **Overwrite mode is conservative by default.** `overwrite` remains fail-closed for child-row mutation, while `overwrite-strict` enables verified child-row replacement for imported runs.
- **Single-machine.** Designed for single-machine use with local SQLite. No distributed or multi-node support.

---

## Testing

107,000+ lines of Rust with 2,000+ tests across unit, integration, and conformance suites.

```bash
# run all library tests
cargo test --lib

# run specific test module
cargo test --lib -- backend::tests
cargo test --lib -- robot::tests
cargo test --lib -- tty_audio::tests

# run integration tests
cargo test --test tty_telemetry_tests
cargo test --test conformance_comparator_tests
cargo test --test gpu_cancellation_tests
cargo test --test robot_contract_tests
cargo test --test e2e_pipeline_tests

# run benchmarks
cargo bench --bench storage_bench
cargo bench --bench normalize_bench
cargo bench --bench pipeline_bench
cargo bench --bench tty_bench
cargo bench --bench sync_bench

# lint (lib targets; some test targets have pre-existing warnings from in-progress work)
cargo clippy --lib -- -D warnings
```

### Remote Quality Gates (`rch`)

When offload policy requires all cargo gates to run through `rch`, use:

```bash
scripts/run_quality_gates_rch.sh
```

Behavior:
- enforces remote-only execution (fails if `rch` falls back to local)
- retries transient worker failures on fresh remote target dirs
- runs `fmt`, `check`, `clippy`, and `test` by default

Useful toggles:

```bash
# skip the full test gate (keeps fmt/check/clippy)
RUN_TEST_GATE=0 scripts/run_quality_gates_rch.sh

# increase retries for noisy worker pools
MAX_ATTEMPTS=6 scripts/run_quality_gates_rch.sh

# temporarily block problematic workers for this run; they are auto-restored on exit
BLOCK_WORKERS=vmi1149989 scripts/run_quality_gates_rch.sh
```

### Test Categories

| Category | Count | Description |
|----------|-------|-------------|
| Backend engine tests | 260+ | Engine trait compliance, native pilot validation |
| Robot contract tests | 150+ | NDJSON schema validation, field presence |
| TTY audio tests | 109 | Handshake, integrity, retransmit, telemetry |
| Conformance tests | 80+ | Cross-engine tolerance, replay envelopes |
| Storage tests | 100+ | SQLite roundtrip, concurrent writes, recovery |
| GPU cancellation tests | 42 | Stream ownership, fence payloads, fallback |
| Conformance comparator | 25 | Drift metrics, WER, shadow-run modes |
| E2E pipeline tests | -- | Full pipeline from input to persisted result |

---

## FAQ

**Q: Do I need all three backends installed?**

No. franken_whisper works with any single backend. The `auto` router will use whatever is available. You can also force a specific backend with `--backend whisper_cpp`.

**Q: What audio formats are supported?**

Common audio formats (MP3, AAC, FLAC, WAV, OGG, Vorbis, ALAC) are decoded natively by the built-in Rust decoder with zero external dependencies. Video files and exotic codecs (AC3, DTS, Opus-in-MKV) fall back to ffmpeg automatically. If ffmpeg is also unavailable, the error message tells you exactly which format was unsupported and how to install ffmpeg.

**Q: Can I use this as a library?**

Yes. `franken_whisper` is both a library crate and a binary. The public API exposes all modules: `backend`, `orchestrator`, `robot`, `storage`, `tty_audio`, `conformance`, etc.

**Q: What's the "replay envelope"?**

Each run produces a `ReplayEnvelope` containing SHA-256 hashes of the input content, backend identity, and output payload. This allows detecting drift when re-running the same input. If the output hash changes, something in the pipeline changed.

**Q: How does cancellation work?**

Ctrl+C sets a global shutdown flag. The `CancellationToken` from `asupersync` propagates through every pipeline stage. Each stage calls `token.checkpoint()` at safe points, which returns `Err(Cancelled)` if shutdown was requested. This ensures no partial writes to SQLite and no orphaned subprocess resources.

**Q: What's the TTY audio module for?**

It enables audio transport over constrained TTY/PTY links where binary data can't flow directly. Audio is compressed (mu-law + zlib), base64-encoded, and transmitted as NDJSON lines with sequence numbers, CRC32, and SHA-256 integrity. A handshake protocol negotiates version and codec, and a retransmit loop recovers lost frames with escalating recovery strategies.

**Q: How does the Bayesian router differ from a simple priority list?**

A priority list always tries backends in the same order. The Bayesian router learns from outcomes: if whisper_cpp starts failing (bad model, corrupted binary), its posterior degrades and the router automatically shifts traffic to insanely_fast. The loss matrix also considers request-specific factors like audio duration and whether diarization was requested. When the router doesn't have enough data (< 5 observations) or its predictions are poorly calibrated (Brier score > 0.35), it falls back to the static priority list automatically. The result is adaptive routing when the model is well-calibrated, with automatic fallback to static priority when it isn't.

**Q: What happens if I Ctrl+C during a long transcription?**

The shutdown controller sets a global atomic flag and propagates cancellation through the pipeline. The currently active stage finishes its current checkpoint boundary (never mid-write), rolls back any uncommitted SQLite transaction, kills any running subprocess (ffmpeg, whisper-cli), runs registered finalizers within a 5-second bounded timeout, and exits with code 130. No data corruption, no orphaned processes, no leaked temp files.

**Q: What's the "alien-artifact engineering contract"?**

A design discipline for adaptive/runtime decision systems. Every adaptive controller in franken_whisper (the backend router, the bitrate controller, the budget tuner) must declare an explicit state space, action space, loss matrix, posterior/confidence terms, calibration metric, deterministic fallback trigger, and evidence ledger artifact. No adaptive behavior ships without a conservative fallback, which prevents the system from making unbounded bad decisions when its model is wrong.

**Q: How are native engines different from bridge adapters?**

Bridge adapters spawn external processes (whisper-cli, python3) and parse their stdout/stderr. Native engines run in-process as deterministic Rust code. Both implement the same `Engine` trait and produce identical `TranscriptionResult` types. Native engines use the `<backend>-native` naming convention (e.g., `whisper.cpp-native`), return the same `BackendKind` as their bridge counterpart, and must declare a capability superset of the bridge. The rollout from bridge to native follows a 5-stage progression (shadow, validated, fallback, primary, sole) with conformance gating at each stage.

**Q: Why SQLite instead of Postgres/Redis/files?**

SQLite fits a single-machine CLI tool well: zero configuration, no daemon, ACID transactions, concurrent reads via WAL mode, and the entire database is a single file that's easy to backup or move. The `fsqlite` crate provides a Rust-native interface without depending on the system `libsqlite3`. JSONL export/import covers portability: export to files, import on another machine, share via git.

**Q: How does the TTY audio retransmit loop guarantee determinism?**

The `RetransmitLoop` struct is fully deterministic: given the same input frames and the same loss pattern, the output and report are byte-identical across runs. There are no timing dependencies; `timeout_ms` is advisory/reporting only, with no sleeps or waits. Frame recovery proceeds in sequence-number order. Strategy escalation follows a fixed chain (Simple -> Redundant -> Escalate). The `inject_loss()` method resets all prior recovery state, ensuring clean separation between test scenarios.

**Q: What's the difference between `fail_closed` and `skip_missing` recovery?**

`fail_closed` (default for decode) aborts on the first violation: any gap, duplicate, CRC mismatch, or zlib failure terminates the stream. Use it when you need guaranteed complete output. `skip_missing` (default for retransmit-plan) continues processing, recording all violations in telemetry counters (gaps, duplicates, integrity_failures, dropped_frames). Use it when you need to assess damage and plan retransmission. Both policies always fail immediately on protocol-level violations (wrong version, unsupported codec, wrong sample rate).

**Q: What's speculative streaming and when should I use it?**

Speculative streaming runs two models simultaneously: a fast model (e.g., `tiny.en`) produces low-latency partial transcripts immediately, while a quality model (e.g., `large-v3`) runs in parallel. When the quality model finishes each window, it either confirms the fast model's output or retracts it and sends a correction. Use `--speculative` when you need both low latency (see results immediately) and high accuracy (corrections arrive shortly after). The `--always-correct` flag forces quality correction on every window for evaluation/benchmarking purposes.

**Q: What's TinyDiarize?**

TinyDiarize is whisper.cpp's built-in speaker-turn detection via the `--tdrz` flag (mapped to `--tiny-diarize` in franken_whisper). It injects speaker-turn tokens during inference without requiring a separate diarization pipeline, HuggingFace token, or pyannote models. It's less accurate than full diarization but much simpler — zero additional dependencies.

**Q: What's a replay pack?**

A self-contained directory of four JSON artifacts (`env.json`, `manifest.json`, `repro.lock`, `tolerance_manifest.json`) that capture everything needed to reproduce and analyze a specific run. Unlike the replay envelope (which is a hash summary), the replay pack includes the routing evidence, conformance tolerances, environment snapshot, and request parameters. Use it for post-hoc debugging: "why did run X produce different output than run Y?"

**Q: Can franken_whisper transcribe video files?**

Yes. Any video file that ffmpeg can decode (MP4, MKV, AVI, MOV, WebM, etc.) is handled automatically. The ffmpeg fallback extracts the audio track using the `-vn` flag. The built-in Rust decoder handles audio-only files; ffmpeg is invoked only when needed for video or exotic codecs.

**Q: What's the difference between `sync export-jsonl` and incremental export?**

Full export dumps every run in the database. Incremental export tracks a cursor (`exports/cursor.json`) and only exports runs created since the last export. Use incremental for regular backups of large databases. Both produce the same JSONL format with SHA-256 checksums in the manifest.

**Q: How does the evidence ledger work?**

Every routing decision records a `RoutingEvidenceLedgerEntry` in a 200-entry circular buffer. Each entry contains: the decision ID, trace ID, observed availability state, chosen backend action, recommended ordering, whether fallback was active, posterior snapshot (Beta distribution parameters), calibration score, Brier score, e-process value, confidence interval width, audio duration bucket, diarize flag, and the actual outcome. This ledger is queryable via `robot routing-history` and included in stage event payloads.

**Q: How do I contribute?**

See the section below.

---

## What Makes This Different

The whisper ecosystem has ~95k stars on OpenAI's repo, ~47k on whisper.cpp, ~20k on WhisperX, ~14k on faster-whisper. There are dozens of wrappers, pipelines, and API services. Here's what franken_whisper does that none of them do.

### No other tool learns which backend to use

WhisperS2T, transcribe-anything, and WhisperLive let you *pick* a backend. franken_whisper *learns* which backend to use based on observed outcomes. The Bayesian router maintains Beta-distribution posteriors per backend, updates them after every run, tracks calibration via Brier scoring on a 50-observation sliding window, and falls back to deterministic priority when the model is uncertain (Brier > 0.35, calibration < 0.3, or < 5 observations). Every routing decision is recorded in a 200-entry evidence ledger with full posterior snapshots, making post-hoc analysis trivial. No other whisper tool does decision-theoretic backend selection.

### No other tool validates cross-engine conformance

When you replace one backend with another (or roll out a native Rust engine to replace an external subprocess), how do you know the output is equivalent? franken_whisper's conformance harness compares segment output across engines using a 50ms canonical timestamp tolerance, text matching, speaker label matching, and WER approximation. The 5-stage native rollout governance (shadow → validated → fallback → primary → sole) prevents a buggy engine from silently degrading quality. No other tool has a formal conformance framework.

### No other tool does dual-model speculative streaming

WhisperLive and WhisperLiveKit stream transcripts in real time, but they use a single model with buffering. franken_whisper's speculative streaming runs a fast model and a quality model in parallel on overlapping windows, emits partial transcripts immediately, and then issues corrections when the quality model disagrees. The `CorrectionTracker` adaptively adjusts confirmation thresholds based on observed fast-vs-quality drift statistics. This is architecturally distinct from every other streaming approach in the ecosystem.

### No other tool persists every run with full audit trail

whisper.cpp, WhisperX, faster-whisper — none of them remember what they transcribed yesterday. franken_whisper persists every run to SQLite with the complete request, result, segments, pipeline events, evidence, and replay envelope. You can query history (`runs --format json`), export to JSONL snapshots (full or incremental with cursor), import on another machine, and validate sync integrity with SHA-256 checksums. Run history is a first-class feature, not an afterthought.

### No other tool treats audio as a zero-dependency data type

WhisperX requires ffmpeg. faster-whisper requires ffmpeg. Even whisper.cpp only reads WAV natively. franken_whisper's built-in Rust decoder (symphonia) handles MP3, AAC, FLAC, WAV, OGG, Vorbis, ALAC, and WavPack natively at ~260ms for a 2-minute file — no subprocess, no external binary, no PATH dependency. ffmpeg is only invoked as a fallback for video files and exotic codecs, and even then it's auto-provisioned if missing.

### No other tool is built for agent consumption first

Most whisper tools produce human-readable terminal output and maybe a `--json` flag. franken_whisper's `robot` subcommand is the *primary* interface — sequenced, timestamped NDJSON events with stable schema versioning (v1.0.0), 12 structured error codes, health diagnostics, routing history, and speculation events. Human output is the exception, not the rule. The TTY audio transport protocol enables audio relay over text-only links (PTY, SSH, serial) with integrity verification and adaptive FEC.

### No other safe-Rust ASR orchestrator exists

whisper-rs provides Rust FFI bindings to whisper.cpp (necessarily unsafe). There is no other pure-safe-Rust ASR orchestrator at this scope. franken_whisper enforces `#![forbid(unsafe_code)]` — not `deny` (which can be overridden per-item), but `forbid` (which cannot). Combined with cooperative cancellation tokens, atomic SQLite transactions with retry, bounded finalizer timeouts, and RAII cleanup, the result is a system that never leaves orphaned processes, corrupted databases, or leaked temp files.

---

## About Contributions

Please don't take this the wrong way, but I do not accept outside contributions for any of my projects. I simply don't have the mental bandwidth to review anything, and it's my name on the thing, so I'm responsible for any problems it causes; thus, the risk-reward is highly asymmetric from my perspective. I'd also have to worry about other "stakeholders," which seems unwise for tools I mostly make for myself for free. Feel free to submit issues, and even PRs if you want to illustrate a proposed fix, but know I won't merge them directly. Instead, I'll have Claude or Codex review submissions via `gh` and independently decide whether and how to address them. Bug reports in particular are welcome. Sorry if this offends, but I want to avoid wasted time and hurt feelings. I understand this isn't in sync with the prevailing open-source ethos that seeks community contributions, but it's the only way I can move at this velocity and keep my sanity.

---

## Key Documentation

| Document | Description |
|----------|-------------|
| [`docs/tty-audio-protocol.md`](docs/tty-audio-protocol.md) | Complete TTY audio protocol specification |
| [`docs/tty-replay-guarantees.md`](docs/tty-replay-guarantees.md) | Deterministic replay/framing guarantees for operators |
| [`docs/native_engine_contract.md`](docs/native_engine_contract.md) | Native engine replacement interface contract |
| [`docs/engine_compatibility_spec.md`](docs/engine_compatibility_spec.md) | 50ms timestamp tolerance specification |
| [`docs/conformance-contract.md`](docs/conformance-contract.md) | Cross-engine conformance test contract |
| [`docs/operational-playbook.md`](docs/operational-playbook.md) | Deployment and monitoring guide |
| [`docs/benchmark_regression_policy.md`](docs/benchmark_regression_policy.md) | Performance regression thresholds |
| [`RECOVERY_RUNBOOK.md`](RECOVERY_RUNBOOK.md) | Disaster recovery procedures |
| [`SYNC_STRATEGY.md`](SYNC_STRATEGY.md) | One-way sync semantics |
| [`PROPOSED_ARCHITECTURE.md`](PROPOSED_ARCHITECTURE.md) | System architecture design document |
| [`FEATURE_PARITY.md`](FEATURE_PARITY.md) | Legacy feature parity matrix |

---

## License

MIT License with OpenAI/Anthropic Rider. See [LICENSE](LICENSE) for the full text.

In short: standard MIT terms apply, with an additional restriction that no rights are granted to OpenAI, Anthropic, or their affiliates without express prior written permission from the author. This rider must be preserved in all copies and derivative works.
