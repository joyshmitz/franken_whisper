# AGENTS.md — franken_whisper

> Guidelines for AI coding agents working in this Rust codebase.

---

## RULE 0 - FUNDAMENTAL OVERRIDE

If the user gives a direct instruction, obey it.

---

## RULE 1 - NO FILE DELETION

Never delete files or directories without explicit written permission from the user.

This includes:
- `rm`, `rm -rf`, `unlink`
- `git reset --hard`
- `git clean -fd`
- any scripted bulk-delete

If unsure whether a command is destructive, stop and ask.

---

## Branch Policy

- Primary branch is `main`.
- Do not reference `master` in docs/scripts.
- If release instructions require sync, push `main:master` after `main`.

---

## Project Mission

`franken_whisper` is a native Rust, memory-safe, high-performance speech system that merges ideas from:
- `legacy_whispercpp` (fast local inference + streaming + VAD)
- `legacy_insanely_fast_whisper` (GPU-first batching + ergonomic CLI)
- `legacy_whisper_diarization` (forced alignment + diarization + punctuation restoration)

And integrates deeply with:
- `/dp/asupersync` for cancel-correct orchestration
- `/dp/frankentorch` and optionally `/dp/frankenjax` for accelerated compute backends
- `/dp/frankensqlite` for durable local state and telemetry
- `/dp/frankentui` for optional human TUI

---

## Product Shape

The project must be both:
1. A reusable Rust library for embedding ASR/diarization pipelines.
2. A standalone CLI binary with:
   - robot mode (agent-first, structured machine output)
   - optional TUI mode for humans

Input requirements:
- Any common audio file format (automatic ffmpeg normalization)
- Microphone or line-in capture where supported
- Optional low-bandwidth TTY/PTY streaming mode (see architecture docs)

---

## Porting Workflow (Spec-First)

Use this workflow order:
1. `PLAN_TO_PORT_WHISPER_STACK_TO_RUST.md`
2. `EXISTING_LEGACY_WHISPER_STRUCTURE.md`
3. `PROPOSED_ARCHITECTURE.md`
4. `FEATURE_PARITY.md`

Implementation should follow spec documents, not ad-hoc copying from legacy code.

---

## SQLite + JSONL Workflow

Source of truth: SQLite.

JSONL is for auditability, git-friendly backup, and recovery.

Hard rules:
- One-way sync at a time (locked + atomic).
- No two-way magic merge.
- Maintain explicit version markers.
- Keep recovery procedures documented in:
  - `SYNC_STRATEGY.md`
  - `RECOVERY_RUNBOOK.md`

---

## Alien-Artifact Engineering Contract

For runtime/adaptive decisions, include:
- explicit state space
- explicit actions
- loss matrix
- posterior/confidence terms
- calibration metric
- deterministic fallback trigger
- evidence ledger artifact

No adaptive controller should ship without conservative fallback.

---

## Code Editing Discipline

- No destructive edits.
- No mass regex scripts that rewrite code blindly.
- Add new files only when they represent a real architectural unit.
- Keep APIs explicit and typed.
- Prefer deterministic behavior and replayability.

---

## Toolchain

- Rust 2024 edition.
- Nightly toolchain (`rust-toolchain.toml`).
- `#![forbid(unsafe_code)]`.
- Cargo only.

---

## Mandatory Checks After Substantive Changes

```bash
cargo fmt --check
cargo check --all-targets
cargo clippy --all-targets -- -D warnings
cargo test
```

If any check fails, fix root causes before handing off.

---

## Testing Policy

Each module should include unit tests for:
- happy path
- edge cases
- error handling

Integration/e2e tests should cover:
- file-based transcription flow
- robot mode JSON/NDJSON contracts
- SQLite + JSONL sync/recovery
- ffmpeg normalization behavior and failure paths

---

## Agent Ergonomics Requirements

Robot mode should be:
- stable schema
- deterministic where possible
- explicit error codes
- line-oriented output (`jsonl`/`ndjson`)
- easy to pipe into other tools

Do not mix human decoration with machine output in robot mode.

---

## Multi-Agent / Beads / BV

If using `br` / `bv`:
- Use `br` for issue CRUD.
- Use only `bv --robot-*` variants (never bare `bv`).

---

## Session Completion

Before finishing:
1. Summarize what changed.
2. List quality gates run and results.
3. Document remaining risks/gaps.
4. Propose concrete next steps.
