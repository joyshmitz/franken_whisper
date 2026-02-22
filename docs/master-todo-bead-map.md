# Master TODO -> Bead Map

This map ties `TODO_IMPLEMENTATION_TRACKER.md` packets to executable `br` issues.
It is intentionally dependency-oriented and mirrors packet `P` as of 2026-02-22 (closure snapshot).

## Status Legend

- `planned`: not started
- `in_progress`: active implementation
- `done`: completed and validated
- `blocked`: waiting on dependency

## Active Packet Mapping (O/P)

| Packet Scope | Bead IDs | Primary Artifacts | Status |
|---|---|---|---|
| P1 Conformance tolerance lock | `bd-1rj.7` | `src/conformance.rs`, `docs/engine_compatibility_spec.md`, `README.md`, conformance tests | done |
| P1 TTY control packet epic | `bd-30v` | tracker packet coordination + dependency root for control-plane tasks | done |
| P1 TTY control CLI surface | `bd-2xe.4` | `src/cli.rs`, `src/main.rs`, `src/tty_audio.rs` | done |
| P1 Retransmit loop automation | `bd-2xe.5` | `src/tty_audio.rs`, CLI dispatch, control parsers | done |
| P1 Control-plane docs | `bd-2xe.6` | `README.md`, `docs/tty-audio-protocol.md` | done |
| P1/P2 Quality-gate closeout | `bd-3pf.19` | gate command outputs + tracker evidence | done |
| P2 Control/retransmit tests | `bd-3pf.13` | `tests/cli_integration.rs`, `tests/backend_mock_tests.rs`, control fixtures | done |
| P2 E2E enablement | `bd-3pf.14` | `tests/e2e_pipeline_tests.rs`, test harness scripts | done |
| P2 Stage order + error codes | `bd-3pf.15` | robot contract + orchestration test suites | done |
| P2 TTY protocol integrity tests | `bd-3pf.16` | `tests/cli_integration.rs`, protocol fixtures | done |
| P2 GPU telemetry tests | `bd-3pf.17` | acceleration/orchestrator telemetry tests | done |
| P2 Bench extensions | `bd-3pf.5.1` | `benches/*` | done |
| P3 Native replacement spec/rollout | `bd-1rj.8` | architecture + conformance/rollout docs | done |
| P3 Whisper native pilot | `bd-1rj.9` | `src/backend/*` native engine implementation | done |
| P3 Insanely-fast native pilot | `bd-1rj.10` | `src/backend/*` native engine implementation | done |
| P3 Diarization native pilot | `bd-1rj.11` | `src/backend/*` native diarization pipeline | done |
| P3 Cross-engine CI comparator | `bd-3pf.18` | conformance harness + CI gate artifacts | done |

Verification note:
- `bd-3pf.5.1` sync bench baselines were captured on 2026-02-22 and recorded in `docs/benchmark_regression_policy.md`.

## Dependency Highlights

- `bd-2xe.4`, `bd-2xe.5`, `bd-2xe.6` are children of `bd-30v`.
- `bd-2xe.5` depends on `bd-2xe.4`.
- `bd-2xe.6` depends on `bd-2xe.4`, `bd-2xe.5`.
- `bd-3pf.13` depends on `bd-2xe.4`, `bd-2xe.5`.
- `bd-3pf.15` depends on `bd-3pf.14`.
- `bd-3pf.16` depends on `bd-2xe.4`, `bd-2xe.5`.
- `bd-3pf.17` depends on `bd-38c.5`.
- `bd-3pf.5.1` depends on `bd-2xe.4`.
- `bd-1rj.9`, `bd-1rj.10`, `bd-1rj.11` depend on `bd-1rj.8`, `bd-1rj.7`.
- `bd-3pf.18` depends on `bd-1rj.7`, `bd-1rj.8`, `bd-1rj.9`, `bd-1rj.10`, `bd-1rj.11`.
- `bd-3pf.19` depends on all P1/P2 closure work listed above.

## Rules

- Every bead row must map to concrete code/doc/test artifacts.
- No packet is closed until mandatory quality gates pass.
- `br dep cycles --json` must remain empty after dependency updates.
