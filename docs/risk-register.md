# Risk Register

## Scale

- Impact: `low`, `medium`, `high`, `critical`
- Likelihood: `low`, `medium`, `high`

## Active Risks

| Risk ID | Risk | Impact | Likelihood | Mitigation | Evidence / Tests | Fallback Trigger |
|---|---|---|---|---|---|---|
| FW-R1 | External backend binaries unavailable (`whisper-cli`, `insanely-fast-whisper`, diarization script, ffmpeg) | high | medium | Runtime diagnostics + explicit availability checks + deterministic backend fallback ordering | `robot backends` payload, backend readiness checks in `src/backend/mod.rs` | Auto policy skips unavailable backend and emits warnings/stage evidence |
| FW-R2 | Dairization path fails due missing HF token | high | medium | Require token for insanely-fast diarization readiness (`FRANKEN_WHISPER_HF_TOKEN` or `HF_TOKEN`) | backend readiness logic + diagnostics token flags | Skip insanely-fast diarization candidate, fall back to next deterministic backend |
| FW-R3 | Long-running stage causes hung pipeline | high | medium | Explicit per-stage budgets + timeout error coding + command timeout enforcement | orchestrator budget event + timeout tests + process timeout path | Emit `*.timeout` stage code and terminate run conservatively |
| FW-R4 | Sync lock stale/corrupt file blocks operations | medium | medium | Archive stale/corrupt lock files and reacquire lock deterministically | `SyncLock` tests for stale/corrupt archival in `src/sync.rs` | Lock acquisition archives invalid lock then retries once |
| FW-R5 | JSONL snapshot corruption/import mismatch | high | low | Manifest schema/version checks + SHA-256 file checks + transactional import | sync import/export tests + checksum mismatch test | Reject import and keep DB unchanged (rollback) |
| FW-R6 | Robot contract drift breaks agent consumers | critical | low | Stable schema tests + deterministic stage code naming discipline | robot envelope tests + stage sequencing tests | Block release until schema compatibility is restored |
| FW-R7 | Stage timeout handling introduces non-deterministic behavior | high | low | Deterministic timeout mapping (`stage.timeout`) + static execution safe mode | orchestrator timeout-code tests | Revert to static deterministic stage policy and fail closed |
| FW-R8 | TTY/PTY compressed transport fidelity issues | medium | medium | Keep mode explicitly prototype, maintain roundtrip tests and clear docs | `src/tty_audio.rs` tests | Disable TTY transport path in production pipelines if corruption detected |
| FW-R9 | Native-engine parity drift (token/timestamp/speaker/calibration mismatch) | critical | medium | Define explicit compatibility envelope + enforce conformance harness before parity claims | conformance invariants + corpus replay comparisons (target packet) | Freeze rollout to bridge-safe mode until conformance thresholds are met |
| FW-R10 | GPU device/stream lifecycle leaks under cancellation | high | medium | Explicit device ownership in runtime contract + cancellation-safe teardown semantics | orchestration device telemetry + cancellation integration tests (target packet) | Auto-fallback to CPU/bridge-safe mode when device teardown confidence drops |

## Review Cadence

- Re-evaluate risks at the end of each active execution packet.
- Any `critical` risk requires explicit mention in handoff summary.
- New adaptive logic must add a risk row before merge.
