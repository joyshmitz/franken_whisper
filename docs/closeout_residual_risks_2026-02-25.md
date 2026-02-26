# Closeout Residual Risks Snapshot (2026-02-25, refreshed 23:50 UTC)

This packet documents current residual risk after U1 closure and packet reconciliation.

## Scope

- Repository: `franken_whisper` (primary), with active cross-repo dependency on
  `frankensqlite`.
- Active status inputs:
  - `bd-1a1`: closed
  - `bd-244`: in progress
  - `bd-217`: closed

## Risk Register

### RR-01: Placeholder pipeline stages remain in production path

- Severity: high
- Evidence:
  - source separation placeholder behavior in `src/orchestrator.rs`;
  - punctuation/diarization stages still include simplified placeholder paths.
- Impact:
  - parity expectations for separation/punctuation/diarization remain partial;
  - quality drift risk remains on complex/noisy audio.
- Mitigation:
  - replace placeholder branches with model-backed adapters behind deterministic
    fallback contracts.
- Exit criteria:
  - placeholder stage notes removed and replacement tests pass (happy/error/cancel paths).

### RR-02: SSI runtime containment remains open

- Severity: medium-high
- Evidence:
  - `bd-244` remains `in_progress` for CI-scale runtime containment and gate closure.
- Impact:
  - practical CI runtime envelope and correctness closure are not yet finalized.
- Mitigation:
  - finish runtime validation + guardrails and rerun mandatory gates via `rch`.
- Exit criteria:
  - both `ci_scale` and `single_writer_smoke` outcomes documented with practical runtime evidence.

### RR-03: Cross-repo local-only artifact dependence can regress reproducibility

- Severity: medium
- Evidence:
  - U1 unblock required restoring ignored local-only path `fuzz/corpus/fuzz_sql_parser`.
- Impact:
  - workers/new hosts may regress to non-reproducible `fuzz_dir_canonicalize` failures
    if local corpus prerequisites are absent.
- Mitigation:
  - keep corpus-setup prerequisite explicit in execution packets and operator notes.
- Exit criteria:
  - deterministic corpus bootstrap documented and automated for worker/offload workflows.

## Remaining Work Boundary

- Completed: `bd-1a1` checksum lane closure and `bd-217` tracker/doc synchronization.
- Blocked: none in the current repo-level bead snapshot.
- Active: `bd-244` runtime containment + mandatory-gate completion.

## Next Steps (Concrete)

1. Complete `bd-244` and publish runtime + gate evidence.
2. Document/automate corpus bootstrap for worker/offload reproducibility.
3. Re-run `bv --robot-triage` for next strictly remaining work.
