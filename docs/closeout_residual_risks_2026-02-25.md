# Closeout Residual Risks Snapshot (2026-02-25, refreshed 23:50 UTC)

This packet documents current residual risk after U1 closure and packet reconciliation.

## Scope

- Repository: `franken_whisper` (primary), with active cross-repo dependency on
  `frankensqlite`.
- Active status inputs:
  - `bd-1a1`: closed
  - `bd-244`: closed
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

### RR-02: Cross-repo local-only artifact dependence can regress reproducibility

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

- Completed: `bd-1a1` checksum lane closure, `bd-244` SSI runtime containment + gate closure, and `bd-217` tracker/doc synchronization.
- Blocked: none in the current repo-level bead snapshot.
- Active: none in the current repo-level bead snapshot.

## Next Steps (Concrete)

1. Document/automate corpus bootstrap for worker/offload reproducibility.
2. Re-run `bv --robot-triage` for next strictly remaining work.
3. Open only new beads backed by current evidence rather than stale packet notes.
