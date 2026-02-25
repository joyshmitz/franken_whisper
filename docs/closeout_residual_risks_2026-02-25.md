# Closeout Residual Risks Snapshot (2026-02-25, refreshed 20:44 UTC)

This packet documents current residual risk after bead reconciliation
(`bd-1a1`, `bd-244`, `bd-217`).

## Scope

- Repository: `franken_whisper` (primary), with active cross-repo dependency on
  `frankensqlite`.
- Active status inputs:
  - `bd-1a1`: blocked
  - `bd-244`: in progress
  - `bd-217`: in progress (closeout synchronization)

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

### RR-02: Golden-checksum drift lane is blocked by missing corpus artifact

- Severity: high
- Evidence:
  - offloaded command:
    `rch exec -- cargo test -p fsqlite-harness --test bd_1lsfu_2_core_sql_golden_checksums -- --nocapture`
  - failure:
    `bead_id=bd-1lsfu.2 case=fuzz_dir_canonicalize ... No such file or directory`.
  - missing path:
    `fuzz/corpus/fuzz_sql_parser`.
- Impact:
  - parser/planner/execution mismatch counts cannot be generated;
  - checksum refresh cannot be safely attributed or validated.
- Mitigation:
  - restore/generate corpus directory before rerunning checksum lane.
- Exit criteria:
  - test advances to mismatch diff or pass state (no corpus-path failure).

### RR-03: SSI runtime containment remains open

- Severity: medium-high
- Evidence:
  - `bd-244` remains `in_progress` for CI-scale runtime containment and gate closure.
- Impact:
  - practical CI runtime envelope is not yet proven for the SSI-scale lane.
- Mitigation:
  - finish runtime validation + guardrails and rerun mandatory gates via `rch`.
- Exit criteria:
  - both `ci_scale` and `single_writer_smoke` outcomes documented with practical runtime evidence.

## Remaining Work Boundary

- Completed: tracker/doc synchronization and ownership clarity (`bd-217` workstream).
- Blocked: `bd-1a1` pending corpus source restoration.
- Active: `bd-244` runtime containment + mandatory-gate completion.

## Next Steps (Concrete)

1. Unblock `bd-1a1` by restoring `fuzz/corpus/fuzz_sql_parser`.
2. Complete `bd-244` and publish runtime + gate evidence.
3. Close `bd-217` once packet docs remain synchronized with the above outcomes.
