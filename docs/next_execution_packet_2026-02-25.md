# Next Execution Packet (2026-02-25, refreshed 20:44 UTC)

This packet is the current execution source-of-truth for remaining cross-repo
work after bead reconciliation (`bd-1a1`, `bd-244`, `bd-217`).

## Remaining Open Work (Current Snapshot)

### EP-01: Unblock `bd-1a1` Golden-Checksum Drift Quantification

- Bead: `bd-1a1`
- Status: `blocked`
- Owner: `PearlAnchor`
- Scope:
  - restore/provide required fuzz corpus path for `bd_1lsfu_2`:
    `fuzz/corpus/fuzz_sql_parser`;
  - rerun checksum gate and collect parser/planner/execution mismatch counts;
  - perform controlled refresh only after mismatch evidence is captured.
- Evidence artifact:
  - `rch exec -- cargo test -p fsqlite-harness --test bd_1lsfu_2_core_sql_golden_checksums -- --nocapture`
  - current failure: `bead_id=bd-1lsfu.2 case=fuzz_dir_canonicalize ... No such file or directory`.
- Verification criteria:
  - checksum test reaches `case=checksum_mismatch` or pass state (not corpus path failure);
  - quantified mismatch breakdown is recorded;
  - update-command path is executed only with explicit artifact capture.

### EP-02: Complete `bd-244` SSI Runtime Containment + Mandatory Gate Closure

- Bead: `bd-244`
- Status: `in_progress`
- Owner: `TealCove`
- Scope:
  - validate practical runtime envelope for `ssi_serialization_correctness_ci_scale`
    and `ssi_serialization_correctness_single_writer_smoke`;
  - add guardrails if CI-scale remains non-practical;
  - rerun mandatory gates after U1/U2 closure.
- Required command style:
  - all cargo gates offloaded via `rch exec -- ...`.
- Verification criteria:
  - runtime findings + guardrail decisions documented with command evidence;
  - mandatory gate matrix reported with explicit pass/fail + blocker provenance.

### EP-03: Close `bd-217` Reconciliation Packet

- Bead: `bd-217`
- Status: `in_progress`
- Owner: `PearlAnchor`
- Scope:
  - keep tracker/doc packet aligned to real bead/mail state;
  - ensure only genuinely remaining work appears as open.
- Verification criteria:
  - `TODO_IMPLEMENTATION_TRACKER.md` reflects real status/evidence;
  - cross-repo summary + residual-risk + next-execution docs are synchronized;
  - bead is closed once synchronization is complete.

## Suggested Execution Order

1. Resolve EP-01 blocker precondition (corpus source path) so U1 can emit real drift diagnostics.
2. Finish EP-02 runtime containment and gate reruns.
3. Close EP-03 packet once EP-01/EP-02 states are stable.

## Done/Blocked Decision Rule

- Mark `done` only with command evidence and updated documentation.
- Mark `blocked` only with concrete artifact evidence (error text + command + path).

## Coordination Notes

- Continue `br` as issue source of truth and `bv --robot-*` for prioritization.
- Continue Agent Mail thread updates per bead id to avoid duplicate cross-repo edits.
