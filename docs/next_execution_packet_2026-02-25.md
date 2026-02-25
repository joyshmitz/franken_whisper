# Next Execution Packet (2026-02-25)

This packet fulfills TODO tracker item `T8.4` by defining concrete remaining
work with owner expectations, file scope, and verification criteria.

## Remaining Open Work (Current Snapshot)

### EP-01: Determinism Invariants for Cancellation/Failure Paths

- Bead: `bd-xp7`
- Current owner: `CrimsonAspen`
- Status: `in_progress`
- Intended scope (from bead): stage-order determinism and robot event-order
  replay invariants under cancellation/failure; expected touches in
  `src/orchestrator.rs` and `tests/`.
- Verification criteria:
  - targeted tests for stage/event ordering exist and are deterministic across
    repeated runs;
  - tests cover both cancellation path and failure path;
  - no schema-order regressions in robot event assertions.

### EP-02: Remote Quality-Gate Outcome Publication

- Bead: `bd-1q4`
- Current owner: `MaroonSalmon`
- Status: `in_progress`
- Scope: run remote gates through `rch` and publish pass/fail matrix.
- Required command set (remote/offloaded only):
  - `cargo fmt --check`
  - `cargo check --all-targets`
  - `cargo clippy --all-targets -- -D warnings`
  - `cargo test`
- Verification criteria:
  - each gate outcome recorded as pass/fail;
  - if any fail, blocker provenance captured with file/error pointers;
  - runtime or worker instability explicitly distinguished from code defects.

### EP-03: Cross-Repo Change Summary Packet

- TODO item: `T8.1` (currently unclaimed)
- Recommended owner: next available agent after EP-01/EP-02 publish.
- Scope:
  - produce a compact file-level change summary for this repo and dependent
    cross-repo context referenced in tracker.
- Verification criteria:
  - references include absolute file paths or clear repo-relative pointers;
  - summary separates completed, in-progress, and blocked items.

## Suggested Execution Order

1. Finish EP-01 (`bd-xp7`) to lock deterministic behavior coverage.
2. Finish EP-02 (`bd-1q4`) to establish current quality-gate truth.
3. Execute EP-03 (`T8.1`) using outputs from EP-01 and EP-02.

## Done/Blocked Decision Rule

- Mark a packet item `done` only when both implementation and evidence are
  present.
- Mark `blocked` only with a concrete blocker artifact (command output, worker
  failure, or dependency drift evidence), not with a generic status note.

## Coordination Notes

- If Agent Mail state resets, re-register agent identities and re-announce open
  bead ownership before editing to avoid collisions.
- Continue using `br`/`bv --robot-*` as source of truth for issue state.
