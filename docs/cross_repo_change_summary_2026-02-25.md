# Cross-Repo Change Summary (2026-02-25)

This packet satisfies TODO tracker item `T8.1` by publishing file-level
change context across the active `franken_whisper` workflow and its immediate
cross-repo dependencies.

## Scope

- Primary repo: `/data/projects/franken_whisper`
- Cross-repo context: `/data/projects/frankensqlite`, `/data/projects/asupersync`
- Evidence sources:
  - tracker rows in `TODO_IMPLEMENTATION_TRACKER.md`
  - closeout packets in `docs/closeout_residual_risks_2026-02-25.md` and
    `docs/next_execution_packet_2026-02-25.md`
  - remote gate run via `scripts/run_quality_gates_rch.sh`

## File-Level Summary

| Repo | Status | Files | Notes |
|---|---|---|---|
| `franken_whisper` | completed | `docs/engine_compatibility_spec.md` | Added/retained explicit compatibility envelope release-gate execution checklist (`9.6`). |
| `franken_whisper` | completed | `docs/tty-replay-guarantees.md` | Added/retained operator-facing verification hooks for replay/framing guarantees. |
| `franken_whisper` | completed | `TODO_IMPLEMENTATION_TRACKER.md` | Reconciled `T7.5`/`T7.7` as done, marked `T7.6` in progress (`bd-xp7`), and published `T8.2` gate outcomes. |
| `franken_whisper` | completed | `docs/closeout_residual_risks_2026-02-25.md` | Residual-risk packet with code-anchored risk register and mitigations. |
| `franken_whisper` | completed | `docs/next_execution_packet_2026-02-25.md` | Next execution packet with explicit bead scope and verification criteria. |
| `franken_whisper` | in progress | `src/orchestrator.rs`, `tests/` | `bd-xp7` (owner `CrimsonAspen`): determinism/event-order tests + safety guards; still actively changing. |
| `frankensqlite` | prior completed context | `/data/projects/frankensqlite/crates/fsqlite-parser/src/parser.rs` | Prior unblock note captured in tracker (`err_here` -> `err_msg`) to restore upstream compile flow during cross-repo gates. |
| `asupersync` | active blocker signal | `/data/projects/asupersync/src/runtime/scheduler/three_lane.rs` | Remote `rch` run observed type-mismatch compile failures on one worker attempt (`Ok(Some(_))` vs `Result<usize, io::Error>` shape). |

## Quality-Gate Snapshot (Remote `rch`)

Executed from `franken_whisper` using `scripts/run_quality_gates_rch.sh`:

- `cargo fmt --check`: pass
- `cargo check --all-targets`: pass (after worker retry)
- `cargo clippy --all-targets -- -D warnings`: pass
- `cargo test`: fail at compile time in `src/orchestrator.rs:1111` with `E0277`
  (`spawn_blocking` future `Unpin` bound at `timeout(...).await`) while
  `bd-xp7` changes are in flight

## Completed vs In-Progress vs Blocked

- Completed:
  - documentation/tracker reconciliation for compatibility envelope and replay
    guarantees (`T7.5`, `T7.7`)
  - quality-gate publication row (`T8.2`) with pass/fail evidence
- In progress:
  - orchestrator determinism hardening (`bd-xp7`)
- Blocked (current snapshot):
  - full `cargo test` completion pending resolution of the orchestrator compile
    error introduced during in-progress orchestrator edits

