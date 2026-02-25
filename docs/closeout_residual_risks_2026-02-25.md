# Closeout Residual Risks Snapshot (2026-02-25)

This packet satisfies TODO tracker item `T8.3` by documenting concrete,
currently active residual risks with code-anchored evidence and next actions.

## Scope

- Repository: `franken_whisper`
- Snapshot date: `2026-02-25`
- Source inputs:
  - `src/orchestrator.rs` placeholder-stage implementations
  - active beads from `br`/`bv` (`bd-xp7`, `bd-1q4`)
  - cross-repo blocker notes recorded in
    `TODO_IMPLEMENTATION_TRACKER.md:1247-1249`

## Risk Register

### RR-01: Placeholder pipeline stages remain in production path

- Severity: high
- Evidence:
  - source separation is explicitly placeholder/no-op style:
    `src/orchestrator.rs:2399-2425`
  - punctuation restoration is rule-based placeholder:
    `src/orchestrator.rs:2516-2517`
  - diarization uses simplified placeholder embeddings/notes:
    `src/orchestrator.rs:2672-2742`
- Impact:
  - feature-parity claims for Demucs/TitaNet/neural punctuation remain partial;
  - downstream quality and speaker-label reliability can diverge from intended
    architecture under noisy/complex audio.
- Mitigation path:
  - replace placeholder implementations with model-backed adapters;
  - keep deterministic fallback/no-regression tests for cancellation and stage
    budgets before promotion.
- Exit criteria:
  - placeholder notes removed from stage outputs;
  - model-backed stage tests pass for happy-path + error + cancellation.

### RR-02: Determinism invariant coverage is still in flight

- Severity: medium-high
- Evidence:
  - `bd-xp7` is currently `in_progress` (owner: `CrimsonAspen`):
    stage/event-order determinism tests for cancellation/failure.
- Impact:
  - regression window remains for subtle stage-order/event-order drift in edge
    paths until the dedicated tests land.
- Mitigation path:
  - complete `bd-xp7` and keep tests in CI-facing gates.
- Exit criteria:
  - `bd-xp7` closed with test evidence and stable reproducible results.

### RR-03: Mandatory quality-gate publication is still in flight

- Severity: medium
- Evidence:
  - `bd-1q4` is currently `in_progress` (owner: `MaroonSalmon`) for remote
    quality gates via `rch`.
  - AGENTS gate requirements remain strict (`fmt`, `check`, `clippy`, `test`).
- Impact:
  - release-readiness and hardening status cannot be declared complete until
    gate outcomes are published.
- Mitigation path:
  - finish `bd-1q4` with pass/fail matrix and blocker detail capture.
- Exit criteria:
  - gate summary published with command-level outcomes and blocker provenance.

### RR-04: Cross-repo test-runtime and golden-drift blockers remain unresolved

- Severity: medium
- Evidence:
  - `TODO_IMPLEMENTATION_TRACKER.md:1248` documents extremely long-running SSI
    test on host (`ssi_serialization_correctness_ci_scale`).
  - `TODO_IMPLEMENTATION_TRACKER.md:1249` documents checksum drift failures in
    `fsqlite-harness` due pre-existing parser/codegen dirty state.
- Impact:
  - full cross-repo completion signal can remain blocked by environment/runtime
    debt outside this repository's direct code path.
- Mitigation path:
  - keep this risk visible in closeout packets (`T8.1/T8.2/T8.4`);
  - separate repo-local completion from cross-repo blocker closure.
- Exit criteria:
  - long SSI test runtime reduced to practical envelope and checksum drift
    reconciled/attributed with clean rerun evidence.

## Already Mitigated in This Session

- TTY replayability guidance is present in
  `docs/tty-replay-guarantees.md` and was revalidated against current
  implementation semantics while closing bead `bd-2kt`.

## Next Execution Packet (Concrete)

1. Close `bd-xp7` with deterministic test artifacts and touched-file summary.
2. Close `bd-1q4` with `rch` gate matrix and blocker provenance.
3. Publish `T8.1` cross-repo change summary tying changed files to the risk
   register above and explicitly marking blocked vs complete items.
