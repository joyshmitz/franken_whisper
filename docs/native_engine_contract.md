# Native-Engine Replacement Contract

## Purpose

This document defines the executable contract that every native Rust engine implementation must satisfy before it can replace a subprocess bridge adapter in production. It covers trait compliance, conformance invariants, shadow-run comparison methodology, rollout gates, and deterministic fallback policy.

This contract is referenced by: bd-1rj.9 (whisper.cpp), bd-1rj.10 (insanely-fast), bd-1rj.11 (diarization), bd-3pf.18 (cross-engine conformance comparator).

## 1. Engine Trait Compliance

Every native engine must implement the `Engine` trait (`src/backend/mod.rs`):

| Method | Requirement |
|---|---|
| `name()` | Stable human-readable identifier. Must be distinct from the bridge adapter it replaces (e.g. `"whisper.cpp-native"` vs `"whisper.cpp"`). |
| `kind()` | Returns the **same** `BackendKind` variant as the bridge adapter it replaces. |
| `capabilities()` | Must declare at least the capabilities the bridge adapter declares. May declare additional capabilities. |
| `is_available()` | Must return `true` when the engine's compiled-in dependencies are functional. Must not shell out to check external binaries. |
| `run()` | Produces a `TranscriptionResult` satisfying all invariants in `docs/engine_compatibility_spec.md`. |

### 1.1 StreamingEngine (optional)

Native engines that support incremental segment delivery should also implement `StreamingEngine`. This is required for whisper.cpp replacement (the bridge adapter already supports streaming). Implementations must:

- Deliver segments in monotonic timestamp order.
- Never deliver a segment that later changes (no retraction).
- Call `on_segment` for each segment before `run_streaming` returns the final result.
- The final `TranscriptionResult` must contain the union of all streamed segments.

### 1.2 Naming Convention

Native engine names follow the pattern `"<backend>-native"` (e.g. `"whisper.cpp-native"`, `"insanely-fast-native"`). The bridge adapter retains the undecorated name (e.g. `"whisper.cpp"`). This allows shadow-run comparison to distinguish engine identity in `ReplayEnvelope.backend_identity`.

## 2. Conformance Invariants

All invariants from `docs/engine_compatibility_spec.md` apply. Key enforcement points:

### 2.1 Segment Invariants (runtime)

Validated by `conformance::validate_segment_invariants()` in the orchestrator after the backend stage:

| Invariant | Threshold |
|---|---|
| Timestamp ordering | `end_sec >= start_sec - 1e-6` |
| Monotonicity (non-overlap) | `start_sec >= prev_end_sec - 1e-6` (default policy) |
| Confidence range | `0.0 <= confidence <= 1.0` (when present) |
| Speaker label | Non-empty after trim (when present) |

Violation triggers `backend.contract_violation` event and fails closed.

### 2.2 Cross-Engine Tolerance (shadow-run)

Validated by `conformance::compare_segments_with_tolerance()`:

| Metric | Tolerance |
|---|---|
| Timestamp drift | `CANONICAL_TIMESTAMP_TOLERANCE_SEC` (50ms, `src/conformance.rs`) |
| Text parity | Exact match (default). Normalized comparison for stochastic paths. |
| Speaker labels | Inexact match by default (`require_speaker_exact = false`). |
| Segment count | Must match (length_mismatch = false). |

### 2.3 Replay Determinism

Each run produces a `ReplayEnvelope` with SHA-256 hashes. For deterministic engines:
- Same input WAV + same engine version must produce identical `output_payload_hash`.
- The `backend_identity` and `backend_version` must be stable across runs.

## 3. Conformance Corpus

### 3.1 Corpus Structure

Golden corpus fixtures live in `tests/fixtures/conformance/corpus/`. Each fixture is a JSON file:

```json
{
  "name": "golden_short_utterance_bridge_cross_engine",
  "tags": ["short_utterance", "bridge"],
  "tolerance": {
    "timestamp_tolerance_sec": 0.05,
    "require_text_exact": true,
    "require_speaker_exact": false
  },
  "pair_drift_caps": {
    "max_timestamp_violations": 0,
    "max_text_mismatches": 0,
    "max_speaker_mismatches": 0,
    "allow_length_mismatch": false
  },
  "canonical": [ ... ],
  "engines": [
    {
      "engine": "whisper_cpp_bridge_short_utterance",
      "format": "json_payload",
      "artifact": "tests/fixtures/golden/whisper_cpp_short_utterance_output.json",
      "expect_within_tolerance": true
    },
    {
      "engine": "insanely_fast_bridge_short_utterance",
      "format": "json_payload",
      "artifact": "tests/fixtures/golden/insanely_fast_short_utterance_output.json",
      "expect_within_tolerance": true
    },
    {
      "engine": "whisper_diarization_bridge_short_utterance",
      "format": "diarization_srt",
      "artifact": "tests/fixtures/golden/diarization_short_utterance_output.srt",
      "expect_within_tolerance": true
    }
  ]
}
```

### 3.2 Minimum Corpus Coverage

Before a native engine can replace its bridge adapter, the executable harness
must satisfy all of these gates:

| Gate | Requirement | Enforcement |
|---|---|---|
| Corpus size | `>= 10` fixtures | `MIN_CORPUS_FIXTURES` in `tests/conformance_harness.rs` |
| Scenario tags | Must include `long_form`, `multilingual`, `multi_speaker_overlap`, `silence_heavy`, `noisy_environment`, `code_switching`, `short_utterance`, `variable_volume_overlap` | `REQUIRED_CORPUS_TAGS` in `tests/conformance_harness.rs` |
| Backend-family presence | For each family (`whisper_cpp`, `insanely_fast`, `whisper_diarization`), at least one bridge + one native engine fixture must be present | `per_backend_family_coverage` gate summary |
| Pairwise drift caps | Every engine pair must remain within per-fixture `pair_drift_caps` | `pair_report_within_caps` gate check |

Harness outputs a machine-readable gate summary in:

`target/conformance/bridge_native_conformance_bundle.json`

### 3.3 Canonical Segments

Each fixture defines `canonical` segments as the ground-truth reference. This is typically derived from the bridge adapter's output on a fixed engine version, manually verified for correctness.

## 4. Shadow-Run Comparison

### 4.1 Shadow-Run Architecture

A shadow run executes both the bridge adapter and native engine on the same input, then compares results. The orchestrator supports this via `execute_with_order`:

1. Primary engine produces the production result.
2. Shadow engine runs in parallel (or sequentially if resource-constrained).
3. Results are compared using `compare_segments_with_tolerance()` and `compare_replay_envelopes()`.
4. Comparison report is logged as evidence (not user-facing).

### 4.2 Comparison Metrics

Each shadow-run comparison produces:

| Metric | Source | Threshold |
|---|---|---|
| `segment_count_match` | `SegmentComparisonReport.length_mismatch` | Must be false |
| `text_mismatch_ratio` | `text_mismatches / expected_segments` | Must be 0.0 for deterministic paths |
| `timestamp_violation_ratio` | `timestamp_violations / expected_segments` | Must be 0.0 |
| `speaker_mismatch_ratio` | `speaker_mismatches / expected_segments` | Informational (not a gate) |
| `output_hash_match` | `ReplayComparisonReport.output_hash_match` | Informational for cross-engine |
| `within_tolerance` | `SegmentComparisonReport.within_tolerance()` | Must be true |

### 4.3 Shadow-Run CI Gate

The CI gate (bd-3pf.18) runs shadow comparisons on the full conformance corpus:

- All `within_tolerance()` must return `true`.
- Zero text mismatches across all fixtures.
- Zero timestamp violations across all fixtures.
- Results logged to `shadow_run_report.json` artifact.

## 5. Rollout Gates

### 5.1 Pre-Rollout Checklist

A native engine may replace its bridge adapter only when ALL gates pass:

| # | Gate | Enforcement |
|---|---|---|
| 1 | Engine trait compiles and passes type checks | `cargo check` |
| 2 | `is_available()` returns true in CI environment | Unit test |
| 3 | All conformance corpus fixtures pass `within_tolerance()` | `tests/conformance_harness.rs` |
| 4 | Shadow-run CI gate passes on full corpus | bd-3pf.18 CI job |
| 5 | Replay determinism: same-input same-engine produces identical output hash across 3 runs | `tests/replay_envelope.rs` |
| 6 | No clippy warnings (`cargo clippy --all-targets -- -D warnings`) | CI |
| 7 | Benchmark regression: native engine latency within 120% of bridge adapter baseline | `cargo bench` |
| 8 | Capabilities superset: native declares >= bridge adapter capabilities | Unit test |
| 9 | Cancellation compliance: `CancellationToken::checkpoint()` honored mid-execution | Integration test |
| 10 | Streaming parity (if applicable): segment delivery order matches bridge adapter | Streaming harness test |
| 11 | Corpus breadth gate: minimum fixture count + required scenario tags | `gate_summary` in conformance artifact |
| 12 | Engine-coverage gate: bridge/native observed for every backend family | `gate_summary.per_backend_family_coverage` |
| 13 | Pairwise drift-cap gate: no pair exceeds fixture-declared cap thresholds | `gate_summary.pairwise_drift_caps_ok` |

### 5.1.1 Compatibility-envelope gate mapping

The parity envelope from `docs/engine_compatibility_spec.md` is enforced with
the following release-gate mapping:

| Envelope axis | Required result | Verification |
|---|---|---|
| Text parity | `length_mismatch == false` and `text_mismatches == 0` | `cargo test --test conformance_harness` |
| Timestamp tolerance | `timestamp_violations == 0` under canonical 50ms tolerance (unless fixture override) | `cargo test --test conformance_harness` |
| Speaker stability | Pairwise `speaker_mismatches` within fixture caps (`pair_drift_caps`) | `cargo test --test conformance_harness` |
| Confidence comparability | All confidence values finite and in `[0,1]` (or `None`) | `cargo test --lib -- conformance::tests::rejects_confidence_` |
| Replay determinism linkage | Replay envelope fields present and drift-comparable | `cargo test --test replay_envelope` |

Native promotion claims must cite these concrete checks and their latest
passing run.

### 5.2 Rollout Stages

```
Stage 0: Native engine exists, shadow-run comparison available
         → BackendKind dispatch still routes to bridge adapter
         → Shadow results logged as evidence only

Stage 1: Shadow-run CI gate enabled and green for 1 week
         → All conformance corpus fixtures pass
         → Native engine registered in all_engines() but not in auto_priority()

Stage 2: Native engine added to auto_priority() behind bridge adapter
         → Only used when bridge adapter is unavailable
         → Fallback path validated in CI

Stage 3: Native engine promoted to primary in auto_priority()
         → Bridge adapter becomes fallback
         → Shadow-run comparison continues (now bridge shadows native)

Stage 4: Bridge adapter deprecated
         → Removed from auto_priority()
         → Kept in codebase for regression comparison
         → is_available() may return false (binary not shipped)
```

### 5.2.1 Runtime rollout selector

`evaluate_backend_selection()` reads `FRANKEN_WHISPER_NATIVE_ROLLOUT_STAGE` (`src/conformance.rs` `NativeEngineRolloutStage`):
- `shadow`/`0`
- `validated`/`1`
- `fallback`/`2`
- `primary`/`3`
- `sole`/`4`

Routing-order behavior:
- `shadow`, `validated`, `fallback`: force static `auto_priority()` ordering.
- `primary`, `sole`: allow adaptive recommended order.

Execution behavior is controlled by both:
- `FRANKEN_WHISPER_NATIVE_ROLLOUT_STAGE`
- `FRANKEN_WHISPER_NATIVE_EXECUTION`

Current deterministic execution policy:
- Native execution disabled (default): bridge adapters execute for all rollout stages.
- Native execution enabled + `primary`: native preferred, bridge fallback on native failure.
- Native execution enabled + `sole`: native only (fail closed if unavailable).

The routing evidence payload includes:
- `native_rollout_stage`
- `native_rollout_forced_static_order`

### 5.3 Rollback Trigger

Immediate rollback to previous stage if any of:
- Conformance corpus regression (any fixture moves from pass to fail).
- Production `backend.contract_violation` event from native engine.
- Replay determinism failure (same input produces different output hashes across runs).
- Latency regression > 150% of bridge adapter baseline on any corpus fixture.

## 6. Fallback Policy

### 6.1 Deterministic Fallback

When a native engine is primary but fails, the orchestrator falls back deterministically:

1. Native engine `run()` returns `Err` → try next engine in `auto_priority()` order.
2. Native engine `run()` returns `Ok` but `validate_segment_invariants()` fails → log `backend.contract_violation`, try next engine.
3. All engines exhausted → return `BackendUnavailable` error with collected failure reasons.

### 6.2 Fallback Evidence

Every fallback event is recorded as:
- `backend.fallback` event in the run event log.
- Evidence values: `{primary_engine, fallback_engine, fallback_reason, attempt_count}`.
- Routing evidence ledger entry (if adaptive routing enabled).

### 6.3 No Silent Degradation

The fallback policy is fail-closed, not fail-open:
- A native engine that passes `is_available()` but produces invalid output is a contract violation, not a silent degradation.
- Contract violations increment the adaptive routing calibration penalty for that engine.
- After `ADAPTIVE_FALLBACK_BRIER_THRESHOLD` (0.35) consecutive poor scores, the engine is deprioritized in routing.

## 7. Implementation Guidance

### 7.1 File Organization

```
src/backend/
  mod.rs                    # Engine trait, dispatch, routing
  whisper_cpp.rs            # Bridge adapter (existing)
  whisper_cpp_native.rs     # Native engine (new, bd-1rj.9)
  insanely_fast.rs          # Bridge adapter (existing)
  insanely_fast_native.rs   # Native engine (new, bd-1rj.10)
  whisper_diarization.rs    # Bridge adapter (existing)
  diarization_native.rs     # Native engine (new, bd-1rj.11)
  normalize.rs              # Output normalization (shared)
```

### 7.2 Registration

Native engines are registered in `all_engines()` alongside bridge adapters. The `auto_priority()` function controls dispatch order per rollout stage.

### 7.3 Testing Pattern

Each native engine should have:

1. **Unit tests** in its source file: `is_available()`, `capabilities()`, basic `run()` with mock input.
2. **Conformance harness fixtures**: JSON fixtures under `tests/fixtures/conformance/corpus/` comparing native output to canonical segments.
3. **Shadow-run integration test**: Side-by-side execution with bridge adapter on real audio.
4. **Replay determinism test**: 3 identical runs produce identical `output_payload_hash`.

## 8. References

- `docs/engine_compatibility_spec.md` — Segment invariants and enforcement points
- `docs/conformance-contract.md` — Conformance axes and parity definitions
- `docs/benchmark_regression_policy.md` — Performance regression thresholds
- `src/conformance.rs` — `CANONICAL_TIMESTAMP_TOLERANCE_SEC`, validation and comparison functions
- `tests/conformance_harness.rs` — Fixture-driven conformance test infrastructure
- `tests/replay_envelope.rs` — Replay determinism test infrastructure
