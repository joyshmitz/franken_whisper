# Engine Compatibility Specification

## Scope

This document defines the concrete invariants and tolerances that every engine implementation (bridge adapter or native Rust) must satisfy. These rules are enforced by `conformance::validate_segment_invariants` at runtime and by the conformance harness in CI.

## 1. Segment Timestamp Invariants

### 1.1 Internal ordering
For every segment with both `start_sec` and `end_sec` present: `end_sec >= start_sec - epsilon`, where `epsilon = 1e-6 seconds`.

### 1.2 Monotonicity (non-overlap)
Under the default `SegmentConformancePolicy` (`allow_overlap = false`), each segment's `start_sec` must be `>= previous_end_sec - epsilon`. Engines that produce overlapping segments (e.g. diarization with cross-talk) may opt into `allow_overlap = true` via an explicit policy override.

### 1.3 Segment boundary tolerance
Default cross-engine timestamp drift tolerance is **50ms** (`timestamp_tolerance_sec = 0.05`, matching `SegmentCompatibilityTolerance::default()`). This is checked by `compare_segments_with_tolerance` with an explicit `SegmentCompatibilityTolerance`; tests may pass a looser explicit tolerance when a scenario requires it.

## 2. Confidence Calibration

### 2.1 Range
Confidence values, when present, must satisfy `0.0 <= confidence <= 1.0`. Values outside this range are rejected at runtime.

### 2.2 Semantics
Confidence represents the engine's estimated probability that the transcribed text is correct. Engines are not required to produce identical confidence values, but calibration behavior must remain consistent within an engine across runs on the same input.

### 2.3 Absence
`None` confidence is always valid. Engines that do not produce confidence scores should leave the field as `None`.

## 3. Speaker Labels

### 3.1 Format
Speaker labels, when present, must be non-empty after trimming whitespace. Empty or whitespace-only speaker labels are rejected at runtime.

### 3.2 Recognized patterns
The SRT parser recognizes these speaker label prefixes (case-insensitive):
- `speaker*` (e.g. `Speaker 0`, `SPEAKER_01`)
- `spk*` (e.g. `spk2`, `SPK_3`)
- `spkr*` (e.g. `SPKR_1`)
- `s<digits>` (e.g. `s0`, `s02`)

Separators: `:`, `-`, `|`.

### 3.3 Cross-engine stability
Speaker label identifiers may differ across engines (e.g. `SPEAKER_00` vs `spk0`). Conformance comparisons use `require_speaker_exact = false` by default; set to `true` only for same-engine regression tests.

## 4. Text Normalization

### 4.1 Trimming
All segment text is trimmed of leading and trailing whitespace before storage.

### 4.2 Cross-engine comparison
Default conformance comparison uses exact text match (`require_text_exact = true`). For stochastic decode paths where minor text variation is expected, tests should use a normalized comparison (lowercased, whitespace-collapsed).

### 4.3 Empty segments
Segments with empty text after trimming are permitted in the segment array but excluded from `transcript_from_segments` output.

## 5. Engine Trait Contract

Every engine implements the `Engine` trait (`src/backend/mod.rs`):

| Method | Contract |
|---|---|
| `name()` | Stable human-readable identifier (e.g. `"whisper.cpp"`). |
| `kind()` | Returns the corresponding `BackendKind` enum variant. |
| `capabilities()` | Returns `EngineCapabilities` describing supported features. Must be accurate. |
| `is_available()` | Returns `true` only when the engine's external dependency is reachable. |
| `run()` | Produces a `TranscriptionResult` satisfying all invariants in this spec. |

## 6. Replay Envelope

Each run produces a `ReplayEnvelope` with:
- `input_content_hash`: SHA-256 of normalized WAV input.
- `backend_identity`: Engine name string.
- `backend_version`: Engine version string (when available).
- `output_payload_hash`: SHA-256 of raw backend output JSON.

Deterministic drift detection compares replay envelopes across runs via `compare_replay_envelopes`.

## 7. Enforcement

| Invariant | Enforcement point |
|---|---|
| Timestamp ordering | `conformance::validate_segment_invariants` in orchestrator after backend stage |
| Confidence bounds | `conformance::validate_segment_invariants` in orchestrator after backend stage |
| Speaker label non-empty | `conformance::validate_segment_invariants` in orchestrator after backend stage |
| Cross-engine drift | `tests/conformance_harness.rs` fixture-driven comparison |
| Corpus breadth + rollout gates | `tests/conformance_harness.rs` gate summary (`fixture_count`, required tags, per-family bridge/native coverage, pairwise drift caps) |
| Replay determinism | `tests/replay_envelope.rs` round-trip and comparator tests |

## 8. Runtime Lifecycle Contract (Frozen)

The runtime lifecycle contract is the machine-visible sequence emitted through `RunEvent` (`src/model.rs`) and produced by orchestrator stage logging (`EventLog::push` in `src/orchestrator.rs`).

### 8.1 Canonical stage names

The following `stage` values are reserved for pipeline lifecycle:

| Stage string | Origin |
|---|---|
| `ingest` | `PipelineStage::Ingest` |
| `normalize` | `PipelineStage::Normalize` |
| `backend` | `PipelineStage::Backend` |
| `acceleration` | `PipelineStage::Accelerate` |
| `persist` | `PipelineStage::Persist` |
| `backend_routing` | routing-decision control plane |
| `orchestration` | run-level control and budget telemetry |

### 8.2 Stage event envelope contract

Each stage event must include:
- `stage`: canonical stage string.
- `code`: stable machine-readable status code (`*.start`, `*.ok`, `*.error`, contract-specific routing codes).
- `message`: human-readable status line.
- `payload`: structured JSON object for deterministic machine parsing.

For backend execution auditing, `backend.ok` payloads must also include:
- `resolved_backend`
- `segments`
- `engine_identity`
- `engine_version`
- `implementation` (`bridge|native`)
- `execution_mode` (`bridge_only|native_preferred|native_only`)
- `native_rollout_stage`
- `native_fallback_error` (`null` unless native-preferred bridge fallback was used)

`replay.envelope` payloads mirror the same execution-path fields so replay artifacts retain routing/runtime provenance even when output text is unchanged.

This contract is regression-tested in `tests/cli_integration.rs` and `tests/robot_contract_tests.rs`.

### 8.3 Native rollout stage control

Native-engine rollout mode is controlled by `FRANKEN_WHISPER_NATIVE_ROLLOUT_STAGE` (see `NativeEngineRolloutStage` in `src/conformance.rs`):
- `shadow` / `0`
- `validated` / `1`
- `fallback` / `2`
- `primary` / `3`
- `sole` / `4`

Execution mode is additionally controlled by `FRANKEN_WHISPER_NATIVE_EXECUTION`:
- unset/`0`/`false`: bridge execution only.
- `1`/`true` + `primary`: native preferred with deterministic bridge fallback.
- `1`/`true` + `sole`: native only.

Routing recommendations remain rollout-gated:
- `shadow`, `validated`, `fallback`: force static `auto_priority` order.
- `primary`, `sole`: allow adaptive recommended order.

The selected rollout stage and static-order forcing status are emitted in routing evidence payloads.

## 9. Compatibility Envelope (Release Gates)

This section is the explicit parity envelope for cross-engine claims. A native
engine is not considered parity-ready unless each axis below satisfies its
release gate.

### 9.1 Text parity target

- Primary target: zero text drift on deterministic fixtures.
- Gate condition:
  - `length_mismatch == false`
  - `text_mismatches == 0`
- Enforcement: `compare_segments_with_tolerance()` and
  `tests/conformance_harness.rs` fixture expectations.

### 9.2 Timestamp tolerance target

- Canonical tolerance: `<= 0.05s` per segment boundary
  (`CANONICAL_TIMESTAMP_TOLERANCE_SEC`).
- Gate condition:
  - `timestamp_violations == 0` for canonical corpus fixtures that do not
    declare looser fixture-specific tolerance.
- Enforcement: `compare_segments_with_tolerance()` and
  `tests/conformance_harness.rs`.

### 9.3 Speaker-label stability target

- Cross-engine identity remapping is allowed by default
  (`require_speaker_exact = false`).
- Same-engine regression tests may require exact speaker IDs
  (`require_speaker_exact = true`).
- Gate condition:
  - cross-engine: `speaker_mismatches` must remain within fixture-declared
    `pair_drift_caps.max_speaker_mismatches`.
  - same-engine exact mode: `speaker_mismatches == 0`.
- Enforcement: `tests/conformance_harness.rs` pairwise drift-cap checks.

### 9.4 Confidence comparability target

- Confidence is treated as a calibrated bounded signal, not an exact
  cross-engine numeric match.
- Gate condition:
  - every populated confidence is finite and within `[0.0, 1.0]`;
  - `None` confidence remains valid for engines that do not emit confidence.
- Enforcement: `validate_segment_invariants()` and confidence-bound unit tests
  in `src/conformance.rs`.

### 9.5 Release-gate matrix

| Envelope axis | Pass criteria | Enforcement artifact |
|---|---|---|
| Text parity | No length mismatch, zero text mismatches | `tests/conformance_harness.rs` |
| Timestamp tolerance | Zero timestamp violations at canonical tolerance | `tests/conformance_harness.rs` |
| Speaker stability | Within fixture pair drift caps (or zero in exact mode) | `tests/conformance_harness.rs` |
| Confidence comparability | All confidences finite and within `[0,1]` | `src/conformance.rs` invariant/unit tests |
| Replay determinism linkage | Replay metadata present and comparable for drift triage | `tests/replay_envelope.rs` + conformance harness replay checks |

### 9.6 Mandatory release-check execution

Run these checks before claiming parity completion:

1. `scripts/run_quality_gates_rch.sh`
2. `rch exec -- env CARGO_TARGET_DIR=/tmp/rch_target_fw_conformance cargo test --test conformance_harness`
3. `rch exec -- env CARGO_TARGET_DIR=/tmp/rch_target_fw_replay cargo test --test replay_envelope`

Release claims are blocked if any gate above fails.
