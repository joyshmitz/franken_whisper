# Conformance Contract

## Purpose

This document defines what "compatible" means while converging from bridge adapters to native Rust engines.

Compatibility is not "byte-for-byte identical output." It is a bounded contract with explicit tolerances and deterministic evidence artifacts.

## Conformance Axes

1. Transcript/text parity
- Target: exact text match where deterministic.
- Tolerance mode: normalized-text equivalence for stochastic decode paths.

2. Timestamp envelope
- Segment start/end timestamps must be monotonic within policy.
- Drift must remain inside an explicit per-segment and aggregate tolerance band.

3. Speaker-label stability
- Label identifiers may be remapped across runs/engines, but segment-to-speaker assignment consistency must remain inside declared tolerance.

4. Confidence calibration comparability
- Confidence values are not expected to be numerically identical.
- Calibration behavior must remain within declared error-budget bounds.

## Runtime Contract (Current)

Current runtime enforcement includes `segment-monotonic-v1`:
- no `end_sec < start_sec`,
- no segment overlap by default policy unless explicitly allowed,
- fail closed with deterministic `backend.contract_violation` stage event.

## Evidence Artifacts Required for Parity Claims

Each parity claim must include:
- corpus manifest (audio inputs + expected envelope class),
- run-level event stream (`run_start`/`stage`/`run_complete` or `run_error`),
- replay metadata (`input_hash`, engine identity/version, output hash) once implemented,
- conformance summary (pass/fail counts + per-axis drift metrics).

## Near-Term Execution Items

- Expand tolerance statements above into broader executable tests.
- Grow golden-corpus harness and drift comparator.
- Gate native-engine rollouts on passing conformance thresholds.

Current seed implementation:
- fixture-driven segment conformance harness in `tests/conformance_harness.rs` with JSON fixtures under `tests/fixtures/conformance/`.
