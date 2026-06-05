# Known Conformance Divergences

> This document tracks intentional or investigating deviations from the 
> `docs/engine_compatibility_spec.md`.

## DISC-001: Floating-point precision in SRT timestamps
- **Reference:** whisper.cpp SRT output rounds to milliseconds.
- **Our impl:** `src/export.rs` uses `.round()` then formats.
- **Impact:** Negligible drift (< 1ms).
- **Resolution:** ACCEPTED.
- **Tests affected:** All fixtures using `diarization_srt` format.
- **Review date:** 2026-04-12

## DISC-002: Speaker label remapping
- **Reference:** Bridge adapters use engine-specific prefixes (e.g., `SPEAKER_00`).
- **Our impl:** Native pilots may use different internal IDs during rollout.
- **Impact:** Cross-engine comparison requires `require_speaker_exact = false`.
- **Resolution:** ACCEPTED per spec §3.3.
- **Tests affected:** `corpus/*_cross_engine.json`.
- **Review date:** 2026-04-12

## DISC-003: Greedy vs beam-search divergence between native engine and whisper-cli defaults
- **Reference:** `whisper-cli` defaults to **beam search** (`-bs 5`).
- **Our impl:** The native in-process engine (`src/native_engine/decode.rs`) decodes **greedily** (temperature 0, no beam) at rollout stages below `primary`.
- **Impact:** Occasional word-choice differences and timestamp drift between the two engines on the same audio. Measured on `jfk.wav` + `tiny.en`: text WER 0.0 but final-segment **end-timestamp drift ~240 ms** (native 11.00s vs bridge 10.76s). The transcript text matches; only the tail segment boundary shifts.
- **Resolution:** **ACCEPTED for rollout stages below `primary`.** The bridge-vs-native conformance gate uses a dedicated **native-rollout tolerance profile** — WER ≤ 0.10 and per-segment timestamps within **0.3 s** — deliberately looser than the canonical 50 ms (`CANONICAL_TIMESTAMP_TOLERANCE_SEC`). **Revisit (tighten back toward canonical) when native beam search lands** and the engine is promoted to `primary`.
- **Tests affected:** `tests/conformance_comparator_tests.rs::gated_bridge_vs_native_conformance_jfk_tiny_en` (the new bridge-vs-native real-engine comparison).
- **Review date:** 2026-06-04
