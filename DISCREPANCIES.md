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
