# Changelog

All notable changes to [franken_whisper](https://github.com/Dicklesworthstone/franken_whisper) are documented in this file.

franken_whisper is an agent-first Rust ASR orchestration stack with adaptive backend routing, real-time NDJSON streaming, and SQLite-backed persistence. It wraps whisper.cpp, insanely-fast-whisper, and whisper-diarization behind a unified 10-stage pipeline.

Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Commit links point to the canonical GitHub repository.

---

## [Unreleased] (since v0.1.0)

72 commits, 236 files changed, +26,077 / -1,858 lines since v0.1.0.

Compare: [`v0.1.0...main`](https://github.com/Dicklesworthstone/franken_whisper/compare/v0.1.0...main)

### Added

- **curl|bash installer** with SHA-256 checksum verification for one-line installs ([`e06d87f`](https://github.com/Dicklesworthstone/franken_whisper/commit/e06d87f1c812d2acfd03459973a4cf36a9ced23c))
- **Adaptive window controller** for speculative streaming pipeline -- dynamically adjusts decode windows based on correction rates ([`61b7d40`](https://github.com/Dicklesworthstone/franken_whisper/commit/61b7d40073f41268280033ceac0a8251e0179923))
- **routing_decision event builder** in robot module; align backends output with NDJSON schema ([`6beedc3`](https://github.com/Dicklesworthstone/franken_whisper/commit/6beedc3db49efcf2c26d3f712ee0d78abf62c125))
- **Persistent worker quarantine with TTL** for quality-gate routing -- quarantined workers are tracked across runs ([`9ecf72d`](https://github.com/Dicklesworthstone/franken_whisper/commit/9ecf72da3615ebe4dd9d667b06cdd42574a81086))
- **flock guard and preblock** for unreachable workers in CI quality-gate routing ([`77900fb`](https://github.com/Dicklesworthstone/franken_whisper/commit/77900fb76984aa0183bc50ca45a04d3bc44ad4e5))
- **XDG_STATE_HOME support** for tool state storage; extract ffmpeg to tmpdir ([`10be321`](https://github.com/Dicklesworthstone/franken_whisper/commit/10be321a493db1ce8617f82172c6d2b12716ef88))
- **Backend module expansion** with improved pipeline orchestration ([`d9166c3`](https://github.com/Dicklesworthstone/franken_whisper/commit/d9166c3a3c42175a472d904b54f660cf28a52a74))
- **Storage hardening**: replay pack validation, migration-safe column detection, defensive query wrappers ([`5248ad5`](https://github.com/Dicklesworthstone/franken_whisper/commit/5248ad5e37e43e1b093f1f82f262898a0dabd593), [`23b5062`](https://github.com/Dicklesworthstone/franken_whisper/commit/23b5062619fa5a9c73efffcda659be61f102d7df))
- **TTY audio demo script** and extended TTY audio pipeline with telemetry tests ([`6a5234d`](https://github.com/Dicklesworthstone/franken_whisper/commit/6a5234dff4a43a6fa375ca6165519fff062a6222), [`b8edfaf`](https://github.com/Dicklesworthstone/franken_whisper/commit/b8edfafa1add287c6d666e185681fb2e071feb4a))
- **Parallel sync pipeline** with progress tracking ([`1999137`](https://github.com/Dicklesworthstone/franken_whisper/commit/1999137e6ba3e2c31b201e5c65381ebaaaea5d1b))
- **Expanded robot output** with structured analysis views (+90 lines) ([`32ba54f`](https://github.com/Dicklesworthstone/franken_whisper/commit/32ba54f2e53c9e5fff07ed92abe854e797a70f21))
- **Expanded CLI** with structured subcommands and output modes (+122 lines) ([`ef221f7`](https://github.com/Dicklesworthstone/franken_whisper/commit/ef221f7ba39c1f8d6bd1caf36f2b34c8454608b6))
- **CLI integration tests** and orchestrator refinement ([`a7cca9e`](https://github.com/Dicklesworthstone/franken_whisper/commit/a7cca9e0c5d627be27ab0bb0aafab48884504c35))

### Fixed

- **Diarization transcript fallback** and command pipe output loss ([`a8328bc`](https://github.com/Dicklesworthstone/franken_whisper/commit/a8328bcdf02b93f67ab788476c7ba198667af819))
- **Audio region clamping**: active region boundaries now clamped to actual audio duration ([`2bca2f9`](https://github.com/Dicklesworthstone/franken_whisper/commit/2bca2f9fd48b5be23b3434bf5bef25ed5953bc1e))
- **Division by zero in layer_norm_cpu**: clamp epsilon floor in accelerate module ([`b3c8b56`](https://github.com/Dicklesworthstone/franken_whisper/commit/b3c8b56e9290a8b73a6a527b72091419e5107d19))
- **Sync timeout overflow**: use `saturating_mul` for timeout calculation; prefer `backend.ok` rollout stage; make atomic rename portable ([`9f94946`](https://github.com/Dicklesworthstone/franken_whisper/commit/9f94946753a775d253da17f5b8d3e88a8195d442))
- **Storage JSON error propagation**: replace silent `unwrap_or_default` with explicit JSON parse error return ([`c2f5474`](https://github.com/Dicklesworthstone/franken_whisper/commit/c2f5474936b844e99b24beec2f27fcedad0e23bb))
- **None-timestamp segment handling** in speculation/TUI; surface run detail load errors ([`dd04e9c`](https://github.com/Dicklesworthstone/franken_whisper/commit/dd04e9cad44621911b5cf68e69c5f03330665e55))
- **Orchestrator elapsed time leak**: clear `stage_start` on checkpoint failure to prevent time leaking across pipeline stages ([`b15578e`](https://github.com/Dicklesworthstone/franken_whisper/commit/b15578ea6b403b1cca148c5fec22b5e3c49dd067))
- **Pipeline stage transitions** refinement and stability ([`5f3f187`](https://github.com/Dicklesworthstone/franken_whisper/commit/5f3f1870d393feed4d3b4116a3cc68664680112f))
- **Sync off-by-one** in sync progress tracking ([`0105d9a`](https://github.com/Dicklesworthstone/franken_whisper/commit/0105d9a29f9b443ec82e692e5ba546cc20d85047))
- **Installer fixes**: strip `v` prefix from version in asset URLs; handle versioned binary names in release archives ([`4cc9b66`](https://github.com/Dicklesworthstone/franken_whisper/commit/4cc9b664c82abbd4f4d30e1145de8b8f56000a2d), [`6a48022`](https://github.com/Dicklesworthstone/franken_whisper/commit/6a480224261a235b942e7e172c5f70ed29fc38e2))
- **Stale-sibling validation**: reject when dependency sync is degraded ([`670a758`](https://github.com/Dicklesworthstone/franken_whisper/commit/670a75803995f1ae372cb0a7fdec9db80e2e2ce8))
- **CLI argument parsing** edge case correction ([`8a8a58e`](https://github.com/Dicklesworthstone/franken_whisper/commit/8a8a58e0baba1df01dc4eecb6672a013f1ae7412))
- **SqliteValue Arc migration**: migrate `SqliteValue::Text(String)` and `Blob(Vec<u8>)` to `Arc`-based types for reduced cloning overhead ([`4066d5d`](https://github.com/Dicklesworthstone/franken_whisper/commit/4066d5d7f14f085c7eb1fcd6d2fcfbca52e27b1c))
- **Pipeline panic elimination**: replace 8 `.expect()` panics in pipeline stages with proper `FwError` returns ([`0c4d557`](https://github.com/Dicklesworthstone/franken_whisper/commit/0c4d557ac5926e094a3db8b8415a4899413576e9))

### Changed

- **TTY audio frame validation**: extract frame validation logic and reject sequence gaps on finalize ([`f6f5f41`](https://github.com/Dicklesworthstone/franken_whisper/commit/f6f5f41ef83ffa347c8c6d75bfce885d3433bc44))
- **Speculation test assertions** simplified with `expect_err` ([`33423d5`](https://github.com/Dicklesworthstone/franken_whisper/commit/33423d582363e619be4c8ea0161fbb98720bac21))
- **Quality-gate retry routing** hardened with worker quarantine and expanded retryable patterns ([`1ff4910`](https://github.com/Dicklesworthstone/franken_whisper/commit/1ff4910c067afb70605d524c3b27d1114ed54ac0))
- **Backend/storage pipeline** streamlined for robustness ([`b19eec9`](https://github.com/Dicklesworthstone/franken_whisper/commit/b19eec97545e06eef5e3b8842dfca10595180cf9))
- **Storage query patterns** simplified; dead code paths removed ([`64d340e`](https://github.com/Dicklesworthstone/franken_whisper/commit/64d340e4d6e6fe933d8da8184a1b0a07f89edcc1), [`42f718a`](https://github.com/Dicklesworthstone/franken_whisper/commit/42f718a2e018e2297acda91a1fcdb9f96146c002))
- **Module streamlining** with beads history rotation ([`ca4f587`](https://github.com/Dicklesworthstone/franken_whisper/commit/ca4f587c5e03a4d3a0e44feb386c9380bb89c0c4))

### Documentation

- Major README refresh with updated architecture and feature documentation (+1,867 lines across three commits) ([`2a2bb9f`](https://github.com/Dicklesworthstone/franken_whisper/commit/2a2bb9fd333df71e1e0069d3086d2e8a49315c7c), [`7d207b8`](https://github.com/Dicklesworthstone/franken_whisper/commit/7d207b8ba23ee18573b12a9a8d9dbeadf16258be), [`ba227a6`](https://github.com/Dicklesworthstone/franken_whisper/commit/ba227a60fab5558e0d09c06171c4e37e5babd49e))
- Robot event catalog updated with `routing_decision`, `backends.discovery`, and streaming events ([`ecf702b`](https://github.com/Dicklesworthstone/franken_whisper/commit/ecf702bea3219c697871788f4976d48b0f9dbb40))
- Implementation tracker and execution packet documentation updated ([`b34893e`](https://github.com/Dicklesworthstone/franken_whisper/commit/b34893e0e637ece3023c75572036e054b635ebf9))

---

## [v0.1.0] -- 2026-03-04

**Initial release.** Tag: [`v0.1.0`](https://github.com/Dicklesworthstone/franken_whisper/releases/tag/v0.1.0) | Published: 2026-03-04 | 59 commits from initial commit.

Compare: [`2e9f2e9...v0.1.0`](https://github.com/Dicklesworthstone/franken_whisper/compare/2e9f2e97e0ce8f37d71c2ae00eece2904f3fdd19...v0.1.0)

### Release Artifacts

| Platform | Archive | Size |
|----------|---------|------|
| Linux x86_64 | `franken_whisper-0.1.0-linux_amd64.tar.gz` | 2.3 MB |
| macOS arm64 (Apple Silicon) | `franken_whisper-0.1.0-darwin_arm64.tar.gz` | 2.1 MB |
| Windows x64 | `franken_whisper-0.1.0-windows_amd64.zip` | 2.2 MB |

SHA-256 checksums included in [`checksums-sha256.txt`](https://github.com/Dicklesworthstone/franken_whisper/releases/download/v0.1.0/checksums-sha256.txt).

### Core Architecture

- **10-stage composable pipeline**: Ingest, Normalize, VAD, Source Separate, Backend, Accelerate, Align, Punctuate, Diarize, Persist ([`2e9f2e9`](https://github.com/Dicklesworthstone/franken_whisper/commit/2e9f2e97e0ce8f37d71c2ae00eece2904f3fdd19))
- **Adaptive Bayesian backend routing** with explicit loss matrix, posterior calibration, and deterministic fallback across whisper.cpp, insanely-fast-whisper, and whisper-diarization
- **Real-time NDJSON streaming** with stable schema (v1.0.0) for agent consumption -- every pipeline stage emits sequenced, timestamped events
- **Cooperative cancellation** via `CancellationToken`-based graceful shutdown with resource cleanup
- **Per-stage timeout budgets** with automatic latency profiling and tuning recommendations
- **12 structured `FW-*` error codes** for machine-readable error handling

### Audio Processing

- **Built-in Rust audio decoder** via symphonia: MP3, AAC, FLAC, WAV, OGG, ALAC decoded natively with zero ffmpeg dependency for common formats ([`934a8f6`](https://github.com/Dicklesworthstone/franken_whisper/commit/934a8f639793aec6ae80b3ad823e74369fa3119c))
- **Automatic ffmpeg provisioning** with bridge-native recovery fallback ([`ca25b05`](https://github.com/Dicklesworthstone/franken_whisper/commit/ca25b051e774e21f2ff9d8bbdaa8cd35d65a73a6))
- **VAD redesign** and audio pipeline hardening ([`80969cb`](https://github.com/Dicklesworthstone/franken_whisper/commit/80969cba857644458005f5263bb82bffbd97749e))
- Audio duration probing hardened for edge cases ([`109ebc1`](https://github.com/Dicklesworthstone/franken_whisper/commit/109ebc1dd671a93f4d1ce4e06f415823b520ecc3))
- Input validation: reject directory paths with `is_file` guard ([`830a443`](https://github.com/Dicklesworthstone/franken_whisper/commit/830a4431e799305223bcf54ab473a069ca651819))

### Speculative Streaming

- **Speculative cancel-correct streaming pipeline** with evidence ledger and file processing -- dual-model fast+quality architecture ([`ad4b54a`](https://github.com/Dicklesworthstone/franken_whisper/commit/ad4b54a39c58a8c5e37e66f4470778d0c48308e6))
- TUI speculation display and TTY transcript control frame support ([`847658d`](https://github.com/Dicklesworthstone/franken_whisper/commit/847658d66d1f3364ccc550344b69f6eb7c7be594))
- Conformance validation extended for speculation events ([`30044ab`](https://github.com/Dicklesworthstone/franken_whisper/commit/30044abd4cdec52cda71be7ea350c5006856560b))

### TTY Audio Transport

- **mulaw+zlib+base64 NDJSON protocol** for low-bandwidth audio relay over PTY links with handshake, integrity checks, and deterministic retransmission
- VAD/separate/punctuate/diarize pipeline stages wired into TTY session protocol ([`f183d08`](https://github.com/Dicklesworthstone/franken_whisper/commit/f183d08a7146949f1ab7544d5a704f68fa7409a7))

### Storage and Persistence

- **SQLite-backed run history** with JSONL export/import and SHA-256 replay envelopes
- **Rollback-safe legacy v1 to v2 runs schema migration** ([`1424a8d`](https://github.com/Dicklesworthstone/franken_whisper/commit/1424a8de48201a640ba86da6da9ff61d71a3bb07))
- **acceleration_json column** propagated through sync pipeline with fail-closed overwrite semantics ([`9c7df9d`](https://github.com/Dicklesworthstone/franken_whisper/commit/9c7df9d426b03261bd4250bac0f01b9e967f4391))
- **Cascade-aware overwrite tracking** with overwrite-strict conflict policy ([`4c7cc45`](https://github.com/Dicklesworthstone/franken_whisper/commit/4c7cc4515d47287cee3337f8793b874f2ac93c35), [`e4db512`](https://github.com/Dicklesworthstone/franken_whisper/commit/e4db512b96c3779693e04788566663342f2e3e5d))
- Migration error messages enriched with contextual details ([`07f24f6`](https://github.com/Dicklesworthstone/franken_whisper/commit/07f24f6afbee2fcd9318f6a178c7252192718f12))
- MVCC visibility retry in concurrent write test ([`4e713a2`](https://github.com/Dicklesworthstone/franken_whisper/commit/4e713a209906f085466d41b8788b88671b4ae639))
- Incremental export corrected to use `finished_at` instead of `started_at` ([`8974203`](https://github.com/Dicklesworthstone/franken_whisper/commit/8974203d9b0bf0816a655e60c5a82f5f45657842))

### Diarization

- **TinyDiarize support**: whisper.cpp's built-in speaker-turn detection (no HF token required) ([`7abe271`](https://github.com/Dicklesworthstone/franken_whisper/commit/7abe27163dfcbd7e146d469c7aba562f531f4dd6))
- Speaker constraints integration tests ([`3412914`](https://github.com/Dicklesworthstone/franken_whisper/commit/3412914ffa6f26b3f20a3f8f85e8d82a1029f89a))

### Orchestrator

- **Parallel processing and retry logic** for transcription orchestration ([`5c70d3d`](https://github.com/Dicklesworthstone/franken_whisper/commit/5c70d3de4fd55080c6a559e1e83d61ef721349a3))
- **whisper_cpp backend hardening** with expanded orchestrator capabilities and pipeline robustness ([`1f6f694`](https://github.com/Dicklesworthstone/franken_whisper/commit/1f6f694603ef1c9682a82a6efe9f17bff3147d1b))
- Routing mode helper extracted; shutdown and storage resilience hardened; sync cursor upgraded to composite key ([`7cad3b5`](https://github.com/Dicklesworthstone/franken_whisper/commit/7cad3b5d3a215cacb8500b121c1794ac14642bca))

### Robot Mode

- **`robot health` subcommand** for database diagnostics ([`aff4551`](https://github.com/Dicklesworthstone/franken_whisper/commit/aff4551254ad96d43c646e5f9257d46de9d9da45))
- Golden test output fixtures added ([`146d225`](https://github.com/Dicklesworthstone/franken_whisper/commit/146d2256ae7e4d30b475bad157e67a4c28e5fff7))
- Integration tests for robot subcommands ([`0433627`](https://github.com/Dicklesworthstone/franken_whisper/commit/0433627cf11c552ed090c0a20a07c99a1fc3d295))

### Backend Engines

- Segment timestamps sanitized at extraction boundary ([`bd99c6c`](https://github.com/Dicklesworthstone/franken_whisper/commit/bd99c6cf07403a0285f6c63daeec16e5160e1ea9))
- Mutex poisoning handled gracefully in two-lane streaming executor ([`b1682e8`](https://github.com/Dicklesworthstone/franken_whisper/commit/b1682e8da91985d92d7bfbabbe571526f7d9700e))
- Comprehensive test coverage for all backend engines and routing layer ([`2640d14`](https://github.com/Dicklesworthstone/franken_whisper/commit/2640d1491a270ce413c22b6ea71725312618984f))
- Duration estimation and native engine edge-case tests ([`22c3a03`](https://github.com/Dicklesworthstone/franken_whisper/commit/22c3a036cdca5b6d4cf276c67efdbadb768664e1))

### Event Fingerprinting

- Event fingerprinting enriched with seq+stage tuples; ingest failure determinism test added ([`acbb240`](https://github.com/Dicklesworthstone/franken_whisper/commit/acbb2406139a8007f0e952c2c253cd4bc7cfd18b))
- Streaming engine trait added ([`80969cb`](https://github.com/Dicklesworthstone/franken_whisper/commit/80969cba857644458005f5263bb82bffbd97749e))

### Bug Fixes

- Prevent stdout/stderr pipe deadlock by reading on dedicated threads ([`dd6cbb2`](https://github.com/Dicklesworthstone/franken_whisper/commit/dd6cbb2421c1e3d333df78d791d7d235e17ac143))
- `ControlFrameType` references updated to `TtyControlFrame` enum ([`d625089`](https://github.com/Dicklesworthstone/franken_whisper/commit/d62508926b3a3a5254a0e26037a4684d4c33bf8a))
- BetaPosterior::new input validation with assertions and tests ([`e4a898f`](https://github.com/Dicklesworthstone/franken_whisper/commit/e4a898f39e6ccf8cab1e17e3980f587cf22bf0d1))

### Testing

- 2,939 tests passing at release (3,500+ by HEAD)
- Massive edge-case test expansion for speculative pipeline ([`a2d5234`](https://github.com/Dicklesworthstone/franken_whisper/commit/a2d5234c220ede7be14a3f7b06a75c1e08eae182))
- Unit test coverage expanded across orchestrator, error, model, accelerate, robot, and process modules ([`1ed8429`](https://github.com/Dicklesworthstone/franken_whisper/commit/1ed8429b9e751190566b080cc5b294fcfaa29793))
- Integration test suites for speculative pipeline ([`bc2a526`](https://github.com/Dicklesworthstone/franken_whisper/commit/bc2a526e0ba15d7714b602c5f365059c5f6e56a5))
- Sync and replay_pack edge-case tests with schema validation ([`8d787f2`](https://github.com/Dicklesworthstone/franken_whisper/commit/8d787f2127921285251c71483236ba0e0735a7c7))
- Conformance harness with 50ms cross-engine timestamp tolerance and drift detection
- ffmpeg provisioning subsystem unit tests ([`e794156`](https://github.com/Dicklesworthstone/franken_whisper/commit/e7941569d294b805a966ba95bae941a82de23b7f))

### Infrastructure

- `#![forbid(unsafe_code)]` enforced -- zero unsafe blocks
- Rust 2024 edition, nightly toolchain
- `cargo clippy --all-targets -- -D warnings` enforced
- Boilerplate and dead code annotations eliminated ([`e70b860`](https://github.com/Dicklesworthstone/franken_whisper/commit/e70b86082ed279ca6032142192a3d305b905bfa5))
- rch quality-gate wrapper added ([`6f5f45e`](https://github.com/Dicklesworthstone/franken_whisper/commit/6f5f45e7ebd3f7010f3fd017729f7f87685824d5))
- MCP agent mail configuration for Codex, Cursor, and Gemini ([`27b1a98`](https://github.com/Dicklesworthstone/franken_whisper/commit/27b1a9822edd4ae9c2b0253ea3e97c5581c4c997))
- GitHub social preview image ([`098da6c`](https://github.com/Dicklesworthstone/franken_whisper/commit/098da6c89e85dfbc2dd52009e327be67da3d03be))

### Initial Commit

- [`2e9f2e9`](https://github.com/Dicklesworthstone/franken_whisper/commit/2e9f2e97e0ce8f37d71c2ae00eece2904f3fdd19) -- 2026-02-22 -- franken_whisper ASR orchestration stack foundation

---

<!-- agent-metadata
  repo: Dicklesworthstone/franken_whisper
  generated: 2026-03-21
  commits_analyzed: 131
  tags: v0.1.0
  releases: v0.1.0 (2026-03-04)
-->

[Unreleased]: https://github.com/Dicklesworthstone/franken_whisper/compare/v0.1.0...main
[v0.1.0]: https://github.com/Dicklesworthstone/franken_whisper/releases/tag/v0.1.0
