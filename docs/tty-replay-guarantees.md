# TTY Replay Guarantees

This note defines operator-facing replayability and framing guarantees for TTY
audio transport. It is a companion to `docs/tty-audio-protocol.md`; that file
remains the wire-format source of truth.

## Scope and Version Envelope

- Audio replay guarantees apply to TTY audio data frames with
  `protocol_version == 1`.
- The stream can include control frames (`frame_type` present), including
  transcript correction control frames used by protocol v2 control semantics.
- Audio decode/replay determinism is defined by the audio frame path
  (`TtyAudioFrame`) and selected recovery policy, independent of transcript
  control messages.

## Deterministic Framing and Parse Rules

- Framing is line-oriented NDJSON: one JSON object per line.
- Empty lines are ignored.
- Frame classification is deterministic:
  - has `frame_type` -> control frame
  - otherwise -> audio frame
- Handshake/control ordering invariants:
  - if handshake is present, it must appear before audio frames;
  - duplicate handshakes are rejected;
  - `handshake_ack` before `handshake` is rejected;
  - if a handshake is present, audio frames must match negotiated version.
- Legacy compatibility:
  - encoder emits handshake first;
  - decoder still accepts audio-only streams with no handshake.

## Data-Frame Ordering and Decode Determinism

- Before decode, frames are sorted by `seq`; out-of-order arrival is normalized.
- Decode then processes a deterministic expected sequence starting at `0`.
- With identical input stream and identical policy, output raw audio bytes and
  decode telemetry are identical.
- Hard-fail invariants (policy-independent):
  - unsupported protocol version;
  - unsupported codec;
  - unsupported sample rate/channels;
  - handshake ordering violations.

## Recovery Policy Contract

- `fail_closed`
  - first sequence/integrity violation aborts decode with deterministic error.
- `skip_missing`
  - violating frames are dropped;
  - decode continues for valid later frames;
  - violations are recorded in telemetry (`gaps`, `duplicates`,
    `integrity_failures`, `dropped_frames`).

Violation classes covered by the policy path:
- sequence gap;
- duplicate sequence;
- base64 decode failure;
- zlib decompression failure;
- CRC32 mismatch (when `crc32` is present);
- SHA-256 mismatch (when `payload_sha256` is present).

## Integrity and Replayability Guarantees

- `crc32` and `payload_sha256` are computed over raw pre-compression audio
  bytes.
- Both fields are optional; if present, each must validate.
- When both are present, both checks are enforced.
- Under `skip_missing`, failed-integrity frames are dropped deterministically
  and reported; under `fail_closed`, they terminate decode.

## Deterministic Retransmit Artifacts

- Retransmit plan generation is deterministic:
  - take union of gap-derived missing sequences and integrity-failure sequences;
  - sort and deduplicate;
  - collapse contiguous values into stable `{start_seq, end_seq}` ranges.
- CLI retransmit loop (`tty-audio control retransmit-loop`) is deterministic:
  - if no missing/corrupt frames: emits one `ack` with `up_to_seq: 0`;
  - otherwise emits `max(rounds, 1)` identical `retransmit_request` frames,
    then one `retransmit_response`.
- Programmatic `RetransmitLoop` is deterministic for fixed input/loss pattern:
  - recover in sequence order;
  - escalate strategy `Simple -> Redundant -> Escalate` with fixed recovery
    counts (1, 2, 4 frames per round).

## Operator Evidence Bundle

For audit-grade replay, preserve:
- original NDJSON frame stream;
- selected decode policy (`fail_closed` or `skip_missing`);
- generated retransmit plan and/or retransmit-loop report;
- if session close is used, `session_close.last_data_seq` evidence.

These artifacts are sufficient to reproduce decode/retransmit outcomes.

## Verification Hooks

- Core decode/replay invariants:
  - `src/tty_audio.rs` unit tests (gap handling, integrity checks, deterministic
    retransmit loop, handshake/control ordering).
- Integration and contract coverage:
  - `tests/tty_control_tests.rs`
  - `tests/tty_correction_tests.rs`
  - `tests/tty_telemetry_tests.rs`
  - `tests/e2e_pipeline_tests.rs`
  - `tests/recovery_smoke.rs`
- Operator smoke checks:
  - `cargo run -- tty-audio retransmit-plan`
  - `cargo run -- tty-audio control retransmit-loop --rounds 3`
  - `cargo run -- tty-audio decode --recovery fail_closed|skip_missing`

## Non-Guarantees

- No guarantee of bit-identical compressed payload bytes across different
  encoders/ffmpeg builds.
- No guarantee about network delivery timing or wall-clock pacing; guarantees
  are content/order based.
- No automatic merge of divergent retransmit sessions.
