# TTY Replay Guarantees

This note defines operator-facing replay guarantees for TTY audio transport.
It is a companion to `docs/tty-audio-protocol.md` and does not replace the
wire-level protocol spec.

## Scope

Guarantees apply to protocol version `1` with NDJSON framing, using the
existing control/data frame model.

## Deterministic Framing Guarantees

- Frame boundaries are line-oriented NDJSON: one JSON object per line.
- Data frame sequencing is strict by `seq` (`u64`), starting at `0`.
- Decoder behavior is deterministic for a fixed input stream and recovery mode:
  - `fail_closed`: first violation terminates decode.
  - `skip_missing`: violations are recorded, valid later frames continue.

## Integrity Guarantees

- Optional `crc32` and `payload_sha256` are interpreted over raw
  pre-compression audio bytes.
- Integrity failures are deterministic outcomes:
  - `fail_closed`: hard failure.
  - `skip_missing`: frame dropped and recorded in telemetry.

## Replayability Guarantees

- Given identical framed input and identical decode policy, decoded output bytes
  are deterministic.
- Retransmit planning is deterministic:
  - requested sequences are sorted, deduplicated;
  - contiguous sequences are collapsed into stable ranges.
- `RetransmitLoop` behavior is deterministic for fixed inputs/loss pattern:
  - stable round progression;
  - stable recovery strategy escalation;
  - stable report contents.

## Event/Run-Log Coupling

- TTY sequence/integrity outcomes should be persisted with run artifacts
  (`events` + replay metadata) so decode/retransmit decisions are auditable.
- Operator replay requires preserving:
  - framed NDJSON stream,
  - decode policy (`fail_closed` or `skip_missing`),
  - retransmit plan/report artifacts.

## Non-Guarantees

- No claim of bit-identical compressed payloads across different encoders.
- No claim of packet/network timing determinism; guarantees are content/order
  based.
- No automatic two-way merge of divergent retransmit sessions.
