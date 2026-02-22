//! Integration tests for TTY protocol handshake/integrity telemetry.
//!
//! Bead bd-3pf.16: Exercise mixed audio+control NDJSON streams including
//! version mismatch, duplicate handshake, missing frames, and integrity
//! failures; assert telemetry counters and recovery outputs match docs.
//!
//! All tests are deterministic (no timing dependencies).

use base64::Engine;
use base64::engine::general_purpose::STANDARD_NO_PAD;
use flate2::Compression;
use flate2::write::ZlibEncoder;
use std::io::Write;

use franken_whisper::tty_audio::*;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Compress raw bytes with zlib (matching the encoder's codec).
fn compress(data: &[u8]) -> Vec<u8> {
    let mut enc = ZlibEncoder::new(Vec::new(), Compression::fast());
    enc.write_all(data).expect("zlib write");
    enc.finish().expect("zlib finish")
}

/// CRC32 of raw bytes (matching crc32fast).
fn crc32_of(data: &[u8]) -> u32 {
    let mut h = crc32fast::Hasher::new();
    h.update(data);
    h.finalize()
}

/// SHA-256 hex digest of raw bytes.
fn sha256_hex(data: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    let d = Sha256::digest(data);
    format!("{d:x}")
}

/// Build a valid `TtyAudioFrame` from arbitrary raw bytes.
fn make_frame(seq: u64, data: &[u8]) -> TtyAudioFrame {
    let compressed = compress(data);
    TtyAudioFrame {
        protocol_version: 1,
        seq,
        codec: "mulaw+zlib+b64".to_owned(),
        sample_rate_hz: 8_000,
        channels: 1,
        payload_b64: STANDARD_NO_PAD.encode(compressed),
        crc32: Some(crc32_of(data)),
        payload_sha256: Some(sha256_hex(data)),
    }
}

/// Serialize a slice of audio frames to NDJSON.
fn frames_to_ndjson(frames: &[TtyAudioFrame]) -> String {
    frames
        .iter()
        .map(|f| serde_json::to_string(f).expect("serialize"))
        .collect::<Vec<_>>()
        .join("\n")
        + "\n"
}

/// Serialize a handshake control frame JSON line.
fn handshake_line(min_v: u32, max_v: u32, codecs: &[&str]) -> String {
    serde_json::to_string(&TtyControlFrame::Handshake {
        min_version: min_v,
        max_version: max_v,
        supported_codecs: codecs.iter().map(|s| (*s).to_owned()).collect(),
    })
    .expect("serialize handshake")
}

/// Serialize a handshake-ack control frame JSON line.
fn handshake_ack_line(version: u32, codec: &str) -> String {
    serde_json::to_string(&TtyControlFrame::HandshakeAck {
        negotiated_version: version,
        negotiated_codec: codec.to_owned(),
    })
    .expect("serialize handshake_ack")
}

// =========================================================================
//  1. Version mismatch handling during handshake
// =========================================================================

#[test]
fn version_mismatch_no_overlap_returns_error() {
    // Local supports only v1, remote requires v5+
    let result = negotiate_version(1, 1, 5, 10);
    assert!(result.is_err());
    let err_msg = result.unwrap_err();
    assert!(
        err_msg.contains("no compatible protocol version"),
        "unexpected error: {err_msg}"
    );
}

#[test]
fn version_mismatch_inverted_ranges_returns_error() {
    // Local [3,5] vs remote [1,2] — no overlap
    let result = negotiate_version(3, 5, 1, 2);
    assert!(result.is_err());
    let err_msg = result.unwrap_err();
    assert!(err_msg.contains("no compatible protocol version"));
}

#[test]
fn version_negotiation_picks_highest_in_overlap() {
    // Local [1,5] vs remote [3,8] => overlap [3,5] => pick 5
    assert_eq!(negotiate_version(1, 5, 3, 8), Ok(5));
}

#[test]
fn version_negotiation_single_point_overlap() {
    // Local [1,3] vs remote [3,7] => overlap [3,3] => pick 3
    assert_eq!(negotiate_version(1, 3, 3, 7), Ok(3));
}

#[test]
fn handshake_version_mismatch_in_stream_rejects_frames() {
    // Handshake advertises only version range [5,10], but the decoder supports
    // only v1 — parse_audio_frames should fail during handshake negotiation.
    let hs = handshake_line(5, 10, &["mulaw+zlib+b64"]);
    let f0 = serde_json::to_string(&make_frame(0, b"data")).unwrap();
    let ndjson = format!("{hs}\n{f0}\n");
    let mut reader = ndjson.as_bytes();

    let err = decode_frames_to_raw(&mut reader).expect_err("version mismatch should fail");
    assert!(
        err.to_string().contains("no compatible protocol version"),
        "unexpected error: {err}"
    );
}

#[test]
fn handshake_frame_version_mismatch_after_negotiation() {
    // A handshake negotiates v1, but then a frame arrives with protocol_version=2.
    let hs = handshake_line(1, 1, &["mulaw+zlib+b64"]);
    let mut f0 = make_frame(0, b"data");
    f0.protocol_version = 2;
    let f0_json = serde_json::to_string(&f0).unwrap();
    let ndjson = format!("{hs}\n{f0_json}\n");
    let mut reader = ndjson.as_bytes();

    let err = decode_frames_to_raw(&mut reader)
        .expect_err("frame version mismatch after handshake should fail");
    assert!(
        err.to_string()
            .contains("does not match negotiated version"),
        "unexpected error: {err}"
    );
}

#[test]
fn handshake_incompatible_codec_returns_error() {
    let hs = handshake_line(1, 1, &["opus+b64"]);
    let f0 = serde_json::to_string(&make_frame(0, b"data")).unwrap();
    let ndjson = format!("{hs}\n{f0}\n");
    let mut reader = ndjson.as_bytes();

    let err = decode_frames_to_raw(&mut reader).expect_err("codec mismatch should fail");
    assert!(
        err.to_string().contains("no compatible tty-audio codec"),
        "unexpected error: {err}"
    );
}

// =========================================================================
//  2. Duplicate handshake rejection
// =========================================================================

#[test]
fn duplicate_handshake_is_rejected() {
    let hs = handshake_line(1, 1, &["mulaw+zlib+b64"]);
    let f0 = serde_json::to_string(&make_frame(0, b"data")).unwrap();
    // Two handshakes before any audio
    let ndjson = format!("{hs}\n{hs}\n{f0}\n");
    let mut reader = ndjson.as_bytes();

    let err = decode_frames_to_raw(&mut reader).expect_err("duplicate handshake should fail");
    assert!(
        err.to_string().contains("duplicate tty-audio handshake"),
        "unexpected error: {err}"
    );
}

#[test]
fn handshake_after_audio_frames_is_rejected() {
    // If a handshake appears after audio has started, it should be rejected.
    let hs = handshake_line(1, 1, &["mulaw+zlib+b64"]);
    let f0 = serde_json::to_string(&make_frame(0, b"data")).unwrap();
    // Audio first, then handshake
    let ndjson = format!("{f0}\n{hs}\n");
    let mut reader = ndjson.as_bytes();

    let err = decode_frames_to_raw(&mut reader).expect_err("late handshake should fail");
    assert!(
        err.to_string()
            .contains("handshake must appear before audio frames"),
        "unexpected error: {err}"
    );
}

#[test]
fn handshake_ack_before_handshake_is_rejected() {
    let ack = handshake_ack_line(1, "mulaw+zlib+b64");
    let f0 = serde_json::to_string(&make_frame(0, b"data")).unwrap();
    let ndjson = format!("{ack}\n{f0}\n");
    let mut reader = ndjson.as_bytes();

    let err = decode_frames_to_raw(&mut reader).expect_err("ack before handshake should fail");
    assert!(
        err.to_string()
            .contains("handshake_ack received before handshake"),
        "unexpected error: {err}"
    );
}

#[test]
fn handshake_ack_with_wrong_version_is_rejected() {
    let hs = handshake_line(1, 1, &["mulaw+zlib+b64"]);
    let ack = handshake_ack_line(99, "mulaw+zlib+b64");
    let f0 = serde_json::to_string(&make_frame(0, b"data")).unwrap();
    let ndjson = format!("{hs}\n{ack}\n{f0}\n");
    let mut reader = ndjson.as_bytes();

    let err = decode_frames_to_raw(&mut reader).expect_err("ack version mismatch should fail");
    assert!(
        err.to_string().contains("unexpected negotiated_version"),
        "unexpected error: {err}"
    );
}

#[test]
fn handshake_ack_with_wrong_codec_is_rejected() {
    let hs = handshake_line(1, 1, &["mulaw+zlib+b64"]);
    let ack = handshake_ack_line(1, "opus+b64");
    let f0 = serde_json::to_string(&make_frame(0, b"data")).unwrap();
    let ndjson = format!("{hs}\n{ack}\n{f0}\n");
    let mut reader = ndjson.as_bytes();

    let err = decode_frames_to_raw(&mut reader).expect_err("ack codec mismatch should fail");
    assert!(
        err.to_string().contains("unsupported negotiated codec"),
        "unexpected error: {err}"
    );
}

// =========================================================================
//  3. Missing frame detection and recovery
// =========================================================================

#[test]
fn missing_frames_fail_closed_reports_gap() {
    let frames = vec![make_frame(0, b"first"), make_frame(5, b"sixth")];
    let ndjson = frames_to_ndjson(&frames);
    let mut reader = ndjson.as_bytes();

    let err = decode_frames_to_raw(&mut reader).expect_err("gap should fail in FailClosed mode");
    assert!(
        err.to_string().contains("missing tty-audio frame sequence"),
        "unexpected error: {err}"
    );
}

#[test]
fn missing_frames_skip_missing_reports_correct_telemetry() {
    // Frames 0, 3, 4 present; frames 1, 2 missing
    let frames = vec![
        make_frame(0, b"chunk-0"),
        make_frame(3, b"chunk-3"),
        make_frame(4, b"chunk-4"),
    ];
    let ndjson = frames_to_ndjson(&frames);
    let mut reader = ndjson.as_bytes();

    let (report, raw) =
        decode_frames_to_raw_with_policy(&mut reader, DecodeRecoveryPolicy::SkipMissing)
            .expect("skip missing should recover");

    // Telemetry counters
    assert_eq!(report.frames_decoded, 3);
    assert_eq!(report.gaps.len(), 1);
    assert_eq!(report.gaps[0].expected, 1);
    assert_eq!(report.gaps[0].got, 3);
    assert!(report.duplicates.is_empty());
    assert!(report.integrity_failures.is_empty());
    assert!(report.dropped_frames.is_empty());
    assert_eq!(report.recovery_policy, DecodeRecoveryPolicy::SkipMissing);

    // Recovery output: skipped frames are not in the raw output
    assert_eq!(raw, b"chunk-0chunk-3chunk-4");
}

#[test]
fn multiple_gaps_tracked_in_telemetry() {
    // Frames 0, 3, 7 present; gaps at [1,3) and [4,7)
    let frames = vec![
        make_frame(0, b"a"),
        make_frame(3, b"d"),
        make_frame(7, b"h"),
    ];
    let ndjson = frames_to_ndjson(&frames);
    let mut reader = ndjson.as_bytes();

    let (report, _) =
        decode_frames_to_raw_with_policy(&mut reader, DecodeRecoveryPolicy::SkipMissing)
            .expect("decode");

    assert_eq!(report.gaps.len(), 2);
    assert_eq!(report.gaps[0].expected, 1);
    assert_eq!(report.gaps[0].got, 3);
    assert_eq!(report.gaps[1].expected, 4);
    assert_eq!(report.gaps[1].got, 7);
}

#[test]
fn missing_frame_detection_generates_retransmit_candidates() {
    let report = DecodeReport {
        frames_decoded: 3,
        gaps: vec![SequenceGap {
            expected: 2,
            got: 5,
        }],
        duplicates: vec![],
        integrity_failures: vec![],
        dropped_frames: vec![],
        recovery_policy: DecodeRecoveryPolicy::SkipMissing,
    };
    let candidates = retransmit_candidates(&report);
    assert_eq!(candidates, vec![2, 3, 4]);
}

#[test]
fn retransmit_plan_from_reader_captures_gaps_and_integrity_failures() {
    // Frame 0 ok, frame 1 corrupt, frame 2 missing (gap), frame 4 ok
    let mut corrupt = make_frame(1, b"bad");
    corrupt.crc32 = Some(corrupt.crc32.unwrap() ^ 0xFFFF);

    let frames = vec![
        make_frame(0, b"ok-0"),
        corrupt,
        // seq 2 and 3 missing (gap from 2 to 4)
        make_frame(4, b"ok-4"),
    ];
    let ndjson = frames_to_ndjson(&frames);
    let mut reader = ndjson.as_bytes();

    let plan = retransmit_plan_from_reader(&mut reader, DecodeRecoveryPolicy::SkipMissing)
        .expect("plan should succeed");

    // Frame 1 had integrity failure, frames 2,3 are gap candidates
    assert!(plan.requested_sequences.contains(&1));
    assert!(plan.requested_sequences.contains(&2));
    assert!(plan.requested_sequences.contains(&3));
    assert_eq!(plan.integrity_failure_count, 1);
    assert_eq!(plan.gap_count, 1);
}

// =========================================================================
//  4. Integrity verification failures
// =========================================================================

#[test]
fn crc32_mismatch_fails_in_fail_closed_mode() {
    let mut frame = make_frame(0, b"original");
    frame.crc32 = Some(frame.crc32.unwrap() ^ 0xDEADBEEF);

    let ndjson = frames_to_ndjson(&[frame]);
    let mut reader = ndjson.as_bytes();

    let err = decode_frames_to_raw(&mut reader).expect_err("CRC mismatch should fail");
    assert!(
        err.to_string().contains("CRC mismatch"),
        "unexpected error: {err}"
    );
}

#[test]
fn sha256_mismatch_fails_in_fail_closed_mode() {
    let mut frame = make_frame(0, b"original");
    frame.payload_sha256 = Some("0".repeat(64));

    let ndjson = frames_to_ndjson(&[frame]);
    let mut reader = ndjson.as_bytes();

    let err = decode_frames_to_raw(&mut reader).expect_err("SHA-256 mismatch should fail");
    assert!(
        err.to_string().contains("SHA-256 mismatch"),
        "unexpected error: {err}"
    );
}

#[test]
fn crc32_mismatch_skip_missing_drops_frame_and_records_telemetry() {
    let mut bad = make_frame(1, b"bad");
    bad.crc32 = Some(bad.crc32.unwrap() ^ 0xFF);
    let frames = vec![make_frame(0, b"ok-0"), bad, make_frame(2, b"ok-2")];
    let ndjson = frames_to_ndjson(&frames);
    let mut reader = ndjson.as_bytes();

    let (report, raw) =
        decode_frames_to_raw_with_policy(&mut reader, DecodeRecoveryPolicy::SkipMissing)
            .expect("should recover");

    assert_eq!(report.integrity_failures, vec![1]);
    assert_eq!(report.dropped_frames, vec![1]);
    assert!(report.gaps.is_empty());
    assert_eq!(raw, b"ok-0ok-2");
}

#[test]
fn sha256_mismatch_skip_missing_drops_frame_and_records_telemetry() {
    let mut bad = make_frame(1, b"bad");
    bad.payload_sha256 = Some("deadbeef".to_owned());
    let frames = vec![make_frame(0, b"ok-0"), bad, make_frame(2, b"ok-2")];
    let ndjson = frames_to_ndjson(&frames);
    let mut reader = ndjson.as_bytes();

    let (report, raw) =
        decode_frames_to_raw_with_policy(&mut reader, DecodeRecoveryPolicy::SkipMissing)
            .expect("should recover");

    assert_eq!(report.integrity_failures, vec![1]);
    assert_eq!(report.dropped_frames, vec![1]);
    assert_eq!(raw, b"ok-0ok-2");
}

#[test]
fn multiple_integrity_failures_tracked_separately() {
    let mut bad1 = make_frame(1, b"bad1");
    bad1.crc32 = Some(bad1.crc32.unwrap() ^ 0xAA);
    let mut bad3 = make_frame(3, b"bad3");
    bad3.payload_sha256 = Some("0".repeat(64));

    let frames = vec![
        make_frame(0, b"ok-0"),
        bad1,
        make_frame(2, b"ok-2"),
        bad3,
        make_frame(4, b"ok-4"),
    ];
    let ndjson = frames_to_ndjson(&frames);
    let mut reader = ndjson.as_bytes();

    let (report, raw) =
        decode_frames_to_raw_with_policy(&mut reader, DecodeRecoveryPolicy::SkipMissing)
            .expect("should recover");

    assert_eq!(report.integrity_failures.len(), 2);
    assert!(report.integrity_failures.contains(&1));
    assert!(report.integrity_failures.contains(&3));
    assert_eq!(report.dropped_frames.len(), 2);
    assert_eq!(raw, b"ok-0ok-2ok-4");
}

#[test]
fn no_integrity_hashes_means_no_integrity_failures() {
    // Frame without CRC or SHA should still decode successfully
    let data = b"raw-data";
    let compressed = compress(data);
    let frame = TtyAudioFrame {
        protocol_version: 1,
        seq: 0,
        codec: "mulaw+zlib+b64".to_owned(),
        sample_rate_hz: 8_000,
        channels: 1,
        payload_b64: STANDARD_NO_PAD.encode(compressed),
        crc32: None,
        payload_sha256: None,
    };
    let ndjson = frames_to_ndjson(&[frame]);
    let mut reader = ndjson.as_bytes();

    let (report, raw) = decode_frames_to_raw(&mut reader).expect("decode should succeed");
    assert!(report.integrity_failures.is_empty());
    assert_eq!(raw, data);
}

// =========================================================================
//  5. Mixed audio+control stream processing
// =========================================================================

#[test]
fn mixed_stream_handshake_ack_and_audio_decodes_correctly() {
    let hs = handshake_line(1, 1, &["mulaw+zlib+b64"]);
    let ack = handshake_ack_line(1, "mulaw+zlib+b64");
    let f0 = serde_json::to_string(&make_frame(0, b"data-0")).unwrap();
    let f1 = serde_json::to_string(&make_frame(1, b"data-1")).unwrap();

    let ndjson = format!("{hs}\n{ack}\n{f0}\n{f1}\n");
    let mut reader = ndjson.as_bytes();

    let (report, raw) = decode_frames_to_raw(&mut reader).expect("decode");
    assert_eq!(report.frames_decoded, 2);
    assert!(report.gaps.is_empty());
    assert!(report.integrity_failures.is_empty());
    assert_eq!(raw, b"data-0data-1");
}

#[test]
fn mixed_stream_handshake_audio_ack_backpressure_audio() {
    let hs = handshake_line(1, 1, &["mulaw+zlib+b64"]);
    let f0 = serde_json::to_string(&make_frame(0, b"chunk-0")).unwrap();
    let ack = serde_json::to_string(&TtyControlFrame::Ack { up_to_seq: 0 }).unwrap();
    let bp = serde_json::to_string(&TtyControlFrame::Backpressure {
        remaining_capacity: 50,
    })
    .unwrap();
    let f1 = serde_json::to_string(&make_frame(1, b"chunk-1")).unwrap();
    let f2 = serde_json::to_string(&make_frame(2, b"chunk-2")).unwrap();

    let ndjson = format!("{hs}\n{f0}\n{ack}\n{bp}\n{f1}\n{f2}\n");
    let mut reader = ndjson.as_bytes();

    let (report, raw) = decode_frames_to_raw(&mut reader).expect("decode");
    assert_eq!(report.frames_decoded, 3);
    assert!(report.gaps.is_empty());
    assert!(report.duplicates.is_empty());
    assert!(report.integrity_failures.is_empty());
    assert!(report.dropped_frames.is_empty());
    assert_eq!(raw, b"chunk-0chunk-1chunk-2");
}

#[test]
fn mixed_stream_with_retransmit_request_between_frames() {
    let hs = handshake_line(1, 1, &["mulaw+zlib+b64"]);
    let f0 = serde_json::to_string(&make_frame(0, b"frame-0")).unwrap();
    let rtx = serde_json::to_string(&TtyControlFrame::RetransmitRequest {
        sequences: vec![99],
    })
    .unwrap();
    let f1 = serde_json::to_string(&make_frame(1, b"frame-1")).unwrap();

    let ndjson = format!("{hs}\n{f0}\n{rtx}\n{f1}\n");
    let mut reader = ndjson.as_bytes();

    let (report, raw) = decode_frames_to_raw(&mut reader).expect("decode");
    assert_eq!(report.frames_decoded, 2);
    assert_eq!(raw, b"frame-0frame-1");
}

#[test]
fn control_frame_before_handshake_or_audio_is_rejected() {
    let ack = serde_json::to_string(&TtyControlFrame::Ack { up_to_seq: 0 }).unwrap();
    let f0 = serde_json::to_string(&make_frame(0, b"data")).unwrap();
    let ndjson = format!("{ack}\n{f0}\n");
    let mut reader = ndjson.as_bytes();

    let err = decode_frames_to_raw(&mut reader).expect_err("control before handshake should fail");
    assert!(
        err.to_string().contains("received before handshake"),
        "unexpected error: {err}"
    );
}

// =========================================================================
//  6. Telemetry counter accuracy after various operations
// =========================================================================

#[test]
fn telemetry_counters_zero_on_clean_decode() {
    let frames = vec![
        make_frame(0, b"a"),
        make_frame(1, b"b"),
        make_frame(2, b"c"),
    ];
    let ndjson = frames_to_ndjson(&frames);
    let mut reader = ndjson.as_bytes();

    let (report, _) = decode_frames_to_raw(&mut reader).expect("decode");
    assert_eq!(report.frames_decoded, 3);
    assert_eq!(report.gaps.len(), 0);
    assert_eq!(report.duplicates.len(), 0);
    assert_eq!(report.integrity_failures.len(), 0);
    assert_eq!(report.dropped_frames.len(), 0);
}

#[test]
fn telemetry_counters_combined_gap_integrity_duplicate() {
    // Build a stream with:
    // - Frame 0: ok
    // - Frame 1: ok
    // - (Frame 2 missing — gap)
    // - Frame 3: integrity failure (bad CRC)
    // - Frame 4: ok
    // - Frame 4: duplicate
    // - Frame 5: ok
    let mut bad3 = make_frame(3, b"bad3");
    bad3.crc32 = Some(bad3.crc32.unwrap() ^ 0xBEEF);

    let frames = vec![
        make_frame(0, b"ok-0"),
        make_frame(1, b"ok-1"),
        // seq 2 missing
        bad3,
        make_frame(4, b"ok-4"),
        make_frame(4, b"ok-4-dup"), // duplicate seq
        make_frame(5, b"ok-5"),
    ];
    let ndjson = frames_to_ndjson(&frames);
    let mut reader = ndjson.as_bytes();

    let (report, raw) =
        decode_frames_to_raw_with_policy(&mut reader, DecodeRecoveryPolicy::SkipMissing)
            .expect("should recover");

    // Telemetry verification
    assert_eq!(report.frames_decoded, 6);
    assert_eq!(report.gaps.len(), 1, "one gap (seq 2)");
    assert_eq!(report.gaps[0].expected, 2);
    assert_eq!(report.gaps[0].got, 3);
    assert_eq!(
        report.integrity_failures.len(),
        1,
        "one integrity failure (seq 3)"
    );
    assert!(report.integrity_failures.contains(&3));
    assert_eq!(report.duplicates.len(), 1, "one duplicate (seq 4)");
    assert!(report.duplicates.contains(&4));
    // Dropped = integrity failures + duplicates
    assert_eq!(report.dropped_frames.len(), 2);
    assert!(report.dropped_frames.contains(&3)); // integrity
    assert!(report.dropped_frames.contains(&4)); // duplicate

    // Raw output should only contain ok frames (0, 1, 4, 5)
    assert_eq!(raw, b"ok-0ok-1ok-4ok-5");
}

#[test]
fn retransmit_plan_telemetry_counters_match_report() {
    let report = DecodeReport {
        frames_decoded: 10,
        gaps: vec![
            SequenceGap {
                expected: 2,
                got: 4,
            },
            SequenceGap {
                expected: 7,
                got: 9,
            },
        ],
        duplicates: vec![5],
        integrity_failures: vec![6],
        dropped_frames: vec![5, 6],
        recovery_policy: DecodeRecoveryPolicy::SkipMissing,
    };

    let plan = retransmit_plan_from_report(&report);

    // Gap candidates: [2,3] + [7,8] = [2,3,7,8]
    // Integrity: [6]
    // Merged and deduped: [2,3,6,7,8]
    assert_eq!(plan.requested_sequences, vec![2, 3, 6, 7, 8]);
    assert_eq!(plan.gap_count, 2);
    assert_eq!(plan.integrity_failure_count, 1);
    assert_eq!(plan.dropped_frame_count, 2);
    assert_eq!(plan.protocol_version, 1);
}

#[test]
fn adaptive_bitrate_telemetry_tracks_delivery_outcomes() {
    let mut abr = AdaptiveBitrateController::new(64_000);

    // Record individual deliveries
    abr.record_delivery(true);
    abr.record_delivery(true);
    abr.record_delivery(false);
    assert_eq!(abr.frames_sent(), 3);
    assert_eq!(abr.frames_lost(), 1);

    // Record a batch
    abr.record_batch(7, 2);
    assert_eq!(abr.frames_sent(), 12);
    assert_eq!(abr.frames_lost(), 3);

    // frame_loss_rate = 3/12 = 0.25 > 0.10 => poor quality
    assert_eq!(abr.critical_frame_redundancy(), 3);
}

#[test]
fn adaptive_bitrate_quality_transitions() {
    let mut abr = AdaptiveBitrateController::new(64_000);

    // Perfect link
    abr.record_batch(1000, 0);
    assert_eq!(abr.critical_frame_redundancy(), 1);
    assert!((abr.current_quality - 1.0).abs() < f64::EPSILON);

    // Add some loss to push into moderate range (target ~5%)
    // total = 1000 + 50 + 3 = 1053, lost = 0 + 3 = 3, rate ~0.28%
    // Need about 1% => let's add a targeted batch
    abr.record_batch(0, 11); // total = 1011, lost = 11, rate ~1.09%
    assert_eq!(abr.critical_frame_redundancy(), 2); // moderate

    // Push into poor territory (>10%)
    abr.record_batch(0, 200); // total = 1211, lost = 211, rate ~17.4%
    assert_eq!(abr.critical_frame_redundancy(), 3); // poor
}

// =========================================================================
//  7. Recovery output format correctness
// =========================================================================

#[test]
fn emit_retransmit_loop_output_format_with_gaps() {
    let frames = vec![make_frame(0, b"a"), make_frame(3, b"d")];
    let ndjson = frames_to_ndjson(&frames);
    let mut reader = ndjson.as_bytes();
    let mut out = Vec::new();

    emit_retransmit_loop_from_reader(&mut reader, DecodeRecoveryPolicy::SkipMissing, 2, &mut out)
        .expect("emit loop");

    let text = String::from_utf8(out).expect("utf8");
    let lines: Vec<&str> = text.lines().collect();

    // 2 rounds of retransmit_request + 1 retransmit_response = 3 lines
    assert_eq!(lines.len(), 3);

    // All lines must be valid JSON
    for line in &lines {
        let _: serde_json::Value =
            serde_json::from_str(line).expect("each line must be valid JSON");
    }

    // First two are retransmit_request
    for line in &lines[..2] {
        let frame: TtyControlFrame = serde_json::from_str(line).expect("parse");
        match frame {
            TtyControlFrame::RetransmitRequest { sequences } => {
                assert_eq!(sequences, vec![1, 2]);
            }
            other => panic!("expected RetransmitRequest, got {other:?}"),
        }
    }

    // Last is retransmit_response
    let last: TtyControlFrame = serde_json::from_str(lines[2]).expect("parse");
    match last {
        TtyControlFrame::RetransmitResponse { sequences } => {
            assert_eq!(sequences, vec![1, 2]);
        }
        other => panic!("expected RetransmitResponse, got {other:?}"),
    }
}

#[test]
fn emit_retransmit_loop_ack_when_no_issues() {
    let frames = vec![
        make_frame(0, b"ok-0"),
        make_frame(1, b"ok-1"),
        make_frame(2, b"ok-2"),
    ];
    let ndjson = frames_to_ndjson(&frames);
    let mut reader = ndjson.as_bytes();
    let mut out = Vec::new();

    emit_retransmit_loop_from_reader(&mut reader, DecodeRecoveryPolicy::SkipMissing, 5, &mut out)
        .expect("emit loop");

    let text = String::from_utf8(out).expect("utf8");
    let parsed: TtyControlFrame = serde_json::from_str(text.trim()).expect("json");
    match parsed {
        TtyControlFrame::Ack { up_to_seq } => assert_eq!(up_to_seq, 0),
        other => panic!("expected Ack, got {other:?}"),
    }
}

#[test]
fn retransmit_plan_ranges_are_collapsed_correctly() {
    let report = DecodeReport {
        frames_decoded: 10,
        gaps: vec![SequenceGap {
            expected: 1,
            got: 6,
        }],
        duplicates: vec![],
        integrity_failures: vec![8],
        dropped_frames: vec![8],
        recovery_policy: DecodeRecoveryPolicy::SkipMissing,
    };

    let plan = retransmit_plan_from_report(&report);

    // Sequences: [1,2,3,4,5] from gap + [8] from integrity = [1,2,3,4,5,8]
    assert_eq!(plan.requested_sequences, vec![1, 2, 3, 4, 5, 8]);
    // Ranges: [1-5] and [8-8]
    assert_eq!(plan.requested_ranges.len(), 2);
    assert_eq!(
        plan.requested_ranges[0],
        RetransmitRange {
            start_seq: 1,
            end_seq: 5
        }
    );
    assert_eq!(
        plan.requested_ranges[1],
        RetransmitRange {
            start_seq: 8,
            end_seq: 8
        }
    );
}

#[test]
fn session_close_validation_normal_matching() {
    let close = ControlFrameType::SessionClose {
        reason: SessionCloseReason::Normal,
        last_data_seq: Some(10),
    };
    validate_session_close(Some(10), &close).expect("should validate");
}

#[test]
fn session_close_validation_mismatch_fails() {
    let close = ControlFrameType::SessionClose {
        reason: SessionCloseReason::Normal,
        last_data_seq: Some(10),
    };
    let err = validate_session_close(Some(5), &close).expect_err("mismatch should fail");
    assert!(err.to_string().contains("mismatch"));
}

#[test]
fn session_close_validation_no_data_but_claimed_fails() {
    let close = ControlFrameType::SessionClose {
        reason: SessionCloseReason::Error,
        last_data_seq: Some(5),
    };
    let err = validate_session_close(None, &close).expect_err("should fail");
    assert!(err.to_string().contains("no data frames were observed"));
}

// =========================================================================
//  8. RetransmitLoop deterministic recovery
// =========================================================================

#[test]
fn retransmit_loop_no_loss_reports_zero_everything() {
    let frames = vec![
        make_frame(0, b"a"),
        make_frame(1, b"b"),
        make_frame(2, b"c"),
    ];
    let mut rtx = RetransmitLoop::new(5, 1000, RecoveryStrategy::Simple, frames);
    rtx.run().expect("run should succeed");

    let report = rtx.report();
    assert_eq!(report.total_frames, 3);
    assert_eq!(report.lost_frames, 0);
    assert_eq!(report.recovered_frames, 0);
    assert_eq!(report.rounds_used, 0);
}

#[test]
fn retransmit_loop_single_loss_simple_strategy() {
    let frames = vec![
        make_frame(0, b"a"),
        make_frame(1, b"b"),
        make_frame(2, b"c"),
    ];
    let mut rtx = RetransmitLoop::new(5, 1000, RecoveryStrategy::Simple, frames);
    rtx.inject_loss(&[1]);
    rtx.run().expect("run");

    let report = rtx.report();
    assert_eq!(report.total_frames, 3);
    assert_eq!(report.lost_frames, 1);
    assert_eq!(report.recovered_frames, 1);
    // Simple recovers 1 per round, so 1 round needed
    assert_eq!(report.rounds_used, 1);
}

#[test]
fn retransmit_loop_multiple_losses_strategy_escalation() {
    // 5 frames, lose 3 of them
    let frames = vec![
        make_frame(0, b"a"),
        make_frame(1, b"b"),
        make_frame(2, b"c"),
        make_frame(3, b"d"),
        make_frame(4, b"e"),
    ];
    let mut rtx = RetransmitLoop::new(10, 1000, RecoveryStrategy::Simple, frames);
    rtx.inject_loss(&[1, 2, 3]);
    rtx.run().expect("run");

    let report = rtx.report();
    assert_eq!(report.total_frames, 5);
    assert_eq!(report.lost_frames, 3);
    assert_eq!(report.recovered_frames, 3);
    // Round 1 (Simple): recover 1 frame => 2 remaining
    // Round 2 (Redundant): recover 2 frames => 0 remaining
    assert_eq!(report.rounds_used, 2);
    // Strategy escalated: Simple -> Redundant -> Escalate (after round 2)
    assert_eq!(report.final_strategy, RecoveryStrategy::Escalate);
}

#[test]
fn retransmit_loop_max_rounds_limits_recovery() {
    // Lose many frames with limited rounds
    let frames: Vec<_> = (0..10).map(|i| make_frame(i, &[i as u8; 4])).collect();
    let mut rtx = RetransmitLoop::new(2, 1000, RecoveryStrategy::Simple, frames);
    rtx.inject_loss(&[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
    rtx.run().expect("run");

    let report = rtx.report();
    assert_eq!(report.total_frames, 10);
    assert_eq!(report.lost_frames, 10);
    // Round 1 (Simple): recover 1
    // Round 2 (Redundant): recover 2
    // Total recovered: 3 out of 10
    assert_eq!(report.recovered_frames, 3);
    assert_eq!(report.rounds_used, 2);
}

#[test]
fn retransmit_loop_inject_loss_ignores_unknown_sequences() {
    let frames = vec![make_frame(0, b"a"), make_frame(1, b"b")];
    let mut rtx = RetransmitLoop::new(5, 1000, RecoveryStrategy::Simple, frames);
    // Seq 99 does not exist in the buffer
    rtx.inject_loss(&[1, 99]);
    rtx.run().expect("run");

    let report = rtx.report();
    assert_eq!(report.lost_frames, 1, "only seq 1 should be lost");
    assert_eq!(report.recovered_frames, 1);
}

#[test]
fn retransmit_loop_inject_loss_resets_state() {
    let frames = vec![
        make_frame(0, b"a"),
        make_frame(1, b"b"),
        make_frame(2, b"c"),
    ];
    let mut rtx = RetransmitLoop::new(5, 1000, RecoveryStrategy::Simple, frames);

    // First run: lose seq 0
    rtx.inject_loss(&[0]);
    rtx.run().expect("run");
    assert_eq!(rtx.report().recovered_frames, 1);

    // Re-inject: new loss pattern resets state
    rtx.inject_loss(&[1, 2]);
    rtx.run().expect("run");

    let report = rtx.report();
    assert_eq!(report.lost_frames, 2);
    assert_eq!(report.recovered_frames, 2);
}

#[test]
fn retransmit_loop_escalate_strategy_ceiling() {
    // RecoveryStrategy::Escalate.escalate() should return Escalate (ceiling)
    assert_eq!(
        RecoveryStrategy::Simple.escalate(),
        RecoveryStrategy::Redundant
    );
    assert_eq!(
        RecoveryStrategy::Redundant.escalate(),
        RecoveryStrategy::Escalate
    );
    assert_eq!(
        RecoveryStrategy::Escalate.escalate(),
        RecoveryStrategy::Escalate
    );
}

#[test]
fn retransmit_loop_starting_at_redundant_strategy() {
    // Start at Redundant, which recovers 2 per round
    let frames = vec![
        make_frame(0, b"a"),
        make_frame(1, b"b"),
        make_frame(2, b"c"),
        make_frame(3, b"d"),
    ];
    let mut rtx = RetransmitLoop::new(5, 1000, RecoveryStrategy::Redundant, frames);
    rtx.inject_loss(&[0, 1]);
    rtx.run().expect("run");

    let report = rtx.report();
    assert_eq!(report.lost_frames, 2);
    assert_eq!(report.recovered_frames, 2);
    // Redundant recovers 2 per round, so 1 round suffices
    assert_eq!(report.rounds_used, 1);
}

#[test]
fn retransmit_loop_starting_at_escalate_strategy() {
    // Start at Escalate, which recovers 4 per round
    let frames: Vec<_> = (0..6).map(|i| make_frame(i, &[i as u8; 4])).collect();
    let mut rtx = RetransmitLoop::new(5, 1000, RecoveryStrategy::Escalate, frames);
    rtx.inject_loss(&[0, 1, 2, 3]);
    rtx.run().expect("run");

    let report = rtx.report();
    assert_eq!(report.lost_frames, 4);
    assert_eq!(report.recovered_frames, 4);
    assert_eq!(report.rounds_used, 1);
    assert_eq!(report.final_strategy, RecoveryStrategy::Escalate);
}

// =========================================================================
//  9. Adaptive bitrate FEC emission format
// =========================================================================

#[test]
fn adaptive_bitrate_fec_emits_correct_ndjson_lines() {
    let mut abr = AdaptiveBitrateController::new(64_000);
    // Moderate link -> redundancy = 2
    abr.record_batch(90, 5);
    assert_eq!(abr.critical_frame_redundancy(), 2);

    let frame = TtyControlFrame::Handshake {
        min_version: 1,
        max_version: 1,
        supported_codecs: vec!["mulaw+zlib+b64".to_owned()],
    };
    let mut buf = Vec::new();
    let count = abr
        .emit_critical_frame_with_fec(&mut buf, &frame)
        .expect("emit");
    assert_eq!(count, 2);

    let text = String::from_utf8(buf).expect("utf8");
    let lines: Vec<&str> = text.lines().collect();
    assert_eq!(lines.len(), 2);

    // Both lines should be identical handshake frames
    for line in &lines {
        let parsed: serde_json::Value = serde_json::from_str(line).expect("json");
        assert_eq!(parsed["frame_type"], "handshake");
    }
}

#[test]
fn adaptive_bitrate_compress_roundtrips() {
    let abr = AdaptiveBitrateController::new(64_000);
    let data = b"test payload for adaptive compression roundtrip";
    let compressed = abr.compress_adaptive(data).expect("compress");

    use flate2::read::ZlibDecoder;
    use std::io::Read;
    let mut decoder = ZlibDecoder::new(&compressed[..]);
    let mut decompressed = Vec::new();
    decoder.read_to_end(&mut decompressed).expect("decompress");
    assert_eq!(decompressed, data);
}

// =========================================================================
// 10. Duplicate frame handling telemetry
// =========================================================================

#[test]
fn duplicate_frame_in_fail_closed_mode_errors() {
    let frames = vec![
        make_frame(0, b"ok"),
        make_frame(1, b"first"),
        make_frame(1, b"dup"),
    ];
    let ndjson = frames_to_ndjson(&frames);
    let mut reader = ndjson.as_bytes();

    let err = decode_frames_to_raw(&mut reader).expect_err("duplicate should fail");
    assert!(
        err.to_string()
            .contains("duplicate tty-audio frame sequence")
    );
}

#[test]
fn duplicate_frame_in_skip_missing_mode_drops_and_records() {
    let frames = vec![
        make_frame(0, b"ok-0"),
        make_frame(1, b"ok-1"),
        make_frame(1, b"dup-1"),
        make_frame(2, b"ok-2"),
    ];
    let ndjson = frames_to_ndjson(&frames);
    let mut reader = ndjson.as_bytes();

    let (report, raw) =
        decode_frames_to_raw_with_policy(&mut reader, DecodeRecoveryPolicy::SkipMissing)
            .expect("should recover");

    assert_eq!(report.duplicates, vec![1]);
    assert_eq!(report.dropped_frames, vec![1]);
    assert_eq!(raw, b"ok-0ok-1ok-2");
}

// =========================================================================
// 11. End-to-end: mixed stream with all anomaly types under SkipMissing
// =========================================================================

#[test]
fn end_to_end_mixed_anomalies_skip_missing() {
    let hs = handshake_line(1, 1, &["mulaw+zlib+b64"]);

    let f0 = serde_json::to_string(&make_frame(0, b"good-0")).unwrap();
    let f1 = serde_json::to_string(&make_frame(1, b"good-1")).unwrap();
    // Gap: frame 2 missing
    let mut bad3 = make_frame(3, b"corrupt-3");
    bad3.crc32 = Some(bad3.crc32.unwrap() ^ 0xDEAD);
    let f3 = serde_json::to_string(&bad3).unwrap();
    let f4 = serde_json::to_string(&make_frame(4, b"good-4")).unwrap();
    let ack = serde_json::to_string(&TtyControlFrame::Ack { up_to_seq: 1 }).unwrap();
    let f5 = serde_json::to_string(&make_frame(5, b"good-5")).unwrap();

    let ndjson = format!("{hs}\n{f0}\n{f1}\n{f3}\n{f4}\n{ack}\n{f5}\n");
    let mut reader = ndjson.as_bytes();

    let (report, raw) =
        decode_frames_to_raw_with_policy(&mut reader, DecodeRecoveryPolicy::SkipMissing)
            .expect("should recover from mixed anomalies");

    // Verify telemetry
    assert_eq!(report.frames_decoded, 5, "5 audio frames in input");
    assert_eq!(report.gaps.len(), 1, "gap at seq 2");
    assert_eq!(report.gaps[0].expected, 2);
    assert_eq!(report.gaps[0].got, 3);
    assert_eq!(report.integrity_failures, vec![3], "seq 3 CRC mismatch");
    assert_eq!(report.dropped_frames, vec![3], "seq 3 dropped");
    assert!(report.duplicates.is_empty());
    assert_eq!(report.recovery_policy, DecodeRecoveryPolicy::SkipMissing);

    // Verify raw output: frames 0, 1, 4, 5 (2 missing, 3 dropped)
    assert_eq!(raw, b"good-0good-1good-4good-5");

    // Verify retransmit plan from this report
    let plan = retransmit_plan_from_report(&report);
    // Gap candidates: seq 2
    // Integrity failures: seq 3
    // Merged: [2, 3]
    assert_eq!(plan.requested_sequences, vec![2, 3]);
    assert_eq!(plan.gap_count, 1);
    assert_eq!(plan.integrity_failure_count, 1);
    assert_eq!(plan.dropped_frame_count, 1);
}
