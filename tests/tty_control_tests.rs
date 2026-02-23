//! Integration tests for TTY control commands and retransmit loop.
//!
//! Covers:
//! - Control frame emission (handshake, eof / session close, reset)
//! - Frame validation (protocol version, codec, integrity checks)
//! - Retransmit loop behavior (gap detection, plan generation, loop emission)
//!
//! Uses the public API from `franken_whisper::tty_audio`.

use base64::Engine;
use base64::engine::general_purpose::STANDARD_NO_PAD;
use flate2::Compression;
use flate2::write::ZlibEncoder;
use std::io::Write;

use franken_whisper::tty_audio::{
    AdaptiveBitrateController, ControlFrameType, DecodeRecoveryPolicy, DecodeReport,
    FixedCountMicSource, MicStreamConfig, RetransmitRange, SequenceGap, SessionCloseReason,
    SliceMicSource, TtyAudioFrame, TtyControlFrame, UnavailableMicSource, decode_frames_to_raw,
    decode_frames_to_raw_with_policy, emit_control_frame_to_writer,
    emit_retransmit_loop_from_reader, emit_session_close, mic_stream_event_value,
    negotiate_version, retransmit_candidates, retransmit_plan_from_reader,
    retransmit_plan_from_report, stream_mic_to_ndjson, validate_session_close,
};

// ---------------------------------------------------------------------------
// Helpers — build valid frames from raw data using the public API types
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

// =========================================================================
//  1. Control frame emission tests
// =========================================================================

#[test]
fn emit_handshake_control_frame_to_writer() {
    let frame = TtyControlFrame::Handshake {
        min_version: 1,
        max_version: 1,
        supported_codecs: vec!["mulaw+zlib+b64".to_owned()],
    };
    let mut buf = Vec::new();
    emit_control_frame_to_writer(&mut buf, &frame).expect("emit should succeed");

    let text = String::from_utf8(buf).expect("utf8");
    let lines: Vec<&str> = text.lines().collect();
    assert_eq!(lines.len(), 1, "should emit exactly one NDJSON line");

    let parsed: serde_json::Value = serde_json::from_str(lines[0]).expect("valid JSON");
    assert_eq!(parsed["frame_type"], "handshake");
    assert_eq!(parsed["min_version"], 1);
    assert_eq!(parsed["max_version"], 1);
    let codecs = parsed["supported_codecs"].as_array().expect("array");
    assert_eq!(codecs.len(), 1);
    assert_eq!(codecs[0], "mulaw+zlib+b64");
}

#[test]
fn emit_handshake_ack_control_frame() {
    let frame = TtyControlFrame::HandshakeAck {
        negotiated_version: 1,
        negotiated_codec: "mulaw+zlib+b64".to_owned(),
    };
    let mut buf = Vec::new();
    emit_control_frame_to_writer(&mut buf, &frame).expect("emit");

    let text = String::from_utf8(buf).expect("utf8");
    let parsed: serde_json::Value = serde_json::from_str(text.trim()).expect("json");
    assert_eq!(parsed["frame_type"], "handshake_ack");
    assert_eq!(parsed["negotiated_version"], 1);
    assert_eq!(parsed["negotiated_codec"], "mulaw+zlib+b64");
}

#[test]
fn emit_session_close_normal() {
    let mut buf = Vec::new();
    emit_session_close(&mut buf, SessionCloseReason::Normal, Some(42)).expect("emit");

    let text = String::from_utf8(buf).expect("utf8");
    let parsed: serde_json::Value = serde_json::from_str(text.trim()).expect("json");
    assert_eq!(parsed["frame_type"], "session_close");
    assert_eq!(parsed["reason"], "normal");
    assert_eq!(parsed["last_data_seq"], 42);
}

#[test]
fn emit_session_close_error_reason() {
    let mut buf = Vec::new();
    emit_session_close(&mut buf, SessionCloseReason::Error, None).expect("emit");

    let text = String::from_utf8(buf).expect("utf8");
    let parsed: serde_json::Value = serde_json::from_str(text.trim()).expect("json");
    assert_eq!(parsed["frame_type"], "session_close");
    assert_eq!(parsed["reason"], "error");
    assert!(parsed["last_data_seq"].is_null());
}

#[test]
fn emit_session_close_timeout_reason() {
    let mut buf = Vec::new();
    emit_session_close(&mut buf, SessionCloseReason::Timeout, Some(10)).expect("emit");

    let parsed: serde_json::Value =
        serde_json::from_str(String::from_utf8(buf).unwrap().trim()).unwrap();
    assert_eq!(parsed["reason"], "timeout");
    assert_eq!(parsed["last_data_seq"], 10);
}

#[test]
fn emit_session_close_peer_requested_reason() {
    let mut buf = Vec::new();
    emit_session_close(&mut buf, SessionCloseReason::PeerRequested, Some(0)).expect("emit");

    let parsed: serde_json::Value =
        serde_json::from_str(String::from_utf8(buf).unwrap().trim()).unwrap();
    assert_eq!(parsed["reason"], "peer_requested");
}

#[test]
fn emit_ack_frame() {
    let frame = TtyControlFrame::Ack { up_to_seq: 99 };
    let mut buf = Vec::new();
    emit_control_frame_to_writer(&mut buf, &frame).expect("emit");

    let text = String::from_utf8(buf).expect("utf8");
    let parsed: serde_json::Value = serde_json::from_str(text.trim()).expect("json");
    assert_eq!(parsed["frame_type"], "ack");
    assert_eq!(parsed["up_to_seq"], 99);
}

#[test]
fn emit_backpressure_frame() {
    let frame = TtyControlFrame::Backpressure {
        remaining_capacity: 50,
    };
    let mut buf = Vec::new();
    emit_control_frame_to_writer(&mut buf, &frame).expect("emit");

    let parsed: serde_json::Value =
        serde_json::from_str(String::from_utf8(buf).unwrap().trim()).unwrap();
    assert_eq!(parsed["frame_type"], "backpressure");
    assert_eq!(parsed["remaining_capacity"], 50);
}

#[test]
fn emit_retransmit_request_frame() {
    let frame = TtyControlFrame::RetransmitRequest {
        sequences: vec![2, 5, 8],
    };
    let mut buf = Vec::new();
    emit_control_frame_to_writer(&mut buf, &frame).expect("emit");

    let parsed: serde_json::Value =
        serde_json::from_str(String::from_utf8(buf).unwrap().trim()).unwrap();
    assert_eq!(parsed["frame_type"], "retransmit_request");
    let seqs: Vec<u64> = parsed["sequences"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap())
        .collect();
    assert_eq!(seqs, vec![2, 5, 8]);
}

#[test]
fn emit_retransmit_response_frame() {
    let frame = TtyControlFrame::RetransmitResponse {
        sequences: vec![10, 20],
    };
    let mut buf = Vec::new();
    emit_control_frame_to_writer(&mut buf, &frame).expect("emit");

    let parsed: serde_json::Value =
        serde_json::from_str(String::from_utf8(buf).unwrap().trim()).unwrap();
    assert_eq!(parsed["frame_type"], "retransmit_response");
    let seqs: Vec<u64> = parsed["sequences"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap())
        .collect();
    assert_eq!(seqs, vec![10, 20]);
}

// =========================================================================
//  2. Frame validation tests
// =========================================================================

#[test]
fn decode_rejects_unsupported_protocol_version() {
    let mut frame = make_frame(0, b"data");
    frame.protocol_version = 99;

    let ndjson = frames_to_ndjson(&[frame]);
    let mut reader = ndjson.as_bytes();
    let err = decode_frames_to_raw(&mut reader).expect_err("should fail");
    assert!(
        err.to_string()
            .contains("unsupported tty-audio protocol_version"),
        "unexpected error: {err}"
    );
}

#[test]
fn decode_rejects_unsupported_codec() {
    let mut frame = make_frame(0, b"data");
    frame.codec = "opus+b64".to_owned();

    let ndjson = frames_to_ndjson(&[frame]);
    let mut reader = ndjson.as_bytes();
    let err = decode_frames_to_raw(&mut reader).expect_err("should fail");
    assert!(
        err.to_string().contains("unsupported tty-audio codec"),
        "unexpected error: {err}"
    );
}

#[test]
fn decode_rejects_wrong_sample_rate() {
    let mut frame = make_frame(0, b"data");
    frame.sample_rate_hz = 44100;

    let ndjson = frames_to_ndjson(&[frame]);
    let mut reader = ndjson.as_bytes();
    let err = decode_frames_to_raw(&mut reader).expect_err("should fail");
    assert!(
        err.to_string().contains("unsupported tty-audio shape"),
        "unexpected error: {err}"
    );
}

#[test]
fn decode_rejects_wrong_channel_count() {
    let mut frame = make_frame(0, b"data");
    frame.channels = 2;

    let ndjson = frames_to_ndjson(&[frame]);
    let mut reader = ndjson.as_bytes();
    let err = decode_frames_to_raw(&mut reader).expect_err("should fail");
    assert!(
        err.to_string().contains("unsupported tty-audio shape"),
        "unexpected error: {err}"
    );
}

#[test]
fn decode_detects_crc32_mismatch() {
    let mut frame = make_frame(0, b"data");
    frame.crc32 = Some(frame.crc32.unwrap() ^ 0xDEAD_BEEF);

    let ndjson = frames_to_ndjson(&[frame]);
    let mut reader = ndjson.as_bytes();
    let err = decode_frames_to_raw(&mut reader).expect_err("should fail");
    assert!(
        err.to_string().contains("CRC mismatch"),
        "unexpected error: {err}"
    );
}

#[test]
fn decode_detects_sha256_mismatch() {
    let mut frame = make_frame(0, b"data");
    frame.payload_sha256 =
        Some("0000000000000000000000000000000000000000000000000000000000000000".to_owned());

    let ndjson = frames_to_ndjson(&[frame]);
    let mut reader = ndjson.as_bytes();
    let err = decode_frames_to_raw(&mut reader).expect_err("should fail");
    assert!(
        err.to_string().contains("SHA-256 mismatch"),
        "unexpected error: {err}"
    );
}

#[test]
fn decode_sequential_frames_succeeds() {
    let frames = vec![
        make_frame(0, b"chunk-a"),
        make_frame(1, b"chunk-b"),
        make_frame(2, b"chunk-c"),
    ];
    let ndjson = frames_to_ndjson(&frames);
    let mut reader = ndjson.as_bytes();

    let (report, raw) = decode_frames_to_raw(&mut reader).expect("decode should succeed");
    assert_eq!(report.frames_decoded, 3);
    assert!(report.gaps.is_empty());
    assert!(report.duplicates.is_empty());
    assert!(report.integrity_failures.is_empty());
    assert_eq!(raw, b"chunk-achunk-bchunk-c");
}

#[test]
fn decode_detects_gaps_as_error_in_fail_closed_mode() {
    let frames = vec![make_frame(0, b"first"), make_frame(5, b"sixth")];
    let ndjson = frames_to_ndjson(&frames);
    let mut reader = ndjson.as_bytes();

    let err = decode_frames_to_raw(&mut reader).expect_err("gap should fail");
    assert!(
        err.to_string().contains("missing tty-audio frame sequence"),
        "unexpected error: {err}"
    );
}

#[test]
fn decode_detects_duplicate_frames_as_error() {
    let frames = vec![
        make_frame(0, b"first"),
        make_frame(1, b"second"),
        make_frame(1, b"second-dup"),
    ];
    let ndjson = frames_to_ndjson(&frames);
    let mut reader = ndjson.as_bytes();

    let err = decode_frames_to_raw(&mut reader).expect_err("duplicate should fail");
    assert!(
        err.to_string()
            .contains("duplicate tty-audio frame sequence"),
        "unexpected error: {err}"
    );
}

#[test]
fn decode_empty_input_returns_error() {
    let ndjson = "";
    let mut reader = ndjson.as_bytes();
    let err = decode_frames_to_raw(&mut reader).expect_err("empty should fail");
    assert!(
        err.to_string().contains("no tty-audio"),
        "unexpected error: {err}"
    );
}

#[test]
fn decode_skip_missing_recovers_from_gap() {
    let frames = vec![
        make_frame(0, b"first"),
        make_frame(3, b"fourth"),
        make_frame(4, b"fifth"),
    ];
    let ndjson = frames_to_ndjson(&frames);
    let mut reader = ndjson.as_bytes();

    let (report, raw) =
        decode_frames_to_raw_with_policy(&mut reader, DecodeRecoveryPolicy::SkipMissing)
            .expect("skip missing should recover");
    assert_eq!(report.frames_decoded, 3);
    assert_eq!(report.gaps.len(), 1);
    assert_eq!(report.gaps[0].expected, 1);
    assert_eq!(report.gaps[0].got, 3);
    assert_eq!(raw, b"firstfourthfifth");
}

#[test]
fn decode_skip_missing_drops_corrupt_frame() {
    let mut bad = make_frame(1, b"bad-data");
    bad.crc32 = Some(bad.crc32.unwrap() ^ 0xFF);
    let frames = vec![make_frame(0, b"ok0"), bad, make_frame(2, b"ok2")];
    let ndjson = frames_to_ndjson(&frames);
    let mut reader = ndjson.as_bytes();

    let (report, raw) =
        decode_frames_to_raw_with_policy(&mut reader, DecodeRecoveryPolicy::SkipMissing)
            .expect("skip missing should recover");
    assert_eq!(report.integrity_failures, vec![1]);
    assert_eq!(report.dropped_frames, vec![1]);
    assert_eq!(raw, b"ok0ok2");
}

#[test]
fn decode_with_handshake_and_interleaved_control() {
    let handshake = serde_json::to_string(&TtyControlFrame::Handshake {
        min_version: 1,
        max_version: 1,
        supported_codecs: vec!["mulaw+zlib+b64".to_owned()],
    })
    .unwrap();
    let ack = serde_json::to_string(&TtyControlFrame::Ack { up_to_seq: 0 }).unwrap();
    let bp = serde_json::to_string(&TtyControlFrame::Backpressure {
        remaining_capacity: 10,
    })
    .unwrap();
    let f0 = serde_json::to_string(&make_frame(0, b"data-0")).unwrap();
    let f1 = serde_json::to_string(&make_frame(1, b"data-1")).unwrap();

    let ndjson = format!("{handshake}\n{f0}\n{ack}\n{bp}\n{f1}\n");
    let mut reader = ndjson.as_bytes();

    let (report, raw) = decode_frames_to_raw(&mut reader).expect("decode should succeed");
    assert_eq!(report.frames_decoded, 2);
    assert_eq!(raw, b"data-0data-1");
}

#[test]
fn validate_session_close_matching_seq() {
    let close = ControlFrameType::SessionClose {
        reason: SessionCloseReason::Normal,
        last_data_seq: Some(42),
    };
    validate_session_close(Some(42), &close).expect("should be valid");
}

#[test]
fn validate_session_close_mismatching_seq() {
    let close = ControlFrameType::SessionClose {
        reason: SessionCloseReason::Normal,
        last_data_seq: Some(42),
    };
    let err = validate_session_close(Some(99), &close).expect_err("should fail");
    assert!(
        err.to_string().contains("mismatch"),
        "unexpected error: {err}"
    );
}

#[test]
fn validate_session_close_no_data_frames_but_claimed() {
    let close = ControlFrameType::SessionClose {
        reason: SessionCloseReason::Error,
        last_data_seq: Some(5),
    };
    let err = validate_session_close(None, &close).expect_err("should fail");
    assert!(
        err.to_string().contains("no data frames were observed"),
        "unexpected error: {err}"
    );
}

#[test]
fn validate_session_close_no_data_frames_and_no_claim() {
    let close = ControlFrameType::SessionClose {
        reason: SessionCloseReason::Normal,
        last_data_seq: None,
    };
    validate_session_close(None, &close).expect("should be valid for empty session");
}

// =========================================================================
//  3. Retransmit loop behavior tests
// =========================================================================

#[test]
fn negotiate_version_overlapping_ranges() {
    assert_eq!(negotiate_version(1, 3, 2, 5), Ok(3));
}

#[test]
fn negotiate_version_exact_match() {
    assert_eq!(negotiate_version(1, 1, 1, 1), Ok(1));
}

#[test]
fn negotiate_version_no_overlap() {
    let result = negotiate_version(1, 2, 3, 5);
    assert!(result.is_err());
    assert!(
        result
            .unwrap_err()
            .contains("no compatible protocol version")
    );
}

#[test]
fn negotiate_version_picks_highest_compatible() {
    assert_eq!(negotiate_version(1, 10, 3, 7), Ok(7));
}

#[test]
fn retransmit_candidates_from_gaps() {
    let report = DecodeReport {
        frames_decoded: 4,
        gaps: vec![
            SequenceGap {
                expected: 1,
                got: 3,
            },
            SequenceGap {
                expected: 5,
                got: 7,
            },
        ],
        duplicates: vec![],
        integrity_failures: vec![],
        dropped_frames: vec![],
        recovery_policy: DecodeRecoveryPolicy::SkipMissing,
    };
    let candidates = retransmit_candidates(&report);
    assert_eq!(candidates, vec![1, 2, 5, 6]);
}

#[test]
fn retransmit_candidates_empty_when_no_gaps() {
    let report = DecodeReport {
        frames_decoded: 5,
        gaps: vec![],
        duplicates: vec![],
        integrity_failures: vec![],
        dropped_frames: vec![],
        recovery_policy: DecodeRecoveryPolicy::FailClosed,
    };
    assert!(retransmit_candidates(&report).is_empty());
}

#[test]
fn retransmit_plan_merges_gaps_and_integrity_failures() {
    let report = DecodeReport {
        frames_decoded: 10,
        gaps: vec![SequenceGap {
            expected: 2,
            got: 5,
        }],
        duplicates: vec![],
        integrity_failures: vec![7, 8],
        dropped_frames: vec![7, 8],
        recovery_policy: DecodeRecoveryPolicy::SkipMissing,
    };
    let plan = retransmit_plan_from_report(&report);
    assert_eq!(plan.protocol_version, 1);
    // gap sequences: 2,3,4 + integrity: 7,8 -> sorted: 2,3,4,7,8
    assert_eq!(plan.requested_sequences, vec![2, 3, 4, 7, 8]);
    assert_eq!(
        plan.requested_ranges,
        vec![
            RetransmitRange {
                start_seq: 2,
                end_seq: 4
            },
            RetransmitRange {
                start_seq: 7,
                end_seq: 8
            },
        ]
    );
    assert_eq!(plan.gap_count, 1);
    assert_eq!(plan.integrity_failure_count, 2);
    assert_eq!(plan.dropped_frame_count, 2);
}

#[test]
fn retransmit_plan_empty_when_no_issues() {
    let report = DecodeReport {
        frames_decoded: 3,
        gaps: vec![],
        duplicates: vec![],
        integrity_failures: vec![],
        dropped_frames: vec![],
        recovery_policy: DecodeRecoveryPolicy::SkipMissing,
    };
    let plan = retransmit_plan_from_report(&report);
    assert!(plan.requested_sequences.is_empty());
    assert!(plan.requested_ranges.is_empty());
    assert_eq!(plan.gap_count, 0);
    assert_eq!(plan.integrity_failure_count, 0);
}

#[test]
fn retransmit_plan_from_reader_with_gaps() {
    // Frames 0 and 3 present, missing 1 and 2
    let frames = vec![make_frame(0, b"a"), make_frame(3, b"d")];
    let ndjson = frames_to_ndjson(&frames);
    let mut reader = ndjson.as_bytes();

    let plan = retransmit_plan_from_reader(&mut reader, DecodeRecoveryPolicy::SkipMissing)
        .expect("plan from reader should succeed");
    assert_eq!(plan.requested_sequences, vec![1, 2]);
    assert_eq!(plan.gap_count, 1);
}

#[test]
fn emit_retransmit_loop_emits_requests_then_response() {
    // Gap: missing seq 1,2 between 0 and 3
    let frames = vec![make_frame(0, b"a"), make_frame(3, b"d")];
    let ndjson = frames_to_ndjson(&frames);
    let mut reader = ndjson.as_bytes();
    let mut out = Vec::new();

    emit_retransmit_loop_from_reader(&mut reader, DecodeRecoveryPolicy::SkipMissing, 2, &mut out)
        .expect("emit loop");

    let text = String::from_utf8(out).expect("utf8");
    let control_frames: Vec<TtyControlFrame> = text
        .lines()
        .map(|line| serde_json::from_str(line).expect("valid control frame JSON"))
        .collect();

    assert_eq!(
        control_frames.len(),
        3,
        "expected 2 requests + 1 response, got {}",
        control_frames.len()
    );
    // First two should be retransmit requests
    for (i, frame) in control_frames.iter().enumerate().take(2) {
        match frame {
            TtyControlFrame::RetransmitRequest { sequences } => {
                assert_eq!(sequences, &vec![1, 2]);
            }
            other => panic!("expected RetransmitRequest at index {i}, got {other:?}"),
        }
    }
    // Last should be retransmit response
    match &control_frames[2] {
        TtyControlFrame::RetransmitResponse { sequences } => {
            assert_eq!(sequences, &vec![1, 2]);
        }
        other => panic!("expected RetransmitResponse, got {other:?}"),
    }
}

#[test]
fn emit_retransmit_loop_emits_ack_when_no_missing() {
    let frames = vec![make_frame(0, b"a"), make_frame(1, b"b")];
    let ndjson = frames_to_ndjson(&frames);
    let mut reader = ndjson.as_bytes();
    let mut out = Vec::new();

    emit_retransmit_loop_from_reader(&mut reader, DecodeRecoveryPolicy::SkipMissing, 3, &mut out)
        .expect("emit loop");

    let text = String::from_utf8(out).expect("utf8");
    let parsed: TtyControlFrame = serde_json::from_str(text.trim()).expect("json");
    match parsed {
        TtyControlFrame::Ack { up_to_seq } => assert_eq!(up_to_seq, 0),
        other => panic!("expected Ack, got {other:?}"),
    }
}

#[test]
fn emit_retransmit_loop_with_single_round() {
    let frames = vec![make_frame(0, b"a"), make_frame(5, b"f")];
    let ndjson = frames_to_ndjson(&frames);
    let mut reader = ndjson.as_bytes();
    let mut out = Vec::new();

    emit_retransmit_loop_from_reader(&mut reader, DecodeRecoveryPolicy::SkipMissing, 1, &mut out)
        .expect("emit loop");

    let text = String::from_utf8(out).expect("utf8");
    let lines: Vec<&str> = text.lines().collect();
    // 1 request + 1 response = 2 lines
    assert_eq!(lines.len(), 2);
}

#[test]
fn emit_retransmit_loop_with_integrity_failures() {
    let mut corrupt = make_frame(2, b"corrupt");
    corrupt.payload_sha256 = Some("deadbeef".to_owned());

    let frames = vec![
        make_frame(0, b"ok-0"),
        make_frame(1, b"ok-1"),
        corrupt,
        make_frame(3, b"ok-3"),
    ];
    let ndjson = frames_to_ndjson(&frames);
    let mut reader = ndjson.as_bytes();
    let mut out = Vec::new();

    emit_retransmit_loop_from_reader(&mut reader, DecodeRecoveryPolicy::SkipMissing, 1, &mut out)
        .expect("emit loop");

    let text = String::from_utf8(out).expect("utf8");
    let control_frames: Vec<TtyControlFrame> = text
        .lines()
        .map(|line| serde_json::from_str(line).expect("json"))
        .collect();
    // 1 request + 1 response
    assert_eq!(control_frames.len(), 2);
    match &control_frames[0] {
        TtyControlFrame::RetransmitRequest { sequences } => {
            assert!(sequences.contains(&2), "should request corrupted seq 2");
        }
        other => panic!("expected RetransmitRequest, got {other:?}"),
    }
}

// =========================================================================
//  4. Adaptive Bitrate Controller tests
// =========================================================================

#[test]
fn adaptive_bitrate_initial_state() {
    let controller = AdaptiveBitrateController::new(64_000);
    assert_eq!(controller.target_bitrate, 64_000);
    assert!((controller.current_quality - 1.0).abs() < f64::EPSILON);
    assert!((controller.frame_loss_rate - 0.0).abs() < f64::EPSILON);
    assert_eq!(controller.frames_sent(), 0);
    assert_eq!(controller.frames_lost(), 0);
}

#[test]
fn adaptive_bitrate_high_quality_link() {
    let mut controller = AdaptiveBitrateController::new(64_000);
    // Deliver 100 frames, lose 0
    controller.record_batch(100, 0);
    assert_eq!(controller.critical_frame_redundancy(), 1);
    // flate2::Compression::fast() is level 1
    assert_eq!(controller.recommended_compression().level(), 1);
}

#[test]
fn adaptive_bitrate_moderate_quality_link() {
    let mut controller = AdaptiveBitrateController::new(64_000);
    // Deliver 90 frames, lose 5 (5% loss)
    controller.record_batch(90, 5);
    assert_eq!(controller.critical_frame_redundancy(), 2);
    assert_eq!(controller.recommended_compression().level(), 6);
}

#[test]
fn adaptive_bitrate_poor_quality_link() {
    let mut controller = AdaptiveBitrateController::new(64_000);
    // Deliver 70 frames, lose 30 (30% loss)
    controller.record_batch(70, 30);
    assert_eq!(controller.critical_frame_redundancy(), 3);
    assert_eq!(controller.recommended_compression().level(), 9);
}

#[test]
fn adaptive_bitrate_fec_emission() {
    let mut controller = AdaptiveBitrateController::new(64_000);
    // Make it a poor link
    controller.record_batch(50, 50);
    let redundancy = controller.critical_frame_redundancy();
    assert_eq!(redundancy, 3);

    let frame = TtyControlFrame::Handshake {
        min_version: 1,
        max_version: 1,
        supported_codecs: vec!["mulaw+zlib+b64".to_owned()],
    };
    let mut buf = Vec::new();
    let count = controller
        .emit_critical_frame_with_fec(&mut buf, &frame)
        .expect("emit");
    assert_eq!(count, 3);

    let text = String::from_utf8(buf).expect("utf8");
    let lines: Vec<&str> = text.lines().collect();
    assert_eq!(lines.len(), 3, "should emit 3 copies for FEC");
}

#[test]
fn adaptive_bitrate_compress_adaptive_produces_valid_zlib() {
    let controller = AdaptiveBitrateController::new(64_000);
    let data = b"test data for adaptive compression";
    let compressed = controller.compress_adaptive(data).expect("compress");
    // Decompress with flate2 to verify
    use flate2::read::ZlibDecoder;
    use std::io::Read;
    let mut decoder = ZlibDecoder::new(&compressed[..]);
    let mut decompressed = Vec::new();
    decoder.read_to_end(&mut decompressed).expect("decompress");
    assert_eq!(decompressed, data);
}

// =========================================================================
//  5. MicStreamConfig validation tests
// =========================================================================

#[test]
fn mic_stream_config_default_is_valid() {
    let config = MicStreamConfig::default();
    config.validate().expect("default should be valid");
    assert_eq!(config.chunk_ms, 200);
    assert_eq!(config.sample_rate_hz, 8_000);
    assert_eq!(config.channels, 1);
}

#[test]
fn mic_stream_config_rejects_wrong_sample_rate() {
    let config = MicStreamConfig {
        sample_rate_hz: 16_000,
        ..MicStreamConfig::default()
    };
    let err = config.validate().expect_err("should reject 16kHz");
    assert!(err.to_string().contains("unsupported"));
}

#[test]
fn mic_stream_config_rejects_stereo() {
    let config = MicStreamConfig {
        channels: 2,
        ..MicStreamConfig::default()
    };
    let err = config.validate().expect_err("should reject stereo");
    assert!(err.to_string().contains("unsupported"));
}

// =========================================================================
//  6. MicAudioSource + stream_mic_to_ndjson tests
// =========================================================================

#[test]
fn slice_mic_source_produces_frames() {
    let data = vec![0x7Fu8; 3200]; // 2 chunks at 1600 bytes each
    let mut source = SliceMicSource::new(&data);
    let config = MicStreamConfig::default();
    let mut out = Vec::new();

    let count = stream_mic_to_ndjson(&config, &mut source, &mut out).expect("stream");
    assert_eq!(
        count, 2,
        "should produce 2 frames from 3200 bytes at chunk_size=1600"
    );

    let text = String::from_utf8(out).expect("utf8");
    let lines: Vec<&str> = text.lines().collect();
    // 1 handshake + 2 mic events
    assert_eq!(lines.len(), 3);
    // First line is the handshake control frame
    let handshake: serde_json::Value = serde_json::from_str(lines[0]).expect("json");
    assert_eq!(handshake["frame_type"], "handshake");
}

#[test]
fn fixed_count_mic_source_emits_exact_count() {
    let mut source = FixedCountMicSource::new(5, 0x80);
    let config = MicStreamConfig::default();
    let mut out = Vec::new();

    let count = stream_mic_to_ndjson(&config, &mut source, &mut out).expect("stream");
    assert_eq!(count, 5);
}

#[test]
fn unavailable_mic_source_returns_error() {
    let mut source = UnavailableMicSource::new("no device found");
    let config = MicStreamConfig::default();
    let mut out = Vec::new();

    let err = stream_mic_to_ndjson(&config, &mut source, &mut out).expect_err("should fail");
    assert!(
        err.to_string().contains("no device found"),
        "unexpected error: {err}"
    );
}

#[test]
fn mic_stream_event_value_wraps_frame_correctly() {
    let frame = make_frame(7, b"payload");
    let event = mic_stream_event_value(&frame);
    assert_eq!(event.event, "mic_audio_chunk");
    assert!(!event.schema_version.is_empty());
    assert_eq!(event.frame.seq, 7);
}

// =========================================================================
//  7. Control frame serde round-trip tests
// =========================================================================

#[test]
fn control_frame_handshake_round_trips_through_json() {
    let original = TtyControlFrame::Handshake {
        min_version: 1,
        max_version: 3,
        supported_codecs: vec!["mulaw+zlib+b64".to_owned(), "opus+b64".to_owned()],
    };
    let json = serde_json::to_string(&original).expect("serialize");
    let parsed: TtyControlFrame = serde_json::from_str(&json).expect("deserialize");
    match parsed {
        TtyControlFrame::Handshake {
            min_version,
            max_version,
            supported_codecs,
        } => {
            assert_eq!(min_version, 1);
            assert_eq!(max_version, 3);
            assert_eq!(supported_codecs.len(), 2);
        }
        other => panic!("expected Handshake, got {other:?}"),
    }
}

#[test]
fn session_close_round_trips_through_json() {
    let original = ControlFrameType::SessionClose {
        reason: SessionCloseReason::Timeout,
        last_data_seq: Some(100),
    };
    let json = serde_json::to_string(&original).expect("serialize");
    let parsed: ControlFrameType = serde_json::from_str(&json).expect("deserialize");
    match parsed {
        ControlFrameType::SessionClose {
            reason,
            last_data_seq,
        } => {
            assert_eq!(reason, SessionCloseReason::Timeout);
            assert_eq!(last_data_seq, Some(100));
        }
        other => panic!("expected SessionClose, got {other:?}"),
    }
}

#[test]
fn audio_frame_without_optional_fields_deserializes() {
    let json = r#"{"seq":0,"codec":"mulaw+zlib+b64","sample_rate_hz":8000,"channels":1,"payload_b64":"abc"}"#;
    let frame: TtyAudioFrame = serde_json::from_str(json).expect("deserialize");
    assert_eq!(frame.protocol_version, 1);
    assert_eq!(frame.seq, 0);
    assert!(frame.crc32.is_none());
    assert!(frame.payload_sha256.is_none());
}

#[test]
fn audio_frame_with_all_fields_round_trips() {
    let frame = make_frame(42, b"full-payload");
    let json = serde_json::to_string(&frame).expect("serialize");
    let parsed: TtyAudioFrame = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(parsed.seq, 42);
    assert_eq!(parsed.crc32, frame.crc32);
    assert_eq!(parsed.payload_sha256, frame.payload_sha256);
    assert_eq!(parsed.protocol_version, 1);
    assert_eq!(parsed.codec, "mulaw+zlib+b64");
    assert_eq!(parsed.sample_rate_hz, 8_000);
    assert_eq!(parsed.channels, 1);
}
