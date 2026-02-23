//! Tests for TTY transcript correction control frames (bd-qlt.16).
//!
//! These tests cover the wire-format serialization, conversion, and emission
//! of `TranscriptSegmentCompact` and the transcript-related `TtyControlFrame`
//! variants (`TranscriptPartial`, `TranscriptRetract`, `TranscriptCorrect`).
//! No TUI feature gate required.

#![forbid(unsafe_code)]

use franken_whisper::model::TranscriptionSegment;
use franken_whisper::speculation::CorrectionEvent;
use franken_whisper::tty_audio::{
    TRANSCRIPT_PROTOCOL_VERSION, TranscriptSegmentCompact, TtyControlFrame,
    emit_tty_transcript_correct, emit_tty_transcript_partial, emit_tty_transcript_retract,
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn make_seg(text: &str, start: f64, end: f64, conf: f64) -> TranscriptionSegment {
    TranscriptionSegment {
        start_sec: Some(start),
        end_sec: Some(end),
        text: text.to_owned(),
        speaker: None,
        confidence: Some(conf),
    }
}

fn make_compact(text: &str, start: f64, end: f64, conf: f64) -> TranscriptSegmentCompact {
    TranscriptSegmentCompact {
        s: Some(start),
        e: Some(end),
        t: text.to_owned(),
        sp: None,
        c: Some(conf),
    }
}

// ---------------------------------------------------------------------------
// 1-5: TranscriptSegmentCompact tests
// ---------------------------------------------------------------------------

#[test]
fn compact_segment_from_model_segment() {
    let seg = make_seg("hello world", 0.0, 1.0, 0.75);
    let compact = TranscriptSegmentCompact::from(&seg);
    assert_eq!(compact.t, "hello world");
    assert_eq!(compact.s, Some(0.0));
    assert_eq!(compact.e, Some(1.0));
    assert_eq!(compact.c, Some(0.75));
    assert_eq!(compact.sp, None);
}

#[test]
fn compact_segment_preserves_text() {
    let seg = TranscriptionSegment {
        start_sec: Some(0.25),
        end_sec: Some(0.5),
        text: "Le texte en fran\u{00e7}ais".to_owned(),
        speaker: Some("Alice".to_owned()),
        confidence: Some(0.875),
    };
    let compact = TranscriptSegmentCompact::from(&seg);
    assert_eq!(compact.t, "Le texte en fran\u{00e7}ais");
    assert_eq!(compact.sp, Some("Alice".to_owned()));
}

#[test]
fn compact_segment_preserves_timestamps() {
    let seg = make_seg("ts", 0.125, 2.5, 0.5);
    let compact = TranscriptSegmentCompact::from(&seg);
    assert_eq!(compact.s, Some(0.125));
    assert_eq!(compact.e, Some(2.5));
}

#[test]
fn compact_segment_preserves_confidence() {
    let seg = make_seg("conf", 0.0, 1.0, 0.25);
    let compact = TranscriptSegmentCompact::from(&seg);
    assert_eq!(compact.c, Some(0.25));
}

#[test]
fn compact_segment_serde_round_trip() {
    let original = make_compact("round trip", 0.5, 1.5, 0.75);
    let json = serde_json::to_string(&original).expect("serialize");
    let restored: TranscriptSegmentCompact = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(restored.t, original.t);
    assert_eq!(restored.s, original.s);
    assert_eq!(restored.e, original.e);
    assert_eq!(restored.c, original.c);
    assert_eq!(restored.sp, original.sp);
}

// ---------------------------------------------------------------------------
// 6-8: TtyControlFrame transcript variant serde round-trips
// ---------------------------------------------------------------------------

#[test]
fn transcript_partial_serde_round_trip() {
    let frame = TtyControlFrame::TranscriptPartial {
        seq: 42,
        window_id: 7,
        segments: vec![make_compact("partial", 0.0, 0.5, 0.75)],
        model_id: "fast-v1".to_owned(),
        speculative: true,
    };
    let json = serde_json::to_string(&frame).expect("serialize");
    let restored: TtyControlFrame = serde_json::from_str(&json).expect("deserialize");
    if let TtyControlFrame::TranscriptPartial {
        seq,
        window_id,
        segments,
        model_id,
        speculative,
    } = restored
    {
        assert_eq!(seq, 42);
        assert_eq!(window_id, 7);
        assert_eq!(segments.len(), 1);
        assert_eq!(segments[0].t, "partial");
        assert_eq!(model_id, "fast-v1");
        assert!(speculative);
    } else {
        panic!("expected TranscriptPartial variant");
    }
}

#[test]
fn transcript_retract_serde_round_trip() {
    let frame = TtyControlFrame::TranscriptRetract {
        retracted_seq: 42,
        window_id: 7,
        reason: "quality model disagreed".to_owned(),
    };
    let json = serde_json::to_string(&frame).expect("serialize");
    let restored: TtyControlFrame = serde_json::from_str(&json).expect("deserialize");
    if let TtyControlFrame::TranscriptRetract {
        retracted_seq,
        window_id,
        reason,
    } = restored
    {
        assert_eq!(retracted_seq, 42);
        assert_eq!(window_id, 7);
        assert_eq!(reason, "quality model disagreed");
    } else {
        panic!("expected TranscriptRetract variant");
    }
}

#[test]
fn transcript_correct_serde_round_trip() {
    let frame = TtyControlFrame::TranscriptCorrect {
        correction_id: 10,
        replaces_seq: 42,
        window_id: 7,
        segments: vec![make_compact("corrected text", 0.0, 1.0, 0.875)],
        model_id: "quality-v2".to_owned(),
        drift_wer: 0.25,
    };
    let json = serde_json::to_string(&frame).expect("serialize");
    let restored: TtyControlFrame = serde_json::from_str(&json).expect("deserialize");
    if let TtyControlFrame::TranscriptCorrect {
        correction_id,
        replaces_seq,
        window_id,
        segments,
        model_id,
        drift_wer,
    } = restored
    {
        assert_eq!(correction_id, 10);
        assert_eq!(replaces_seq, 42);
        assert_eq!(window_id, 7);
        assert_eq!(segments.len(), 1);
        assert_eq!(segments[0].t, "corrected text");
        assert_eq!(model_id, "quality-v2");
        assert_eq!(drift_wer, 0.25);
    } else {
        panic!("expected TranscriptCorrect variant");
    }
}

// ---------------------------------------------------------------------------
// 9-11: type field checks
// ---------------------------------------------------------------------------

#[test]
fn transcript_partial_has_type_field() {
    let frame = TtyControlFrame::TranscriptPartial {
        seq: 1,
        window_id: 0,
        segments: vec![],
        model_id: "m".to_owned(),
        speculative: false,
    };
    let val: serde_json::Value = serde_json::to_value(&frame).expect("to_value");
    assert_eq!(val["frame_type"], "transcript_partial");
}

#[test]
fn transcript_retract_has_type_field() {
    let frame = TtyControlFrame::TranscriptRetract {
        retracted_seq: 1,
        window_id: 0,
        reason: "r".to_owned(),
    };
    let val: serde_json::Value = serde_json::to_value(&frame).expect("to_value");
    assert_eq!(val["frame_type"], "transcript_retract");
}

#[test]
fn transcript_correct_has_type_field() {
    let frame = TtyControlFrame::TranscriptCorrect {
        correction_id: 1,
        replaces_seq: 0,
        window_id: 0,
        segments: vec![],
        model_id: "m".to_owned(),
        drift_wer: 0.0,
    };
    let val: serde_json::Value = serde_json::to_value(&frame).expect("to_value");
    assert_eq!(val["frame_type"], "transcript_correct");
}

// ---------------------------------------------------------------------------
// 12: wire format validity
// ---------------------------------------------------------------------------

#[test]
fn transcript_frames_are_valid_json() {
    let frames: Vec<TtyControlFrame> = vec![
        TtyControlFrame::TranscriptPartial {
            seq: 1,
            window_id: 0,
            segments: vec![make_compact("a", 0.0, 0.5, 0.75)],
            model_id: "fast".to_owned(),
            speculative: true,
        },
        TtyControlFrame::TranscriptRetract {
            retracted_seq: 1,
            window_id: 0,
            reason: "wer too high".to_owned(),
        },
        TtyControlFrame::TranscriptCorrect {
            correction_id: 1,
            replaces_seq: 1,
            window_id: 0,
            segments: vec![make_compact("b", 0.0, 0.5, 0.875)],
            model_id: "quality".to_owned(),
            drift_wer: 0.125,
        },
    ];
    for frame in &frames {
        let json_str = serde_json::to_string(frame).expect("serialize");
        let parsed: serde_json::Value = serde_json::from_str(&json_str).expect("valid JSON");
        assert!(parsed.is_object(), "frame should serialize as JSON object");
    }
}

// ---------------------------------------------------------------------------
// 13: compact segment uses short field names
// ---------------------------------------------------------------------------

#[test]
fn compact_segment_uses_short_field_names() {
    let seg = make_compact("test", 0.25, 0.75, 0.5);
    let val: serde_json::Value = serde_json::to_value(&seg).expect("to_value");
    let obj = val.as_object().expect("should be object");
    assert!(
        obj.contains_key("s"),
        "expected short field name 's' for start"
    );
    assert!(
        obj.contains_key("e"),
        "expected short field name 'e' for end"
    );
    assert!(
        obj.contains_key("t"),
        "expected short field name 't' for text"
    );
    assert!(
        obj.contains_key("c"),
        "expected short field name 'c' for confidence"
    );
    // Long names must NOT appear.
    assert!(
        !obj.contains_key("start"),
        "unexpected long field name 'start'"
    );
    assert!(!obj.contains_key("end"), "unexpected long field name 'end'");
    assert!(
        !obj.contains_key("text"),
        "unexpected long field name 'text'"
    );
    assert!(
        !obj.contains_key("confidence"),
        "unexpected long field name 'confidence'"
    );
}

// ---------------------------------------------------------------------------
// 14: protocol version
// ---------------------------------------------------------------------------

#[test]
fn protocol_version_is_2() {
    assert_eq!(TRANSCRIPT_PROTOCOL_VERSION, 2);
}

// ---------------------------------------------------------------------------
// 15-17: emission helpers produce valid NDJSON
// ---------------------------------------------------------------------------

#[test]
fn emit_transcript_partial_produces_ndjson() {
    let segments = vec![make_seg("hello", 0.0, 1.0, 0.75)];
    let mut buf: Vec<u8> = Vec::new();
    emit_tty_transcript_partial(&mut buf, 1, 0, &segments, "fast-v1", true)
        .expect("emit should succeed");
    let output = String::from_utf8(buf).expect("valid utf8");
    // NDJSON: exactly one trailing newline.
    assert!(output.ends_with('\n'), "NDJSON line must end with newline");
    let trimmed = output.trim_end();
    assert!(
        !trimmed.contains('\n'),
        "NDJSON should be a single line (no embedded newlines)"
    );
    let parsed: serde_json::Value = serde_json::from_str(trimmed).expect("valid JSON");
    assert_eq!(parsed["frame_type"], "transcript_partial");
    assert_eq!(parsed["seq"], 1);
    assert_eq!(parsed["speculative"], true);
}

#[test]
fn emit_transcript_retract_produces_ndjson() {
    let mut buf: Vec<u8> = Vec::new();
    emit_tty_transcript_retract(&mut buf, 42, 7, "wer exceeded threshold")
        .expect("emit should succeed");
    let output = String::from_utf8(buf).expect("valid utf8");
    assert!(output.ends_with('\n'));
    let trimmed = output.trim_end();
    assert!(!trimmed.contains('\n'));
    let parsed: serde_json::Value = serde_json::from_str(trimmed).expect("valid JSON");
    assert_eq!(parsed["frame_type"], "transcript_retract");
    assert_eq!(parsed["retracted_seq"], 42);
    assert_eq!(parsed["window_id"], 7);
    assert_eq!(parsed["reason"], "wer exceeded threshold");
}

#[test]
fn emit_transcript_correct_produces_ndjson() {
    let fast_segments = vec![make_seg("helo wrld", 0.0, 1.0, 0.5)];
    let corrected_segments = vec![make_seg("hello world", 0.0, 1.0, 0.875)];
    let correction = CorrectionEvent::new(
        10,
        42,
        7,
        "quality-v2".to_owned(),
        corrected_segments,
        250,
        "2026-02-22T12:00:00Z".to_owned(),
        &fast_segments,
    );
    let mut buf: Vec<u8> = Vec::new();
    emit_tty_transcript_correct(&mut buf, &correction).expect("emit should succeed");
    let output = String::from_utf8(buf).expect("valid utf8");
    assert!(output.ends_with('\n'));
    let trimmed = output.trim_end();
    assert!(!trimmed.contains('\n'));
    let parsed: serde_json::Value = serde_json::from_str(trimmed).expect("valid JSON");
    assert_eq!(parsed["frame_type"], "transcript_correct");
    assert_eq!(parsed["correction_id"], 10);
    assert_eq!(parsed["replaces_seq"], 42);
    assert_eq!(parsed["window_id"], 7);
    assert_eq!(parsed["model_id"], "quality-v2");
    // Segments should be present.
    let segs = parsed["segments"].as_array().expect("segments array");
    assert_eq!(segs.len(), 1);
    assert_eq!(segs[0]["t"], "hello world");
}

// ---------------------------------------------------------------------------
// 18-22: edge cases
// ---------------------------------------------------------------------------

#[test]
fn compact_segment_none_timestamps() {
    let seg = TranscriptionSegment {
        start_sec: None,
        end_sec: None,
        text: "no timestamps".to_owned(),
        speaker: None,
        confidence: Some(0.5),
    };
    let compact = TranscriptSegmentCompact::from(&seg);
    assert_eq!(compact.s, None);
    assert_eq!(compact.e, None);
    assert_eq!(compact.t, "no timestamps");
    // Round-trip through JSON.
    let json = serde_json::to_string(&compact).expect("serialize");
    let restored: TranscriptSegmentCompact = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(restored.s, None);
    assert_eq!(restored.e, None);
}

#[test]
fn compact_segment_none_confidence() {
    let seg = TranscriptionSegment {
        start_sec: Some(0.0),
        end_sec: Some(1.0),
        text: "no confidence".to_owned(),
        speaker: None,
        confidence: None,
    };
    let compact = TranscriptSegmentCompact::from(&seg);
    assert_eq!(compact.c, None);
    // When confidence is None, `c` should be absent from JSON (skip_serializing_if).
    let val: serde_json::Value = serde_json::to_value(&compact).expect("to_value");
    assert!(
        !val.as_object().unwrap().contains_key("c"),
        "None confidence should be omitted from JSON"
    );
}

#[test]
fn compact_segment_empty_text() {
    let seg = TranscriptionSegment {
        start_sec: Some(0.0),
        end_sec: Some(0.0),
        text: String::new(),
        speaker: None,
        confidence: Some(0.0),
    };
    let compact = TranscriptSegmentCompact::from(&seg);
    assert_eq!(compact.t, "");
    let json = serde_json::to_string(&compact).expect("serialize");
    let restored: TranscriptSegmentCompact = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(restored.t, "");
}

#[test]
fn transcript_partial_empty_segments() {
    let frame = TtyControlFrame::TranscriptPartial {
        seq: 99,
        window_id: 0,
        segments: vec![],
        model_id: "m".to_owned(),
        speculative: false,
    };
    let json = serde_json::to_string(&frame).expect("serialize");
    let restored: TtyControlFrame = serde_json::from_str(&json).expect("deserialize");
    if let TtyControlFrame::TranscriptPartial { segments, .. } = restored {
        assert!(segments.is_empty());
    } else {
        panic!("expected TranscriptPartial");
    }
}

#[test]
fn transcript_correct_multiple_segments() {
    let segments = vec![
        make_compact("first", 0.0, 0.5, 0.75),
        make_compact("second", 0.5, 1.0, 0.875),
        make_compact("third", 1.0, 1.5, 0.625),
    ];
    let frame = TtyControlFrame::TranscriptCorrect {
        correction_id: 5,
        replaces_seq: 3,
        window_id: 1,
        segments,
        model_id: "quality".to_owned(),
        drift_wer: 0.125,
    };
    let json = serde_json::to_string(&frame).expect("serialize");
    let restored: TtyControlFrame = serde_json::from_str(&json).expect("deserialize");
    if let TtyControlFrame::TranscriptCorrect {
        segments,
        drift_wer,
        ..
    } = restored
    {
        assert_eq!(segments.len(), 3);
        assert_eq!(segments[0].t, "first");
        assert_eq!(segments[1].t, "second");
        assert_eq!(segments[2].t, "third");
        assert_eq!(drift_wer, 0.125);
    } else {
        panic!("expected TranscriptCorrect");
    }
}
