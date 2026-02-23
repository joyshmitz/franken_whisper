//! Integration tests for core speculation types and robot events.
//!
//! Bead: bd-qlt.12

#![forbid(unsafe_code)]

use franken_whisper::model::TranscriptionSegment;
use franken_whisper::robot::{
    self, ROBOT_SCHEMA_VERSION, SPECULATION_STATS_REQUIRED_FIELDS,
    TRANSCRIPT_CORRECT_REQUIRED_FIELDS, TRANSCRIPT_RETRACT_REQUIRED_FIELDS,
};
use franken_whisper::speculation::{
    CorrectionDrift, CorrectionEvent, PartialStatus, PartialTranscript, SpeculationStats,
    SpeculationWindow,
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn make_segment(text: &str, confidence: f64) -> TranscriptionSegment {
    TranscriptionSegment {
        start_sec: Some(0.0),
        end_sec: Some(1.0),
        text: text.to_owned(),
        speaker: None,
        confidence: Some(confidence),
    }
}

fn make_segment_at(text: &str, start: f64, end: f64, confidence: f64) -> TranscriptionSegment {
    TranscriptionSegment {
        start_sec: Some(start),
        end_sec: Some(end),
        text: text.to_owned(),
        speaker: None,
        confidence: Some(confidence),
    }
}

// ===========================================================================
// SpeculationWindow tests
// ===========================================================================

#[test]
fn speculation_window_new_and_duration() {
    let w = SpeculationWindow::new(
        1,
        "run-001".to_owned(),
        1000,
        3000,
        200,
        "abc123".to_owned(),
    );
    assert_eq!(w.window_id, 1);
    assert_eq!(w.run_id, "run-001");
    assert_eq!(w.start_ms, 1000);
    assert_eq!(w.end_ms, 3000);
    assert_eq!(w.overlap_ms, 200);
    assert_eq!(w.audio_hash, "abc123");
    assert_eq!(w.duration_ms(), 2000);
}

#[test]
fn speculation_window_contains_ms_boundaries() {
    let w = SpeculationWindow::new(
        1,
        "run-001".to_owned(),
        1000,
        3000,
        200,
        "abc123".to_owned(),
    );
    // start is inclusive
    assert!(w.contains_ms(1000));
    // end is exclusive
    assert!(!w.contains_ms(3000));
    // middle is inside
    assert!(w.contains_ms(2000));
    // outside (below start)
    assert!(!w.contains_ms(999));
    // outside (above end)
    assert!(!w.contains_ms(3001));
}

#[test]
fn speculation_window_serde_roundtrip() {
    let w = SpeculationWindow::new(
        42,
        "run-rt".to_owned(),
        500,
        1500,
        100,
        "deadbeef".to_owned(),
    );
    let json_str = serde_json::to_string(&w).expect("serialize");
    let w2: SpeculationWindow = serde_json::from_str(&json_str).expect("deserialize");
    assert_eq!(w.window_id, w2.window_id);
    assert_eq!(w.run_id, w2.run_id);
    assert_eq!(w.start_ms, w2.start_ms);
    assert_eq!(w.end_ms, w2.end_ms);
    assert_eq!(w.overlap_ms, w2.overlap_ms);
    assert_eq!(w.audio_hash, w2.audio_hash);
}

#[test]
fn speculation_window_zero_duration() {
    let w = SpeculationWindow::new(1, "run-zd".to_owned(), 5000, 5000, 0, "hash".to_owned());
    assert_eq!(w.duration_ms(), 0);
    // When start == end, contains should return false for all values
    assert!(!w.contains_ms(5000));
    assert!(!w.contains_ms(4999));
    assert!(!w.contains_ms(5001));
}

// ===========================================================================
// PartialTranscript tests
// ===========================================================================

#[test]
fn partial_transcript_computes_confidence_mean() {
    let segments = vec![
        make_segment("hello", 0.8),
        make_segment("world", 0.6),
        make_segment("foo", 1.0),
    ];
    let pt = PartialTranscript::new(
        1,
        10,
        "fast-model".to_owned(),
        segments,
        50,
        "2026-01-01T00:00:00Z".to_owned(),
    );
    // mean of 0.8, 0.6, 1.0 = 2.4 / 3 = 0.8
    let expected = (0.8 + 0.6 + 1.0) / 3.0;
    assert!(
        (pt.confidence_mean - expected).abs() < 1e-9,
        "confidence_mean was {}, expected {}",
        pt.confidence_mean,
        expected,
    );
}

#[test]
fn partial_transcript_empty_segments_confidence_zero() {
    let pt = PartialTranscript::new(
        1,
        10,
        "fast-model".to_owned(),
        vec![],
        50,
        "2026-01-01T00:00:00Z".to_owned(),
    );
    assert!(
        (pt.confidence_mean - 0.0).abs() < 1e-9,
        "confidence_mean for empty segments should be 0.0, got {}",
        pt.confidence_mean,
    );
}

#[test]
fn partial_transcript_status_transitions() {
    let mut pt = PartialTranscript::new(
        1,
        10,
        "fast-model".to_owned(),
        vec![make_segment("hi", 0.9)],
        50,
        "2026-01-01T00:00:00Z".to_owned(),
    );
    assert_eq!(pt.status, PartialStatus::Pending);

    pt.confirm();
    assert_eq!(pt.status, PartialStatus::Confirmed);

    // Create a fresh one and retract
    let mut pt2 = PartialTranscript::new(
        2,
        10,
        "fast-model".to_owned(),
        vec![make_segment("hi", 0.9)],
        50,
        "2026-01-01T00:00:00Z".to_owned(),
    );
    assert_eq!(pt2.status, PartialStatus::Pending);
    pt2.retract();
    assert_eq!(pt2.status, PartialStatus::Retracted);
}

#[test]
fn partial_transcript_serde_roundtrip() {
    let mut pt = PartialTranscript::new(
        7,
        20,
        "fast-v1".to_owned(),
        vec![make_segment("test", 0.95)],
        42,
        "2026-01-01T00:00:00Z".to_owned(),
    );
    // Pending roundtrip
    let json_str = serde_json::to_string(&pt).expect("serialize");
    assert!(
        json_str.contains("\"pending\""),
        "status should serialize as snake_case string 'pending', got: {json_str}"
    );
    let pt_back: PartialTranscript = serde_json::from_str(&json_str).expect("deserialize");
    assert_eq!(pt_back.status, PartialStatus::Pending);
    assert_eq!(pt_back.seq, 7);
    assert_eq!(pt_back.window_id, 20);

    // Confirmed roundtrip
    pt.confirm();
    let json_str2 = serde_json::to_string(&pt).expect("serialize confirmed");
    assert!(
        json_str2.contains("\"confirmed\""),
        "status should serialize as 'confirmed', got: {json_str2}"
    );
    let pt_back2: PartialTranscript =
        serde_json::from_str(&json_str2).expect("deserialize confirmed");
    assert_eq!(pt_back2.status, PartialStatus::Confirmed);

    // Retracted roundtrip
    let mut pt3 = PartialTranscript::new(
        8,
        20,
        "fast-v1".to_owned(),
        vec![],
        10,
        "2026-01-01T00:00:00Z".to_owned(),
    );
    pt3.retract();
    let json_str3 = serde_json::to_string(&pt3).expect("serialize retracted");
    assert!(
        json_str3.contains("\"retracted\""),
        "status should serialize as 'retracted', got: {json_str3}"
    );
    let pt_back3: PartialTranscript =
        serde_json::from_str(&json_str3).expect("deserialize retracted");
    assert_eq!(pt_back3.status, PartialStatus::Retracted);
}

// ===========================================================================
// CorrectionDrift tests
// ===========================================================================

#[test]
fn correction_drift_identical_segments() {
    let segments = vec![make_segment("the cat sat", 0.9)];
    let drift = CorrectionDrift::compute(&segments, &segments);
    assert!(
        (drift.wer_approx - 0.0).abs() < 1e-9,
        "identical segments should have wer_approx=0, got {}",
        drift.wer_approx,
    );
    assert_eq!(drift.text_edit_distance, 0);
    assert!(
        (drift.confidence_delta - 0.0).abs() < 1e-9,
        "identical confidence should have delta=0, got {}",
        drift.confidence_delta,
    );
    assert_eq!(drift.segment_count_delta, 0);
}

#[test]
fn correction_drift_completely_different() {
    let fast = vec![make_segment("alpha beta gamma", 0.9)];
    let quality = vec![make_segment("delta epsilon zeta", 0.9)];
    let drift = CorrectionDrift::compute(&fast, &quality);
    // All three words differ -> wer = 3/3 = 1.0
    assert!(
        drift.wer_approx > 0.9,
        "completely different text should have high WER, got {}",
        drift.wer_approx,
    );
}

#[test]
fn correction_drift_segment_count_mismatch() {
    let fast = vec![make_segment("a", 0.9)];
    let quality = vec![
        make_segment("a", 0.9),
        make_segment("b", 0.8),
        make_segment("c", 0.7),
    ];
    let drift = CorrectionDrift::compute(&fast, &quality);
    // segment_count_delta = quality_count - fast_count = 3 - 1 = 2
    assert_eq!(drift.segment_count_delta, 2);
}

#[test]
fn correction_drift_empty_inputs() {
    let drift = CorrectionDrift::compute(&[], &[]);
    assert!(
        (drift.wer_approx - 0.0).abs() < 1e-9,
        "empty inputs should have wer=0, got {}",
        drift.wer_approx,
    );
    assert_eq!(drift.text_edit_distance, 0);
    assert!(
        (drift.confidence_delta - 0.0).abs() < 1e-9,
        "empty inputs should have confidence_delta=0, got {}",
        drift.confidence_delta,
    );
    assert_eq!(drift.segment_count_delta, 0);
}

#[test]
fn correction_drift_known_wer() {
    // "the cat sat" vs "the dog sat" => 1 word differs out of 3 => wer ~ 0.333
    let fast = vec![make_segment("the cat sat", 0.9)];
    let quality = vec![make_segment("the dog sat", 0.9)];
    let drift = CorrectionDrift::compute(&fast, &quality);
    let expected_wer = 1.0 / 3.0;
    assert!(
        (drift.wer_approx - expected_wer).abs() < 0.01,
        "wer should be ~0.333, got {}",
        drift.wer_approx,
    );
}

#[test]
fn correction_drift_edit_distance_known() {
    // Character-level Levenshtein: "kitten" vs "sitting"
    // Known distance = 3 (but note segments are joined by space, so we test with single-segment)
    let fast = vec![make_segment("kitten", 0.9)];
    let quality = vec![make_segment("sitting", 0.9)];
    let drift = CorrectionDrift::compute(&fast, &quality);
    // "kitten" -> "sitting": k->s, e->i, n->ng => edit distance = 3
    assert_eq!(
        drift.text_edit_distance, 3,
        "Levenshtein('kitten', 'sitting') should be 3, got {}",
        drift.text_edit_distance,
    );

    // Also test a trivial case: "abc" vs "abc" = 0
    let fast2 = vec![make_segment("abc", 0.9)];
    let quality2 = vec![make_segment("abc", 0.9)];
    let drift2 = CorrectionDrift::compute(&fast2, &quality2);
    assert_eq!(drift2.text_edit_distance, 0);

    // "abc" vs "axc" = 1 substitution
    let fast3 = vec![make_segment("abc", 0.9)];
    let quality3 = vec![make_segment("axc", 0.9)];
    let drift3 = CorrectionDrift::compute(&fast3, &quality3);
    assert_eq!(drift3.text_edit_distance, 1);
}

// ===========================================================================
// CorrectionEvent tests
// ===========================================================================

#[test]
fn correction_event_computes_drift_and_confidence() {
    let fast_segs = vec![make_segment("the cat sat", 0.8)];
    let quality_segs = vec![make_segment("the dog sat", 0.95)];
    let ce = CorrectionEvent::new(
        1,
        10,
        5,
        "quality-v1".to_owned(),
        quality_segs,
        200,
        "2026-01-01T00:00:00Z".to_owned(),
        &fast_segs,
    );
    // Drift should be computed from fast_segs and quality_segs
    let expected_wer = 1.0 / 3.0;
    assert!(
        (ce.drift.wer_approx - expected_wer).abs() < 0.01,
        "drift wer should be ~0.333, got {}",
        ce.drift.wer_approx,
    );
    // quality_confidence_mean should be computed from corrected_segments
    assert!(
        (ce.quality_confidence_mean - 0.95).abs() < 1e-9,
        "quality_confidence_mean should be 0.95, got {}",
        ce.quality_confidence_mean,
    );
}

#[test]
fn correction_event_is_significant() {
    let fast_segs = vec![make_segment("a b c", 0.9)];
    let quality_segs = vec![make_segment("x y z", 0.9)];
    let ce = CorrectionEvent::new(
        1,
        10,
        5,
        "quality-v1".to_owned(),
        quality_segs,
        200,
        "2026-01-01T00:00:00Z".to_owned(),
        &fast_segs,
    );
    // wer should be 1.0 (all words different)
    assert!(
        ce.is_significant(0.5),
        "should be significant at threshold 0.5 with wer={}",
        ce.drift.wer_approx,
    );
    assert!(
        !ce.is_significant(1.0),
        "should NOT be significant at threshold 1.0 with wer={} (need > threshold)",
        ce.drift.wer_approx,
    );
    // Also verify exact threshold behavior
    assert!(
        !ce.is_significant(ce.drift.wer_approx),
        "should NOT be significant when threshold equals wer (need strictly greater)",
    );
}

#[test]
fn correction_event_serde_roundtrip() {
    let fast_segs = vec![make_segment_at("hello world", 0.0, 1.5, 0.85)];
    let quality_segs = vec![make_segment_at("hello earth", 0.0, 1.5, 0.92)];
    let ce = CorrectionEvent::new(
        42,
        7,
        3,
        "quality-model-v2".to_owned(),
        quality_segs,
        150,
        "2026-02-22T12:00:00Z".to_owned(),
        &fast_segs,
    );
    let json_str = serde_json::to_string(&ce).expect("serialize CorrectionEvent");
    let ce2: CorrectionEvent =
        serde_json::from_str(&json_str).expect("deserialize CorrectionEvent");
    assert_eq!(ce.correction_id, ce2.correction_id);
    assert_eq!(ce.retracted_seq, ce2.retracted_seq);
    assert_eq!(ce.window_id, ce2.window_id);
    assert_eq!(ce.quality_model_id, ce2.quality_model_id);
    assert_eq!(ce.quality_latency_ms, ce2.quality_latency_ms);
    assert_eq!(ce.corrected_at_rfc3339, ce2.corrected_at_rfc3339);
    assert!(
        (ce.quality_confidence_mean - ce2.quality_confidence_mean).abs() < 1e-9,
        "confidence_mean mismatch",
    );
    assert!(
        (ce.drift.wer_approx - ce2.drift.wer_approx).abs() < 1e-9,
        "drift.wer_approx mismatch",
    );
    assert_eq!(ce.drift.text_edit_distance, ce2.drift.text_edit_distance);
    assert_eq!(ce.corrected_segments.len(), ce2.corrected_segments.len());
}

// ===========================================================================
// Robot event tests
// ===========================================================================

#[test]
fn robot_retract_event_has_required_fields() {
    let val = robot::transcript_retract_value("run-abc", 42, 7, "quality_correction", "quality-v1");
    let obj = val
        .as_object()
        .expect("retract event should be a JSON object");
    for field in TRANSCRIPT_RETRACT_REQUIRED_FIELDS {
        assert!(
            obj.contains_key(*field),
            "transcript.retract event missing required field: '{field}'",
        );
    }
}

#[test]
fn robot_correct_event_has_required_fields() {
    let fast_segs = vec![make_segment("fast text", 0.8)];
    let quality_segs = vec![make_segment("corrected text", 0.95)];
    let ce = CorrectionEvent::new(
        1,
        10,
        5,
        "quality-v1".to_owned(),
        quality_segs,
        200,
        "2026-01-01T00:00:00Z".to_owned(),
        &fast_segs,
    );
    let val = robot::transcript_correct_value("run-abc", &ce);
    let obj = val
        .as_object()
        .expect("correct event should be a JSON object");
    for field in TRANSCRIPT_CORRECT_REQUIRED_FIELDS {
        assert!(
            obj.contains_key(*field),
            "transcript.correct event missing required field: '{field}'",
        );
    }
}

#[test]
fn robot_speculation_stats_has_required_fields() {
    let stats = SpeculationStats {
        windows_processed: 100,
        corrections_emitted: 15,
        confirmations_emitted: 85,
        correction_rate: 0.15,
        mean_fast_latency_ms: 45.0,
        mean_quality_latency_ms: 250.0,
        current_window_size_ms: 3000,
        mean_drift_wer: 0.12,
    };
    let val = robot::speculation_stats_value("run-xyz", &stats);
    let obj = val
        .as_object()
        .expect("speculation_stats event should be a JSON object");
    for field in SPECULATION_STATS_REQUIRED_FIELDS {
        assert!(
            obj.contains_key(*field),
            "transcript.speculation_stats event missing required field: '{field}'",
        );
    }
}

#[test]
fn robot_retract_event_schema_version() {
    let val = robot::transcript_retract_value("run-sv", 1, 1, "quality_correction", "model-q");
    let sv = val
        .get("schema_version")
        .expect("should have schema_version")
        .as_str()
        .expect("schema_version should be a string");
    assert_eq!(
        sv, ROBOT_SCHEMA_VERSION,
        "schema_version should match ROBOT_SCHEMA_VERSION ({ROBOT_SCHEMA_VERSION}), got '{sv}'",
    );
}

#[test]
fn robot_correct_event_contains_drift() {
    let fast_segs = vec![make_segment("the cat sat", 0.8)];
    let quality_segs = vec![make_segment("the dog sat", 0.95)];
    let ce = CorrectionEvent::new(
        99,
        5,
        3,
        "quality-v2".to_owned(),
        quality_segs,
        180,
        "2026-01-15T08:00:00Z".to_owned(),
        &fast_segs,
    );
    let val = robot::transcript_correct_value("run-drift", &ce);
    let drift_obj = val
        .get("drift")
        .expect("transcript.correct event should contain 'drift' subobject");
    let drift_map = drift_obj
        .as_object()
        .expect("'drift' should be a JSON object");

    // Verify all expected drift fields are present
    assert!(
        drift_map.contains_key("wer_approx"),
        "drift missing 'wer_approx'"
    );
    assert!(
        drift_map.contains_key("confidence_delta"),
        "drift missing 'confidence_delta'"
    );
    assert!(
        drift_map.contains_key("segment_count_delta"),
        "drift missing 'segment_count_delta'"
    );
    assert!(
        drift_map.contains_key("text_edit_distance"),
        "drift missing 'text_edit_distance'"
    );

    // Verify drift values are reasonable
    let wer = drift_map["wer_approx"]
        .as_f64()
        .expect("wer_approx should be a number");
    assert!(
        (wer - 1.0 / 3.0).abs() < 0.01,
        "drift wer_approx should be ~0.333, got {wer}",
    );
}
