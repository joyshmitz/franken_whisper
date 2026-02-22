//! Integration tests for WindowManager and CorrectionTracker (bd-qlt.13).

use franken_whisper::model::TranscriptionSegment;
use franken_whisper::speculation::{
    CorrectionDecision, CorrectionTolerance, CorrectionTracker, PartialStatus, PartialTranscript,
    WindowManager, WindowStatus,
};

fn make_seg(text: &str, conf: f64) -> TranscriptionSegment {
    TranscriptionSegment {
        start_sec: Some(0.0),
        end_sec: Some(1.0),
        text: text.to_owned(),
        speaker: None,
        confidence: Some(conf),
    }
}

fn make_seg_at(text: &str, start: f64, end: f64, conf: f64) -> TranscriptionSegment {
    TranscriptionSegment {
        start_sec: Some(start),
        end_sec: Some(end),
        text: text.to_owned(),
        speaker: None,
        confidence: Some(conf),
    }
}

fn make_partial(
    seq: u64,
    window_id: u64,
    segments: Vec<TranscriptionSegment>,
) -> PartialTranscript {
    PartialTranscript::new(
        seq,
        window_id,
        "whisper-tiny".to_owned(),
        segments,
        50,
        "2026-01-01T00:00:00Z".to_owned(),
    )
}

// ===== WindowManager tests =====

#[test]
fn wm_next_window_sequential_ids() {
    let mut wm = WindowManager::new("run1", 3000, 500);
    let w0 = wm.next_window(0, "hash0");
    let w1 = wm.next_window(2500, "hash1");
    let w2 = wm.next_window(5000, "hash2");
    assert_eq!(w0.window_id, 0);
    assert_eq!(w1.window_id, 1);
    assert_eq!(w2.window_id, 2);
}

#[test]
fn wm_next_window_correct_bounds() {
    let mut wm = WindowManager::new("run1", 3000, 500);
    let w = wm.next_window(1000, "hash");
    assert_eq!(w.start_ms, 1000);
    assert_eq!(w.end_ms, 4000);
    assert_eq!(w.overlap_ms, 500);
}

#[test]
fn wm_record_fast_result_updates_state() {
    let mut wm = WindowManager::new("run1", 3000, 500);
    let w = wm.next_window(0, "hash");
    let partial = make_partial(0, w.window_id, vec![make_seg("hello", 0.9)]);
    wm.record_fast_result(w.window_id, partial);
    let ws = wm.get_window(0).unwrap();
    assert_eq!(ws.status, WindowStatus::FastComplete);
    assert!(ws.fast_result.is_some());
}

#[test]
fn wm_record_quality_result_updates_state() {
    let mut wm = WindowManager::new("run1", 3000, 500);
    let w = wm.next_window(0, "hash");
    wm.record_quality_result(w.window_id, vec![make_seg("hello", 0.95)]);
    let ws = wm.get_window(0).unwrap();
    assert_eq!(ws.status, WindowStatus::QualityComplete);
    assert!(ws.quality_result.is_some());
}

#[test]
fn wm_resolve_sets_resolved() {
    let mut wm = WindowManager::new("run1", 3000, 500);
    let w = wm.next_window(0, "hash");
    wm.resolve_window(w.window_id);
    let ws = wm.get_window(0).unwrap();
    assert_eq!(ws.status, WindowStatus::Resolved);
}

#[test]
fn wm_windows_resolved_count() {
    let mut wm = WindowManager::new("run1", 3000, 500);
    let w0 = wm.next_window(0, "h0");
    let _w1 = wm.next_window(2500, "h1");
    wm.resolve_window(w0.window_id);
    assert_eq!(wm.windows_resolved(), 1);
    assert_eq!(wm.windows_pending(), 1);
}

#[test]
fn wm_windows_pending_count() {
    let mut wm = WindowManager::new("run1", 3000, 500);
    wm.next_window(0, "h0");
    wm.next_window(2500, "h1");
    wm.next_window(5000, "h2");
    assert_eq!(wm.windows_pending(), 3);
    assert_eq!(wm.windows_resolved(), 0);
}

#[test]
fn wm_set_window_size_applies_to_next() {
    let mut wm = WindowManager::new("run1", 3000, 500);
    let w0 = wm.next_window(0, "h0");
    assert_eq!(w0.end_ms - w0.start_ms, 3000);
    wm.set_window_size(5000);
    let w1 = wm.next_window(2500, "h1");
    assert_eq!(w1.end_ms - w1.start_ms, 5000);
}

#[test]
fn wm_set_window_size_clamps_min() {
    let mut wm = WindowManager::new("run1", 3000, 500);
    wm.set_window_size(100);
    assert_eq!(wm.current_window_size(), 1000);
}

#[test]
fn wm_set_window_size_clamps_max() {
    let mut wm = WindowManager::new("run1", 3000, 500);
    wm.set_window_size(999_999);
    assert_eq!(wm.current_window_size(), 30_000);
}

#[test]
fn wm_merge_segments_ordered() {
    let mut wm = WindowManager::new("run1", 3000, 0);
    let w0 = wm.next_window(0, "h0");
    let w1 = wm.next_window(3000, "h1");
    wm.record_quality_result(w0.window_id, vec![make_seg_at("first", 0.0, 1.5, 0.9)]);
    wm.record_quality_result(w1.window_id, vec![make_seg_at("second", 3.0, 4.5, 0.9)]);
    wm.resolve_window(w0.window_id);
    wm.resolve_window(w1.window_id);
    let merged = wm.merge_segments();
    assert_eq!(merged.len(), 2);
    assert_eq!(merged[0].text, "first");
    assert_eq!(merged[1].text, "second");
}

#[test]
fn wm_merge_single_window() {
    let mut wm = WindowManager::new("run1", 3000, 0);
    let w = wm.next_window(0, "h0");
    wm.record_quality_result(w.window_id, vec![make_seg("only", 0.9)]);
    wm.resolve_window(w.window_id);
    let merged = wm.merge_segments();
    assert_eq!(merged.len(), 1);
    assert_eq!(merged[0].text, "only");
}

#[test]
fn wm_get_window_returns_some() {
    let mut wm = WindowManager::new("run1", 3000, 500);
    let w = wm.next_window(0, "h0");
    assert!(wm.get_window(w.window_id).is_some());
}

#[test]
fn wm_get_window_returns_none() {
    let wm = WindowManager::new("run1", 3000, 500);
    assert!(wm.get_window(999).is_none());
}

// ===== CorrectionTracker tests =====

#[test]
fn ct_register_partial_stores() {
    let mut ct = CorrectionTracker::new(CorrectionTolerance::default());
    let partial = make_partial(0, 0, vec![make_seg("hello", 0.9)]);
    let seq = ct.register_partial(partial);
    assert_eq!(seq, 0);
    assert!(ct.get_partial(0).is_some());
}

#[test]
fn ct_identical_output_confirms() {
    let mut ct = CorrectionTracker::new(CorrectionTolerance::default());
    let segs = vec![make_seg("hello world", 0.9)];
    let partial = make_partial(0, 0, segs.clone());
    ct.register_partial(partial);
    let decision = ct
        .submit_quality_result(0, "whisper-large", segs, 100)
        .unwrap();
    assert!(matches!(decision, CorrectionDecision::Confirm { .. }));
}

#[test]
fn ct_different_output_corrects() {
    let mut ct = CorrectionTracker::new(CorrectionTolerance::default());
    let fast_segs = vec![make_seg("the cat sat on mat", 0.8)];
    let quality_segs = vec![make_seg("completely different text here now", 0.95)];
    let partial = make_partial(0, 0, fast_segs);
    ct.register_partial(partial);
    let decision = ct
        .submit_quality_result(0, "whisper-large", quality_segs, 200)
        .unwrap();
    assert!(matches!(decision, CorrectionDecision::Correct { .. }));
}

#[test]
fn ct_always_correct_forces_correction() {
    let mut ct = CorrectionTracker::new(CorrectionTolerance {
        always_correct: true,
        ..CorrectionTolerance::default()
    });
    let segs = vec![make_seg("hello", 0.9)];
    let partial = make_partial(0, 0, segs.clone());
    ct.register_partial(partial);
    let decision = ct
        .submit_quality_result(0, "whisper-large", segs, 100)
        .unwrap();
    assert!(matches!(decision, CorrectionDecision::Correct { .. }));
}

#[test]
fn ct_unknown_window_id_errors() {
    let mut ct = CorrectionTracker::new(CorrectionTolerance::default());
    let result = ct.submit_quality_result(999, "whisper-large", vec![], 100);
    assert!(result.is_err());
}

#[test]
fn ct_correction_rate_computed() {
    let mut ct = CorrectionTracker::new(CorrectionTolerance {
        always_correct: true,
        ..CorrectionTolerance::default()
    });
    // Register and submit 2 windows — both will correct (always_correct=true)
    let p0 = make_partial(0, 0, vec![make_seg("a", 0.9)]);
    ct.register_partial(p0);
    ct.submit_quality_result(0, "q", vec![make_seg("a", 0.9)], 100)
        .unwrap();

    // Now create tracker with mixed results
    let mut ct2 = CorrectionTracker::new(CorrectionTolerance::default());
    let p1 = make_partial(1, 1, vec![make_seg("hello", 0.9)]);
    ct2.register_partial(p1);
    ct2.submit_quality_result(1, "q", vec![make_seg("hello", 0.9)], 100)
        .unwrap(); // confirm
    let p2 = make_partial(2, 2, vec![make_seg("abc", 0.9)]);
    ct2.register_partial(p2);
    ct2.submit_quality_result(
        2,
        "q",
        vec![make_seg("xyz completely different", 0.95)],
        100,
    )
    .unwrap(); // correct
    // rate = 1 correction / 2 total = 0.5
    assert!((ct2.correction_rate() - 0.5).abs() < 0.01);
}

#[test]
fn ct_mean_wer_computed() {
    let mut ct = CorrectionTracker::new(CorrectionTolerance::default());
    let p0 = make_partial(0, 0, vec![make_seg("the cat sat", 0.9)]);
    ct.register_partial(p0);
    ct.submit_quality_result(0, "q", vec![make_seg("the dog sat", 0.95)], 100)
        .unwrap();
    let wer = ct.mean_wer();
    // "the cat sat" vs "the dog sat" -> 1/3 WER
    assert!(wer > 0.3 && wer < 0.4, "expected ~0.333, got {wer}");
}

#[test]
fn ct_all_resolved_false_when_pending() {
    let mut ct = CorrectionTracker::new(CorrectionTolerance::default());
    let partial = make_partial(0, 0, vec![make_seg("hello", 0.9)]);
    ct.register_partial(partial);
    assert!(!ct.all_resolved());
}

#[test]
fn ct_all_resolved_true_after_all() {
    let mut ct = CorrectionTracker::new(CorrectionTolerance::default());
    let partial = make_partial(0, 0, vec![make_seg("hello", 0.9)]);
    ct.register_partial(partial);
    ct.submit_quality_result(0, "q", vec![make_seg("hello", 0.9)], 100)
        .unwrap();
    assert!(ct.all_resolved());
}

#[test]
fn ct_high_tolerance_confirms_everything() {
    let mut ct = CorrectionTracker::new(CorrectionTolerance {
        max_wer: 1.0,
        max_confidence_delta: 1.0,
        max_edit_distance: 100_000,
        always_correct: false,
    });
    let p = make_partial(0, 0, vec![make_seg("aaa", 0.9)]);
    ct.register_partial(p);
    let decision = ct
        .submit_quality_result(0, "q", vec![make_seg("zzz completely different", 0.5)], 100)
        .unwrap();
    assert!(matches!(decision, CorrectionDecision::Confirm { .. }));
}

#[test]
fn ct_stats_accumulate() {
    let mut ct = CorrectionTracker::new(CorrectionTolerance::default());
    let p0 = make_partial(0, 0, vec![make_seg("hello", 0.9)]);
    let p1 = make_partial(1, 1, vec![make_seg("world", 0.9)]);
    ct.register_partial(p0);
    ct.register_partial(p1);
    ct.submit_quality_result(0, "q", vec![make_seg("hello", 0.9)], 100)
        .unwrap();
    ct.submit_quality_result(1, "q", vec![make_seg("world", 0.9)], 200)
        .unwrap();
    let stats = ct.stats();
    assert_eq!(stats.windows_processed, 2);
    assert_eq!(stats.total_quality_latency_ms, 300);
}

#[test]
fn ct_corrections_list_grows() {
    let mut ct = CorrectionTracker::new(CorrectionTolerance {
        always_correct: true,
        ..CorrectionTolerance::default()
    });
    let p = make_partial(0, 0, vec![make_seg("hello", 0.9)]);
    ct.register_partial(p);
    ct.submit_quality_result(0, "q", vec![make_seg("hello", 0.9)], 100)
        .unwrap();
    assert_eq!(ct.corrections().len(), 1);
}

#[test]
fn ct_partial_status_after_confirm() {
    let mut ct = CorrectionTracker::new(CorrectionTolerance::default());
    let p = make_partial(0, 0, vec![make_seg("hello", 0.9)]);
    ct.register_partial(p);
    ct.submit_quality_result(0, "q", vec![make_seg("hello", 0.9)], 100)
        .unwrap();
    let partial = ct.get_partial(0).unwrap();
    assert_eq!(partial.status, PartialStatus::Confirmed);
}

#[test]
fn ct_partial_status_after_correct() {
    let mut ct = CorrectionTracker::new(CorrectionTolerance {
        always_correct: true,
        ..CorrectionTolerance::default()
    });
    let p = make_partial(0, 0, vec![make_seg("hello", 0.9)]);
    ct.register_partial(p);
    ct.submit_quality_result(0, "q", vec![make_seg("hello", 0.9)], 100)
        .unwrap();
    let partial = ct.get_partial(0).unwrap();
    assert_eq!(partial.status, PartialStatus::Retracted);
}
