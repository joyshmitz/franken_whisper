//! Integration tests for SpeculativeStreamingPipeline (bd-qlt.14).
//!
//! Tests exercise pipeline construction, WindowManager + CorrectionTracker
//! integration, event accumulation, config impact, and edge cases.

#![forbid(unsafe_code)]

use franken_whisper::model::TranscriptionSegment;
use franken_whisper::speculation::{
    CorrectionDecision, CorrectionTolerance, CorrectionTracker, PartialTranscript, WindowManager,
};
use franken_whisper::streaming::{SpeculativeConfig, SpeculativeStreamingPipeline};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

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

fn default_config() -> SpeculativeConfig {
    SpeculativeConfig {
        window_size_ms: 3000,
        overlap_ms: 500,
        fast_model_name: "test-fast".to_owned(),
        quality_model_name: "test-quality".to_owned(),
        tolerance: CorrectionTolerance::default(),
        adaptive: false,
        emit_events: true,
    }
}

/// Simulate what the pipeline does internally for a single window:
/// register a partial with the tracker, then submit quality result.
fn simulate_window(
    wm: &mut WindowManager,
    tracker: &mut CorrectionTracker,
    window_pos_ms: u64,
    audio_hash: &str,
    fast_segments: Vec<TranscriptionSegment>,
    quality_segments: Vec<TranscriptionSegment>,
    seq: u64,
) -> CorrectionDecision {
    let window = wm.next_window(window_pos_ms, audio_hash);
    let window_id = window.window_id;

    let partial = PartialTranscript::new(
        seq,
        window_id,
        "test-fast".to_owned(),
        fast_segments.clone(),
        50,
        chrono::Utc::now().to_rfc3339(),
    );

    let fast_partial_for_wm = PartialTranscript::new(
        seq,
        window_id,
        "test-fast".to_owned(),
        fast_segments,
        50,
        chrono::Utc::now().to_rfc3339(),
    );

    tracker.register_partial(partial);
    wm.record_fast_result(window_id, fast_partial_for_wm);
    wm.record_quality_result(window_id, quality_segments.clone());

    let decision = tracker
        .submit_quality_result(window_id, "test-quality", quality_segments, 100)
        .expect("submit_quality_result should succeed");

    wm.resolve_window(window_id);
    decision
}

// ===========================================================================
// Pipeline construction tests
// ===========================================================================

#[test]
fn pipeline_new_creates_with_default_config() {
    let config = SpeculativeConfig::default();
    let pipeline = SpeculativeStreamingPipeline::new(config, "run-001".to_owned());
    // Construction succeeds — we can access the run_id.
    assert_eq!(pipeline.run_id(), "run-001");
}

#[test]
fn pipeline_config_preserved() {
    let config = SpeculativeConfig {
        window_size_ms: 5000,
        overlap_ms: 750,
        fast_model_name: "tiny".to_owned(),
        quality_model_name: "large-v3".to_owned(),
        tolerance: CorrectionTolerance {
            max_wer: 0.25,
            max_confidence_delta: 0.5,
            max_edit_distance: 100,
            always_correct: false,
        },
        adaptive: true,
        emit_events: false,
    };

    let pipeline = SpeculativeStreamingPipeline::new(config, "run-cfg".to_owned());

    // Window manager should reflect the configured window size.
    assert_eq!(pipeline.window_manager().current_window_size(), 5000);
}

#[test]
fn pipeline_initial_state_clean() {
    let pipeline = SpeculativeStreamingPipeline::new(default_config(), "run-clean".to_owned());

    assert!(pipeline.events().is_empty(), "no events initially");
    assert_eq!(
        pipeline.stats().windows_processed,
        0,
        "no windows processed initially"
    );
    assert_eq!(
        pipeline.stats().corrections_emitted,
        0,
        "no corrections initially"
    );
    assert_eq!(
        pipeline.stats().confirmations_emitted,
        0,
        "no confirmations initially"
    );
    assert!(
        pipeline.merged_transcript().is_empty(),
        "no segments initially"
    );
}

#[test]
fn pipeline_run_id_set() {
    let pipeline = SpeculativeStreamingPipeline::new(default_config(), "run-id-test".to_owned());
    assert_eq!(pipeline.run_id(), "run-id-test");
}

// ===========================================================================
// WindowManager + CorrectionTracker integration
// ===========================================================================

#[test]
fn pipeline_single_window_confirm_flow() {
    let mut wm = WindowManager::new("run-confirm", 3000, 500);
    let mut tracker = CorrectionTracker::new(CorrectionTolerance::default());

    // Fast and quality produce identical text => Confirm.
    let fast = vec![make_seg("hello world", 0.75)];
    let quality = vec![make_seg("hello world", 0.875)];

    let decision = simulate_window(&mut wm, &mut tracker, 0, "hash-0", fast, quality, 0);

    assert!(
        matches!(decision, CorrectionDecision::Confirm { .. }),
        "identical text should confirm"
    );
    assert_eq!(tracker.stats().confirmations_emitted, 1);
    assert_eq!(tracker.stats().corrections_emitted, 0);
    assert_eq!(wm.windows_resolved(), 1);
}

#[test]
fn pipeline_single_window_correct_flow() {
    let mut wm = WindowManager::new("run-correct", 3000, 500);
    let mut tracker = CorrectionTracker::new(CorrectionTolerance {
        max_wer: 0.0625, // Tight tolerance to force correction.
        max_confidence_delta: 0.0625,
        max_edit_distance: 2,
        always_correct: false,
    });

    let fast = vec![make_seg("helo wrld", 0.5)];
    let quality = vec![make_seg("hello world", 0.875)];

    let decision = simulate_window(&mut wm, &mut tracker, 0, "hash-0", fast, quality, 0);

    assert!(
        matches!(decision, CorrectionDecision::Correct { .. }),
        "different text should correct with tight tolerance"
    );
    assert_eq!(tracker.stats().corrections_emitted, 1);
    assert_eq!(tracker.stats().confirmations_emitted, 0);
    assert_eq!(wm.windows_resolved(), 1);
}

#[test]
fn pipeline_multiple_windows_sequential() {
    let mut wm = WindowManager::new("run-multi", 3000, 500);
    let mut tracker = CorrectionTracker::new(CorrectionTolerance::default());

    let texts = ["hello world", "this is a test", "of the system"];
    for (i, text) in texts.iter().enumerate() {
        let fast = vec![make_seg_at(text, i as f64, i as f64 + 1.0, 0.75)];
        let quality = vec![make_seg_at(text, i as f64, i as f64 + 1.0, 0.875)];
        let pos_ms = (i as u64) * 2500;

        let decision = simulate_window(
            &mut wm,
            &mut tracker,
            pos_ms,
            &format!("hash-{i}"),
            fast,
            quality,
            i as u64,
        );

        assert!(
            matches!(decision, CorrectionDecision::Confirm { .. }),
            "window {i} should confirm (identical text)"
        );
    }

    assert_eq!(tracker.stats().windows_processed, 3);
    assert_eq!(tracker.stats().confirmations_emitted, 3);
    assert_eq!(tracker.stats().corrections_emitted, 0);
    assert_eq!(wm.windows_resolved(), 3);
}

#[test]
fn pipeline_mixed_corrections() {
    let mut wm = WindowManager::new("run-mixed", 3000, 500);
    let mut tracker = CorrectionTracker::new(CorrectionTolerance {
        max_wer: 0.125,
        max_confidence_delta: 0.125,
        max_edit_distance: 5,
        always_correct: false,
    });

    // Window 0: identical => Confirm.
    let decision0 = simulate_window(
        &mut wm,
        &mut tracker,
        0,
        "h0",
        vec![make_seg_at("hello world", 0.0, 1.0, 0.75)],
        vec![make_seg_at("hello world", 0.0, 1.0, 0.875)],
        0,
    );
    assert!(matches!(decision0, CorrectionDecision::Confirm { .. }));

    // Window 1: very different => Correct.
    let decision1 = simulate_window(
        &mut wm,
        &mut tracker,
        2500,
        "h1",
        vec![make_seg_at("completely wrong text here", 1.0, 2.0, 0.5)],
        vec![make_seg_at(
            "the actual correct transcript",
            1.0,
            2.0,
            0.875,
        )],
        1,
    );
    assert!(matches!(decision1, CorrectionDecision::Correct { .. }));

    // Window 2: identical => Confirm.
    let decision2 = simulate_window(
        &mut wm,
        &mut tracker,
        5000,
        "h2",
        vec![make_seg_at("goodbye", 2.0, 3.0, 0.75)],
        vec![make_seg_at("goodbye", 2.0, 3.0, 0.875)],
        2,
    );
    assert!(matches!(decision2, CorrectionDecision::Confirm { .. }));

    assert_eq!(tracker.stats().windows_processed, 3);
    assert_eq!(tracker.stats().confirmations_emitted, 2);
    assert_eq!(tracker.stats().corrections_emitted, 1);
}

#[test]
fn pipeline_stats_reflect_corrections() {
    let mut wm = WindowManager::new("run-stats", 3000, 500);
    let mut tracker = CorrectionTracker::new(CorrectionTolerance {
        max_wer: 0.125,
        max_confidence_delta: 0.125,
        max_edit_distance: 3,
        always_correct: false,
    });

    // 2 confirms, 2 corrections => correction_rate = 0.5.
    let pairs: Vec<(&str, &str)> = vec![
        ("hello", "hello"),                // confirm
        ("wrng txt", "correct text here"), // correct
        ("goodbye", "goodbye"),            // confirm
        ("bad bad", "good good output"),   // correct
    ];

    for (i, (fast_text, quality_text)) in pairs.iter().enumerate() {
        let fast = vec![make_seg_at(fast_text, i as f64, i as f64 + 1.0, 0.75)];
        let quality = vec![make_seg_at(quality_text, i as f64, i as f64 + 1.0, 0.875)];
        simulate_window(
            &mut wm,
            &mut tracker,
            i as u64 * 2500,
            &format!("h{i}"),
            fast,
            quality,
            i as u64,
        );
    }

    assert_eq!(tracker.stats().windows_processed, 4);
    assert_eq!(tracker.stats().corrections_emitted, 2);
    assert_eq!(tracker.stats().confirmations_emitted, 2);
    assert!((tracker.correction_rate() - 0.5).abs() < 1e-10);
}

#[test]
fn pipeline_merged_segments_ordered() {
    let mut wm = WindowManager::new("run-merge", 3000, 500);
    let mut tracker = CorrectionTracker::new(CorrectionTolerance::default());

    // Create windows with segments at different time positions.
    // Simulate them out of order but expect merge to sort by start_sec.
    simulate_window(
        &mut wm,
        &mut tracker,
        5000,
        "h2",
        vec![make_seg_at("third", 5.0, 6.0, 0.75)],
        vec![make_seg_at("third", 5.0, 6.0, 0.875)],
        2,
    );

    simulate_window(
        &mut wm,
        &mut tracker,
        0,
        "h0",
        vec![make_seg_at("first", 0.0, 1.0, 0.75)],
        vec![make_seg_at("first", 0.0, 1.0, 0.875)],
        0,
    );

    simulate_window(
        &mut wm,
        &mut tracker,
        2500,
        "h1",
        vec![make_seg_at("second", 2.5, 3.5, 0.75)],
        vec![make_seg_at("second", 2.5, 3.5, 0.875)],
        1,
    );

    let merged = wm.merge_segments();
    assert_eq!(merged.len(), 3);
    assert_eq!(merged[0].text, "first");
    assert_eq!(merged[1].text, "second");
    assert_eq!(merged[2].text, "third");

    // Verify time ordering.
    for i in 1..merged.len() {
        assert!(
            merged[i].start_sec.unwrap_or(0.0) >= merged[i - 1].start_sec.unwrap_or(0.0),
            "segments should be sorted by start_sec"
        );
    }
}

// ===========================================================================
// Event accumulation
// ===========================================================================

#[test]
fn pipeline_events_accumulate() {
    let config = default_config();
    let mut pipeline = SpeculativeStreamingPipeline::new(config, "run-events".to_owned());

    assert!(pipeline.events().is_empty());

    // Process a window using process_window (which spawns threads internally).
    let fast_segs = vec![make_seg("hello", 0.75)];
    let quality_segs = vec![make_seg("hello", 0.875)];

    let result = pipeline.process_window("hash-0", 0, move || fast_segs, move || quality_segs);
    assert!(result.is_ok());

    // At minimum we expect a transcript.partial event and a confirm/correct event.
    assert!(
        pipeline.events().len() >= 2,
        "expected at least 2 events after one window, got {}",
        pipeline.events().len()
    );

    // Process a second window.
    let fast_segs2 = vec![make_seg("world", 0.75)];
    let quality_segs2 = vec![make_seg("world", 0.875)];

    let result2 =
        pipeline.process_window("hash-1", 3000, move || fast_segs2, move || quality_segs2);
    assert!(result2.is_ok());

    assert!(
        pipeline.events().len() >= 4,
        "expected at least 4 events after two windows, got {}",
        pipeline.events().len()
    );
}

#[test]
fn pipeline_events_contain_correction_data() {
    let config = SpeculativeConfig {
        tolerance: CorrectionTolerance {
            max_wer: 0.0625,
            max_confidence_delta: 0.0625,
            max_edit_distance: 2,
            always_correct: false,
        },
        ..default_config()
    };

    let mut pipeline = SpeculativeStreamingPipeline::new(config, "run-corr-event".to_owned());

    // Force a correction with very different text.
    let fast_segs = vec![make_seg("completely wrong", 0.5)];
    let quality_segs = vec![make_seg("actual correct text", 0.875)];

    let result = pipeline.process_window("hash-0", 0, move || fast_segs, move || quality_segs);
    assert!(result.is_ok());
    let decision = result.unwrap();
    assert!(matches!(decision, CorrectionDecision::Correct { .. }));

    // Check events contain retract and correct events.
    let event_codes: Vec<&str> = pipeline.events().iter().map(|e| e.code.as_str()).collect();
    assert!(
        event_codes.contains(&"transcript.partial"),
        "should have partial event"
    );
    assert!(
        event_codes.contains(&"transcript.retract"),
        "should have retract event"
    );
    assert!(
        event_codes.contains(&"transcript.correct"),
        "should have correct event"
    );

    // All events should be in the "speculation" stage.
    for event in pipeline.events() {
        assert_eq!(event.stage, "speculation");
    }
}

// ===========================================================================
// Config impact
// ===========================================================================

#[test]
fn pipeline_always_correct_forces_corrections() {
    let config = SpeculativeConfig {
        tolerance: CorrectionTolerance {
            always_correct: true,
            ..CorrectionTolerance::default()
        },
        ..default_config()
    };

    let mut pipeline = SpeculativeStreamingPipeline::new(config, "run-always".to_owned());

    // Identical text — should still correct because always_correct=true.
    let fast_segs = vec![make_seg("identical text", 0.875)];
    let quality_segs = vec![make_seg("identical text", 0.875)];

    let result = pipeline.process_window("hash-0", 0, move || fast_segs, move || quality_segs);
    assert!(result.is_ok());
    let decision = result.unwrap();

    assert!(
        matches!(decision, CorrectionDecision::Correct { .. }),
        "always_correct=true should force correction even for identical text"
    );
    assert_eq!(pipeline.stats().corrections_emitted, 1);
    assert_eq!(pipeline.stats().confirmations_emitted, 0);
}

#[test]
fn pipeline_high_tolerance_confirms_everything() {
    let config = SpeculativeConfig {
        tolerance: CorrectionTolerance {
            max_wer: 1.0,
            max_confidence_delta: 1.0,
            max_edit_distance: usize::MAX,
            always_correct: false,
        },
        ..default_config()
    };

    let mut pipeline = SpeculativeStreamingPipeline::new(config, "run-tolerant".to_owned());

    // Very different text — should still confirm because tolerance is maxed out.
    let fast_segs = vec![make_seg("totally wrong garbage", 0.25)];
    let quality_segs = vec![make_seg("completely different correct text", 0.875)];

    let result = pipeline.process_window("hash-0", 0, move || fast_segs, move || quality_segs);
    assert!(result.is_ok());
    let decision = result.unwrap();

    assert!(
        matches!(decision, CorrectionDecision::Confirm { .. }),
        "max tolerance should confirm even very different text"
    );
    assert_eq!(pipeline.stats().confirmations_emitted, 1);
    assert_eq!(pipeline.stats().corrections_emitted, 0);
}

#[test]
fn pipeline_custom_window_size() {
    let config = SpeculativeConfig {
        window_size_ms: 7000,
        overlap_ms: 1000,
        ..default_config()
    };

    let pipeline = SpeculativeStreamingPipeline::new(config, "run-custom-ws".to_owned());

    assert_eq!(pipeline.window_manager().current_window_size(), 7000);
}

// ===========================================================================
// Edge cases
// ===========================================================================

#[test]
fn pipeline_empty_segments_handled() {
    let mut wm = WindowManager::new("run-empty", 3000, 500);
    let mut tracker = CorrectionTracker::new(CorrectionTolerance::default());

    // Both fast and quality return empty segments.
    let decision = simulate_window(&mut wm, &mut tracker, 0, "hash-0", vec![], vec![], 0);

    // Empty vs empty => WER = 0, edit distance = 0 => Confirm.
    assert!(
        matches!(decision, CorrectionDecision::Confirm { .. }),
        "empty segments should confirm"
    );
    assert_eq!(tracker.stats().windows_processed, 1);
}

#[test]
fn pipeline_single_segment_window() {
    let mut wm = WindowManager::new("run-single-seg", 3000, 500);
    let mut tracker = CorrectionTracker::new(CorrectionTolerance::default());

    let fast = vec![make_seg("one segment", 0.75)];
    let quality = vec![make_seg("one segment", 0.875)];

    let decision = simulate_window(&mut wm, &mut tracker, 0, "hash-0", fast, quality, 0);

    assert!(matches!(decision, CorrectionDecision::Confirm { .. }));
    let merged = wm.merge_segments();
    assert_eq!(merged.len(), 1);
    assert_eq!(merged[0].text, "one segment");
}

#[test]
fn pipeline_long_text_segments() {
    let mut wm = WindowManager::new("run-long", 3000, 500);
    let mut tracker = CorrectionTracker::new(CorrectionTolerance::default());

    let long_text = "word ".repeat(500);
    let fast = vec![make_seg(&long_text, 0.75)];
    let quality = vec![make_seg(&long_text, 0.875)];

    let decision = simulate_window(&mut wm, &mut tracker, 0, "hash-0", fast, quality, 0);

    assert!(
        matches!(decision, CorrectionDecision::Confirm { .. }),
        "long identical text should confirm"
    );
    assert_eq!(tracker.stats().windows_processed, 1);
}

// ===========================================================================
// process_duration_with_models tests
// ===========================================================================

#[test]
fn pipeline_process_duration_zero_returns_empty_result() {
    let config = default_config();
    let mut pipeline = SpeculativeStreamingPipeline::new(config, "run-zero-dur".to_owned());

    let result = pipeline
        .process_duration_with_models_no_checkpoint(0, "seed", |_start, _end| Ok((vec![], vec![])))
        .expect("zero duration should succeed");

    assert!(result.segments.is_empty());
    assert!(result.transcript.is_empty());
}

#[test]
fn pipeline_process_duration_single_window() {
    let config = SpeculativeConfig {
        window_size_ms: 5000,
        overlap_ms: 500,
        ..default_config()
    };
    let mut pipeline = SpeculativeStreamingPipeline::new(config, "run-single-dur".to_owned());

    let result = pipeline
        .process_duration_with_models_no_checkpoint(3000, "seed", |_start, _end| {
            let fast = vec![make_seg_at("hello world", 0.0, 3.0, 0.75)];
            let quality = vec![make_seg_at("hello world", 0.0, 3.0, 0.875)];
            Ok((fast, quality))
        })
        .expect("single window duration should succeed");

    assert_eq!(result.segments.len(), 1);
    assert_eq!(result.transcript, "hello world");
    assert_eq!(pipeline.stats().windows_processed, 1);
}

#[test]
fn pipeline_process_duration_multiple_windows() {
    let config = SpeculativeConfig {
        window_size_ms: 2000,
        overlap_ms: 500,
        ..default_config()
    };
    let mut pipeline = SpeculativeStreamingPipeline::new(config, "run-multi-dur".to_owned());

    let mut call_count = 0u32;
    let result = pipeline
        .process_duration_with_models(
            5000,
            "seed",
            || Ok(()),
            |start_ms, end_ms| {
                call_count += 1;
                let text = format!("segment at {start_ms}-{end_ms}");
                let start_sec = start_ms as f64 / 1000.0;
                let end_sec = end_ms as f64 / 1000.0;
                let fast = vec![make_seg_at(&text, start_sec, end_sec, 0.75)];
                let quality = vec![make_seg_at(&text, start_sec, end_sec, 0.875)];
                Ok((fast, quality))
            },
        )
        .expect("multi-window duration should succeed");

    assert!(
        result.segments.len() >= 2,
        "5000ms audio with 2000ms windows should produce multiple segments"
    );
    assert!(
        pipeline.stats().windows_processed >= 2,
        "should have processed at least 2 windows"
    );
}

// ===========================================================================
// Build result tests
// ===========================================================================

#[test]
fn pipeline_build_result_joins_segments() {
    let config = default_config();
    let mut pipeline = SpeculativeStreamingPipeline::new(config, "run-build".to_owned());

    let fast1 = vec![make_seg_at("hello", 0.0, 1.0, 0.75)];
    let quality1 = vec![make_seg_at("hello", 0.0, 1.0, 0.875)];

    pipeline
        .process_window("h0", 0, move || fast1, move || quality1)
        .unwrap();

    let fast2 = vec![make_seg_at("world", 1.0, 2.0, 0.75)];
    let quality2 = vec![make_seg_at("world", 1.0, 2.0, 0.875)];

    pipeline
        .process_window("h1", 3000, move || fast2, move || quality2)
        .unwrap();

    let result = pipeline.build_result();
    assert_eq!(result.transcript, "hello world");
    assert_eq!(result.segments.len(), 2);
    assert_eq!(result.language, Some("en".to_owned()));
}

// ===========================================================================
// Stats accuracy tests
// ===========================================================================

#[test]
fn pipeline_stats_zero_windows_defaults() {
    let pipeline = SpeculativeStreamingPipeline::new(default_config(), "run-zero-stats".to_owned());

    let stats = pipeline.stats();
    assert_eq!(stats.windows_processed, 0);
    assert_eq!(stats.corrections_emitted, 0);
    assert_eq!(stats.confirmations_emitted, 0);
    assert!((stats.correction_rate - 0.0).abs() < 1e-10);
    assert!((stats.mean_fast_latency_ms - 0.0).abs() < 1e-10);
    assert!((stats.mean_quality_latency_ms - 0.0).abs() < 1e-10);
    assert!((stats.mean_drift_wer - 0.0).abs() < 1e-10);
    assert_eq!(stats.current_window_size_ms, 3000);
}

#[test]
fn pipeline_correction_tracker_all_resolved() {
    let mut wm = WindowManager::new("run-resolved", 3000, 500);
    let mut tracker = CorrectionTracker::new(CorrectionTolerance::default());

    simulate_window(
        &mut wm,
        &mut tracker,
        0,
        "h0",
        vec![make_seg("a", 0.75)],
        vec![make_seg("a", 0.875)],
        0,
    );
    simulate_window(
        &mut wm,
        &mut tracker,
        2500,
        "h1",
        vec![make_seg("b", 0.75)],
        vec![make_seg("b", 0.875)],
        1,
    );

    assert!(
        tracker.all_resolved(),
        "all partials should be resolved after quality submission"
    );
}

// ===========================================================================
// WindowManager merge deduplication
// ===========================================================================

#[test]
fn pipeline_merge_deduplicates_overlapping_segments() {
    let mut wm = WindowManager::new("run-dedup", 3000, 500);
    let mut tracker = CorrectionTracker::new(CorrectionTolerance::default());

    // Two windows that produce segments at the same time position.
    // The deduplication logic should keep only the higher confidence one.
    simulate_window(
        &mut wm,
        &mut tracker,
        0,
        "h0",
        vec![make_seg_at("hello", 0.0, 1.0, 0.5)],
        vec![make_seg_at("hello", 0.0, 1.0, 0.75)],
        0,
    );

    // Second window with a segment at a very similar position (within 0.1s).
    simulate_window(
        &mut wm,
        &mut tracker,
        500,
        "h1",
        vec![make_seg_at("hello better", 0.0, 1.0, 0.5)],
        vec![make_seg_at("hello better", 0.0, 1.0, 0.875)],
        1,
    );

    let merged = wm.merge_segments();
    // Both segments are at (0.0, 1.0) so they should be deduplicated.
    assert_eq!(
        merged.len(),
        1,
        "overlapping segments should be deduplicated"
    );
    assert!(
        merged[0].confidence.unwrap_or(0.0) >= 0.875,
        "should keep higher-confidence segment"
    );
}

// ===========================================================================
// Window manager window size mutability
// ===========================================================================

#[test]
fn pipeline_window_manager_set_window_size_clamped() {
    let mut wm = WindowManager::new("run-clamp", 3000, 500);

    // Default min = 1000, max = 30_000.
    wm.set_window_size(500);
    assert_eq!(
        wm.current_window_size(),
        1000,
        "should clamp to min_window_ms"
    );

    wm.set_window_size(50_000);
    assert_eq!(
        wm.current_window_size(),
        30_000,
        "should clamp to max_window_ms"
    );

    wm.set_window_size(5000);
    assert_eq!(wm.current_window_size(), 5000, "normal value should stick");
}
