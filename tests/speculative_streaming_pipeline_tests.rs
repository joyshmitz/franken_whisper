//! Integration tests for `SpeculativeStreamingPipeline` (bd-qlt.6).

#![forbid(unsafe_code)]

use franken_whisper::error::FwError;
use franken_whisper::model::TranscriptionSegment;
use franken_whisper::robot::{
    SPECULATION_STATS_REQUIRED_FIELDS, TRANSCRIPT_CONFIRM_REQUIRED_FIELDS,
    TRANSCRIPT_CORRECT_REQUIRED_FIELDS, TRANSCRIPT_PARTIAL_REQUIRED_FIELDS,
    TRANSCRIPT_RETRACT_REQUIRED_FIELDS,
};
use franken_whisper::speculation::{CorrectionDecision, CorrectionTolerance};
use franken_whisper::streaming::{SpeculativeConfig, SpeculativeStreamingPipeline};

fn seg(text: &str, start_ms: u64, end_ms: u64, confidence: f64) -> TranscriptionSegment {
    TranscriptionSegment {
        start_sec: Some(start_ms as f64 / 1000.0),
        end_sec: Some(end_ms as f64 / 1000.0),
        text: text.to_owned(),
        speaker: None,
        confidence: Some(confidence),
    }
}

#[test]
fn process_window_confirm_records_confirm_event_and_stats() {
    let mut pipeline = SpeculativeStreamingPipeline::new(
        SpeculativeConfig {
            emit_events: true,
            ..SpeculativeConfig::default()
        },
        "run-confirm".to_owned(),
    );

    let decision = pipeline
        .process_window(
            "h0",
            0,
            || vec![seg("hello world", 0, 300, 0.9)],
            || vec![seg("hello world", 0, 300, 0.95)],
        )
        .expect("process_window should succeed");

    assert!(
        matches!(decision, CorrectionDecision::Confirm { .. }),
        "expected confirm decision"
    );

    let stats = pipeline.stats();
    assert_eq!(stats.windows_processed, 1);
    assert_eq!(stats.confirmations_emitted, 1);
    assert_eq!(stats.corrections_emitted, 0);
    assert_eq!(pipeline.window_manager().windows_resolved(), 1);

    let codes: Vec<&str> = pipeline
        .events()
        .iter()
        .map(|event| event.code.as_str())
        .collect();
    assert_eq!(codes, vec!["transcript.partial", "transcript.confirm"]);

    let partial_payload = &pipeline.events()[0].payload;
    for field in TRANSCRIPT_PARTIAL_REQUIRED_FIELDS {
        assert!(
            partial_payload.get(field).is_some(),
            "partial payload missing required field `{field}`"
        );
    }

    let confirm_payload = &pipeline.events()[1].payload;
    for field in TRANSCRIPT_CONFIRM_REQUIRED_FIELDS {
        assert!(
            confirm_payload.get(field).is_some(),
            "confirm payload missing required field `{field}`"
        );
    }
}

#[test]
fn process_window_correct_emits_retract_and_correct_events() {
    let mut pipeline = SpeculativeStreamingPipeline::new(
        SpeculativeConfig {
            emit_events: true,
            tolerance: CorrectionTolerance {
                always_correct: true,
                ..CorrectionTolerance::default()
            },
            ..SpeculativeConfig::default()
        },
        "run-correct".to_owned(),
    );

    let decision = pipeline
        .process_window(
            "h1",
            0,
            || vec![seg("alpha beta gamma", 0, 300, 0.8)],
            || vec![seg("alpha beta corrected", 0, 300, 0.95)],
        )
        .expect("process_window should succeed");

    assert!(
        matches!(decision, CorrectionDecision::Correct { .. }),
        "expected correction decision"
    );

    let stats = pipeline.stats();
    assert_eq!(stats.windows_processed, 1);
    assert_eq!(stats.confirmations_emitted, 0);
    assert_eq!(stats.corrections_emitted, 1);
    assert_eq!(pipeline.correction_tracker().corrections().len(), 1);

    let codes: Vec<&str> = pipeline
        .events()
        .iter()
        .map(|event| event.code.as_str())
        .collect();
    assert_eq!(
        codes,
        vec![
            "transcript.partial",
            "transcript.retract",
            "transcript.correct",
        ]
    );

    let retract_payload = &pipeline.events()[1].payload;
    for field in TRANSCRIPT_RETRACT_REQUIRED_FIELDS {
        assert!(
            retract_payload.get(field).is_some(),
            "retract payload missing required field `{field}`"
        );
    }

    let correct_payload = &pipeline.events()[2].payload;
    for field in TRANSCRIPT_CORRECT_REQUIRED_FIELDS {
        assert!(
            correct_payload.get(field).is_some(),
            "correct payload missing required field `{field}`"
        );
    }
}

#[test]
fn process_duration_with_models_runs_bounded_windows_and_emits_stats_event() {
    let mut pipeline = SpeculativeStreamingPipeline::new(
        SpeculativeConfig {
            emit_events: true,
            window_size_ms: 1000,
            overlap_ms: 250,
            ..SpeculativeConfig::default()
        },
        "run-duration".to_owned(),
    );

    let result = pipeline
        .process_duration_with_models_no_checkpoint(2600, "seed", |start_ms, end_ms| {
            let text = format!("window-{start_ms}");
            let fast = vec![seg(&text, start_ms, (start_ms + 200).min(end_ms), 0.9)];
            let quality = vec![seg(&text, start_ms, (start_ms + 200).min(end_ms), 0.95)];
            Ok((fast, quality))
        })
        .expect("duration loop should succeed");

    let stats = pipeline.stats();
    assert_eq!(stats.windows_processed, 4);
    assert_eq!(stats.confirmations_emitted, 4);
    assert_eq!(stats.corrections_emitted, 0);
    assert_eq!(result.segments.len(), 4);

    let codes: Vec<&str> = pipeline
        .events()
        .iter()
        .map(|event| event.code.as_str())
        .collect();
    assert_eq!(codes.last().copied(), Some("transcript.speculation_stats"));
    assert_eq!(codes.len(), stats.windows_processed as usize * 2 + 1);

    let stats_payload = &pipeline.events()[codes.len() - 1].payload;
    for field in SPECULATION_STATS_REQUIRED_FIELDS {
        assert!(
            stats_payload.get(field).is_some(),
            "speculation stats payload missing required field `{field}`"
        );
    }
}

#[test]
fn process_duration_checkpoint_can_cancel_mid_stream() {
    let mut pipeline = SpeculativeStreamingPipeline::new(
        SpeculativeConfig {
            emit_events: true,
            window_size_ms: 1000,
            overlap_ms: 250,
            ..SpeculativeConfig::default()
        },
        "run-cancel".to_owned(),
    );

    let mut checkpoints = 0usize;
    let mut model_calls = 0usize;

    let error = pipeline
        .process_duration_with_models(
            4000,
            "seed-cancel",
            || {
                checkpoints += 1;
                if checkpoints > 1 {
                    Err(FwError::Cancelled("checkpoint cancelled".to_owned()))
                } else {
                    Ok(())
                }
            },
            |start_ms, end_ms| {
                model_calls += 1;
                let sample = vec![seg("ok", start_ms, (start_ms + 200).min(end_ms), 0.9)];
                Ok((sample.clone(), sample))
            },
        )
        .expect_err("second checkpoint should cancel loop");

    assert!(matches!(error, FwError::Cancelled(_)));
    assert_eq!(model_calls, 1, "only the first window should execute");
}

#[test]
fn process_duration_zero_length_still_runs_checkpoint_once() {
    let mut pipeline = SpeculativeStreamingPipeline::new(
        SpeculativeConfig {
            emit_events: true,
            ..SpeculativeConfig::default()
        },
        "run-zero-duration".to_owned(),
    );

    let mut checkpoints = 0usize;
    let result = pipeline
        .process_duration_with_models(
            0,
            "seed-zero",
            || {
                checkpoints += 1;
                Ok(())
            },
            |_start_ms, _end_ms| Ok((Vec::new(), Vec::new())),
        )
        .expect("zero-duration loop should succeed");

    assert_eq!(checkpoints, 1, "checkpoint should run exactly once");
    assert!(result.segments.is_empty());
}
