//! bd-3pf.15: Deterministic stage-order and error-code contract tests.
//!
//! Verifies:
//! - The 10-stage pipeline executes in correct canonical order.
//! - Robot error codes are deterministic (same input -> same code).
//! - Stage dependency validation (Backend cannot run before Ingest, etc.).
//! - `skip_stages` / `without()` properly excludes stages from the config.

#![forbid(unsafe_code)]

use franken_whisper::error::FwError;
use franken_whisper::orchestrator::{PipelineBuilder, PipelineConfig, PipelineStage};

// ---------------------------------------------------------------------------
// Canonical stage order
// ---------------------------------------------------------------------------

/// The 10-stage pipeline in its required execution order.
const CANONICAL_ORDER: [PipelineStage; 10] = [
    PipelineStage::Ingest,
    PipelineStage::Normalize,
    PipelineStage::Vad,
    PipelineStage::Separate,
    PipelineStage::Backend,
    PipelineStage::Accelerate,
    PipelineStage::Align,
    PipelineStage::Punctuate,
    PipelineStage::Diarize,
    PipelineStage::Persist,
];

// ---------------------------------------------------------------------------
// 1. Default pipeline config matches canonical order
// ---------------------------------------------------------------------------

#[test]
fn stage_order_default_config_matches_canonical_order() {
    let config = PipelineConfig::default();
    let stages = config.stages();
    assert_eq!(
        stages.len(),
        10,
        "default pipeline should have exactly 10 stages, got {}",
        stages.len()
    );
    for (i, (actual, expected)) in stages.iter().zip(CANONICAL_ORDER.iter()).enumerate() {
        assert_eq!(
            actual, expected,
            "stage {i}: expected {:?}, got {:?}",
            expected, actual
        );
    }
}

// ---------------------------------------------------------------------------
// 2. PipelineBuilder::default_stages() matches canonical order
// ---------------------------------------------------------------------------

#[test]
fn stage_order_builder_default_stages_matches_canonical() {
    let config = PipelineBuilder::default_stages()
        .build()
        .expect("default_stages builder should produce a valid config");
    let stages = config.stages();
    assert_eq!(stages.len(), 10);
    for (i, (actual, expected)) in stages.iter().zip(CANONICAL_ORDER.iter()).enumerate() {
        assert_eq!(actual, expected, "builder default stage {i} mismatch");
    }
}

// ---------------------------------------------------------------------------
// 3. Each stage label is unique and non-empty
// ---------------------------------------------------------------------------

#[test]
fn stage_order_labels_are_unique_and_nonempty() {
    let mut seen = std::collections::HashSet::new();
    for stage in &CANONICAL_ORDER {
        let label = stage.label();
        assert!(!label.is_empty(), "stage {:?} has empty label", stage);
        assert!(seen.insert(label), "duplicate stage label: '{}'", label);
    }
}

// ---------------------------------------------------------------------------
// 4. Stage Display matches label()
// ---------------------------------------------------------------------------

#[test]
fn stage_order_display_matches_label() {
    for stage in &CANONICAL_ORDER {
        let display = format!("{stage}");
        assert_eq!(
            display,
            stage.label(),
            "Display for {:?} should match label()",
            stage
        );
    }
}

// ---------------------------------------------------------------------------
// 5. Default config validates successfully
// ---------------------------------------------------------------------------

#[test]
fn stage_order_default_config_validates() {
    let config = PipelineConfig::default();
    config
        .validate()
        .expect("default config should validate without error");
}

// ---------------------------------------------------------------------------
// 6. Backend cannot run before Ingest (transitive via Normalize)
// ---------------------------------------------------------------------------

#[test]
fn stage_order_backend_cannot_run_before_ingest() {
    // Backend requires Normalize, which requires Ingest.
    // Putting Backend before both should fail.
    let config = PipelineConfig::new(vec![
        PipelineStage::Backend,
        PipelineStage::Ingest,
        PipelineStage::Normalize,
    ]);
    let result = config.validate();
    assert!(
        result.is_err(),
        "Backend before Normalize should fail validation"
    );
    let err_text = result.unwrap_err().to_string();
    assert!(
        err_text.contains("Backend") || err_text.contains("Normalize"),
        "error should mention Backend or Normalize dependency: {err_text}"
    );
}

// ---------------------------------------------------------------------------
// 7. Backend requires Normalize before it
// ---------------------------------------------------------------------------

#[test]
fn stage_order_backend_requires_normalize() {
    let config = PipelineConfig::new(vec![PipelineStage::Ingest, PipelineStage::Backend]);
    let result = config.validate();
    assert!(
        result.is_err(),
        "Backend without Normalize should fail validation"
    );
    let err_text = result.unwrap_err().to_string();
    assert!(
        err_text.contains("Backend") && err_text.contains("Normalize"),
        "error should mention Backend requires Normalize: {err_text}"
    );
}

// ---------------------------------------------------------------------------
// 8. Normalize requires Ingest before it
// ---------------------------------------------------------------------------

#[test]
fn stage_order_normalize_requires_ingest() {
    let config = PipelineConfig::new(vec![PipelineStage::Normalize, PipelineStage::Ingest]);
    let result = config.validate();
    assert!(
        result.is_err(),
        "Normalize before Ingest should fail validation"
    );
    let err_text = result.unwrap_err().to_string();
    assert!(
        err_text.contains("Normalize") && err_text.contains("Ingest"),
        "error should mention Normalize requires Ingest: {err_text}"
    );
}

// ---------------------------------------------------------------------------
// 9. Accelerate requires Backend before it
// ---------------------------------------------------------------------------

#[test]
fn stage_order_accelerate_requires_backend() {
    let config = PipelineConfig::new(vec![
        PipelineStage::Ingest,
        PipelineStage::Normalize,
        PipelineStage::Accelerate,
    ]);
    let result = config.validate();
    assert!(
        result.is_err(),
        "Accelerate without Backend should fail validation"
    );
    let err_text = result.unwrap_err().to_string();
    assert!(
        err_text.contains("Accelerate") && err_text.contains("Backend"),
        "error should mention Accelerate requires Backend: {err_text}"
    );
}

// ---------------------------------------------------------------------------
// 10. Align requires Backend before it
// ---------------------------------------------------------------------------

#[test]
fn stage_order_align_requires_backend() {
    let config = PipelineConfig::new(vec![
        PipelineStage::Ingest,
        PipelineStage::Normalize,
        PipelineStage::Align,
    ]);
    let result = config.validate();
    assert!(
        result.is_err(),
        "Align without Backend should fail validation"
    );
    let err_text = result.unwrap_err().to_string();
    assert!(
        err_text.contains("Align") && err_text.contains("Backend"),
        "error should mention Align requires Backend: {err_text}"
    );
}

// ---------------------------------------------------------------------------
// 11. Vad requires Normalize before it
// ---------------------------------------------------------------------------

#[test]
fn stage_order_vad_requires_normalize() {
    let config = PipelineConfig::new(vec![PipelineStage::Ingest, PipelineStage::Vad]);
    let result = config.validate();
    assert!(
        result.is_err(),
        "Vad without Normalize should fail validation"
    );
    let err_text = result.unwrap_err().to_string();
    assert!(
        err_text.contains("Vad") && err_text.contains("Normalize"),
        "error should mention Vad requires Normalize: {err_text}"
    );
}

// ---------------------------------------------------------------------------
// 12. Separate requires Normalize before it
// ---------------------------------------------------------------------------

#[test]
fn stage_order_separate_requires_normalize() {
    let config = PipelineConfig::new(vec![PipelineStage::Ingest, PipelineStage::Separate]);
    let result = config.validate();
    assert!(
        result.is_err(),
        "Separate without Normalize should fail validation"
    );
    let err_text = result.unwrap_err().to_string();
    assert!(
        err_text.contains("Separate") && err_text.contains("Normalize"),
        "error should mention Separate requires Normalize: {err_text}"
    );
}

// ---------------------------------------------------------------------------
// 13. Punctuate requires Backend before it
// ---------------------------------------------------------------------------

#[test]
fn stage_order_punctuate_requires_backend() {
    let config = PipelineConfig::new(vec![
        PipelineStage::Ingest,
        PipelineStage::Normalize,
        PipelineStage::Punctuate,
    ]);
    let result = config.validate();
    assert!(
        result.is_err(),
        "Punctuate without Backend should fail validation"
    );
    let err_text = result.unwrap_err().to_string();
    assert!(
        err_text.contains("Punctuate") && err_text.contains("Backend"),
        "error should mention Punctuate requires Backend: {err_text}"
    );
}

// ---------------------------------------------------------------------------
// 14. Diarize requires Backend before it
// ---------------------------------------------------------------------------

#[test]
fn stage_order_diarize_requires_backend() {
    let config = PipelineConfig::new(vec![
        PipelineStage::Ingest,
        PipelineStage::Normalize,
        PipelineStage::Diarize,
    ]);
    let result = config.validate();
    assert!(
        result.is_err(),
        "Diarize without Backend should fail validation"
    );
    let err_text = result.unwrap_err().to_string();
    assert!(
        err_text.contains("Diarize") && err_text.contains("Backend"),
        "error should mention Diarize requires Backend: {err_text}"
    );
}

// ---------------------------------------------------------------------------
// 15. Duplicate stages are rejected
// ---------------------------------------------------------------------------

#[test]
fn stage_order_duplicate_stages_rejected() {
    let config = PipelineConfig::new(vec![
        PipelineStage::Ingest,
        PipelineStage::Normalize,
        PipelineStage::Backend,
        PipelineStage::Ingest, // duplicate
    ]);
    let result = config.validate();
    assert!(
        result.is_err(),
        "duplicate Ingest stage should fail validation"
    );
    let err_text = result.unwrap_err().to_string();
    assert!(
        err_text.contains("duplicate"),
        "error should mention 'duplicate': {err_text}"
    );
}

// ---------------------------------------------------------------------------
// 16. skip_stages (without) properly excludes stages
// ---------------------------------------------------------------------------

#[test]
fn stage_order_skip_stages_excludes_vad() {
    let config = PipelineBuilder::default_stages()
        .without(PipelineStage::Vad)
        .build()
        .expect("removing Vad should still validate");
    assert!(
        !config.has_stage(PipelineStage::Vad),
        "Vad should not be present after skip"
    );
    assert_eq!(
        config.stages().len(),
        9,
        "should have 9 stages after removing Vad"
    );
}

#[test]
fn stage_order_skip_stages_excludes_separate() {
    let config = PipelineBuilder::default_stages()
        .without(PipelineStage::Separate)
        .build()
        .expect("removing Separate should still validate");
    assert!(
        !config.has_stage(PipelineStage::Separate),
        "Separate should not be present after skip"
    );
}

#[test]
fn stage_order_skip_stages_excludes_diarize() {
    let config = PipelineBuilder::default_stages()
        .without(PipelineStage::Diarize)
        .build()
        .expect("removing Diarize should still validate");
    assert!(
        !config.has_stage(PipelineStage::Diarize),
        "Diarize should not be present after skip"
    );
}

#[test]
fn stage_order_skip_stages_excludes_persist() {
    let config = PipelineBuilder::default_stages()
        .without(PipelineStage::Persist)
        .build()
        .expect("removing Persist should still validate");
    assert!(
        !config.has_stage(PipelineStage::Persist),
        "Persist should not be present after skip"
    );
}

#[test]
fn stage_order_skip_multiple_stages() {
    let config = PipelineBuilder::default_stages()
        .without(PipelineStage::Vad)
        .without(PipelineStage::Separate)
        .without(PipelineStage::Align)
        .without(PipelineStage::Punctuate)
        .without(PipelineStage::Diarize)
        .without(PipelineStage::Persist)
        .build()
        .expect("minimal pipeline should validate");
    assert_eq!(
        config.stages().len(),
        4,
        "should have 4 stages: Ingest, Normalize, Backend, Accelerate"
    );
    assert!(config.has_stage(PipelineStage::Ingest));
    assert!(config.has_stage(PipelineStage::Normalize));
    assert!(config.has_stage(PipelineStage::Backend));
    assert!(config.has_stage(PipelineStage::Accelerate));
}

// ---------------------------------------------------------------------------
// 17. Skipping a required dependency breaks validation
// ---------------------------------------------------------------------------

#[test]
fn stage_order_skip_normalize_breaks_backend() {
    let result = PipelineBuilder::default_stages()
        .without(PipelineStage::Normalize)
        .build();
    assert!(
        result.is_err(),
        "removing Normalize should break Backend dependency"
    );
}

#[test]
fn stage_order_skip_ingest_breaks_normalize() {
    let result = PipelineBuilder::default_stages()
        .without(PipelineStage::Ingest)
        .build();
    assert!(
        result.is_err(),
        "removing Ingest should break Normalize dependency"
    );
}

#[test]
fn stage_order_skip_backend_breaks_accelerate() {
    let result = PipelineBuilder::default_stages()
        .without(PipelineStage::Backend)
        .build();
    assert!(
        result.is_err(),
        "removing Backend should break Accelerate dependency"
    );
}

// ---------------------------------------------------------------------------
// 18. Robot error codes are deterministic
// ---------------------------------------------------------------------------

#[test]
fn stage_order_robot_error_codes_deterministic() {
    // Run the same error construction 10 times and verify the code is stable.
    for _ in 0..10 {
        let io_err = FwError::Io(std::io::Error::other("disk failure"));
        assert_eq!(io_err.robot_error_code(), "FW-ROBOT-EXEC");
        assert_eq!(io_err.error_code(), "FW-IO");

        let timeout_err = FwError::StageTimeout {
            stage: "backend".to_owned(),
            budget_ms: 30_000,
        };
        assert_eq!(timeout_err.robot_error_code(), "FW-ROBOT-TIMEOUT");
        assert_eq!(timeout_err.error_code(), "FW-STAGE-TIMEOUT");

        let backend_err = FwError::BackendUnavailable("gone".to_owned());
        assert_eq!(backend_err.robot_error_code(), "FW-ROBOT-BACKEND");
        assert_eq!(backend_err.error_code(), "FW-BACKEND-UNAVAILABLE");

        let request_err = FwError::InvalidRequest("bad input".to_owned());
        assert_eq!(request_err.robot_error_code(), "FW-ROBOT-REQUEST");
        assert_eq!(request_err.error_code(), "FW-INVALID-REQUEST");

        let storage_err = FwError::Storage("db locked".to_owned());
        assert_eq!(storage_err.robot_error_code(), "FW-ROBOT-STORAGE");
        assert_eq!(storage_err.error_code(), "FW-STORAGE");

        let cancelled_err = FwError::Cancelled("user abort".to_owned());
        assert_eq!(cancelled_err.robot_error_code(), "FW-ROBOT-CANCELLED");
        assert_eq!(cancelled_err.error_code(), "FW-CANCELLED");
    }
}

// ---------------------------------------------------------------------------
// 19. Same error variant with same data always yields same error_code
// ---------------------------------------------------------------------------

#[test]
fn stage_order_error_code_same_input_same_output() {
    let inputs = vec![
        ("disk fail", "FW-IO"),
        ("network fail", "FW-IO"),
        ("permission denied", "FW-IO"),
    ];

    for (msg, expected_code) in inputs {
        let err1 = FwError::Io(std::io::Error::other(msg));
        let err2 = FwError::Io(std::io::Error::other(msg));
        assert_eq!(err1.error_code(), expected_code);
        assert_eq!(err2.error_code(), expected_code);
        assert_eq!(err1.error_code(), err2.error_code());
    }
}

// ---------------------------------------------------------------------------
// 20. Error codes start with "FW-" prefix
// ---------------------------------------------------------------------------

#[test]
fn stage_order_all_error_codes_have_fw_prefix() {
    let errors: Vec<FwError> = vec![
        FwError::Io(std::io::Error::other("x")),
        FwError::Json(serde_json::from_str::<serde_json::Value>("{").unwrap_err()),
        FwError::CommandMissing {
            command: "x".to_owned(),
        },
        FwError::CommandFailed {
            command: "x".to_owned(),
            status: 1,
            stderr_suffix: String::new(),
        },
        FwError::CommandTimedOut {
            command: "x".to_owned(),
            timeout_ms: 1,
            stderr_suffix: String::new(),
        },
        FwError::BackendUnavailable("x".to_owned()),
        FwError::InvalidRequest("x".to_owned()),
        FwError::Storage("x".to_owned()),
        FwError::Unsupported("x".to_owned()),
        FwError::MissingArtifact(std::path::PathBuf::from("x")),
        FwError::Cancelled("x".to_owned()),
        FwError::StageTimeout {
            stage: "x".to_owned(),
            budget_ms: 1,
        },
    ];

    for err in &errors {
        let code = err.error_code();
        assert!(
            code.starts_with("FW-"),
            "error_code must start with FW-: got '{code}' for {:?}",
            err
        );
        let robot_code = err.robot_error_code();
        assert!(
            robot_code.starts_with("FW-ROBOT-"),
            "robot_error_code must start with FW-ROBOT-: got '{robot_code}' for {:?}",
            err
        );
    }
}

// ---------------------------------------------------------------------------
// 21. PipelineConfig::has_stage works correctly
// ---------------------------------------------------------------------------

#[test]
fn stage_order_has_stage_reports_correctly() {
    let config = PipelineConfig::default();
    for stage in &CANONICAL_ORDER {
        assert!(
            config.has_stage(*stage),
            "default config should have stage {:?}",
            stage
        );
    }

    let minimal = PipelineBuilder::new()
        .stage(PipelineStage::Ingest)
        .stage(PipelineStage::Normalize)
        .stage(PipelineStage::Backend)
        .build()
        .expect("minimal valid pipeline");
    assert!(minimal.has_stage(PipelineStage::Ingest));
    assert!(minimal.has_stage(PipelineStage::Normalize));
    assert!(minimal.has_stage(PipelineStage::Backend));
    assert!(!minimal.has_stage(PipelineStage::Vad));
    assert!(!minimal.has_stage(PipelineStage::Separate));
    assert!(!minimal.has_stage(PipelineStage::Accelerate));
    assert!(!minimal.has_stage(PipelineStage::Align));
    assert!(!minimal.has_stage(PipelineStage::Punctuate));
    assert!(!minimal.has_stage(PipelineStage::Diarize));
    assert!(!minimal.has_stage(PipelineStage::Persist));
}

// ---------------------------------------------------------------------------
// 22. Builder chain constructs exact stage sequence
// ---------------------------------------------------------------------------

#[test]
fn stage_order_builder_chain_constructs_exact_sequence() {
    let config = PipelineBuilder::new()
        .stage(PipelineStage::Ingest)
        .stage(PipelineStage::Normalize)
        .stage(PipelineStage::Vad)
        .stage(PipelineStage::Separate)
        .stage(PipelineStage::Backend)
        .stage(PipelineStage::Accelerate)
        .stage(PipelineStage::Align)
        .stage(PipelineStage::Punctuate)
        .stage(PipelineStage::Diarize)
        .stage(PipelineStage::Persist)
        .build()
        .expect("full explicit pipeline should validate");

    let stages = config.stages();
    assert_eq!(stages.len(), 10);
    for (i, (actual, expected)) in stages.iter().zip(CANONICAL_ORDER.iter()).enumerate() {
        assert_eq!(actual, expected, "explicit builder stage {i} mismatch");
    }
}

// ---------------------------------------------------------------------------
// 23. Empty pipeline config validates (no stages to violate constraints)
// ---------------------------------------------------------------------------

#[test]
fn stage_order_empty_pipeline_validates() {
    let config = PipelineConfig::new(vec![]);
    config.validate().expect("empty pipeline should validate");
    assert_eq!(config.stages().len(), 0);
}

// ---------------------------------------------------------------------------
// 24. build_unchecked bypasses validation
// ---------------------------------------------------------------------------

#[test]
fn stage_order_build_unchecked_skips_validation() {
    // This would fail validation, but build_unchecked should succeed.
    let config = PipelineBuilder::new()
        .stage(PipelineStage::Backend)
        .stage(PipelineStage::Ingest)
        .build_unchecked();
    assert_eq!(config.stages().len(), 2);
    assert_eq!(config.stages()[0], PipelineStage::Backend);
    assert_eq!(config.stages()[1], PipelineStage::Ingest);
    // But validation should still fail on this config.
    assert!(config.validate().is_err());
}

// ---------------------------------------------------------------------------
// 25. Minimal valid pipeline: Ingest -> Normalize -> Backend
// ---------------------------------------------------------------------------

#[test]
fn stage_order_minimal_valid_pipeline() {
    let config = PipelineBuilder::new()
        .stage(PipelineStage::Ingest)
        .stage(PipelineStage::Normalize)
        .stage(PipelineStage::Backend)
        .build()
        .expect("Ingest -> Normalize -> Backend should be minimal valid pipeline");
    assert_eq!(config.stages().len(), 3);
}

// ---------------------------------------------------------------------------
// 26. Stage order preserved after without() calls
// ---------------------------------------------------------------------------

#[test]
fn stage_order_preserved_after_without() {
    let config = PipelineBuilder::default_stages()
        .without(PipelineStage::Vad)
        .without(PipelineStage::Separate)
        .without(PipelineStage::Align)
        .without(PipelineStage::Punctuate)
        .without(PipelineStage::Diarize)
        .without(PipelineStage::Persist)
        .build()
        .expect("should validate");

    let stages = config.stages();
    // Should be: Ingest, Normalize, Backend, Accelerate -- in that order.
    assert_eq!(stages[0], PipelineStage::Ingest);
    assert_eq!(stages[1], PipelineStage::Normalize);
    assert_eq!(stages[2], PipelineStage::Backend);
    assert_eq!(stages[3], PipelineStage::Accelerate);
}

// ---------------------------------------------------------------------------
// 27. Robot error code grouping is consistent
// ---------------------------------------------------------------------------

#[test]
fn stage_order_robot_error_code_grouping_contract() {
    // Timeout variants group together.
    let cmd_timeout = FwError::CommandTimedOut {
        command: "ffmpeg".to_owned(),
        timeout_ms: 5000,
        stderr_suffix: String::new(),
    };
    let stage_timeout = FwError::StageTimeout {
        stage: "normalize".to_owned(),
        budget_ms: 10_000,
    };
    assert_eq!(
        cmd_timeout.robot_error_code(),
        stage_timeout.robot_error_code()
    );
    assert_eq!(cmd_timeout.robot_error_code(), "FW-ROBOT-TIMEOUT");

    // But their fine-grained error_code() values differ.
    assert_ne!(cmd_timeout.error_code(), stage_timeout.error_code());
}
