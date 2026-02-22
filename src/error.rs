use std::path::PathBuf;

use thiserror::Error;

pub type FwResult<T> = Result<T, FwError>;

#[derive(Debug, Error)]
pub enum FwError {
    #[error("i/o failure: {0}")]
    Io(#[from] std::io::Error),

    #[error("json failure: {0}")]
    Json(#[from] serde_json::Error),

    #[error("missing command `{command}` on PATH")]
    CommandMissing { command: String },

    #[error("command failed: `{command}` (status: {status}){stderr_suffix}")]
    CommandFailed {
        command: String,
        status: i32,
        stderr_suffix: String,
    },

    #[error("command timed out after {timeout_ms}ms: `{command}`{stderr_suffix}")]
    CommandTimedOut {
        command: String,
        timeout_ms: u64,
        stderr_suffix: String,
    },

    #[error("backend unavailable: {0}")]
    BackendUnavailable(String),

    #[error("invalid request: {0}")]
    InvalidRequest(String),

    #[error("storage error: {0}")]
    Storage(String),

    #[error("unsupported operation: {0}")]
    Unsupported(String),

    #[error("missing expected artifact at `{0}`")]
    MissingArtifact(PathBuf),

    #[error("pipeline cancelled: {0}")]
    Cancelled(String),

    #[error("stage `{stage}` exceeded budget of {budget_ms}ms")]
    StageTimeout { stage: String, budget_ms: u64 },
}

impl FwError {
    #[must_use]
    pub fn from_command_failure(command: String, status: i32, stderr: String) -> Self {
        let trimmed = stderr.trim();
        let stderr_suffix = if trimmed.is_empty() {
            String::new()
        } else {
            format!("; stderr: {trimmed}")
        };
        Self::CommandFailed {
            command,
            status,
            stderr_suffix,
        }
    }

    #[must_use]
    pub fn from_command_timeout(command: String, timeout_ms: u64, stderr: String) -> Self {
        let trimmed = stderr.trim();
        let stderr_suffix = if trimmed.is_empty() {
            String::new()
        } else {
            format!("; stderr: {trimmed}")
        };
        Self::CommandTimedOut {
            command,
            timeout_ms,
            stderr_suffix,
        }
    }

    #[must_use]
    pub const fn robot_error_code(&self) -> &'static str {
        match self {
            Self::CommandTimedOut { .. } | Self::StageTimeout { .. } => "FW-ROBOT-TIMEOUT",
            Self::BackendUnavailable(_) => "FW-ROBOT-BACKEND",
            Self::InvalidRequest(_) => "FW-ROBOT-REQUEST",
            Self::Storage(_) => "FW-ROBOT-STORAGE",
            Self::Cancelled(_) => "FW-ROBOT-CANCELLED",
            _ => "FW-ROBOT-EXEC",
        }
    }

    /// Stable, unique, machine-readable error code for every variant.
    /// Unlike `robot_error_code()` which groups variants, this gives each
    /// variant its own distinct code string.
    #[must_use]
    pub const fn error_code(&self) -> &'static str {
        match self {
            Self::Io(_) => "FW-IO",
            Self::Json(_) => "FW-JSON",
            Self::CommandMissing { .. } => "FW-CMD-MISSING",
            Self::CommandFailed { .. } => "FW-CMD-FAILED",
            Self::CommandTimedOut { .. } => "FW-CMD-TIMEOUT",
            Self::BackendUnavailable(_) => "FW-BACKEND-UNAVAILABLE",
            Self::InvalidRequest(_) => "FW-INVALID-REQUEST",
            Self::Storage(_) => "FW-STORAGE",
            Self::Unsupported(_) => "FW-UNSUPPORTED",
            Self::MissingArtifact(_) => "FW-MISSING-ARTIFACT",
            Self::Cancelled(_) => "FW-CANCELLED",
            Self::StageTimeout { .. } => "FW-STAGE-TIMEOUT",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::FwError;

    #[test]
    fn timeout_errors_map_to_timeout_robot_code() {
        let stage_timeout = FwError::StageTimeout {
            stage: "backend".to_owned(),
            budget_ms: 10,
        };
        assert_eq!(stage_timeout.robot_error_code(), "FW-ROBOT-TIMEOUT");

        let command_timeout = FwError::from_command_timeout(
            "ffmpeg -i in.wav out.wav".to_owned(),
            1000,
            String::new(),
        );
        assert_eq!(command_timeout.robot_error_code(), "FW-ROBOT-TIMEOUT");
    }

    #[test]
    fn robot_error_code_mapping_regression_matrix() {
        let matrix = vec![
            (
                FwError::Io(std::io::Error::other("disk read failed")),
                "FW-ROBOT-EXEC",
            ),
            (
                FwError::Json(serde_json::from_str::<serde_json::Value>("{").unwrap_err()),
                "FW-ROBOT-EXEC",
            ),
            (
                FwError::BackendUnavailable("missing backend".to_owned()),
                "FW-ROBOT-BACKEND",
            ),
            (
                FwError::InvalidRequest("bad request".to_owned()),
                "FW-ROBOT-REQUEST",
            ),
            (FwError::Storage("db fail".to_owned()), "FW-ROBOT-STORAGE"),
            (
                FwError::Cancelled("cancelled".to_owned()),
                "FW-ROBOT-CANCELLED",
            ),
            (
                FwError::StageTimeout {
                    stage: "normalize".to_owned(),
                    budget_ms: 42,
                },
                "FW-ROBOT-TIMEOUT",
            ),
            (
                FwError::from_command_timeout(
                    "ffmpeg -i in.wav out.wav".to_owned(),
                    5_000,
                    String::new(),
                ),
                "FW-ROBOT-TIMEOUT",
            ),
            (
                FwError::CommandFailed {
                    command: "ffmpeg -i in.wav out.wav".to_owned(),
                    status: 1,
                    stderr_suffix: "; stderr: boom".to_owned(),
                },
                "FW-ROBOT-EXEC",
            ),
            (
                FwError::CommandMissing {
                    command: "ffmpeg".to_owned(),
                },
                "FW-ROBOT-EXEC",
            ),
            (
                FwError::MissingArtifact(std::path::PathBuf::from("missing.json")),
                "FW-ROBOT-EXEC",
            ),
            (
                FwError::Unsupported("not supported".to_owned()),
                "FW-ROBOT-EXEC",
            ),
        ];

        for (error, expected_code) in matrix {
            assert_eq!(error.robot_error_code(), expected_code);
        }
    }

    #[test]
    fn from_command_failure_with_empty_stderr() {
        let err = FwError::from_command_failure("cmd".to_owned(), 1, String::new());
        let text = err.to_string();
        assert!(text.contains("cmd"));
        assert!(text.contains("status: 1"));
        // No stderr suffix when stderr is empty.
        assert!(!text.contains("stderr"));
    }

    #[test]
    fn from_command_failure_with_nonempty_stderr() {
        let err = FwError::from_command_failure("prog arg".to_owned(), 2, "  oh no  \n".to_owned());
        let text = err.to_string();
        assert!(text.contains("prog arg"));
        assert!(text.contains("status: 2"));
        assert!(text.contains("stderr: oh no"), "should trim stderr: {text}");
    }

    #[test]
    fn from_command_timeout_with_empty_stderr() {
        let err = FwError::from_command_timeout("slow".to_owned(), 5000, String::new());
        let text = err.to_string();
        assert!(text.contains("5000ms"));
        assert!(!text.contains("stderr"));
    }

    #[test]
    fn from_command_timeout_with_nonempty_stderr() {
        let err =
            FwError::from_command_timeout("slow".to_owned(), 1000, "  partial output  ".to_owned());
        let text = err.to_string();
        assert!(text.contains("1000ms"));
        assert!(
            text.contains("stderr: partial output"),
            "should trim stderr: {text}"
        );
    }

    #[test]
    fn from_command_failure_multiline_stderr_is_trimmed() {
        let stderr = "  line one\nline two\n  line three  \n".to_owned();
        let err = FwError::from_command_failure("cmd".to_owned(), 1, stderr);
        let text = err.to_string();
        // Trim only strips leading/trailing whitespace, not internal newlines.
        assert!(
            text.contains("line one\nline two\n  line three"),
            "multiline stderr should preserve internal newlines: {text}"
        );
    }

    #[test]
    fn from_command_failure_whitespace_only_stderr_treated_as_empty() {
        let err = FwError::from_command_failure("cmd".to_owned(), 1, "   \n\t  ".to_owned());
        let text = err.to_string();
        assert!(
            !text.contains("stderr"),
            "whitespace-only stderr should be omitted: {text}"
        );
    }

    #[test]
    fn from_command_timeout_multiline_stderr_is_trimmed() {
        let stderr = "\nerror detail\nmore detail\n".to_owned();
        let err = FwError::from_command_timeout("slow".to_owned(), 5000, stderr);
        let text = err.to_string();
        assert!(
            text.contains("error detail\nmore detail"),
            "multiline stderr preserved: {text}"
        );
    }

    #[test]
    fn missing_artifact_displays_path() {
        let err = FwError::MissingArtifact(std::path::PathBuf::from("/tmp/output/result.json"));
        let text = err.to_string();
        assert!(
            text.contains("/tmp/output/result.json"),
            "should include full path: {text}"
        );
    }

    #[test]
    fn stage_timeout_displays_stage_and_budget() {
        let err = FwError::StageTimeout {
            stage: "normalize".to_owned(),
            budget_ms: 180_000,
        };
        let text = err.to_string();
        assert!(text.contains("normalize"), "should mention stage: {text}");
        assert!(text.contains("180000"), "should mention budget: {text}");
    }

    #[test]
    fn command_missing_displays_command_name() {
        let err = FwError::CommandMissing {
            command: "whisper-cli".to_owned(),
        };
        let text = err.to_string();
        assert!(
            text.contains("whisper-cli"),
            "should mention command: {text}"
        );
    }

    #[test]
    fn cancelled_error_displays_reason() {
        let err = FwError::Cancelled("pipeline deadline exceeded at ingest stage".to_owned());
        let text = err.to_string();
        assert!(
            text.contains("pipeline deadline exceeded"),
            "should include reason: {text}"
        );
    }

    #[test]
    fn display_messages_for_all_variants() {
        let cases: Vec<(FwError, &str)> = vec![
            (
                FwError::Io(std::io::Error::other("disk fail")),
                "i/o failure",
            ),
            (
                FwError::Json(serde_json::from_str::<serde_json::Value>("{").unwrap_err()),
                "json failure",
            ),
            (
                FwError::CommandMissing {
                    command: "ffmpeg".to_owned(),
                },
                "missing command",
            ),
            (
                FwError::CommandFailed {
                    command: "cmd".to_owned(),
                    status: 1,
                    stderr_suffix: String::new(),
                },
                "command failed",
            ),
            (
                FwError::CommandTimedOut {
                    command: "slow".to_owned(),
                    timeout_ms: 5000,
                    stderr_suffix: String::new(),
                },
                "command timed out",
            ),
            (
                FwError::BackendUnavailable("gone".to_owned()),
                "backend unavailable",
            ),
            (FwError::InvalidRequest("bad".to_owned()), "invalid request"),
            (FwError::Storage("db".to_owned()), "storage error"),
            (FwError::Unsupported("nope".to_owned()), "unsupported"),
            (
                FwError::MissingArtifact(std::path::PathBuf::from("out.json")),
                "missing expected artifact",
            ),
            (FwError::Cancelled("done".to_owned()), "pipeline cancelled"),
            (
                FwError::StageTimeout {
                    stage: "be".to_owned(),
                    budget_ms: 10,
                },
                "exceeded budget",
            ),
        ];

        // Verify we cover all 12 variants.
        assert_eq!(cases.len(), 12, "test should cover every FwError variant");

        for (error, expected_substring) in cases {
            let text = error.to_string();
            assert!(
                text.contains(expected_substring),
                "expected `{expected_substring}` in: {text}"
            );
        }
    }

    #[test]
    fn io_error_from_conversion() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
        let fw_err: FwError = io_err.into();
        assert!(matches!(fw_err, FwError::Io(_)));
        let text = fw_err.to_string();
        assert!(text.contains("file not found"), "got: {text}");
    }

    #[test]
    fn json_error_from_conversion() {
        let json_err = serde_json::from_str::<serde_json::Value>("not json").unwrap_err();
        let fw_err: FwError = json_err.into();
        assert!(matches!(fw_err, FwError::Json(_)));
        let text = fw_err.to_string();
        assert!(
            text.contains("json failure"),
            "should start with 'json failure': {text}"
        );
    }

    #[test]
    fn stage_timeout_zero_budget() {
        let err = FwError::StageTimeout {
            stage: "ingest".to_owned(),
            budget_ms: 0,
        };
        let text = err.to_string();
        assert!(text.contains("0ms"), "should show 0ms budget: {text}");
        assert!(text.contains("ingest"), "should show stage: {text}");
    }

    #[test]
    fn backend_unavailable_display() {
        let err = FwError::BackendUnavailable(
            "whisper-cli not found; insanely-fast-whisper not found".to_owned(),
        );
        let text = err.to_string();
        assert!(
            text.contains("whisper-cli not found"),
            "should contain detail: {text}"
        );
    }

    #[test]
    fn invalid_request_display() {
        let err = FwError::InvalidRequest("multiple inputs specified".to_owned());
        let text = err.to_string();
        assert!(
            text.contains("multiple inputs"),
            "should contain detail: {text}"
        );
    }

    #[test]
    fn storage_error_display() {
        let err = FwError::Storage("database locked".to_owned());
        let text = err.to_string();
        assert!(
            text.contains("database locked"),
            "should contain detail: {text}"
        );
    }

    #[test]
    fn unsupported_display() {
        let err = FwError::Unsupported("streaming not implemented".to_owned());
        let text = err.to_string();
        assert!(
            text.contains("streaming not implemented"),
            "should contain detail: {text}"
        );
    }

    #[test]
    fn command_timed_out_large_timeout() {
        let err = FwError::CommandTimedOut {
            command: "whisper-cli --model large-v3".to_owned(),
            timeout_ms: 3_600_000,
            stderr_suffix: String::new(),
        };
        let text = err.to_string();
        assert!(
            text.contains("3600000ms"),
            "should show large timeout: {text}"
        );
    }

    #[test]
    fn from_command_timeout_whitespace_only_stderr_treated_as_empty() {
        let err = FwError::from_command_timeout("slow".to_owned(), 5000, "   \n\t  \n".to_owned());
        let text = err.to_string();
        assert!(
            !text.contains("stderr"),
            "whitespace-only stderr should be omitted in timeout too: {text}"
        );
    }

    #[test]
    fn from_command_failure_zero_and_negative_status() {
        let zero = FwError::from_command_failure("cmd".to_owned(), 0, String::new());
        assert!(
            zero.to_string().contains("status: 0"),
            "zero status displayed"
        );

        let neg = FwError::from_command_failure("cmd".to_owned(), -9, "killed".to_owned());
        let text = neg.to_string();
        assert!(text.contains("status: -9"), "negative status: {text}");
        assert!(text.contains("stderr: killed"), "stderr present: {text}");
    }

    #[test]
    fn fw_error_is_send_and_sync() {
        fn assert_send<T: Send>() {}
        fn assert_sync<T: Sync>() {}
        assert_send::<FwError>();
        assert_sync::<FwError>();
    }

    #[test]
    fn missing_artifact_unicode_path() {
        let err = FwError::MissingArtifact(std::path::PathBuf::from("/tmp/données/résultat.json"));
        let text = err.to_string();
        assert!(
            text.contains("résultat.json"),
            "unicode path preserved: {text}"
        );
    }

    #[test]
    fn robot_error_code_returns_only_known_codes() {
        let known_codes = [
            "FW-ROBOT-EXEC",
            "FW-ROBOT-TIMEOUT",
            "FW-ROBOT-BACKEND",
            "FW-ROBOT-REQUEST",
            "FW-ROBOT-STORAGE",
            "FW-ROBOT-CANCELLED",
        ];

        let all_errors: Vec<FwError> = vec![
            FwError::Io(std::io::Error::other("test")),
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

        for error in &all_errors {
            let code = error.robot_error_code();
            assert!(
                known_codes.contains(&code),
                "unexpected robot_error_code `{code}` for {:?}",
                error
            );
        }
    }

    // ── error_code() tests ──

    /// Construct every FwError variant and verify error_code() returns a
    /// non-empty string starting with "FW-".
    #[test]
    fn test_every_variant_has_error_code() {
        let all_errors: Vec<FwError> = vec![
            FwError::Io(std::io::Error::other("test")),
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

        // Ensure we cover all 12 variants.
        assert_eq!(
            all_errors.len(),
            12,
            "test should cover every FwError variant"
        );

        for error in &all_errors {
            let code = error.error_code();
            assert!(
                !code.is_empty(),
                "error_code() must not be empty for {:?}",
                error
            );
            assert!(
                code.starts_with("FW-"),
                "error_code() must start with FW- but got `{code}` for {:?}",
                error
            );
        }
    }

    /// Verify no two variants share the same error_code().
    #[test]
    fn test_error_codes_are_unique() {
        let all_errors: Vec<FwError> = vec![
            FwError::Io(std::io::Error::other("test")),
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

        let codes: Vec<&str> = all_errors.iter().map(|e| e.error_code()).collect();
        let mut seen = std::collections::HashSet::new();
        for code in &codes {
            assert!(seen.insert(code), "duplicate error_code detected: `{code}`");
        }
    }

    /// Verify all error codes match the pattern FW-[A-Z-]+.
    #[test]
    fn test_error_code_format() {
        let all_errors: Vec<FwError> = vec![
            FwError::Io(std::io::Error::other("test")),
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

        for error in &all_errors {
            let code = error.error_code();
            assert!(
                code.starts_with("FW-"),
                "code must start with FW-: `{code}`"
            );
            let suffix = &code[3..]; // everything after "FW-"
            assert!(
                !suffix.is_empty() && suffix.chars().all(|c| c.is_ascii_uppercase() || c == '-'),
                "code suffix must match [A-Z-]+ but got `{suffix}` in `{code}`"
            );
        }
    }
}
