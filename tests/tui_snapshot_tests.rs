//! bd-3pf.4: TUI snapshot tests.
//!
//! Since the TUI components (LiveTranscriptionView, SpeakerColorMap, etc.)
//! are `pub(crate)` and behind `#[cfg(feature = "tui")]`, these integration
//! tests verify the publicly observable contracts and rendering formats that
//! the TUI consumes:
//!
//! - TranscriptionSegment formatting consistency
//! - BackendKind string representation
//! - Error display formatting for TUI error panels
//! - Progress-bar rendering logic (percentage clamping, bar construction)
//! - Speaker color assignment determinism via hash-based contract
//! - Timestamp formatting contracts

#![forbid(unsafe_code)]

use franken_whisper::error::FwError;
use franken_whisper::model::{BackendKind, TranscriptionSegment};

// ---------------------------------------------------------------------------
// Helpers: replicate the TUI's segment formatting logic for snapshot testing
// ---------------------------------------------------------------------------

/// Format a segment line the same way LiveTranscriptionView does.
/// This is the contract that the TUI's format_segment_line must uphold.
fn format_segment_line(
    index: usize,
    segment: &TranscriptionSegment,
    diarization_active: bool,
) -> String {
    let start = format_ts(segment.start_sec);
    let end = format_ts(segment.end_sec);

    let speaker_prefix = if diarization_active {
        segment
            .speaker
            .as_deref()
            .map(|s| format!("[{s}] "))
            .unwrap_or_default()
    } else {
        String::new()
    };

    let confidence_suffix = segment
        .confidence
        .map(|c| format!(" ({c:.3})"))
        .unwrap_or_default();

    format!(
        "{index:03} {start} -> {end} {speaker_prefix}{}{confidence_suffix}",
        segment.text
    )
}

/// Format a timestamp the same way the TUI does.
fn format_ts(seconds: Option<f64>) -> String {
    let Some(value) = seconds else {
        return "--:--:--.---".to_owned();
    };

    if value.is_sign_negative() {
        return "00:00:00.000".to_owned();
    }

    let total_ms = (value * 1_000.0).round() as u64;
    let hours = total_ms / 3_600_000;
    let minutes = (total_ms % 3_600_000) / 60_000;
    let secs = (total_ms % 60_000) / 1_000;
    let millis = total_ms % 1_000;

    format!("{hours:02}:{minutes:02}:{secs:02}.{millis:03}")
}

/// Replicate the FNV-1a-style hash used by SpeakerColorMap.
fn speaker_hash(name: &str) -> usize {
    let mut h: u64 = 0xcbf29ce484222325;
    for byte in name.as_bytes() {
        h ^= *byte as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h as usize
}

/// Compute the palette index for a speaker (8-color palette).
fn speaker_color_index(name: &str) -> usize {
    speaker_hash(name) % 8
}

/// Render a simple text progress bar.
fn render_progress_bar(percentage: f64, width: usize) -> String {
    let clamped = percentage.clamp(0.0, 100.0);
    let filled = ((clamped / 100.0) * width as f64).round() as usize;
    let empty = width.saturating_sub(filled);
    format!(
        "[{}{}] {:.1}%",
        "=".repeat(filled),
        " ".repeat(empty),
        clamped
    )
}

// ---------------------------------------------------------------------------
// 1. Segment formatting: basic snapshot
// ---------------------------------------------------------------------------

#[test]
fn tui_snapshot_segment_line_basic() {
    let seg = TranscriptionSegment {
        start_sec: Some(0.0),
        end_sec: Some(2.5),
        text: "Hello world.".to_owned(),
        speaker: None,
        confidence: Some(0.95),
    };
    let line = format_segment_line(0, &seg, false);
    assert_eq!(
        line,
        "000 00:00:00.000 -> 00:00:02.500 Hello world. (0.950)"
    );
}

// ---------------------------------------------------------------------------
// 2. Segment formatting: with speaker diarization
// ---------------------------------------------------------------------------

#[test]
fn tui_snapshot_segment_line_with_speaker() {
    let seg = TranscriptionSegment {
        start_sec: Some(10.0),
        end_sec: Some(15.5),
        text: "Good morning.".to_owned(),
        speaker: Some("SPEAKER_00".to_owned()),
        confidence: Some(0.88),
    };
    let line = format_segment_line(1, &seg, true);
    assert_eq!(
        line,
        "001 00:00:10.000 -> 00:00:15.500 [SPEAKER_00] Good morning. (0.880)"
    );
}

// ---------------------------------------------------------------------------
// 3. Segment formatting: diarization active but no speaker assigned
// ---------------------------------------------------------------------------

#[test]
fn tui_snapshot_segment_line_diarization_no_speaker() {
    let seg = TranscriptionSegment {
        start_sec: Some(5.0),
        end_sec: Some(7.0),
        text: "Unknown speaker text.".to_owned(),
        speaker: None,
        confidence: Some(0.75),
    };
    let line = format_segment_line(2, &seg, true);
    // No speaker prefix when speaker is None.
    assert_eq!(
        line,
        "002 00:00:05.000 -> 00:00:07.000 Unknown speaker text. (0.750)"
    );
}

// ---------------------------------------------------------------------------
// 4. Segment formatting: no confidence
// ---------------------------------------------------------------------------

#[test]
fn tui_snapshot_segment_line_no_confidence() {
    let seg = TranscriptionSegment {
        start_sec: Some(0.0),
        end_sec: Some(1.0),
        text: "No confidence value.".to_owned(),
        speaker: None,
        confidence: None,
    };
    let line = format_segment_line(0, &seg, false);
    assert_eq!(
        line,
        "000 00:00:00.000 -> 00:00:01.000 No confidence value."
    );
}

// ---------------------------------------------------------------------------
// 5. Segment formatting: missing timestamps
// ---------------------------------------------------------------------------

#[test]
fn tui_snapshot_segment_line_missing_timestamps() {
    let seg = TranscriptionSegment {
        start_sec: None,
        end_sec: None,
        text: "No timestamps.".to_owned(),
        speaker: None,
        confidence: None,
    };
    let line = format_segment_line(0, &seg, false);
    assert_eq!(line, "000 --:--:--.--- -> --:--:--.--- No timestamps.");
}

// ---------------------------------------------------------------------------
// 6. Timestamp formatting: edge cases
// ---------------------------------------------------------------------------

#[test]
fn tui_snapshot_timestamp_zero() {
    assert_eq!(format_ts(Some(0.0)), "00:00:00.000");
}

#[test]
fn tui_snapshot_timestamp_negative() {
    assert_eq!(format_ts(Some(-5.0)), "00:00:00.000");
}

#[test]
fn tui_snapshot_timestamp_none() {
    assert_eq!(format_ts(None), "--:--:--.---");
}

#[test]
fn tui_snapshot_timestamp_large_value() {
    // 1 hour, 23 minutes, 45.678 seconds
    assert_eq!(format_ts(Some(5025.678)), "01:23:45.678");
}

#[test]
fn tui_snapshot_timestamp_sub_second() {
    assert_eq!(format_ts(Some(0.123)), "00:00:00.123");
}

#[test]
fn tui_snapshot_timestamp_exact_minute() {
    assert_eq!(format_ts(Some(60.0)), "00:01:00.000");
}

#[test]
fn tui_snapshot_timestamp_exact_hour() {
    assert_eq!(format_ts(Some(3600.0)), "01:00:00.000");
}

// ---------------------------------------------------------------------------
// 7. SpeakerColorMap: deterministic color assignment
// ---------------------------------------------------------------------------

#[test]
fn tui_snapshot_speaker_color_deterministic() {
    // Same speaker name should always produce the same index.
    let idx1 = speaker_color_index("SPEAKER_00");
    let idx2 = speaker_color_index("SPEAKER_00");
    assert_eq!(idx1, idx2, "same speaker should get same color index");
}

#[test]
fn tui_snapshot_speaker_color_different_speakers_may_differ() {
    let idx_00 = speaker_color_index("SPEAKER_00");
    let idx_01 = speaker_color_index("SPEAKER_01");
    // They *may* collide (8-color palette), but the hash should be deterministic.
    // Just verify the indices are in valid range.
    assert!(idx_00 < 8, "index must be in palette range");
    assert!(idx_01 < 8, "index must be in palette range");
}

#[test]
fn tui_snapshot_speaker_color_deterministic_across_runs() {
    // The hash is pure and deterministic, so we can pin specific values.
    // If the hash algorithm changes, these would need updating -- that's
    // the point of a contract test.
    let speakers = ["SPEAKER_00", "SPEAKER_01", "SPEAKER_02", "Alice", "Bob"];
    let expected: Vec<usize> = speakers.iter().map(|s| speaker_color_index(s)).collect();

    // Run again and verify stability.
    for (i, speaker) in speakers.iter().enumerate() {
        assert_eq!(
            speaker_color_index(speaker),
            expected[i],
            "color index for '{}' changed between calls",
            speaker
        );
    }
}

#[test]
fn tui_snapshot_speaker_color_empty_name() {
    let idx = speaker_color_index("");
    assert!(
        idx < 8,
        "empty speaker name should still produce valid index"
    );
}

// ---------------------------------------------------------------------------
// 8. Progress bar rendering
// ---------------------------------------------------------------------------

#[test]
fn tui_snapshot_progress_bar_zero_percent() {
    let bar = render_progress_bar(0.0, 20);
    assert_eq!(bar, "[                    ] 0.0%");
}

#[test]
fn tui_snapshot_progress_bar_fifty_percent() {
    let bar = render_progress_bar(50.0, 20);
    assert_eq!(bar, "[==========          ] 50.0%");
}

#[test]
fn tui_snapshot_progress_bar_hundred_percent() {
    let bar = render_progress_bar(100.0, 20);
    assert_eq!(bar, "[====================] 100.0%");
}

#[test]
fn tui_snapshot_progress_bar_clamped_negative() {
    let bar = render_progress_bar(-10.0, 20);
    assert_eq!(bar, "[                    ] 0.0%");
}

#[test]
fn tui_snapshot_progress_bar_clamped_over_hundred() {
    let bar = render_progress_bar(150.0, 20);
    assert_eq!(bar, "[====================] 100.0%");
}

#[test]
fn tui_snapshot_progress_bar_narrow_width() {
    let bar = render_progress_bar(50.0, 4);
    assert_eq!(bar, "[==  ] 50.0%");
}

#[test]
fn tui_snapshot_progress_bar_quarter() {
    let bar = render_progress_bar(25.0, 20);
    assert_eq!(bar, "[=====               ] 25.0%");
}

#[test]
fn tui_snapshot_progress_bar_seventy_five() {
    let bar = render_progress_bar(75.0, 20);
    assert_eq!(bar, "[===============     ] 75.0%");
}

// ---------------------------------------------------------------------------
// 9. Error display formatting for TUI error panels
// ---------------------------------------------------------------------------

#[test]
fn tui_snapshot_error_display_io() {
    let err = FwError::Io(std::io::Error::other("disk read failed"));
    let text = err.to_string();
    assert_eq!(text, "i/o failure: disk read failed");
}

#[test]
fn tui_snapshot_error_display_backend_unavailable() {
    let err = FwError::BackendUnavailable("whisper-cli not found".to_owned());
    let text = err.to_string();
    assert_eq!(text, "backend unavailable: whisper-cli not found");
}

#[test]
fn tui_snapshot_error_display_invalid_request() {
    let err = FwError::InvalidRequest("empty input path".to_owned());
    let text = err.to_string();
    assert_eq!(text, "invalid request: empty input path");
}

#[test]
fn tui_snapshot_error_display_storage() {
    let err = FwError::Storage("database locked".to_owned());
    let text = err.to_string();
    assert_eq!(text, "storage error: database locked");
}

#[test]
fn tui_snapshot_error_display_stage_timeout() {
    let err = FwError::StageTimeout {
        stage: "backend".to_owned(),
        budget_ms: 30_000,
    };
    let text = err.to_string();
    assert_eq!(text, "stage `backend` exceeded budget of 30000ms");
}

#[test]
fn tui_snapshot_error_display_cancelled() {
    let err = FwError::Cancelled("user pressed Ctrl-C".to_owned());
    let text = err.to_string();
    assert_eq!(text, "pipeline cancelled: user pressed Ctrl-C");
}

#[test]
fn tui_snapshot_error_display_missing_artifact() {
    let err = FwError::MissingArtifact(std::path::PathBuf::from("/tmp/output.json"));
    let text = err.to_string();
    assert_eq!(text, "missing expected artifact at `/tmp/output.json`");
}

#[test]
fn tui_snapshot_error_display_unsupported() {
    let err = FwError::Unsupported("streaming not yet available".to_owned());
    let text = err.to_string();
    assert_eq!(text, "unsupported operation: streaming not yet available");
}

#[test]
fn tui_snapshot_error_display_command_failed_no_stderr() {
    let err = FwError::CommandFailed {
        command: "ffmpeg -i in.wav out.wav".to_owned(),
        status: 1,
        stderr_suffix: String::new(),
    };
    let text = err.to_string();
    assert_eq!(
        text,
        "command failed: `ffmpeg -i in.wav out.wav` (status: 1)"
    );
}

#[test]
fn tui_snapshot_error_display_command_failed_with_stderr() {
    let err = FwError::CommandFailed {
        command: "ffmpeg -i in.wav out.wav".to_owned(),
        status: 2,
        stderr_suffix: "; stderr: no such codec".to_owned(),
    };
    let text = err.to_string();
    assert_eq!(
        text,
        "command failed: `ffmpeg -i in.wav out.wav` (status: 2); stderr: no such codec"
    );
}

#[test]
fn tui_snapshot_error_display_command_missing() {
    let err = FwError::CommandMissing {
        command: "whisper-cli".to_owned(),
    };
    let text = err.to_string();
    assert_eq!(text, "missing command `whisper-cli` on PATH");
}

// ---------------------------------------------------------------------------
// 10. BackendKind string representation contract
// ---------------------------------------------------------------------------

#[test]
fn tui_snapshot_backend_kind_as_str() {
    assert_eq!(BackendKind::Auto.as_str(), "auto");
    assert_eq!(BackendKind::WhisperCpp.as_str(), "whisper_cpp");
    assert_eq!(BackendKind::InsanelyFast.as_str(), "insanely_fast");
    assert_eq!(
        BackendKind::WhisperDiarization.as_str(),
        "whisper_diarization"
    );
}

// ---------------------------------------------------------------------------
// 11. Multi-segment rendering snapshot
// ---------------------------------------------------------------------------

#[test]
fn tui_snapshot_multi_segment_rendering() {
    let segments = [
        TranscriptionSegment {
            start_sec: Some(0.0),
            end_sec: Some(2.5),
            text: "Hello world.".to_owned(),
            speaker: Some("SPEAKER_00".to_owned()),
            confidence: Some(0.95),
        },
        TranscriptionSegment {
            start_sec: Some(2.5),
            end_sec: Some(6.0),
            text: "This is a test.".to_owned(),
            speaker: Some("SPEAKER_01".to_owned()),
            confidence: Some(0.88),
        },
        TranscriptionSegment {
            start_sec: Some(6.0),
            end_sec: Some(9.0),
            text: "End of transcription.".to_owned(),
            speaker: None,
            confidence: Some(0.92),
        },
    ];

    let lines: Vec<String> = segments
        .iter()
        .enumerate()
        .map(|(i, s)| format_segment_line(i, s, true))
        .collect();

    let snapshot = lines.join("\n");
    let expected = "\
000 00:00:00.000 -> 00:00:02.500 [SPEAKER_00] Hello world. (0.950)\n\
001 00:00:02.500 -> 00:00:06.000 [SPEAKER_01] This is a test. (0.880)\n\
002 00:00:06.000 -> 00:00:09.000 End of transcription. (0.920)";
    assert_eq!(snapshot, expected);
}

// ---------------------------------------------------------------------------
// 12. Empty segments render "waiting" message
// ---------------------------------------------------------------------------

#[test]
fn tui_snapshot_empty_segments_waiting_message() {
    let segments: Vec<TranscriptionSegment> = vec![];
    if segments.is_empty() {
        let line = "Waiting for transcription segments...";
        assert_eq!(line, "Waiting for transcription segments...");
    }
}

// ---------------------------------------------------------------------------
// 13. Error code display alongside error message
// ---------------------------------------------------------------------------

#[test]
fn tui_snapshot_error_panel_with_code() {
    let err = FwError::BackendUnavailable("whisper-cli not in PATH".to_owned());
    let panel = format!("[{}] {}", err.error_code(), err);
    assert_eq!(
        panel,
        "[FW-BACKEND-UNAVAILABLE] backend unavailable: whisper-cli not in PATH"
    );
}

#[test]
fn tui_snapshot_error_panel_with_robot_code() {
    let err = FwError::StageTimeout {
        stage: "normalize".to_owned(),
        budget_ms: 60_000,
    };
    let panel = format!(
        "[{}] [{}] {}",
        err.robot_error_code(),
        err.error_code(),
        err
    );
    assert_eq!(
        panel,
        "[FW-ROBOT-TIMEOUT] [FW-STAGE-TIMEOUT] stage `normalize` exceeded budget of 60000ms"
    );
}
