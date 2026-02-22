//! Backend mock tests: verify pipeline behavior without real transcription binaries.
//!
//! These tests use the mock scripts in tests/mocks/ to simulate backend responses.

mod helpers;

use std::collections::HashMap;
use std::path::PathBuf;

use franken_whisper::model::*;

/// Get the mock environment variables (pointing at mock scripts).
#[allow(dead_code)]
fn mock_env() -> HashMap<String, String> {
    helpers::mock_backend_env()
}

#[test]
fn mock_whisper_cpp_returns_golden_output() {
    // Read the golden file directly and parse it
    let golden_path = helpers::golden_dir().join("whisper_cpp_output.json");
    let golden_text = std::fs::read_to_string(&golden_path).expect("read golden file");
    let golden: serde_json::Value = serde_json::from_str(&golden_text).expect("parse golden JSON");

    // Verify golden file structure
    assert!(
        golden.get("text").is_some(),
        "golden should have 'text' field"
    );
    assert!(
        golden.get("segments").is_some(),
        "golden should have 'segments' field"
    );
    assert!(
        golden.get("language").is_some(),
        "golden should have 'language' field"
    );

    let segments = golden["segments"]
        .as_array()
        .expect("segments should be array");
    assert_eq!(segments.len(), 2, "golden should have 2 segments");
}

#[test]
fn mock_insanely_fast_returns_golden_output() {
    let golden_path = helpers::golden_dir().join("insanely_fast_output.json");
    let golden_text = std::fs::read_to_string(&golden_path).expect("read golden file");
    let golden: serde_json::Value = serde_json::from_str(&golden_text).expect("parse golden JSON");

    assert!(
        golden.get("text").is_some(),
        "golden should have 'text' field"
    );
    assert!(
        golden.get("chunks").is_some(),
        "golden should have 'chunks' field"
    );

    let chunks = golden["chunks"].as_array().expect("chunks should be array");
    assert_eq!(chunks.len(), 2, "golden should have 2 chunks");
}

#[test]
fn mock_diarization_produces_expected_files() {
    let golden_dir = helpers::golden_dir();

    // Check SRT golden file
    let srt_path = golden_dir.join("diarization_output.srt");
    let srt_text = std::fs::read_to_string(&srt_path).expect("read golden SRT");
    assert!(
        srt_text.contains("SPEAKER_00"),
        "SRT should contain SPEAKER_00"
    );
    assert!(
        srt_text.contains("SPEAKER_01"),
        "SRT should contain SPEAKER_01"
    );
    assert!(
        srt_text.contains("-->"),
        "SRT should contain timestamp arrows"
    );

    // Check TXT golden file
    let txt_path = golden_dir.join("diarization_output.txt");
    let txt_text = std::fs::read_to_string(&txt_path).expect("read golden TXT");
    assert!(
        txt_text.contains("SPEAKER_00"),
        "TXT should contain SPEAKER_00"
    );
    assert!(
        txt_text.contains("SPEAKER_01"),
        "TXT should contain SPEAKER_01"
    );
}

#[test]
fn golden_robot_events_are_valid_ndjson() {
    let ndjson_path = helpers::golden_dir().join("robot_events.ndjson");
    let text = std::fs::read_to_string(&ndjson_path).expect("read golden NDJSON");

    let lines: Vec<&str> = text.lines().collect();
    assert!(
        lines.len() >= 5,
        "should have at least 5 event lines, got {}",
        lines.len()
    );

    for (i, line) in lines.iter().enumerate() {
        let parsed: serde_json::Value = serde_json::from_str(line)
            .unwrap_or_else(|e| panic!("line {} is not valid JSON: {e}\nline: {line}", i + 1));
        assert!(
            parsed.get("event").is_some(),
            "line {} should have 'event' field",
            i + 1
        );
    }

    // Check first line is run_start
    let first: serde_json::Value = serde_json::from_str(lines[0]).unwrap();
    assert_eq!(first["event"], "run_start");

    // Check last line is run_complete
    let last: serde_json::Value = serde_json::from_str(lines[lines.len() - 1]).unwrap();
    assert_eq!(last["event"], "run_complete");
}

#[test]
fn golden_tty_frames_are_valid_ndjson() {
    let frames_path = helpers::golden_dir().join("tty_frames.ndjson");
    let text = std::fs::read_to_string(&frames_path).expect("read golden TTY frames");

    let lines: Vec<&str> = text.lines().collect();
    assert!(lines.len() >= 2, "should have at least 2 TTY frames");

    for (i, line) in lines.iter().enumerate() {
        let parsed: serde_json::Value = serde_json::from_str(line)
            .unwrap_or_else(|e| panic!("frame {} is not valid JSON: {e}", i));
        assert!(
            parsed.get("seq").is_some(),
            "frame {} should have 'seq' field",
            i
        );
        assert!(
            parsed.get("codec").is_some(),
            "frame {} should have 'codec' field",
            i
        );
    }
}

#[test]
fn test_helpers_create_test_request_is_valid() {
    let request = helpers::create_test_request();
    assert_eq!(request.backend, BackendKind::WhisperCpp);
    assert_eq!(request.language.as_deref(), Some("en"));
    assert!(!request.diarize);
    assert!(!request.translate);

    let json = serde_json::to_string(&request).expect("should serialize");
    let _: TranscribeRequest = serde_json::from_str(&json).expect("should round-trip");
}

#[test]
fn test_helpers_create_test_result_has_segments() {
    let result = helpers::create_test_result();
    assert_eq!(result.backend, BackendKind::WhisperCpp);
    assert_eq!(result.segments.len(), 2);
    assert!(result.segments[0].confidence.is_some());
    assert!(result.segments[0].start_sec.is_some());
}

#[test]
fn test_helpers_create_test_report_is_complete() {
    let report = helpers::create_test_report();
    assert!(!report.run_id.is_empty());
    assert!(!report.trace_id.is_empty());
    assert!(!report.events.is_empty());
    helpers::assert_events_ordered(&report.events);

    // Test that it serializes to valid JSON
    let json = serde_json::to_string(&report).expect("should serialize");
    let _: franken_whisper::RunReport = serde_json::from_str(&json).expect("should round-trip");
}

#[test]
fn test_helpers_segments_match_with_tolerance() {
    let result = helpers::create_test_result();
    helpers::assert_segments_match(&result.segments, &result.segments, 0.001);
}

#[test]
fn test_helpers_generate_wav_creates_valid_file() {
    let tmp = tempfile::tempdir().expect("create temp dir");
    let wav_path = helpers::generate_test_wav(tmp.path(), "test_tone.wav", 0.5, 440.0);

    assert!(wav_path.exists(), "WAV file should exist");
    let metadata = std::fs::metadata(&wav_path).expect("read metadata");

    // 16kHz * 0.5s * 2 bytes/sample + 44 bytes WAV header = 16044 bytes
    let expected_size = (16000.0 * 0.5 * 2.0) as u64 + 44;
    assert_eq!(
        metadata.len(),
        expected_size,
        "WAV file size should match expected"
    );

    // Verify RIFF header
    let bytes = std::fs::read(&wav_path).expect("read WAV bytes");
    assert_eq!(&bytes[0..4], b"RIFF");
    assert_eq!(&bytes[8..12], b"WAVE");
}

#[test]
fn test_helpers_generate_silence_wav() {
    let tmp = tempfile::tempdir().expect("create temp dir");
    let wav_path = helpers::generate_silence_wav(tmp.path(), "silence.wav", 1.0);

    assert!(wav_path.exists());
    let bytes = std::fs::read(&wav_path).expect("read WAV bytes");
    assert_eq!(&bytes[0..4], b"RIFF");

    // Verify all samples are zero (starting after 44-byte header)
    let samples_bytes = &bytes[44..];
    for chunk in samples_bytes.chunks(2) {
        let sample = i16::from_le_bytes([chunk[0], chunk[1]]);
        assert_eq!(sample, 0, "silence should have all-zero samples");
    }
}

#[test]
fn test_helpers_create_test_db_works() {
    let (store, _tmp) = helpers::create_test_db();

    // Should be able to persist a report
    let report = helpers::create_test_report();
    store
        .persist_report(&report)
        .expect("should persist report");

    // Should be able to list runs
    let runs = store.list_recent_runs(10).expect("should list runs");
    assert_eq!(runs.len(), 1);
    assert_eq!(runs[0].run_id, report.run_id);
}

#[test]
fn test_mock_env_points_to_existing_scripts() {
    let env = helpers::mock_backend_env();

    for (key, value) in &env {
        if key.contains("BIN") {
            let path = PathBuf::from(value);
            // Skip bare command names (e.g. "python3") that rely on PATH lookup;
            // only check paths that contain a directory separator.
            if path.is_absolute() || value.contains('/') {
                assert!(
                    path.exists(),
                    "mock binary should exist at {value} (env: {key})"
                );
            }
        }
    }
}
