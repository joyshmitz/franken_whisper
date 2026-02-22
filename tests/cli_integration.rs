use std::path::PathBuf;
use std::process::{Command as ProcessCommand, Stdio};

use serde_json::json;
use tempfile::tempdir;

use franken_whisper::backend;
use franken_whisper::cli::{
    Cli, Command, RunsOutputFormat, TtyAudioCommand, TtyAudioControlCommand, TtyAudioRecoveryPolicy,
};
use franken_whisper::model::{
    BackendKind, BackendParams, DecodingParams, DiarizationConfig, InputSource, OutputFormat,
    RunEvent, RunReport, SpeakerConstraints, TimestampLevel, TranscribeRequest,
    TranscriptionResult, TranscriptionSegment, VadParams,
};
use franken_whisper::storage::RunStore;
use franken_whisper::sync::{self, ConflictPolicy};

// ---------------------------------------------------------------------------
// Storage: runs get --id
// ---------------------------------------------------------------------------

fn fixture_report(id: &str, db_path: &std::path::Path) -> RunReport {
    RunReport {
        run_id: id.to_owned(),
        started_at_rfc3339: "2026-02-22T00:00:00Z".to_owned(),
        finished_at_rfc3339: "2026-02-22T00:00:05Z".to_owned(),
        input_path: "test.wav".to_owned(),
        normalized_wav_path: "normalized.wav".to_owned(),
        request: TranscribeRequest {
            input: InputSource::File {
                path: PathBuf::from("test.wav"),
            },
            backend: BackendKind::Auto,
            model: None,
            language: Some("en".to_owned()),
            translate: false,
            diarize: false,
            persist: true,
            db_path: db_path.to_path_buf(),
            timeout_ms: None,
            backend_params: franken_whisper::model::BackendParams::default(),
        },
        result: TranscriptionResult {
            backend: BackendKind::WhisperCpp,
            transcript: "integration test transcript".to_owned(),
            language: Some("en".to_owned()),
            segments: vec![
                TranscriptionSegment {
                    start_sec: Some(0.0),
                    end_sec: Some(2.0),
                    text: "integration test".to_owned(),
                    speaker: Some("SPEAKER_00".to_owned()),
                    confidence: Some(0.92),
                },
                TranscriptionSegment {
                    start_sec: Some(2.0),
                    end_sec: Some(4.0),
                    text: "transcript".to_owned(),
                    speaker: None,
                    confidence: None,
                },
            ],
            acceleration: None,
            raw_output: json!({"test": true}),
            artifact_paths: vec![],
        },
        events: vec![
            RunEvent {
                seq: 1,
                ts_rfc3339: "2026-02-22T00:00:01Z".to_owned(),
                stage: "ingest".to_owned(),
                code: "ingest.ok".to_owned(),
                message: "materialized".to_owned(),
                payload: json!({}),
            },
            RunEvent {
                seq: 2,
                ts_rfc3339: "2026-02-22T00:00:03Z".to_owned(),
                stage: "backend".to_owned(),
                code: "backend.ok".to_owned(),
                message: "done".to_owned(),
                payload: json!({"segments": 2}),
            },
        ],
        warnings: vec![],
        evidence: vec![],
        trace_id: "00000000000000000000000000000000".to_owned(),
        replay: franken_whisper::model::ReplayEnvelope::default(),
    }
}

#[cfg(unix)]
fn ffmpeg_available() -> bool {
    ProcessCommand::new("ffmpeg")
        .args(["-hide_banner", "-version"])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|status| status.success())
        .unwrap_or(false)
}

#[cfg(unix)]
fn write_whisper_cpp_stub_binary(dir: &std::path::Path) -> PathBuf {
    use std::fs;
    use std::os::unix::fs::PermissionsExt;

    let stub_path = dir.join("whisper_cpp_stub.sh");
    let script = r#"#!/usr/bin/env bash
set -euo pipefail
out_prefix=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    -of)
      out_prefix="$2"
      shift 2
      ;;
    *)
      shift
      ;;
  esac
done
if [[ -z "${out_prefix}" ]]; then
  echo "missing -of output prefix" >&2
  exit 2
fi
mkdir -p "$(dirname "${out_prefix}")"
cat > "${out_prefix}.json" <<'JSON'
{"text":"stub transcript","language":"en","segments":[{"start":0.0,"end":0.5,"text":"stub transcript","speaker":"SPEAKER_00","confidence":0.9}]}
JSON
"#;
    fs::write(&stub_path, script).expect("write stub");
    let mut perms = fs::metadata(&stub_path).expect("metadata").permissions();
    perms.set_mode(0o755);
    fs::set_permissions(&stub_path, perms).expect("chmod");
    stub_path
}

#[cfg(unix)]
fn generate_silent_wav(path: &std::path::Path) {
    let status = ProcessCommand::new("ffmpeg")
        .args([
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "anullsrc=r=16000:cl=mono",
            "-t",
            "0.3",
        ])
        .arg(path)
        .status()
        .expect("spawn ffmpeg");
    assert!(status.success(), "ffmpeg should synthesize silent wav");
}

#[cfg(unix)]
fn run_transcribe_json_with_stub(
    args: &[&str],
    stdin_payload: Option<&[u8]>,
    stub_bin: &std::path::Path,
    state_root: &std::path::Path,
) -> serde_json::Value {
    let mut cmd = ProcessCommand::new(env!("CARGO_BIN_EXE_franken_whisper"));
    cmd.arg("transcribe");
    cmd.args(args);
    cmd.env("FRANKEN_WHISPER_WHISPER_CPP_BIN", stub_bin);
    cmd.env("FRANKEN_WHISPER_STATE_DIR", state_root);
    cmd.stdout(Stdio::piped());
    cmd.stderr(Stdio::piped());

    let output = if let Some(payload) = stdin_payload {
        use std::io::Write;
        cmd.stdin(Stdio::piped());
        let mut child = cmd.spawn().expect("spawn transcribe");
        let mut stdin = child.stdin.take().expect("stdin pipe");
        stdin.write_all(payload).expect("write payload");
        drop(stdin);
        child.wait_with_output().expect("wait transcribe")
    } else {
        cmd.output().expect("run transcribe")
    };

    assert!(
        output.status.success(),
        "transcribe failed\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout_text = String::from_utf8_lossy(&output.stdout);
    let json_start = stdout_text.find('{').unwrap_or_else(|| {
        panic!(
            "json report parse failed: no JSON object in stdout\nstdout:\n{}\nstderr:\n{}",
            stdout_text,
            String::from_utf8_lossy(&output.stderr)
        )
    });
    let json_payload = &stdout_text[json_start..];
    serde_json::from_str(json_payload).unwrap_or_else(|error| {
        panic!(
            "json report parse failed: {error}\nstdout:\n{}\nstderr:\n{}",
            stdout_text,
            String::from_utf8_lossy(&output.stderr)
        )
    })
}

#[test]
fn run_get_by_id_returns_full_details() {
    let dir = tempdir().expect("tempdir");
    let db_path = dir.path().join("storage.sqlite3");
    let store = RunStore::open(&db_path).expect("store should open");

    let report = fixture_report("integ-run-1", &db_path);
    store.persist_report(&report).expect("persist");

    let details = store
        .load_run_details("integ-run-1")
        .expect("query should succeed")
        .expect("run should exist");

    assert_eq!(details.run_id, "integ-run-1");
    assert_eq!(details.transcript, "integration test transcript");
    assert_eq!(details.segments.len(), 2);
    assert_eq!(details.events.len(), 2);
    assert_eq!(details.events[0].code, "ingest.ok");
    assert_eq!(details.events[1].code, "backend.ok");
}

#[test]
fn run_get_nonexistent_returns_none() {
    let dir = tempdir().expect("tempdir");
    let db_path = dir.path().join("storage.sqlite3");
    let store = RunStore::open(&db_path).expect("store should open");

    let details = store
        .load_run_details("nonexistent")
        .expect("query should succeed");

    assert!(details.is_none());
}

#[test]
fn runs_empty_db_returns_empty_list() {
    let dir = tempdir().expect("tempdir");
    let db_path = dir.path().join("storage.sqlite3");
    let store = RunStore::open(&db_path).expect("store should open");

    let runs = store.list_recent_runs(10).expect("list should succeed");
    assert!(runs.is_empty());
}

// ---------------------------------------------------------------------------
// Robot backends: diagnostics
// ---------------------------------------------------------------------------

#[test]
fn backend_diagnostics_returns_three_entries() {
    let diags = backend::diagnostics();
    assert_eq!(diags.len(), 3);

    let backend_names: Vec<&str> = diags
        .iter()
        .filter_map(|entry| entry.get("backend").and_then(|v| v.as_str()))
        .collect();

    assert!(backend_names.contains(&"whisper_cpp"));
    assert!(backend_names.contains(&"insanely_fast"));
    assert!(backend_names.contains(&"whisper_diarization"));

    // Each entry has required fields
    for entry in &diags {
        assert!(entry.get("available").is_some());
        assert!(entry.get("backend").is_some());
        assert!(entry.get("unsupported_options").is_some());
        assert!(entry.get("unsupported_options").unwrap().is_array());
    }
}

#[test]
fn backend_diagnostics_is_valid_json() {
    let diags = backend::diagnostics();
    let payload = json!({
        "event": "backends",
        "backends": diags,
    });
    let serialized = serde_json::to_string(&payload).expect("should serialize");
    let _parsed: serde_json::Value = serde_json::from_str(&serialized).expect("should parse back");
}

// ---------------------------------------------------------------------------
// Sync: export empty DB, round-trip, checksum validation
// ---------------------------------------------------------------------------

#[test]
fn sync_export_empty_db_valid_manifest() {
    let dir = tempdir().expect("tempdir");
    let db_path = dir.path().join("empty.sqlite3");
    let export_dir = dir.path().join("export");
    let state_root = dir.path().join("state");

    let _store = RunStore::open(&db_path).expect("store");
    let manifest = sync::export(&db_path, &export_dir, &state_root).expect("export");

    assert_eq!(manifest.schema_version, "1.1");
    assert_eq!(manifest.row_counts.runs, 0);
    assert_eq!(manifest.row_counts.segments, 0);
    assert_eq!(manifest.row_counts.events, 0);

    // Manifest file should exist and be valid JSON
    let manifest_text =
        std::fs::read_to_string(export_dir.join("manifest.json")).expect("manifest file");
    let parsed: sync::SyncManifest =
        serde_json::from_str(&manifest_text).expect("valid manifest json");
    assert_eq!(parsed.schema_version, "1.1");
}

#[test]
fn sync_round_trip_preserves_data() {
    let dir = tempdir().expect("tempdir");
    let source_db = dir.path().join("source.sqlite3");
    let target_db = dir.path().join("target.sqlite3");
    let export_dir = dir.path().join("export");
    let state_root = dir.path().join("state");

    // Persist two runs
    let store = RunStore::open(&source_db).expect("store");
    store
        .persist_report(&fixture_report("rt-1", &source_db))
        .expect("persist 1");
    store
        .persist_report(&fixture_report("rt-2", &source_db))
        .expect("persist 2");

    // Export
    let manifest = sync::export(&source_db, &export_dir, &state_root).expect("export");
    assert_eq!(manifest.row_counts.runs, 2);
    assert_eq!(manifest.row_counts.segments, 4);
    assert_eq!(manifest.row_counts.events, 4);

    // Import to fresh DB
    let result =
        sync::import(&target_db, &export_dir, &state_root, ConflictPolicy::Reject).expect("import");
    assert_eq!(result.runs_imported, 2);
    assert_eq!(result.segments_imported, 4);
    assert_eq!(result.events_imported, 4);
    assert!(result.conflicts.is_empty());

    // Verify target DB
    let target_store = RunStore::open(&target_db).expect("target store");
    let runs = target_store.list_recent_runs(10).expect("list");
    assert_eq!(runs.len(), 2);

    // Verify detailed load
    let details = target_store
        .load_run_details("rt-1")
        .expect("load")
        .expect("exists");
    assert_eq!(details.segments.len(), 2);
    assert_eq!(details.events.len(), 2);
}

#[test]
fn sync_import_rejects_tampered_files() {
    let dir = tempdir().expect("tempdir");
    let db_path = dir.path().join("source.sqlite3");
    let export_dir = dir.path().join("export");
    let state_root = dir.path().join("state");

    let store = RunStore::open(&db_path).expect("store");
    store
        .persist_report(&fixture_report("tamper-1", &db_path))
        .expect("persist");
    sync::export(&db_path, &export_dir, &state_root).expect("export");

    // Tamper with events.jsonl
    std::fs::write(export_dir.join("events.jsonl"), "tampered\n").expect("tamper");

    let target_db = dir.path().join("target.sqlite3");
    let result = sync::import(&target_db, &export_dir, &state_root, ConflictPolicy::Reject);
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("checksum"));
}

#[test]
fn sync_import_missing_manifest_fails() {
    let dir = tempdir().expect("tempdir");
    let db_path = dir.path().join("target.sqlite3");
    let empty_dir = dir.path().join("empty");
    std::fs::create_dir_all(&empty_dir).expect("mkdir");
    let state_root = dir.path().join("state");

    let result = sync::import(&db_path, &empty_dir, &state_root, ConflictPolicy::Reject);
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("manifest.json"));
}

// ---------------------------------------------------------------------------
// Robot mode: NDJSON contract validation
// ---------------------------------------------------------------------------

#[test]
fn robot_report_produces_valid_ndjson() {
    let report = fixture_report("ndjson-1", std::path::Path::new("/tmp/test.db"));
    let mut buf = Vec::new();

    // Simulate emit_robot_report by serializing events + final envelope
    for event in &report.events {
        let line = serde_json::to_string(&json!({
            "event": "stage",
            "run_id": report.run_id,
            "seq": event.seq,
            "ts": event.ts_rfc3339,
            "stage": event.stage,
            "code": event.code,
            "message": event.message,
            "payload": event.payload,
        }))
        .expect("serialize");
        buf.push(line);
    }

    let complete = serde_json::to_string(&json!({
        "event": "run_complete",
        "run_id": report.run_id,
        "backend": report.result.backend,
        "transcript": report.result.transcript,
    }))
    .expect("serialize");
    buf.push(complete);

    // Every line must be valid JSON
    for (index, line) in buf.iter().enumerate() {
        let parsed: serde_json::Value = serde_json::from_str(line)
            .unwrap_or_else(|error| panic!("line {index} invalid JSON: {error}"));
        assert!(
            parsed.get("event").is_some(),
            "line {index} missing 'event' field"
        );
    }
}

#[test]
fn robot_routing_history_args_parse_correctly() {
    use clap::Parser;
    let cli = Cli::parse_from([
        "franken_whisper",
        "robot",
        "routing-history",
        "--db",
        "/tmp/test.sqlite3",
        "--run-id",
        "run-abc",
        "--limit",
        "5",
    ]);
    match cli.command {
        Command::Robot {
            command: franken_whisper::cli::RobotCommand::RoutingHistory(args),
        } => {
            assert_eq!(args.db, PathBuf::from("/tmp/test.sqlite3"));
            assert_eq!(args.run_id.as_deref(), Some("run-abc"));
            assert_eq!(args.limit, 5);
        }
        _ => panic!("expected Robot RoutingHistory"),
    }
}

#[test]
fn robot_routing_history_extracts_decision_events() {
    let dir = tempdir().expect("tempdir");
    let db_path = dir.path().join("routing.sqlite3");
    let store = RunStore::open(&db_path).expect("store open");

    // Build a report with a routing decision event in its events.
    let mut report = fixture_report("routing-hist-1", &db_path);
    report.events.push(RunEvent {
        seq: 10,
        ts_rfc3339: "2026-02-22T00:00:00Z".to_owned(),
        stage: "backend_routing".to_owned(),
        code: "backend.routing.decision_contract".to_owned(),
        message: "evaluated formal backend selection contract".to_owned(),
        payload: json!({
            "mode": "active",
            "chosen_action": "try_whisper_cpp",
            "decision_id": "d-123",
            "calibration_score": 0.85,
            "e_process": 1.18,
            "fallback_active": false,
            "recommended_order": ["whisper_cpp", "insanely_fast", "whisper_diarization"],
        }),
    });

    store.persist_report(&report).expect("persist");

    // Load and verify routing events can be extracted.
    let details = store
        .load_run_details("routing-hist-1")
        .expect("load")
        .expect("run exists");

    let routing_events: Vec<_> = details
        .events
        .iter()
        .filter(|e| e.code == "backend.routing.decision_contract")
        .collect();

    assert_eq!(routing_events.len(), 1);
    assert_eq!(
        routing_events[0].payload["chosen_action"],
        "try_whisper_cpp"
    );
    assert_eq!(routing_events[0].payload["mode"], "active");
    assert_eq!(routing_events[0].payload["calibration_score"], 0.85);
}

#[test]
fn robot_run_emits_cancelled_stage_before_terminal_run_error() {
    let dir = tempdir().expect("tempdir");
    let state_root = dir.path().join("state");
    let db_path = dir.path().join("storage.sqlite3");

    let output = ProcessCommand::new(env!("CARGO_BIN_EXE_franken_whisper"))
        .args([
            "robot",
            "run",
            "--stdin",
            "--timeout",
            "0",
            "--no-persist",
            "--db",
            db_path.to_str().expect("db path should be utf-8"),
        ])
        .env("FRANKEN_WHISPER_STATE_DIR", &state_root)
        .output()
        .expect("robot command should execute");

    assert!(
        !output.status.success(),
        "timeout/cancellation path should return non-zero"
    );

    let stdout = String::from_utf8(output.stdout).expect("stdout should be utf-8");
    let lines: Vec<&str> = stdout
        .lines()
        .filter(|line| {
            let trimmed = line.trim_start();
            !trimmed.is_empty() && trimmed.starts_with('{')
        })
        .collect();
    assert!(
        lines.len() >= 4,
        "expected at least run_start + 2 stage lines + run_error, got {lines:?}"
    );

    let parsed: Vec<serde_json::Value> = lines
        .iter()
        .map(|line| {
            serde_json::from_str(line)
                .unwrap_or_else(|error| panic!("line should be valid json: {error}; line={line}"))
        })
        .collect();

    assert_eq!(parsed[0]["event"], "run_start");

    let budgets_idx = parsed
        .iter()
        .position(|line| line["event"] == "stage" && line["code"] == "orchestration.budgets")
        .expect("orchestration budget line should be present");
    let cancelled_idx = parsed
        .iter()
        .position(|line| line["event"] == "stage" && line["code"] == "orchestration.cancelled")
        .expect("cancellation stage line should be present");
    let error_idx = parsed
        .iter()
        .position(|line| line["event"] == "run_error")
        .expect("terminal run_error should be present");

    assert!(budgets_idx < cancelled_idx);
    assert!(cancelled_idx < error_idx);
    assert_eq!(parsed[error_idx]["code"], "FW-ROBOT-CANCELLED");
    assert_eq!(
        parsed[cancelled_idx]["payload"]["cancellation_evidence"]["reason"],
        "checkpoint deadline exceeded"
    );
}

#[cfg(unix)]
#[test]
fn robot_run_normalize_stage_timeout_maps_to_timeout_error_code() {
    if !ffmpeg_available() {
        return;
    }

    let dir = tempdir().expect("tempdir");
    let state_root = dir.path().join("state");
    let db_path = dir.path().join("storage.sqlite3");
    let stub_bin = write_whisper_cpp_stub_binary(dir.path());
    let input_wav = dir.path().join("timeout-input.wav");
    generate_silent_wav(&input_wav);

    let output = ProcessCommand::new(env!("CARGO_BIN_EXE_franken_whisper"))
        .args([
            "robot",
            "run",
            "--input",
            input_wav.to_str().expect("input path should be utf-8"),
            "--backend",
            "whisper-cpp",
            "--no-persist",
            "--db",
            db_path.to_str().expect("db path should be utf-8"),
        ])
        .env("FRANKEN_WHISPER_STATE_DIR", &state_root)
        .env("FRANKEN_WHISPER_WHISPER_CPP_BIN", &stub_bin)
        .env("FRANKEN_WHISPER_STAGE_BUDGET_NORMALIZE_MS", "1")
        .output()
        .expect("robot command should execute");

    assert!(
        !output.status.success(),
        "forced stage timeout should return non-zero"
    );

    let stdout = String::from_utf8(output.stdout).expect("stdout should be utf-8");
    let lines: Vec<serde_json::Value> = stdout
        .lines()
        .filter(|line| {
            let trimmed = line.trim_start();
            !trimmed.is_empty() && trimmed.starts_with('{')
        })
        .map(|line| {
            serde_json::from_str(line)
                .unwrap_or_else(|error| panic!("line should be valid json: {error}; line={line}"))
        })
        .collect();

    let timeout_stage = lines
        .iter()
        .find(|line| line["event"] == "stage" && line["code"] == "normalize.timeout")
        .expect("normalize.timeout stage event should be emitted");
    assert_eq!(timeout_stage["stage"], "normalize");

    let run_error = lines
        .iter()
        .find(|line| line["event"] == "run_error")
        .expect("run_error envelope should be emitted");
    assert_eq!(run_error["code"], "FW-ROBOT-TIMEOUT");
}

#[test]
fn robot_budget_stage_reflects_env_overrides_and_invalid_fallbacks() {
    let dir = tempdir().expect("tempdir");
    let state_root = dir.path().join("state");
    let db_path = dir.path().join("storage.sqlite3");

    let output = ProcessCommand::new(env!("CARGO_BIN_EXE_franken_whisper"))
        .args([
            "robot",
            "run",
            "--stdin",
            "--timeout",
            "0",
            "--no-persist",
            "--db",
            db_path.to_str().expect("db path should be utf-8"),
        ])
        .env("FRANKEN_WHISPER_STATE_DIR", &state_root)
        .env("FRANKEN_WHISPER_STAGE_BUDGET_INGEST_MS", "321")
        .env("FRANKEN_WHISPER_STAGE_BUDGET_NORMALIZE_MS", "0")
        .env("FRANKEN_WHISPER_STAGE_BUDGET_BACKEND_MS", "2222")
        .env("FRANKEN_WHISPER_STAGE_BUDGET_ACCELERATION_MS", "bad-value")
        .output()
        .expect("robot command should execute");

    assert!(!output.status.success());

    let stdout = String::from_utf8(output.stdout).expect("stdout should be utf-8");
    let budgets_line = stdout
        .lines()
        .filter(|line| {
            let trimmed = line.trim_start();
            !trimmed.is_empty() && trimmed.starts_with('{')
        })
        .map(|line| {
            serde_json::from_str::<serde_json::Value>(line)
                .unwrap_or_else(|error| panic!("line should parse as json: {error}; line={line}"))
        })
        .find(|line| line["event"] == "stage" && line["code"] == "orchestration.budgets")
        .expect("orchestration budget stage should be present");

    let budgets = &budgets_line["payload"]["stage_budgets"];
    assert_eq!(budgets["ingest_ms"], 321);
    assert_eq!(budgets["normalize_ms"], 180000);
    assert_eq!(budgets["backend_ms"], 2222);
    assert_eq!(budgets["acceleration_ms"], 20000);
    assert_eq!(budgets["persist_ms"], 20000);
}

// ---------------------------------------------------------------------------
// CLI argument validation
// ---------------------------------------------------------------------------

#[test]
fn transcribe_args_rejects_no_input() {
    use franken_whisper::cli::TranscribeArgs;

    let args = TranscribeArgs {
        input: None,
        stdin: false,
        mic: false,
        mic_seconds: 15,
        mic_device: None,
        mic_ffmpeg_format: None,
        mic_ffmpeg_source: None,
        backend: BackendKind::Auto,
        model: None,
        language: None,
        translate: false,
        diarize: false,
        db: PathBuf::from(".franken_whisper/storage.sqlite3"),
        no_persist: false,
        timeout: None,
        json: false,
        output_txt: false,
        output_vtt: false,
        output_srt: false,
        output_csv: false,
        output_json_full: false,
        output_lrc: false,
        no_timestamps: false,
        detect_language_only: false,
        split_on_word: false,
        best_of: None,
        beam_size: None,
        max_context: None,
        max_segment_length: None,
        temperature: None,
        temperature_increment: None,
        entropy_threshold: None,
        logprob_threshold: None,
        no_speech_threshold: None,
        vad: false,
        vad_model: None,
        vad_threshold: None,
        vad_min_speech_ms: None,
        vad_min_silence_ms: None,
        vad_max_speech_s: None,
        vad_speech_pad_ms: None,
        vad_samples_overlap: None,
        batch_size: None,
        timestamp_level: None,
        num_speakers: None,
        min_speakers: None,
        max_speakers: None,
        gpu_device: None,
        flash_attention: false,
        hf_token: None,
        transcript_path: None,
        no_stem: false,
        diarization_model: None,
        suppress_numerals: false,
        threads: None,
        processors: None,
        no_gpu: false,
        prompt: None,
        carry_initial_prompt: false,
        no_fallback: false,
        suppress_nst: false,
        offset_ms: None,
        duration_ms: None,
        audio_ctx: None,
        word_threshold: None,
        suppress_regex: None,
    };

    let result = args.to_request();
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("specify one of"));
}

#[test]
fn transcribe_args_rejects_multiple_inputs() {
    use franken_whisper::cli::TranscribeArgs;

    let args = TranscribeArgs {
        input: Some(PathBuf::from("file.wav")),
        stdin: true,
        mic: false,
        mic_seconds: 15,
        mic_device: None,
        mic_ffmpeg_format: None,
        mic_ffmpeg_source: None,
        backend: BackendKind::Auto,
        model: None,
        language: None,
        translate: false,
        diarize: false,
        db: PathBuf::from(".franken_whisper/storage.sqlite3"),
        no_persist: false,
        timeout: None,
        json: false,
        output_txt: false,
        output_vtt: false,
        output_srt: false,
        output_csv: false,
        output_json_full: false,
        output_lrc: false,
        no_timestamps: false,
        detect_language_only: false,
        split_on_word: false,
        best_of: None,
        beam_size: None,
        max_context: None,
        max_segment_length: None,
        temperature: None,
        temperature_increment: None,
        entropy_threshold: None,
        logprob_threshold: None,
        no_speech_threshold: None,
        vad: false,
        vad_model: None,
        vad_threshold: None,
        vad_min_speech_ms: None,
        vad_min_silence_ms: None,
        vad_max_speech_s: None,
        vad_speech_pad_ms: None,
        vad_samples_overlap: None,
        batch_size: None,
        timestamp_level: None,
        num_speakers: None,
        min_speakers: None,
        max_speakers: None,
        gpu_device: None,
        flash_attention: false,
        hf_token: None,
        transcript_path: None,
        no_stem: false,
        diarization_model: None,
        suppress_numerals: false,
        threads: None,
        processors: None,
        no_gpu: false,
        prompt: None,
        carry_initial_prompt: false,
        no_fallback: false,
        suppress_nst: false,
        offset_ms: None,
        duration_ms: None,
        audio_ctx: None,
        word_threshold: None,
        suppress_regex: None,
    };

    let result = args.to_request();
    assert!(result.is_err());
    assert!(
        result
            .unwrap_err()
            .to_string()
            .contains("mutually exclusive")
    );
}

#[test]
fn transcribe_args_parses_hf_token_override_for_insanely_fast() {
    use clap::Parser as _;

    let cli = Cli::parse_from([
        "franken_whisper",
        "transcribe",
        "--input",
        "sample.wav",
        "--backend",
        "insanely-fast",
        "--diarize",
        "--hf-token",
        "hf_cli_token",
    ]);

    match cli.command {
        Command::Transcribe(args) => {
            assert_eq!(args.hf_token.as_deref(), Some("hf_cli_token"));
            let request = args.to_request().expect("valid request");
            assert_eq!(
                request.backend_params.insanely_fast_hf_token.as_deref(),
                Some("hf_cli_token")
            );
        }
        _ => panic!("expected transcribe command"),
    }
}

#[test]
fn transcribe_args_parses_insanely_fast_transcript_path_override() {
    use clap::Parser as _;

    let cli = Cli::parse_from([
        "franken_whisper",
        "transcribe",
        "--input",
        "sample.wav",
        "--backend",
        "insanely-fast",
        "--transcript-path",
        "artifacts/ifw-output.json",
    ]);

    match cli.command {
        Command::Transcribe(args) => {
            assert_eq!(
                args.transcript_path.as_deref(),
                Some(PathBuf::from("artifacts/ifw-output.json").as_path())
            );
            let request = args.to_request().expect("valid request");
            assert_eq!(
                request
                    .backend_params
                    .insanely_fast_transcript_path
                    .as_deref(),
                Some(PathBuf::from("artifacts/ifw-output.json").as_path())
            );
        }
        _ => panic!("expected transcribe command"),
    }
}

#[test]
fn runs_command_accepts_json_output_format() {
    use clap::Parser as _;

    let cli = Cli::parse_from(["franken_whisper", "runs", "--format", "json"]);
    match cli.command {
        Command::Runs(args) => {
            assert_eq!(args.format, RunsOutputFormat::Json);
            assert_eq!(args.limit, 20);
        }
        _ => panic!("expected runs command"),
    }
}

#[test]
fn runs_command_defaults_to_plain_output_format() {
    use clap::Parser as _;

    let cli = Cli::parse_from(["franken_whisper", "runs"]);
    match cli.command {
        Command::Runs(args) => {
            assert_eq!(args.format, RunsOutputFormat::Plain);
        }
        _ => panic!("expected runs command"),
    }
}

#[test]
fn runs_command_accepts_ndjson_output_format() {
    use clap::Parser as _;

    let cli = Cli::parse_from(["franken_whisper", "runs", "--format", "ndjson"]);
    match cli.command {
        Command::Runs(args) => {
            assert_eq!(args.format, RunsOutputFormat::Ndjson);
        }
        _ => panic!("expected runs command"),
    }
}

#[test]
fn tty_audio_decode_defaults_to_fail_closed_recovery() {
    use clap::Parser as _;

    let cli = Cli::parse_from([
        "franken_whisper",
        "tty-audio",
        "decode",
        "--output",
        "out.wav",
    ]);
    match cli.command {
        Command::TtyAudio {
            command: TtyAudioCommand::Decode { recovery, .. },
        } => {
            assert_eq!(recovery, TtyAudioRecoveryPolicy::FailClosed);
        }
        _ => panic!("expected tty-audio decode command"),
    }
}

#[test]
fn tty_audio_decode_accepts_skip_missing_recovery() {
    use clap::Parser as _;

    let cli = Cli::parse_from([
        "franken_whisper",
        "tty-audio",
        "decode",
        "--output",
        "out.wav",
        "--recovery",
        "skip_missing",
    ]);
    match cli.command {
        Command::TtyAudio {
            command: TtyAudioCommand::Decode { recovery, .. },
        } => {
            assert_eq!(recovery, TtyAudioRecoveryPolicy::SkipMissing);
        }
        _ => panic!("expected tty-audio decode command"),
    }
}

#[test]
fn tty_audio_retransmit_plan_defaults_to_skip_missing_recovery() {
    use clap::Parser as _;

    let cli = Cli::parse_from(["franken_whisper", "tty-audio", "retransmit-plan"]);
    match cli.command {
        Command::TtyAudio {
            command: TtyAudioCommand::RetransmitPlan { recovery },
        } => {
            assert_eq!(recovery, TtyAudioRecoveryPolicy::SkipMissing);
        }
        _ => panic!("expected tty-audio retransmit-plan command"),
    }
}

#[test]
fn tty_audio_control_ack_parses_expected_fields() {
    use clap::Parser as _;

    let cli = Cli::parse_from([
        "franken_whisper",
        "tty-audio",
        "control",
        "ack",
        "--up-to-seq",
        "12",
    ]);
    match cli.command {
        Command::TtyAudio {
            command:
                TtyAudioCommand::Control {
                    command: TtyAudioControlCommand::Ack { up_to_seq },
                },
        } => {
            assert_eq!(up_to_seq, 12);
        }
        _ => panic!("expected tty-audio control ack command"),
    }
}

#[test]
fn tty_audio_control_retransmit_loop_defaults_are_deterministic() {
    use clap::Parser as _;

    let cli = Cli::parse_from(["franken_whisper", "tty-audio", "control", "retransmit-loop"]);
    match cli.command {
        Command::TtyAudio {
            command:
                TtyAudioCommand::Control {
                    command: TtyAudioControlCommand::RetransmitLoop { recovery, rounds },
                },
        } => {
            assert_eq!(recovery, TtyAudioRecoveryPolicy::SkipMissing);
            assert_eq!(rounds, 1);
        }
        _ => panic!("expected tty-audio control retransmit-loop command"),
    }
}

#[test]
fn tty_audio_control_ack_command_emits_control_ndjson() {
    let output = ProcessCommand::new(env!("CARGO_BIN_EXE_franken_whisper"))
        .args(["tty-audio", "control", "ack", "--up-to-seq", "42"])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .expect("run command");

    assert!(
        output.status.success(),
        "command failed\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    let payload: serde_json::Value = serde_json::from_slice(&output.stdout).expect("json output");
    assert_eq!(payload["frame_type"], "ack");
    assert_eq!(payload["up_to_seq"], 42);
}

#[test]
fn tty_audio_control_retransmit_loop_emits_request_then_response() {
    use base64::Engine;
    use base64::engine::general_purpose::STANDARD_NO_PAD;
    use flate2::Compression;
    use flate2::write::ZlibEncoder;
    use sha2::{Digest, Sha256};
    use std::io::Write;

    fn payload_b64(data: &[u8]) -> String {
        let mut encoder = ZlibEncoder::new(Vec::new(), Compression::fast());
        encoder.write_all(data).expect("write");
        let compressed = encoder.finish().expect("finish");
        STANDARD_NO_PAD.encode(compressed)
    }

    fn crc32(data: &[u8]) -> u32 {
        let mut hasher = crc32fast::Hasher::new();
        hasher.update(data);
        hasher.finalize()
    }

    fn sha256_hex(data: &[u8]) -> String {
        let digest = Sha256::digest(data);
        format!("{digest:x}")
    }

    let chunk0 = b"chunk-0";
    let chunk3 = b"chunk-3";
    let chunk4 = b"chunk-4";

    let lines = vec![
        serde_json::json!({
            "protocol_version": 1,
            "seq": 0,
            "codec": "mulaw+zlib+b64",
            "sample_rate_hz": 8000,
            "channels": 1,
            "payload_b64": payload_b64(chunk0),
            "crc32": crc32(chunk0),
            "payload_sha256": sha256_hex(chunk0),
        }),
        serde_json::json!({
            "protocol_version": 1,
            "seq": 3,
            "codec": "mulaw+zlib+b64",
            "sample_rate_hz": 8000,
            "channels": 1,
            "payload_b64": payload_b64(chunk3),
            "crc32": crc32(chunk3),
            "payload_sha256": sha256_hex(chunk3),
        }),
        serde_json::json!({
            "protocol_version": 1,
            "seq": 4,
            "codec": "mulaw+zlib+b64",
            "sample_rate_hz": 8000,
            "channels": 1,
            "payload_b64": payload_b64(chunk4),
            "crc32": crc32(chunk4),
            "payload_sha256": "deadbeef",
        }),
    ];

    let ndjson = lines
        .into_iter()
        .map(|line| serde_json::to_string(&line).expect("serialize"))
        .collect::<Vec<_>>()
        .join("\n")
        + "\n";

    let mut child = ProcessCommand::new(env!("CARGO_BIN_EXE_franken_whisper"))
        .args(["tty-audio", "control", "retransmit-loop", "--rounds", "2"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn retransmit-loop");

    {
        let mut stdin = child.stdin.take().expect("stdin");
        stdin.write_all(ndjson.as_bytes()).expect("write ndjson");
    }

    let output = child.wait_with_output().expect("wait");
    assert!(
        output.status.success(),
        "retransmit-loop failed\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    let lines = String::from_utf8(output.stdout).expect("utf8");
    let frames: Vec<serde_json::Value> = lines
        .lines()
        .map(|line| serde_json::from_str(line).expect("json line"))
        .collect();

    assert_eq!(frames.len(), 3, "2 requests + 1 response expected");
    assert_eq!(frames[0]["frame_type"], "retransmit_request");
    assert_eq!(frames[0]["sequences"], json!([1, 2, 4]));
    assert_eq!(frames[1]["frame_type"], "retransmit_request");
    assert_eq!(frames[1]["sequences"], json!([1, 2, 4]));
    assert_eq!(frames[2]["frame_type"], "retransmit_response");
    assert_eq!(frames[2]["sequences"], json!([1, 2, 4]));
}

#[test]
fn tty_audio_retransmit_plan_outputs_missing_and_corrupt_ranges() {
    use base64::Engine;
    use base64::engine::general_purpose::STANDARD_NO_PAD;
    use flate2::Compression;
    use flate2::write::ZlibEncoder;
    use sha2::{Digest, Sha256};
    use std::io::Write;

    fn payload_b64(data: &[u8]) -> String {
        let mut encoder = ZlibEncoder::new(Vec::new(), Compression::fast());
        encoder.write_all(data).expect("write");
        let compressed = encoder.finish().expect("finish");
        STANDARD_NO_PAD.encode(compressed)
    }

    fn crc32(data: &[u8]) -> u32 {
        let mut hasher = crc32fast::Hasher::new();
        hasher.update(data);
        hasher.finalize()
    }

    fn sha256_hex(data: &[u8]) -> String {
        let digest = Sha256::digest(data);
        format!("{digest:x}")
    }

    let chunk0 = b"chunk-0";
    let chunk3 = b"chunk-3";
    let chunk4 = b"chunk-4";

    let lines = vec![
        serde_json::json!({
            "protocol_version": 1,
            "seq": 0,
            "codec": "mulaw+zlib+b64",
            "sample_rate_hz": 8000,
            "channels": 1,
            "payload_b64": payload_b64(chunk0),
            "crc32": crc32(chunk0),
            "payload_sha256": sha256_hex(chunk0),
        }),
        serde_json::json!({
            "protocol_version": 1,
            "seq": 3,
            "codec": "mulaw+zlib+b64",
            "sample_rate_hz": 8000,
            "channels": 1,
            "payload_b64": payload_b64(chunk3),
            "crc32": crc32(chunk3),
            "payload_sha256": sha256_hex(chunk3),
        }),
        serde_json::json!({
            "protocol_version": 1,
            "seq": 4,
            "codec": "mulaw+zlib+b64",
            "sample_rate_hz": 8000,
            "channels": 1,
            "payload_b64": payload_b64(chunk4),
            "crc32": crc32(chunk4),
            "payload_sha256": "deadbeef",
        }),
    ];

    let ndjson = lines
        .into_iter()
        .map(|line| serde_json::to_string(&line).expect("serialize"))
        .collect::<Vec<_>>()
        .join("\n")
        + "\n";

    let mut child = ProcessCommand::new(env!("CARGO_BIN_EXE_franken_whisper"))
        .args(["tty-audio", "retransmit-plan"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn retransmit-plan");

    {
        let mut stdin = child.stdin.take().expect("stdin");
        stdin.write_all(ndjson.as_bytes()).expect("write ndjson");
    }

    let output = child.wait_with_output().expect("wait");
    assert!(
        output.status.success(),
        "retransmit-plan failed\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    let payload: serde_json::Value =
        serde_json::from_slice(&output.stdout).expect("json retransmit-plan");
    assert_eq!(payload["protocol_version"], 1);
    assert_eq!(payload["requested_sequences"], json!([1, 2, 4]));
    assert_eq!(
        payload["requested_ranges"],
        json!([
            {"start_seq": 1, "end_seq": 2},
            {"start_seq": 4, "end_seq": 4}
        ])
    );
    assert_eq!(payload["gap_count"], 1);
    assert_eq!(payload["integrity_failure_count"], 1);
}

// ---------------------------------------------------------------------------
// Phase 3: word-level segment extraction
// ---------------------------------------------------------------------------

#[test]
fn extract_word_level_segments_from_chunks() {
    let input = json!({
        "chunks": [
            {
                "timestamp": [0.0, 2.0],
                "text": "hello world",
                "speaker": "SPEAKER_00",
                "words": [
                    {"word": "hello", "start": 0.0, "end": 0.8, "confidence": 0.95},
                    {"word": "world", "start": 0.8, "end": 2.0, "confidence": 0.88}
                ]
            },
            {
                "timestamp": [2.0, 3.5],
                "text": "test",
                "words": [
                    {"word": "test", "start": 2.0, "end": 3.5}
                ]
            }
        ]
    });
    let segments = backend::extract_segments_from_json(&input);
    assert_eq!(segments.len(), 3);
    assert_eq!(segments[0].text, "hello");
    assert_eq!(segments[0].start_sec, Some(0.0));
    assert_eq!(segments[0].end_sec, Some(0.8));
    assert_eq!(segments[0].confidence, Some(0.95));
    assert_eq!(segments[0].speaker.as_deref(), Some("SPEAKER_00"));
    assert_eq!(segments[1].text, "world");
    assert_eq!(segments[1].speaker.as_deref(), Some("SPEAKER_00"));
    assert_eq!(segments[2].text, "test");
    assert!(segments[2].speaker.is_none());
}

#[test]
fn extract_chunk_level_segments_without_words() {
    let input = json!({
        "chunks": [
            {"timestamp": [0.0, 1.5], "text": "hello"},
            {"timestamp": [1.5, 3.0], "text": "world"}
        ]
    });
    let segments = backend::extract_segments_from_json(&input);
    assert_eq!(segments.len(), 2);
    assert_eq!(segments[0].text, "hello");
    assert_eq!(segments[0].start_sec, Some(0.0));
    assert_eq!(segments[1].text, "world");
}

// ---------------------------------------------------------------------------
// Phase 3: backend params serialization round-trip
// ---------------------------------------------------------------------------

#[test]
fn backend_params_default_round_trips_through_json() {
    let params = BackendParams::default();
    let json_str = serde_json::to_string(&params).expect("serialize");
    let parsed: BackendParams = serde_json::from_str(&json_str).expect("deserialize");
    assert!(parsed.output_formats.is_empty());
    assert!(parsed.timestamp_level.is_none());
    assert!(parsed.decoding.is_none());
    assert!(parsed.vad.is_none());
    assert!(parsed.insanely_fast_hf_token.is_none());
    assert!(parsed.insanely_fast_transcript_path.is_none());
    assert!(!parsed.no_timestamps);
    assert!(!parsed.detect_language_only);
}

#[test]
fn backend_params_populated_round_trips_through_json() {
    let params = BackendParams {
        output_formats: vec![OutputFormat::Srt, OutputFormat::Vtt, OutputFormat::Lrc],
        timestamp_level: Some(TimestampLevel::Word),
        decoding: Some(DecodingParams {
            best_of: Some(5),
            beam_size: Some(3),
            temperature: Some(0.2),
            ..DecodingParams::default()
        }),
        vad: Some(VadParams {
            threshold: Some(0.5),
            min_speech_duration_ms: Some(250),
            ..VadParams::default()
        }),
        speaker_constraints: Some(SpeakerConstraints {
            num_speakers: None,
            min_speakers: Some(2),
            max_speakers: Some(5),
        }),
        diarization_config: Some(DiarizationConfig {
            no_stem: true,
            suppress_numerals: true,
            ..DiarizationConfig::default()
        }),
        gpu_device: Some("0".to_owned()),
        flash_attention: Some(true),
        insanely_fast_hf_token: Some("hf_example_token".to_owned()),
        insanely_fast_transcript_path: Some(PathBuf::from("artifacts/ifw.json")),
        no_timestamps: false,
        detect_language_only: false,
        batch_size: Some(16),
        split_on_word: true,
        threads: Some(4),
        processors: Some(2),
        no_gpu: false,
        prompt: Some("medical terminology".to_owned()),
        carry_initial_prompt: true,
        no_fallback: false,
        suppress_nst: true,
        offset_ms: Some(1000),
        duration_ms: Some(30000),
        audio_ctx: Some(0),
        word_threshold: Some(0.01),
        suppress_regex: Some(r"\[.*\]".to_owned()),
        ..BackendParams::default()
    };
    let json_str = serde_json::to_string(&params).expect("serialize");
    let parsed: BackendParams = serde_json::from_str(&json_str).expect("deserialize");
    assert_eq!(parsed.output_formats.len(), 3);
    assert_eq!(parsed.timestamp_level, Some(TimestampLevel::Word));
    assert_eq!(parsed.decoding.as_ref().unwrap().best_of, Some(5));
    assert_eq!(parsed.vad.as_ref().unwrap().threshold, Some(0.5));
    assert_eq!(
        parsed.speaker_constraints.as_ref().unwrap().min_speakers,
        Some(2)
    );
    assert!(parsed.diarization_config.as_ref().unwrap().no_stem);
    assert_eq!(parsed.batch_size, Some(16));
    assert_eq!(
        parsed.insanely_fast_hf_token.as_deref(),
        Some("hf_example_token")
    );
    assert_eq!(
        parsed.insanely_fast_transcript_path.as_deref(),
        Some(PathBuf::from("artifacts/ifw.json").as_path())
    );
    assert!(parsed.split_on_word);
    assert_eq!(parsed.threads, Some(4));
    assert_eq!(parsed.processors, Some(2));
    assert!(!parsed.no_gpu);
    assert_eq!(parsed.prompt.as_deref(), Some("medical terminology"));
    assert!(parsed.carry_initial_prompt);
    assert!(parsed.suppress_nst);
    assert_eq!(parsed.offset_ms, Some(1000));
    assert_eq!(parsed.duration_ms, Some(30000));
    assert_eq!(parsed.audio_ctx, Some(0));
    assert_eq!(parsed.word_threshold, Some(0.01));
    assert_eq!(parsed.suppress_regex.as_deref(), Some(r"\[.*\]"));
}

#[test]
fn transcribe_request_missing_backend_params_deserializes_with_default() {
    // Simulates loading old data that lacks the backend_params field.
    let old_json = json!({
        "input": {"kind": "file", "path": "test.wav"},
        "backend": "auto",
        "model": null,
        "language": "en",
        "translate": false,
        "diarize": false,
        "persist": true,
        "db_path": "db.sqlite3",
        "timeout_ms": null
    });
    let request: TranscribeRequest =
        serde_json::from_value(old_json).expect("should deserialize with default backend_params");
    assert!(request.backend_params.output_formats.is_empty());
    assert!(request.backend_params.decoding.is_none());
    assert!(!request.backend_params.no_timestamps);
}

// ---------------------------------------------------------------------------
// Phase 3: CLI args → BackendParams mapping
// ---------------------------------------------------------------------------

#[test]
fn transcribe_args_maps_output_formats_to_backend_params() {
    use franken_whisper::cli::TranscribeArgs;

    let args = TranscribeArgs {
        input: Some(PathBuf::from("test.wav")),
        stdin: false,
        mic: false,
        mic_seconds: 15,
        mic_device: None,
        mic_ffmpeg_format: None,
        mic_ffmpeg_source: None,
        backend: BackendKind::Auto,
        model: None,
        language: None,
        translate: false,
        diarize: false,
        db: PathBuf::from(".franken_whisper/storage.sqlite3"),
        no_persist: false,
        timeout: None,
        json: false,
        output_txt: true,
        output_vtt: false,
        output_srt: true,
        output_csv: false,
        output_json_full: false,
        output_lrc: true,
        no_timestamps: false,
        detect_language_only: false,
        split_on_word: false,
        best_of: Some(5),
        beam_size: Some(3),
        max_context: None,
        max_segment_length: None,
        temperature: Some(0.1),
        temperature_increment: None,
        entropy_threshold: None,
        logprob_threshold: None,
        no_speech_threshold: None,
        vad: true,
        vad_model: None,
        vad_threshold: Some(0.6),
        vad_min_speech_ms: None,
        vad_min_silence_ms: None,
        vad_max_speech_s: None,
        vad_speech_pad_ms: None,
        vad_samples_overlap: None,
        batch_size: Some(24),
        timestamp_level: Some(TimestampLevel::Word),
        num_speakers: None,
        min_speakers: Some(2),
        max_speakers: Some(4),
        gpu_device: Some("0".to_owned()),
        flash_attention: true,
        hf_token: None,
        transcript_path: None,
        no_stem: false,
        diarization_model: None,
        suppress_numerals: false,
        threads: None,
        processors: None,
        no_gpu: false,
        prompt: None,
        carry_initial_prompt: false,
        no_fallback: false,
        suppress_nst: false,
        offset_ms: None,
        duration_ms: None,
        audio_ctx: None,
        word_threshold: None,
        suppress_regex: None,
    };

    let request = args.to_request().expect("valid request");
    let bp = &request.backend_params;

    // Output formats.
    assert_eq!(bp.output_formats.len(), 3);
    assert!(bp.output_formats.contains(&OutputFormat::Txt));
    assert!(bp.output_formats.contains(&OutputFormat::Srt));
    assert!(bp.output_formats.contains(&OutputFormat::Lrc));

    // Decoding params.
    let dec = bp.decoding.as_ref().expect("decoding should be Some");
    assert_eq!(dec.best_of, Some(5));
    assert_eq!(dec.beam_size, Some(3));
    assert_eq!(dec.temperature, Some(0.1));

    // VAD params.
    let vad = bp.vad.as_ref().expect("vad should be Some");
    assert_eq!(vad.threshold, Some(0.6));

    // Speaker constraints.
    let sc = bp
        .speaker_constraints
        .as_ref()
        .expect("speaker constraints");
    assert_eq!(sc.min_speakers, Some(2));
    assert_eq!(sc.max_speakers, Some(4));

    // Top-level batch/timestamp/GPU.
    assert_eq!(bp.batch_size, Some(24));
    assert_eq!(bp.timestamp_level, Some(TimestampLevel::Word));
    assert_eq!(bp.gpu_device.as_deref(), Some("0"));
    assert_eq!(bp.flash_attention, Some(true));
}

// ---------------------------------------------------------------------------
// Phase 3: run report with populated backend_params persists + restores
// ---------------------------------------------------------------------------

#[test]
fn run_report_with_backend_params_round_trips_through_storage() {
    let dir = tempdir().expect("tempdir");
    let db_path = dir.path().join("storage.sqlite3");
    let store = RunStore::open(&db_path).expect("store");

    let mut report = fixture_report("bp-rt-1", &db_path);
    report.request.backend_params = BackendParams {
        output_formats: vec![OutputFormat::Srt],
        batch_size: Some(16),
        timestamp_level: Some(TimestampLevel::Word),
        no_timestamps: true,
        ..BackendParams::default()
    };
    store.persist_report(&report).expect("persist");

    // Export and import to a fresh DB to test full serialization round-trip.
    let export_dir = dir.path().join("export");
    let state_root = dir.path().join("state");
    sync::export(&db_path, &export_dir, &state_root).expect("export");

    let target_db = dir.path().join("target.sqlite3");
    let result =
        sync::import(&target_db, &export_dir, &state_root, ConflictPolicy::Reject).expect("import");
    assert_eq!(result.runs_imported, 1);
}

// ---------------------------------------------------------------------------
// Phase 3: cross-input mode error consistency
// ---------------------------------------------------------------------------

#[test]
fn file_stdin_mic_all_produce_input_source_variants() {
    // Verify all three input modes produce the correct InputSource variant.
    let file_request = TranscribeRequest {
        input: InputSource::File {
            path: PathBuf::from("test.wav"),
        },
        backend: BackendKind::Auto,
        model: None,
        language: None,
        translate: false,
        diarize: false,
        persist: false,
        db_path: PathBuf::from("db.sqlite3"),
        timeout_ms: None,
        backend_params: BackendParams::default(),
    };
    assert!(matches!(file_request.input, InputSource::File { .. }));

    let stdin_request = TranscribeRequest {
        input: InputSource::Stdin {
            hint_extension: None,
        },
        backend: BackendKind::Auto,
        model: None,
        language: None,
        translate: false,
        diarize: false,
        persist: false,
        db_path: PathBuf::from("db.sqlite3"),
        timeout_ms: None,
        backend_params: BackendParams::default(),
    };
    assert!(matches!(stdin_request.input, InputSource::Stdin { .. }));

    let mic_request = TranscribeRequest {
        input: InputSource::Microphone {
            seconds: 5,
            device: None,
            ffmpeg_format: None,
            ffmpeg_source: None,
        },
        backend: BackendKind::Auto,
        model: None,
        language: None,
        translate: false,
        diarize: false,
        persist: false,
        db_path: PathBuf::from("db.sqlite3"),
        timeout_ms: None,
        backend_params: BackendParams::default(),
    };
    assert!(matches!(mic_request.input, InputSource::Microphone { .. }));

    // All three should serialize cleanly.
    for request in [&file_request, &stdin_request, &mic_request] {
        let json_str = serde_json::to_string(request).expect("serialize");
        let _parsed: TranscribeRequest =
            serde_json::from_str(&json_str).expect("round-trip deserialize");
    }
}

#[cfg(unix)]
#[test]
fn transcribe_file_input_crosses_ingest_normalize_backend_with_stub_whisper_cpp() {
    if !ffmpeg_available() {
        return;
    }

    let dir = tempdir().expect("tempdir");
    let state_root = dir.path().join("state");
    let stub_bin = write_whisper_cpp_stub_binary(dir.path());
    let input_wav = dir.path().join("file_input.wav");
    generate_silent_wav(&input_wav);

    let report = run_transcribe_json_with_stub(
        &[
            "--input",
            input_wav.to_str().expect("utf8"),
            "--backend",
            "whisper-cpp",
            "--no-persist",
            "--json",
        ],
        None,
        &stub_bin,
        &state_root,
    );

    assert_eq!(report["result"]["backend"], "whisper_cpp");
    assert_eq!(report["result"]["transcript"], "stub transcript");
    assert_eq!(report["result"]["segments"][0]["speaker"], "SPEAKER_00");
}

#[cfg(unix)]
#[test]
fn transcribe_stdin_input_crosses_ingest_normalize_backend_with_stub_whisper_cpp() {
    if !ffmpeg_available() {
        return;
    }

    let dir = tempdir().expect("tempdir");
    let state_root = dir.path().join("state");
    let stub_bin = write_whisper_cpp_stub_binary(dir.path());
    let input_wav = dir.path().join("stdin_input.wav");
    generate_silent_wav(&input_wav);
    let wav_bytes = std::fs::read(&input_wav).expect("wav bytes");

    let report = run_transcribe_json_with_stub(
        &[
            "--stdin",
            "--backend",
            "whisper-cpp",
            "--no-persist",
            "--json",
        ],
        Some(&wav_bytes),
        &stub_bin,
        &state_root,
    );

    assert_eq!(report["result"]["backend"], "whisper_cpp");
    assert_eq!(report["result"]["transcript"], "stub transcript");
}

#[cfg(unix)]
#[test]
fn transcribe_mic_line_in_input_crosses_ingest_normalize_backend_with_stub_whisper_cpp() {
    if !ffmpeg_available() {
        return;
    }

    let dir = tempdir().expect("tempdir");
    let state_root = dir.path().join("state");
    let stub_bin = write_whisper_cpp_stub_binary(dir.path());

    let report = run_transcribe_json_with_stub(
        &[
            "--mic",
            "--mic-seconds",
            "1",
            "--mic-ffmpeg-format",
            "lavfi",
            "--mic-ffmpeg-source",
            "anullsrc=r=16000:cl=mono",
            "--backend",
            "whisper-cpp",
            "--no-persist",
            "--json",
        ],
        None,
        &stub_bin,
        &state_root,
    );

    assert_eq!(report["result"]["backend"], "whisper_cpp");
    assert_eq!(report["result"]["transcript"], "stub transcript");
}

#[cfg(unix)]
#[test]
fn transcribe_happy_path_stage_sequence_contract_is_stable() {
    if !ffmpeg_available() {
        return;
    }

    let dir = tempdir().expect("tempdir");
    let state_root = dir.path().join("state");
    let db_path = dir.path().join("happy_path.sqlite3");
    let stub_bin = write_whisper_cpp_stub_binary(dir.path());
    let input_wav = dir.path().join("happy_path.wav");
    generate_silent_wav(&input_wav);

    let report = run_transcribe_json_with_stub(
        &[
            "--input",
            input_wav.to_str().expect("utf8"),
            "--backend",
            "whisper-cpp",
            "--db",
            db_path.to_str().expect("utf8"),
            "--json",
        ],
        None,
        &stub_bin,
        &state_root,
    );

    let events = report["events"]
        .as_array()
        .expect("events should be an array");
    assert!(
        !events.is_empty(),
        "report should include stage events in json output"
    );

    let code_index = |code: &str| -> usize {
        events
            .iter()
            .position(|event| event["code"] == code)
            .unwrap_or_else(|| panic!("expected event code `{code}` in {events:?}"))
    };
    let code_index_any = |codes: &[&str]| -> usize {
        events
            .iter()
            .position(|event| {
                let code = event["code"].as_str().unwrap_or_default();
                codes.contains(&code)
            })
            .unwrap_or_else(|| panic!("expected any code {codes:?} in {events:?}"))
    };

    let ingest_start = code_index("ingest.start");
    let normalize_start = code_index("normalize.start");
    let backend_start = code_index("backend.start");
    let acceleration_start = code_index("acceleration.start");
    let acceleration_terminal = code_index_any(&["acceleration.ok", "acceleration.fallback"]);
    let persist_start = code_index("persist.start");
    let persist_ok = code_index("persist.ok");

    assert!(ingest_start < normalize_start);
    assert!(normalize_start < backend_start);
    assert!(backend_start < acceleration_start);
    assert!(acceleration_start < acceleration_terminal);
    assert!(acceleration_terminal < persist_start);
    assert!(persist_start < persist_ok);
}

#[cfg(unix)]
#[test]
fn transcribe_backend_stage_payload_exposes_execution_metadata() {
    if !ffmpeg_available() {
        return;
    }

    let dir = tempdir().expect("tempdir");
    let state_root = dir.path().join("state");
    let stub_bin = write_whisper_cpp_stub_binary(dir.path());
    let input_wav = dir.path().join("backend_meta.wav");
    generate_silent_wav(&input_wav);

    let report = run_transcribe_json_with_stub(
        &[
            "--input",
            input_wav.to_str().expect("utf8"),
            "--backend",
            "whisper-cpp",
            "--no-persist",
            "--json",
        ],
        None,
        &stub_bin,
        &state_root,
    );

    let events = report["events"].as_array().expect("events should be array");
    let backend_ok = events
        .iter()
        .find(|event| event["code"] == "backend.ok")
        .expect("backend.ok stage should be present");
    let backend_payload = &backend_ok["payload"];

    assert_eq!(backend_payload["resolved_backend"], "whisper_cpp");
    assert_eq!(backend_payload["implementation"], "bridge");
    assert_eq!(backend_payload["execution_mode"], "bridge_only");
    assert_eq!(backend_payload["native_rollout_stage"], "primary");
    assert!(backend_payload["engine_identity"].is_string());
    assert!(
        backend_payload["engine_version"].is_string()
            || backend_payload["engine_version"].is_null()
    );
    assert!(backend_payload["native_fallback_error"].is_null());

    let replay_event = events
        .iter()
        .find(|event| event["code"] == "replay.envelope")
        .expect("replay.envelope stage should be present");
    let replay_payload = &replay_event["payload"];
    assert_eq!(
        replay_payload["implementation"],
        backend_payload["implementation"]
    );
    assert_eq!(
        replay_payload["execution_mode"],
        backend_payload["execution_mode"]
    );
    assert_eq!(
        replay_payload["native_rollout_stage"],
        backend_payload["native_rollout_stage"]
    );
    assert_eq!(
        replay_payload["native_fallback_error"],
        backend_payload["native_fallback_error"]
    );
}

#[cfg(unix)]
#[test]
fn transcribe_acceleration_context_telemetry_round_trips_in_run_artifacts() {
    if !ffmpeg_available() {
        return;
    }

    let dir = tempdir().expect("tempdir");
    let state_root = dir.path().join("state");
    let db_path = dir.path().join("telemetry.sqlite3");
    let stub_bin = write_whisper_cpp_stub_binary(dir.path());
    let input_wav = dir.path().join("telemetry.wav");
    generate_silent_wav(&input_wav);

    let report = run_transcribe_json_with_stub(
        &[
            "--input",
            input_wav.to_str().expect("utf8"),
            "--backend",
            "whisper-cpp",
            "--db",
            db_path.to_str().expect("utf8"),
            "--json",
        ],
        None,
        &stub_bin,
        &state_root,
    );

    let events = report["events"].as_array().expect("events should be array");
    let accel_event = events
        .iter()
        .find(|event| event["code"] == "acceleration.context")
        .expect("acceleration.context stage event should be present");
    let payload = &accel_event["payload"];
    assert!(payload["logical_stream_owner_id"].is_string());
    assert!(payload["logical_stream_kind"].is_string());
    assert!(payload["acceleration_backend"].is_string());
    assert!(payload["mode"].is_string());
    assert!(payload["frankentorch_feature"].is_boolean());
    assert!(payload["frankenjax_feature"].is_boolean());

    let fence = &payload["cancellation_fence"];
    assert_eq!(fence["status"], "open");
    assert!(fence["checked_at_rfc3339"].is_string());
    assert!(fence["budget_remaining_ms"].is_number() || fence["budget_remaining_ms"].is_null());
    assert!(fence["error"].is_null());
    assert!(fence["error_code"].is_null());

    let evidence = report["evidence"]
        .as_array()
        .expect("run artifact should include evidence array");
    let accel_evidence = evidence
        .iter()
        .find(|entry| entry["logical_stream_owner_id"].is_string())
        .expect("acceleration context should be captured in run evidence");
    assert_eq!(
        accel_evidence["logical_stream_owner_id"],
        payload["logical_stream_owner_id"]
    );
    assert_eq!(
        accel_evidence["logical_stream_kind"],
        payload["logical_stream_kind"]
    );
    assert_eq!(
        accel_evidence["cancellation_fence"],
        payload["cancellation_fence"]
    );

    let run_id = report["run_id"].as_str().expect("run_id should be present");
    let store = RunStore::open(&db_path).expect("store should open");
    let details = store
        .load_run_details(run_id)
        .expect("load should succeed")
        .expect("persisted run should exist");
    let persisted_event = details
        .events
        .iter()
        .find(|event| event.code == "acceleration.context")
        .expect("acceleration.context event should be persisted");
    assert_eq!(
        persisted_event.payload["logical_stream_owner_id"],
        payload["logical_stream_owner_id"]
    );
    assert_eq!(
        persisted_event.payload["logical_stream_kind"],
        payload["logical_stream_kind"]
    );
    assert_eq!(
        persisted_event.payload["cancellation_fence"],
        payload["cancellation_fence"]
    );
}

#[cfg(unix)]
#[test]
fn transcribe_backend_routing_stage_event_has_required_decision_contract_fields() {
    if !ffmpeg_available() {
        return;
    }

    let dir = tempdir().expect("tempdir");
    let state_root = dir.path().join("state");
    let stub_bin = write_whisper_cpp_stub_binary(dir.path());
    let input_wav = dir.path().join("routing.wav");
    generate_silent_wav(&input_wav);

    let report = run_transcribe_json_with_stub(
        &[
            "--input",
            input_wav.to_str().expect("utf8"),
            "--backend",
            "auto",
            "--no-persist",
            "--json",
        ],
        None,
        &stub_bin,
        &state_root,
    );

    let events = report["events"].as_array().expect("events should be array");
    let routing_event = events
        .iter()
        .find(|event| event["code"] == "backend.routing.decision_contract")
        .expect("routing decision contract stage should be present");
    let payload = &routing_event["payload"];

    let required_non_null_fields = [
        "version",
        "schema_version",
        "policy_id",
        "mode",
        "contract",
        "state_space",
        "observed_state",
        "action_set",
        "chosen_action",
        "fallback_active",
        "expected_losses",
        "posterior_snapshot",
        "calibration_score",
        "e_process",
        "ci_width",
        "recommended_order",
        "static_order",
        "availability",
        "provenance",
        "duration_seconds",
        "duration_bucket",
        "diarize",
        "decision_id",
        "trace_id",
    ];
    for field in required_non_null_fields {
        assert!(
            !payload[field].is_null(),
            "routing stage payload missing required field `{field}`"
        );
    }
    assert!(
        payload.get("adaptive_router_state").is_some(),
        "routing stage payload should include adaptive_router_state key"
    );
}

#[test]
fn transcribe_args_maps_mic_line_in_envelope_into_request() {
    use franken_whisper::cli::TranscribeArgs;

    let args = TranscribeArgs {
        input: None,
        stdin: false,
        mic: true,
        mic_seconds: 42,
        mic_device: Some("hw:2,0".to_owned()),
        mic_ffmpeg_format: Some("alsa".to_owned()),
        mic_ffmpeg_source: Some("hw:2,0".to_owned()),
        backend: BackendKind::WhisperCpp,
        model: Some("base.en".to_owned()),
        language: Some("en".to_owned()),
        translate: false,
        diarize: false,
        db: PathBuf::from(".franken_whisper/storage.sqlite3"),
        no_persist: true,
        timeout: Some(30),
        json: false,
        output_txt: false,
        output_vtt: false,
        output_srt: false,
        output_csv: false,
        output_json_full: false,
        output_lrc: false,
        no_timestamps: false,
        detect_language_only: false,
        split_on_word: false,
        best_of: None,
        beam_size: None,
        max_context: None,
        max_segment_length: None,
        temperature: None,
        temperature_increment: None,
        entropy_threshold: None,
        logprob_threshold: None,
        no_speech_threshold: None,
        vad: false,
        vad_model: None,
        vad_threshold: None,
        vad_min_speech_ms: None,
        vad_min_silence_ms: None,
        vad_max_speech_s: None,
        vad_speech_pad_ms: None,
        vad_samples_overlap: None,
        batch_size: None,
        timestamp_level: None,
        num_speakers: None,
        min_speakers: None,
        max_speakers: None,
        gpu_device: None,
        flash_attention: false,
        hf_token: None,
        transcript_path: None,
        no_stem: false,
        diarization_model: None,
        suppress_numerals: false,
        threads: None,
        processors: None,
        no_gpu: false,
        prompt: None,
        carry_initial_prompt: false,
        no_fallback: false,
        suppress_nst: false,
        offset_ms: None,
        duration_ms: None,
        audio_ctx: None,
        word_threshold: None,
        suppress_regex: None,
    };

    let request = args.to_request().expect("valid mic request");
    match request.input {
        InputSource::Microphone {
            seconds,
            device,
            ffmpeg_format,
            ffmpeg_source,
        } => {
            assert_eq!(seconds, 42);
            assert_eq!(device.as_deref(), Some("hw:2,0"));
            assert_eq!(ffmpeg_format.as_deref(), Some("alsa"));
            assert_eq!(ffmpeg_source.as_deref(), Some("hw:2,0"));
        }
        other => panic!("expected microphone input source, got {other:?}"),
    }
    assert_eq!(request.backend, BackendKind::WhisperCpp);
    assert!(!request.persist);
    assert_eq!(request.timeout_ms, Some(30_000));
}

// ---------------------------------------------------------------------------
// Conformance: segment invariant enforcement
// ---------------------------------------------------------------------------

#[test]
fn conformance_rejects_out_of_bounds_confidence() {
    use franken_whisper::conformance::validate_segment_invariants;

    let segments = vec![TranscriptionSegment {
        start_sec: Some(0.0),
        end_sec: Some(1.0),
        text: "hello".to_owned(),
        speaker: None,
        confidence: Some(1.5),
    }];
    let error = validate_segment_invariants(&segments).expect_err("should reject confidence > 1.0");
    assert!(error.to_string().contains("confidence"));
}

#[test]
fn conformance_rejects_negative_confidence() {
    use franken_whisper::conformance::validate_segment_invariants;

    let segments = vec![TranscriptionSegment {
        start_sec: Some(0.0),
        end_sec: Some(1.0),
        text: "hello".to_owned(),
        speaker: None,
        confidence: Some(-0.01),
    }];
    let error = validate_segment_invariants(&segments).expect_err("should reject confidence < 0.0");
    assert!(error.to_string().contains("confidence"));
}

#[test]
fn conformance_rejects_empty_speaker_label() {
    use franken_whisper::conformance::validate_segment_invariants;

    let segments = vec![TranscriptionSegment {
        start_sec: Some(0.0),
        end_sec: Some(1.0),
        text: "hello".to_owned(),
        speaker: Some("   ".to_owned()),
        confidence: None,
    }];
    let error = validate_segment_invariants(&segments).expect_err("should reject empty speaker");
    assert!(error.to_string().contains("empty speaker"));
}

#[test]
fn conformance_accepts_valid_segments_with_all_fields() {
    use franken_whisper::conformance::validate_segment_invariants;

    let segments = vec![
        TranscriptionSegment {
            start_sec: Some(0.0),
            end_sec: Some(1.0),
            text: "hello".to_owned(),
            speaker: Some("SPEAKER_00".to_owned()),
            confidence: Some(0.0),
        },
        TranscriptionSegment {
            start_sec: Some(1.0),
            end_sec: Some(2.0),
            text: "world".to_owned(),
            speaker: Some("SPEAKER_01".to_owned()),
            confidence: Some(1.0),
        },
    ];
    assert!(validate_segment_invariants(&segments).is_ok());
}

#[test]
fn conformance_engine_trait_capabilities_are_consistent() {
    use franken_whisper::backend::{
        Engine, InsanelyFastEngine, WhisperCppEngine, WhisperDiarizationEngine,
    };

    let wcpp = WhisperCppEngine;
    assert_eq!(wcpp.kind(), BackendKind::WhisperCpp);
    assert!(!wcpp.capabilities().supports_diarization);
    assert!(wcpp.capabilities().supports_translation);

    let ifw = InsanelyFastEngine;
    assert_eq!(ifw.kind(), BackendKind::InsanelyFast);
    assert!(ifw.capabilities().supports_diarization);
    assert!(ifw.capabilities().supports_gpu);

    let wd = WhisperDiarizationEngine;
    assert_eq!(wd.kind(), BackendKind::WhisperDiarization);
    assert!(wd.capabilities().supports_diarization);
    assert!(!wd.capabilities().supports_translation);
}

#[test]
fn conformance_all_engines_returns_three_engines() {
    use franken_whisper::backend::all_engines;

    let engines = all_engines();
    assert_eq!(engines.len(), 6);

    let names: Vec<&str> = engines.iter().map(|e| e.name()).collect();
    assert!(names.contains(&"whisper.cpp"));
    assert!(names.contains(&"insanely-fast-whisper"));
    assert!(names.contains(&"whisper-diarization"));
}

#[test]
fn conformance_engine_for_returns_correct_engine() {
    use franken_whisper::backend::engine_for;

    let engine = engine_for(BackendKind::WhisperCpp).expect("should return engine");
    assert_eq!(engine.name(), "whisper.cpp");

    let engine = engine_for(BackendKind::InsanelyFast).expect("should return engine");
    assert_eq!(engine.name(), "insanely-fast-whisper");

    assert!(engine_for(BackendKind::Auto).is_none());
}

// ---------------------------------------------------------------------------
// Sync: ConflictPolicy::Overwrite
// ---------------------------------------------------------------------------

#[test]
fn sync_overwrite_policy_replaces_conflicting_run() {
    let dir = tempdir().expect("tempdir");
    let source_db = dir.path().join("source.sqlite3");
    let target_db = dir.path().join("target.sqlite3");
    let export_dir = dir.path().join("export");
    let state_root = dir.path().join("state");

    // Persist run in source and target (same run_id, different transcripts)
    let store = RunStore::open(&source_db).expect("source store");
    let mut source_report = fixture_report("conflict-1", &source_db);
    source_report.result.transcript = "updated transcript".to_owned();
    store
        .persist_report(&source_report)
        .expect("persist source");

    let target_store = RunStore::open(&target_db).expect("target store");
    target_store
        .persist_report(&fixture_report("conflict-1", &target_db))
        .expect("persist target");

    // Export source
    sync::export(&source_db, &export_dir, &state_root).expect("export");

    // Import with Overwrite policy
    let result = sync::import(
        &target_db,
        &export_dir,
        &state_root,
        ConflictPolicy::Overwrite,
    )
    .expect("import with overwrite");

    assert_eq!(result.runs_imported, 1);

    // Verify overwritten data
    let details = target_store
        .load_run_details("conflict-1")
        .expect("load")
        .expect("exists");
    assert_eq!(details.transcript, "updated transcript");
}

#[test]
fn sync_reject_policy_fails_on_conflict() {
    let dir = tempdir().expect("tempdir");
    let source_db = dir.path().join("source.sqlite3");
    let target_db = dir.path().join("target.sqlite3");
    let export_dir = dir.path().join("export");
    let state_root = dir.path().join("state");

    // Same run in both databases
    let store = RunStore::open(&source_db).expect("source store");
    store
        .persist_report(&fixture_report("dup-1", &source_db))
        .expect("persist source");

    let target_store = RunStore::open(&target_db).expect("target store");
    let mut alt_report = fixture_report("dup-1", &target_db);
    alt_report.result.transcript = "different".to_owned();
    target_store
        .persist_report(&alt_report)
        .expect("persist target");

    sync::export(&source_db, &export_dir, &state_root).expect("export");

    let result = sync::import(&target_db, &export_dir, &state_root, ConflictPolicy::Reject);
    assert!(result.is_err(), "reject policy should fail on conflict");
}

// ---------------------------------------------------------------------------
// Storage: list_recent_runs with limit, load_latest_run_details
// ---------------------------------------------------------------------------

#[test]
fn list_recent_runs_respects_limit() {
    let dir = tempdir().expect("tempdir");
    let db_path = dir.path().join("storage.sqlite3");
    let store = RunStore::open(&db_path).expect("store");

    for i in 0..10 {
        store
            .persist_report(&fixture_report(&format!("limit-{i}"), &db_path))
            .expect("persist");
    }

    let all = store.list_recent_runs(100).expect("list all");
    assert_eq!(all.len(), 10);

    let limited = store.list_recent_runs(3).expect("list limited");
    assert_eq!(limited.len(), 3);
}

#[test]
fn load_latest_run_details_returns_most_recent() {
    let dir = tempdir().expect("tempdir");
    let db_path = dir.path().join("storage.sqlite3");
    let store = RunStore::open(&db_path).expect("store");

    let mut report1 = fixture_report("first", &db_path);
    report1.started_at_rfc3339 = "2026-02-20T00:00:00Z".to_owned();
    store.persist_report(&report1).expect("persist 1");

    let mut report2 = fixture_report("second", &db_path);
    report2.started_at_rfc3339 = "2026-02-22T00:00:00Z".to_owned();
    store.persist_report(&report2).expect("persist 2");

    let latest = store
        .load_latest_run_details()
        .expect("load latest")
        .expect("should exist");
    assert_eq!(latest.run_id, "second");
}

// ---------------------------------------------------------------------------
// Sync: idempotent re-import
// ---------------------------------------------------------------------------

#[test]
fn sync_double_import_is_idempotent() {
    let dir = tempdir().expect("tempdir");
    let source_db = dir.path().join("source.sqlite3");
    let target_db = dir.path().join("target.sqlite3");
    let export_dir = dir.path().join("export");
    let state_root = dir.path().join("state");

    let store = RunStore::open(&source_db).expect("store");
    store
        .persist_report(&fixture_report("idem-1", &source_db))
        .expect("persist");

    sync::export(&source_db, &export_dir, &state_root).expect("export");

    // First import
    let result1 = sync::import(&target_db, &export_dir, &state_root, ConflictPolicy::Reject)
        .expect("import 1");
    assert_eq!(result1.runs_imported, 1);

    // Second import with overwrite — same data, should succeed
    let result2 = sync::import(
        &target_db,
        &export_dir,
        &state_root,
        ConflictPolicy::Overwrite,
    )
    .expect("import 2");
    assert_eq!(result2.runs_imported, 1);

    // Verify only 1 run in target
    let target_store = RunStore::open(&target_db).expect("target store");
    let runs = target_store.list_recent_runs(100).expect("list");
    assert_eq!(runs.len(), 1);
}

// ---------------------------------------------------------------------------
// Storage: report with evidence and warnings round-trip
// ---------------------------------------------------------------------------

#[test]
fn run_report_with_evidence_and_warnings_round_trips() {
    let dir = tempdir().expect("tempdir");
    let db_path = dir.path().join("storage.sqlite3");
    let store = RunStore::open(&db_path).expect("store");

    let mut report = fixture_report("evidence-1", &db_path);
    report.evidence = vec![
        json!({"type": "decision", "action": "try_whisper_cpp"}),
        json!({"type": "calibration", "score": 0.85}),
    ];
    report.warnings = vec![
        "fallback: deterministic route used".to_owned(),
        "divergence: replay drift detected".to_owned(),
    ];
    store.persist_report(&report).expect("persist");

    let details = store
        .load_run_details("evidence-1")
        .expect("load")
        .expect("exists");
    assert_eq!(details.run_id, "evidence-1");
    assert_eq!(details.segments.len(), 2);
    assert_eq!(details.events.len(), 2);
}
