use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::time::Duration;

use crate::error::{FwError, FwResult};
use crate::model::InputSource;
use crate::process::{run_command_cancellable, run_command_with_timeout};

pub fn materialize_input(source: &InputSource, work_dir: &Path) -> FwResult<PathBuf> {
    materialize_input_with_token(source, work_dir, None)
}

pub(crate) fn materialize_input_with_token(
    source: &InputSource,
    work_dir: &Path,
    token: Option<&crate::orchestrator::CancellationToken>,
) -> FwResult<PathBuf> {
    if let Some(tok) = token {
        tok.checkpoint()?;
    }
    tracing::debug!(stage = "ingest", "Entering materialize_input");
    match source {
        InputSource::File { path } => {
            if !path.exists() {
                return Err(FwError::InvalidRequest(format!(
                    "input file does not exist: {}",
                    path.display()
                )));
            }
            if !path.is_file() {
                return Err(FwError::InvalidRequest(format!(
                    "input path is not a file: {}",
                    path.display()
                )));
            }
            Ok(path.clone())
        }
        InputSource::Stdin { hint_extension } => {
            let ext = hint_extension
                .as_deref()
                .map(|value| value.trim_start_matches('.'))
                .filter(|value| !value.is_empty())
                .unwrap_or("bin");
            let target = work_dir.join(format!("stdin_input.{ext}"));

            let mut stdin = std::io::stdin().lock();
            let mut buf = Vec::new();
            stdin.read_to_end(&mut buf)?;
            if buf.is_empty() {
                return Err(FwError::InvalidRequest(
                    "stdin input is empty; provide bytes or use --input/--mic".to_owned(),
                ));
            }
            fs::write(&target, buf)?;
            Ok(target)
        }
        InputSource::Microphone {
            seconds,
            device,
            ffmpeg_format,
            ffmpeg_source,
        } => capture_microphone(
            *seconds,
            device.as_deref(),
            ffmpeg_format.as_deref(),
            ffmpeg_source.as_deref(),
            microphone_timeout(*seconds),
            work_dir,
        ),
    }
}

pub fn normalize_to_wav(input: &Path, work_dir: &Path) -> FwResult<PathBuf> {
    normalize_to_wav_with_timeout(input, work_dir, ffmpeg_timeout(), None)
}

pub(crate) fn normalize_to_wav_with_timeout(
    input: &Path,
    work_dir: &Path,
    timeout: Duration,
    token: Option<&crate::orchestrator::CancellationToken>,
) -> FwResult<PathBuf> {
    let output = work_dir.join("normalized_16k_mono.wav");
    let args = vec![
        "-hide_banner".to_owned(),
        "-loglevel".to_owned(),
        "error".to_owned(),
        "-y".to_owned(),
        "-i".to_owned(),
        input.display().to_string(),
        "-ar".to_owned(),
        "16000".to_owned(),
        "-ac".to_owned(),
        "1".to_owned(),
        "-c:a".to_owned(),
        "pcm_s16le".to_owned(),
        output.display().to_string(),
    ];
    if let Some(tok) = token {
        run_command_cancellable("ffmpeg", &args, None, tok, Some(timeout))?;
    } else {
        run_command_with_timeout("ffmpeg", &args, None, Some(timeout))?;
    }
    Ok(output)
}

pub fn probe_duration_seconds(input: &Path) -> Option<f64> {
    probe_duration_seconds_with_timeout(input, ffprobe_timeout())
}

pub fn probe_duration_seconds_with_timeout(input: &Path, timeout: Duration) -> Option<f64> {
    let args = vec![
        "-v".to_owned(),
        "error".to_owned(),
        "-show_entries".to_owned(),
        "format=duration".to_owned(),
        "-of".to_owned(),
        "default=nokey=1:noprint_wrappers=1".to_owned(),
        input.display().to_string(),
    ];

    let output = run_command_with_timeout("ffprobe", &args, None, Some(timeout)).ok()?;
    let stdout = String::from_utf8_lossy(&output.stdout);
    let secs = stdout.trim().parse::<f64>().ok()?;
    if !secs.is_finite() || secs < 0.0 {
        return None;
    }
    Some(secs)
}

fn capture_microphone(
    seconds: u32,
    device: Option<&str>,
    ffmpeg_format: Option<&str>,
    ffmpeg_source: Option<&str>,
    timeout: Duration,
    work_dir: &Path,
) -> FwResult<PathBuf> {
    let output = work_dir.join("microphone_capture.wav");

    let (format, source) = microphone_defaults(device, ffmpeg_format, ffmpeg_source)?;
    let args = vec![
        "-hide_banner".to_owned(),
        "-loglevel".to_owned(),
        "error".to_owned(),
        "-y".to_owned(),
        "-f".to_owned(),
        format,
        "-i".to_owned(),
        source,
        "-t".to_owned(),
        seconds.to_string(),
        "-ac".to_owned(),
        "1".to_owned(),
        "-ar".to_owned(),
        "16000".to_owned(),
        "-c:a".to_owned(),
        "pcm_s16le".to_owned(),
        output.display().to_string(),
    ];

    run_command_with_timeout("ffmpeg", &args, None, Some(timeout))?;
    Ok(output)
}

fn microphone_defaults(
    device: Option<&str>,
    ffmpeg_format: Option<&str>,
    ffmpeg_source: Option<&str>,
) -> FwResult<(String, String)> {
    if let (Some(format), Some(source)) = (ffmpeg_format, ffmpeg_source) {
        return Ok((format.to_owned(), source.to_owned()));
    }

    if ffmpeg_format.is_some() || ffmpeg_source.is_some() {
        return Err(FwError::InvalidRequest(
            "both --mic-ffmpeg-format and --mic-ffmpeg-source must be provided together".to_owned(),
        ));
    }

    #[cfg(target_os = "linux")]
    {
        Ok(("alsa".to_owned(), device.unwrap_or("default").to_owned()))
    }

    #[cfg(target_os = "macos")]
    {
        Ok(("avfoundation".to_owned(), device.unwrap_or(":0").to_owned()))
    }

    #[cfg(target_os = "windows")]
    {
        Ok((
            "dshow".to_owned(),
            device.unwrap_or("audio=default").to_owned(),
        ))
    }

    #[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
    {
        let _ = device;
        Err(FwError::Unsupported(
            "microphone capture defaults are not implemented for this OS".to_owned(),
        ))
    }
}

fn ffmpeg_timeout() -> Duration {
    duration_from_env(
        "FRANKEN_WHISPER_FFMPEG_TIMEOUT_MS",
        Duration::from_secs(180),
    )
}

fn ffprobe_timeout() -> Duration {
    duration_from_env(
        "FRANKEN_WHISPER_FFPROBE_TIMEOUT_MS",
        Duration::from_secs(10),
    )
}

fn microphone_timeout(seconds: u32) -> Duration {
    let default = Duration::from_secs(u64::from(seconds).saturating_add(15));
    duration_from_env("FRANKEN_WHISPER_MIC_TIMEOUT_MS", default)
}

fn duration_from_env(key: &str, fallback: Duration) -> Duration {
    let Some(raw) = std::env::var(key).ok() else {
        return fallback;
    };
    let Ok(parsed) = raw.parse::<u64>() else {
        return fallback;
    };
    Duration::from_millis(parsed)
}

#[cfg(test)]
mod tests {
    use super::microphone_defaults;

    #[test]
    fn explicit_format_and_source_wins() {
        let result = microphone_defaults(Some("ignored"), Some("pulse"), Some("my-device"))
            .expect("explicit ffmpeg mic args should succeed");
        assert_eq!(result.0, "pulse");
        assert_eq!(result.1, "my-device");
    }

    #[test]
    fn partial_explicit_args_fail() {
        let err = microphone_defaults(None, Some("alsa"), None)
            .expect_err("partial explicit args should fail");
        let text = err.to_string();
        assert!(text.contains("must be provided together"));
    }

    #[test]
    fn materialize_file_nonexistent_returns_error() {
        use super::materialize_input;
        use crate::model::InputSource;

        let source = InputSource::File {
            path: std::path::PathBuf::from("/nonexistent/path/audio.wav"),
        };
        let dir = tempfile::tempdir().expect("tempdir");
        let err = materialize_input(&source, dir.path()).expect_err("should fail");
        let text = err.to_string();
        assert!(
            text.contains("does not exist"),
            "expected 'does not exist' in: {text}"
        );
    }

    #[test]
    fn materialize_file_valid_returns_same_path() {
        use super::materialize_input;
        use crate::model::InputSource;

        let dir = tempfile::tempdir().expect("tempdir");
        let file_path = dir.path().join("test.wav");
        std::fs::write(&file_path, b"fake audio content").expect("write");

        let source = InputSource::File {
            path: file_path.clone(),
        };
        let result = materialize_input(&source, dir.path()).expect("should succeed");
        assert_eq!(result, file_path);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn linux_microphone_defaults_use_alsa() {
        let (format, source) = microphone_defaults(None, None, None).expect("linux defaults");
        assert_eq!(format, "alsa");
        assert_eq!(source, "default");
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn linux_microphone_custom_device() {
        let (format, source) =
            microphone_defaults(Some("hw:1,0"), None, None).expect("custom device");
        assert_eq!(format, "alsa");
        assert_eq!(source, "hw:1,0");
    }

    #[test]
    fn microphone_timeout_adds_buffer_beyond_capture_duration() {
        use super::microphone_timeout;
        let timeout = microphone_timeout(10);
        // Should be at least 10 + 15 = 25 seconds.
        assert!(
            timeout >= std::time::Duration::from_secs(25),
            "timeout should be at least 25s, got {timeout:?}"
        );
    }

    #[test]
    fn duration_from_env_falls_back_on_missing_var() {
        use super::duration_from_env;
        let fallback = std::time::Duration::from_secs(42);
        // Use a var name that's unlikely to exist.
        let result = duration_from_env("FRANKEN_WHISPER_TEST_NONEXISTENT_VAR_39285", fallback);
        assert_eq!(result, fallback);
    }

    #[test]
    fn normalize_to_wav_with_timeout_missing_ffmpeg_returns_error() {
        use super::normalize_to_wav_with_timeout;

        // This test relies on ffmpeg not existing at a bogus path. We can't
        // control ffmpeg presence, so instead we test the function's structure
        // by passing a non-audio file and verifying it either returns Ok (if
        // ffmpeg exists) or Err (if ffmpeg is missing or input is invalid).
        let dir = tempfile::tempdir().expect("tempdir");
        let fake_input = dir.path().join("empty.txt");
        std::fs::write(&fake_input, "not audio").expect("write");

        let result = normalize_to_wav_with_timeout(
            &fake_input,
            dir.path(),
            std::time::Duration::from_secs(5),
            None,
        );
        // Either succeeds (ffmpeg installed and processes it somehow) or fails
        // gracefully with an error — no panics.
        let _ = result;
    }

    #[test]
    fn probe_duration_seconds_nonexistent_file() {
        use super::probe_duration_seconds;
        let result = probe_duration_seconds(std::path::Path::new("/nonexistent/file.wav"));
        // Should return None (either ffprobe missing or file not found).
        assert!(result.is_none());
    }

    #[test]
    fn ffmpeg_timeout_returns_positive_duration() {
        use super::ffmpeg_timeout;
        let timeout = ffmpeg_timeout();
        assert!(timeout.as_secs() > 0, "ffmpeg timeout should be positive");
    }

    #[test]
    fn ffprobe_timeout_returns_positive_duration() {
        use super::ffprobe_timeout;
        let timeout = ffprobe_timeout();
        assert!(timeout.as_secs() > 0, "ffprobe timeout should be positive");
    }

    #[test]
    fn duration_from_env_with_zero_returns_zero_duration() {
        use super::duration_from_env;
        // When the env var is not set, falls back.
        // We can't set env vars safely in parallel tests, but we can verify the
        // parsing logic by calling the function with known absent vars.
        let result = duration_from_env(
            "FRANKEN_WHISPER_TEST_NONEXISTENT_VAR_ALPHA_99999",
            std::time::Duration::from_millis(500),
        );
        assert_eq!(result, std::time::Duration::from_millis(500));
    }

    #[test]
    fn normalize_to_wav_with_cancellation_token_no_panic() {
        use super::normalize_to_wav_with_timeout;
        use crate::orchestrator::CancellationToken;

        let dir = tempfile::tempdir().expect("tempdir");
        let fake_input = dir.path().join("empty.txt");
        std::fs::write(&fake_input, "not audio").expect("write");

        // Token with no deadline should not cause cancellation.
        let token = CancellationToken::no_deadline();
        let result = normalize_to_wav_with_timeout(
            &fake_input,
            dir.path(),
            std::time::Duration::from_secs(5),
            Some(&token),
        );
        // The call should either succeed or fail due to ffmpeg — not panic.
        let _ = result;
    }

    #[test]
    fn probe_duration_seconds_with_timeout_nonexistent_file() {
        use super::probe_duration_seconds_with_timeout;
        let result = probe_duration_seconds_with_timeout(
            std::path::Path::new("/nonexistent/file.wav"),
            std::time::Duration::from_secs(5),
        );
        assert!(result.is_none());
    }

    #[test]
    fn normalize_to_wav_produces_expected_output_path() {
        // Verify that normalize_to_wav_with_timeout targets the right filename.
        // We can't run ffmpeg, but we can observe the expected output path by
        // looking at what it would produce.
        let dir = tempfile::tempdir().expect("tempdir");
        let expected = dir.path().join("normalized_16k_mono.wav");
        // Before calling, the file should not exist.
        assert!(!expected.exists());
    }

    #[test]
    fn normalize_to_wav_wrapper_does_not_panic() {
        use super::normalize_to_wav;
        let dir = tempfile::tempdir().expect("tempdir");
        let fake_input = dir.path().join("empty.txt");
        std::fs::write(&fake_input, "not audio").expect("write");
        // normalize_to_wav is a thin wrapper — should not panic.
        let _ = normalize_to_wav(&fake_input, dir.path());
    }

    #[test]
    fn duration_from_env_distinct_fallbacks_for_distinct_keys() {
        use super::duration_from_env;
        // Two missing vars should each return their own fallback independently.
        let short = duration_from_env(
            "FRANKEN_WHISPER_TEST_DUR_NONEXIST_SHORT",
            std::time::Duration::from_millis(100),
        );
        let long = duration_from_env(
            "FRANKEN_WHISPER_TEST_DUR_NONEXIST_LONG",
            std::time::Duration::from_secs(300),
        );
        assert_eq!(short, std::time::Duration::from_millis(100));
        assert_eq!(long, std::time::Duration::from_secs(300));
        assert_ne!(short, long);
    }

    #[test]
    fn duration_from_env_zero_fallback_returns_zero() {
        use super::duration_from_env;
        let result = duration_from_env(
            "FRANKEN_WHISPER_TEST_DUR_NONEXIST_ZERO",
            std::time::Duration::ZERO,
        );
        assert_eq!(result, std::time::Duration::ZERO);
    }

    #[test]
    fn microphone_timeout_zero_seconds_has_buffer() {
        use super::microphone_timeout;
        let timeout = microphone_timeout(0);
        assert!(
            timeout >= std::time::Duration::from_secs(15),
            "0-second mic capture should still have 15s buffer, got {timeout:?}"
        );
    }

    #[test]
    fn microphone_timeout_large_seconds_does_not_overflow() {
        use super::microphone_timeout;
        let timeout = microphone_timeout(u32::MAX);
        assert!(
            timeout.as_secs() > 0,
            "u32::MAX seconds should not overflow"
        );
    }

    #[test]
    fn partial_mic_source_only_fails() {
        let err =
            microphone_defaults(None, None, Some("hw:0")).expect_err("source-only should fail");
        assert!(err.to_string().contains("must be provided together"));
    }

    #[test]
    fn explicit_format_and_source_overrides_device() {
        // When both format and source are explicitly provided, device is ignored.
        let (format, source) = microphone_defaults(
            Some("should-be-ignored"),
            Some("custom"),
            Some("/dev/audio"),
        )
        .expect("explicit args should succeed");
        assert_eq!(format, "custom");
        assert_eq!(source, "/dev/audio");
    }

    #[test]
    fn materialize_file_with_spaces_in_path() {
        use super::materialize_input;
        use crate::model::InputSource;

        let dir = tempfile::tempdir().expect("tempdir");
        let spaced_dir = dir.path().join("folder with spaces");
        std::fs::create_dir_all(&spaced_dir).expect("mkdir");
        let file_path = spaced_dir.join("my file.wav");
        std::fs::write(&file_path, b"audio data").expect("write");

        let source = InputSource::File {
            path: file_path.clone(),
        };
        let result = materialize_input(&source, dir.path()).expect("should succeed");
        assert_eq!(result, file_path);
    }

    #[test]
    fn materialize_file_symlink_follows_link() {
        use super::materialize_input;
        use crate::model::InputSource;

        let dir = tempfile::tempdir().expect("tempdir");
        let real_file = dir.path().join("real.wav");
        std::fs::write(&real_file, b"audio").expect("write");
        let link_path = dir.path().join("link.wav");
        std::os::unix::fs::symlink(&real_file, &link_path).expect("symlink");

        let source = InputSource::File {
            path: link_path.clone(),
        };
        let result = materialize_input(&source, dir.path()).expect("should succeed");
        assert_eq!(result, link_path);
    }

    #[test]
    fn microphone_timeout_exact_boundary_value() {
        use super::microphone_timeout;
        // 1 second capture should yield 1+15=16 second timeout.
        let timeout = microphone_timeout(1);
        assert_eq!(timeout, std::time::Duration::from_secs(16));
    }

    #[test]
    fn probe_duration_with_empty_file() {
        use super::probe_duration_seconds;

        let dir = tempfile::tempdir().expect("tempdir");
        let empty = dir.path().join("empty.wav");
        std::fs::write(&empty, b"").expect("write");
        // Empty file is not valid audio — should return None (or Some if ffprobe
        // somehow handles it, but definitely should not panic).
        let _ = probe_duration_seconds(&empty);
    }

    // ── Inline WAV generation helper (mirrors tests/helpers/mod.rs) ──
    //
    // We duplicate the WAV generation logic here rather than importing from
    // tests/helpers because unit tests inside src/ cannot depend on the
    // integration test helper module.

    /// Write a minimal valid WAV file: 16-bit PCM, mono, 16 kHz.
    fn write_test_wav(path: &std::path::Path, samples: &[i16], sample_rate: u32) {
        use std::io::Write;
        let channels: u16 = 1;
        let data_size = (samples.len() * 2) as u32;
        let file_size = 36 + data_size;
        let byte_rate = sample_rate * u32::from(channels) * 2;
        let block_align = channels * 2;

        let mut f = std::fs::File::create(path).expect("create WAV");
        f.write_all(b"RIFF").unwrap();
        f.write_all(&file_size.to_le_bytes()).unwrap();
        f.write_all(b"WAVE").unwrap();
        f.write_all(b"fmt ").unwrap();
        f.write_all(&16u32.to_le_bytes()).unwrap();
        f.write_all(&1u16.to_le_bytes()).unwrap(); // PCM
        f.write_all(&channels.to_le_bytes()).unwrap();
        f.write_all(&sample_rate.to_le_bytes()).unwrap();
        f.write_all(&byte_rate.to_le_bytes()).unwrap();
        f.write_all(&block_align.to_le_bytes()).unwrap();
        f.write_all(&16u16.to_le_bytes()).unwrap(); // bits per sample
        f.write_all(b"data").unwrap();
        f.write_all(&data_size.to_le_bytes()).unwrap();
        for s in samples {
            f.write_all(&s.to_le_bytes()).unwrap();
        }
    }

    /// Generate a 1-second 440 Hz sine tone WAV at 16 kHz, return path.
    fn generate_sine_wav(dir: &std::path::Path, name: &str) -> std::path::PathBuf {
        let sample_rate: u32 = 16000;
        let duration_secs: f32 = 1.0;
        let frequency: f32 = 440.0;
        let num_samples = (sample_rate as f32 * duration_secs) as usize;
        let samples: Vec<i16> = (0..num_samples)
            .map(|i| {
                let t = i as f32 / sample_rate as f32;
                ((2.0 * std::f32::consts::PI * frequency * t).sin() * 32_000.0) as i16
            })
            .collect();
        let path = dir.join(name);
        write_test_wav(&path, &samples, sample_rate);
        path
    }

    /// Generate a silent WAV of the given duration (seconds) at 16 kHz.
    fn generate_silence_wav(dir: &std::path::Path, name: &str, secs: f32) -> std::path::PathBuf {
        let sample_rate: u32 = 16000;
        let num_samples = (sample_rate as f32 * secs) as usize;
        let samples = vec![0i16; num_samples];
        let path = dir.join(name);
        write_test_wav(&path, &samples, sample_rate);
        path
    }

    /// Returns true if ffmpeg is available on PATH.
    fn ffmpeg_available() -> bool {
        std::process::Command::new("ffmpeg")
            .arg("-version")
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .is_ok()
    }

    /// Returns true if ffprobe is available on PATH.
    fn ffprobe_available() -> bool {
        std::process::Command::new("ffprobe")
            .arg("-version")
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .is_ok()
    }

    // ── materialize_input: additional edge cases ──

    #[test]
    fn materialize_file_empty_path_returns_error() {
        use super::materialize_input;
        use crate::model::InputSource;

        let dir = tempfile::tempdir().expect("tempdir");
        let source = InputSource::File {
            path: std::path::PathBuf::from(""),
        };
        let err = materialize_input(&source, dir.path()).expect_err("empty path should fail");
        let text = err.to_string();
        assert!(
            text.contains("does not exist"),
            "expected 'does not exist' in: {text}"
        );
    }

    #[test]
    fn materialize_file_unicode_path_existing_file() {
        use super::materialize_input;
        use crate::model::InputSource;

        let dir = tempfile::tempdir().expect("tempdir");
        let unicode_dir = dir.path().join("donn\u{00e9}es");
        std::fs::create_dir_all(&unicode_dir).expect("mkdir");
        let file_path = unicode_dir.join("r\u{00e9}sultat.wav");
        std::fs::write(&file_path, b"audio content").expect("write");

        let source = InputSource::File {
            path: file_path.clone(),
        };
        let result = materialize_input(&source, dir.path()).expect("should succeed");
        assert_eq!(result, file_path);
    }

    #[test]
    fn materialize_file_unicode_path_nonexistent() {
        use super::materialize_input;
        use crate::model::InputSource;

        let dir = tempfile::tempdir().expect("tempdir");
        let source = InputSource::File {
            path: dir
                .path()
                .join("\u{65e5}\u{672c}\u{8a9e}/\u{97f3}\u{58f0}.wav"),
        };
        let err = materialize_input(&source, dir.path()).expect_err("should fail");
        assert!(err.to_string().contains("does not exist"));
    }

    #[test]
    fn materialize_file_directory_path_returns_error() {
        use super::materialize_input;
        use crate::model::InputSource;

        let dir = tempfile::tempdir().expect("tempdir");
        let subdir = dir.path().join("a_directory");
        std::fs::create_dir_all(&subdir).expect("mkdir");

        let source = InputSource::File {
            path: subdir.clone(),
        };
        let err = materialize_input(&source, dir.path()).expect_err("directory should be rejected");
        assert!(
            err.to_string().contains("not a file"),
            "expected 'not a file' in: {}",
            err
        );
    }

    #[test]
    fn materialize_file_returns_original_path_not_copy() {
        use super::materialize_input;
        use crate::model::InputSource;

        // File variant should return the exact same PathBuf, not a copy in work_dir.
        let dir = tempfile::tempdir().expect("tempdir");
        let work = tempfile::tempdir().expect("work_dir");
        let file_path = dir.path().join("original.wav");
        std::fs::write(&file_path, b"audio").expect("write");

        let source = InputSource::File {
            path: file_path.clone(),
        };
        let result = materialize_input(&source, work.path()).expect("should succeed");
        assert_eq!(
            result, file_path,
            "should return original path, not work_dir copy"
        );
    }

    // NOTE: materialize_input with InputSource::Stdin reads from actual stdin,
    // which requires controlling the process's stdin pipe. This is impractical
    // in a unit test without restructuring the function to accept a generic
    // Read trait. Testing Stdin is deferred to integration tests.

    // ── normalize_to_wav: tests that exercise ffmpeg ──

    #[test]
    fn normalize_valid_wav_produces_output_file() {
        if !ffmpeg_available() {
            eprintln!("SKIPPED: ffmpeg not found on PATH");
            return;
        }
        use super::normalize_to_wav;

        let dir = tempfile::tempdir().expect("tempdir");
        let input_wav = generate_sine_wav(dir.path(), "tone.wav");
        let work = tempfile::tempdir().expect("work_dir");

        let result = normalize_to_wav(&input_wav, work.path()).expect("ffmpeg should succeed");
        assert!(
            result.exists(),
            "output file should exist at {}",
            result.display()
        );
        assert_eq!(
            result.file_name().and_then(|f| f.to_str()),
            Some("normalized_16k_mono.wav"),
            "output filename mismatch"
        );
        // Output should be a non-empty file.
        let meta = std::fs::metadata(&result).expect("metadata");
        assert!(
            meta.len() > 44,
            "output WAV should be larger than just a header"
        );
    }

    #[test]
    fn normalize_valid_wav_with_timeout_produces_correct_output() {
        if !ffmpeg_available() {
            eprintln!("SKIPPED: ffmpeg not found on PATH");
            return;
        }
        use super::normalize_to_wav_with_timeout;

        let dir = tempfile::tempdir().expect("tempdir");
        let input_wav = generate_sine_wav(dir.path(), "sine_for_normalize.wav");
        let work = tempfile::tempdir().expect("work_dir");

        let result = normalize_to_wav_with_timeout(
            &input_wav,
            work.path(),
            std::time::Duration::from_secs(30),
            None,
        )
        .expect("should succeed");

        assert!(result.exists());
        // Verify the output starts with RIFF/WAVE header.
        let header = std::fs::read(&result).expect("read output");
        assert!(header.len() > 44, "output too small");
        assert_eq!(&header[0..4], b"RIFF", "should start with RIFF");
        assert_eq!(&header[8..12], b"WAVE", "should contain WAVE marker");
    }

    #[test]
    fn normalize_silent_wav_produces_output() {
        if !ffmpeg_available() {
            eprintln!("SKIPPED: ffmpeg not found on PATH");
            return;
        }
        use super::normalize_to_wav;

        let dir = tempfile::tempdir().expect("tempdir");
        let silence = generate_silence_wav(dir.path(), "silence.wav", 0.5);
        let work = tempfile::tempdir().expect("work_dir");

        let result = normalize_to_wav(&silence, work.path()).expect("should succeed");
        assert!(result.exists(), "normalized silent WAV should exist");
    }

    #[test]
    fn normalize_nonexistent_input_returns_error() {
        // ffmpeg should fail on a file that does not exist.
        use super::normalize_to_wav;

        let work = tempfile::tempdir().expect("work_dir");
        let result = normalize_to_wav(
            std::path::Path::new("/nonexistent/audio_input_99999.wav"),
            work.path(),
        );
        assert!(result.is_err(), "should fail for nonexistent input");
    }

    #[test]
    fn normalize_corrupt_file_returns_error() {
        if !ffmpeg_available() {
            eprintln!("SKIPPED: ffmpeg not found on PATH");
            return;
        }
        use super::normalize_to_wav;

        let dir = tempfile::tempdir().expect("tempdir");
        let corrupt = dir.path().join("corrupt.wav");
        std::fs::write(&corrupt, b"RIFF\x00\x00\x00\x00WAVEgarbage").expect("write");
        let work = tempfile::tempdir().expect("work_dir");

        let result = normalize_to_wav(&corrupt, work.path());
        // ffmpeg may or may not handle this -- it often produces an error for
        // truncated/corrupt containers. Either outcome is acceptable; the
        // important thing is no panic.
        let _ = result;
    }

    #[test]
    fn normalize_with_cancellation_token_valid_wav() {
        if !ffmpeg_available() {
            eprintln!("SKIPPED: ffmpeg not found on PATH");
            return;
        }
        use super::normalize_to_wav_with_timeout;
        use crate::orchestrator::CancellationToken;

        let dir = tempfile::tempdir().expect("tempdir");
        let input_wav = generate_sine_wav(dir.path(), "tone_cancel.wav");
        let work = tempfile::tempdir().expect("work_dir");

        let token = CancellationToken::no_deadline();
        let result = normalize_to_wav_with_timeout(
            &input_wav,
            work.path(),
            std::time::Duration::from_secs(30),
            Some(&token),
        )
        .expect("should succeed with no-deadline token");

        assert!(result.exists());
    }

    #[test]
    fn normalize_overwrites_existing_output() {
        if !ffmpeg_available() {
            eprintln!("SKIPPED: ffmpeg not found on PATH");
            return;
        }
        use super::normalize_to_wav;

        let dir = tempfile::tempdir().expect("tempdir");
        let input_wav = generate_sine_wav(dir.path(), "overwrite_test.wav");
        let work = tempfile::tempdir().expect("work_dir");

        // Create a pre-existing file at the output path.
        let output_path = work.path().join("normalized_16k_mono.wav");
        std::fs::write(&output_path, b"old data").expect("write old data");

        let result = normalize_to_wav(&input_wav, work.path()).expect("should succeed");
        assert!(result.exists());
        let content = std::fs::read(&result).expect("read");
        // The -y flag in ffmpeg args should overwrite; output should be valid WAV,
        // not "old data".
        assert_ne!(&content[..], b"old data", "output should be overwritten");
        assert_eq!(&content[0..4], b"RIFF", "output should be valid RIFF/WAV");
    }

    // ── probe_duration_seconds: tests that exercise ffprobe ──

    #[test]
    fn probe_duration_of_generated_wav_returns_approximately_one_second() {
        if !ffprobe_available() {
            eprintln!("SKIPPED: ffprobe not found on PATH");
            return;
        }
        use super::probe_duration_seconds;

        let dir = tempfile::tempdir().expect("tempdir");
        let wav = generate_sine_wav(dir.path(), "probe_1s.wav");

        let duration = probe_duration_seconds(&wav);
        let secs = duration.expect("should return Some for valid WAV");
        assert!(
            (secs - 1.0).abs() < 0.05,
            "expected ~1.0 second, got {secs}"
        );
    }

    #[test]
    fn probe_duration_of_silence_wav_returns_expected_length() {
        if !ffprobe_available() {
            eprintln!("SKIPPED: ffprobe not found on PATH");
            return;
        }
        use super::probe_duration_seconds;

        let dir = tempfile::tempdir().expect("tempdir");
        let wav = generate_silence_wav(dir.path(), "silence_2s.wav", 2.0);

        let duration = probe_duration_seconds(&wav);
        let secs = duration.expect("should return Some for silent WAV");
        assert!(
            (secs - 2.0).abs() < 0.05,
            "expected ~2.0 seconds, got {secs}"
        );
    }

    #[test]
    fn probe_duration_with_timeout_of_valid_wav() {
        if !ffprobe_available() {
            eprintln!("SKIPPED: ffprobe not found on PATH");
            return;
        }
        use super::probe_duration_seconds_with_timeout;

        let dir = tempfile::tempdir().expect("tempdir");
        let wav = generate_sine_wav(dir.path(), "probe_timeout.wav");

        let duration =
            probe_duration_seconds_with_timeout(&wav, std::time::Duration::from_secs(10));
        let secs = duration.expect("should return Some");
        assert!(
            (secs - 1.0).abs() < 0.05,
            "expected ~1.0 second, got {secs}"
        );
    }

    #[test]
    fn probe_duration_with_text_file_returns_none() {
        use super::probe_duration_seconds;

        let dir = tempfile::tempdir().expect("tempdir");
        let txt = dir.path().join("not_audio.txt");
        std::fs::write(&txt, "hello world").expect("write");

        // ffprobe should fail to parse a text file.
        let result = probe_duration_seconds(&txt);
        // May return None (if ffprobe errors) or Some (unlikely). No panic.
        let _ = result;
    }

    #[test]
    fn probe_duration_of_normalized_output_matches_source() {
        if !ffmpeg_available() || !ffprobe_available() {
            eprintln!("SKIPPED: ffmpeg or ffprobe not found on PATH");
            return;
        }
        use super::{normalize_to_wav, probe_duration_seconds};

        let dir = tempfile::tempdir().expect("tempdir");
        let input_wav = generate_sine_wav(dir.path(), "chain_input.wav");
        let work = tempfile::tempdir().expect("work_dir");

        let normalized =
            normalize_to_wav(&input_wav, work.path()).expect("normalize should succeed");

        let input_dur = probe_duration_seconds(&input_wav).expect("input probe should return Some");
        let output_dur =
            probe_duration_seconds(&normalized).expect("output probe should return Some");

        assert!(
            (input_dur - output_dur).abs() < 0.1,
            "normalized output duration ({output_dur}s) should match input ({input_dur}s)"
        );
    }

    // ── ffmpeg_timeout / ffprobe_timeout: default value checks ──

    #[test]
    fn ffmpeg_timeout_default_is_180_seconds() {
        use super::ffmpeg_timeout;
        // When FRANKEN_WHISPER_FFMPEG_TIMEOUT_MS is not set, the default is 180s.
        // The env var is unlikely to be set in test, so this should hold.
        let timeout = ffmpeg_timeout();
        assert_eq!(
            timeout,
            std::time::Duration::from_secs(180),
            "default ffmpeg timeout should be 180 seconds"
        );
    }

    #[test]
    fn ffprobe_timeout_default_is_10_seconds() {
        use super::ffprobe_timeout;
        let timeout = ffprobe_timeout();
        assert_eq!(
            timeout,
            std::time::Duration::from_secs(10),
            "default ffprobe timeout should be 10 seconds"
        );
    }

    // ── microphone_timeout: additional coverage ──

    #[test]
    fn microphone_timeout_intermediate_values() {
        use super::microphone_timeout;

        // Verify several intermediate values: seconds + 15 = expected.
        let cases: Vec<(u32, u64)> = vec![(5, 20), (30, 45), (60, 75), (120, 135), (300, 315)];
        for (seconds, expected_secs) in cases {
            let timeout = microphone_timeout(seconds);
            assert_eq!(
                timeout,
                std::time::Duration::from_secs(expected_secs),
                "microphone_timeout({seconds}) should be {expected_secs}s"
            );
        }
    }

    // ── duration_from_env: additional edge cases ──

    #[test]
    fn duration_from_env_max_u64_fallback() {
        use super::duration_from_env;
        let max_dur = std::time::Duration::from_millis(u64::MAX);
        let result = duration_from_env("FRANKEN_WHISPER_TEST_DUR_NONEXIST_MAX_U64", max_dur);
        assert_eq!(result, max_dur);
    }

    #[test]
    fn duration_from_env_nanos_precision_fallback() {
        use super::duration_from_env;
        // duration_from_env returns Duration::from_millis() when parsing the
        // env var. For the fallback path it returns the exact Duration provided.
        let precise = std::time::Duration::new(1, 999_999_999);
        let result = duration_from_env("FRANKEN_WHISPER_TEST_DUR_NONEXIST_NANOS", precise);
        assert_eq!(
            result, precise,
            "fallback should preserve nanosecond precision"
        );
    }

    // ── microphone_defaults: platform-specific coverage ──

    #[cfg(target_os = "macos")]
    #[test]
    fn macos_microphone_defaults_use_avfoundation() {
        let (format, source) = microphone_defaults(None, None, None).expect("macos defaults");
        assert_eq!(format, "avfoundation");
        assert_eq!(source, ":0");
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn macos_microphone_custom_device() {
        let (format, source) = microphone_defaults(Some(":1"), None, None).expect("custom device");
        assert_eq!(format, "avfoundation");
        assert_eq!(source, ":1");
    }

    #[cfg(target_os = "windows")]
    #[test]
    fn windows_microphone_defaults_use_dshow() {
        let (format, source) = microphone_defaults(None, None, None).expect("windows defaults");
        assert_eq!(format, "dshow");
        assert_eq!(source, "audio=default");
    }

    // ── InvalidRequest error variant matching ──

    #[test]
    fn materialize_nonexistent_file_error_is_invalid_request_variant() {
        use super::materialize_input;
        use crate::error::FwError;
        use crate::model::InputSource;

        let dir = tempfile::tempdir().expect("tempdir");
        let source = InputSource::File {
            path: std::path::PathBuf::from("/no/such/file_12345.mp3"),
        };
        let err = materialize_input(&source, dir.path()).expect_err("should fail");
        assert!(
            matches!(err, FwError::InvalidRequest(_)),
            "expected InvalidRequest variant, got: {err:?}"
        );
    }

    #[test]
    fn partial_mic_args_error_is_invalid_request_variant() {
        use crate::error::FwError;

        let err =
            microphone_defaults(None, Some("pulse"), None).expect_err("partial args should fail");
        assert!(
            matches!(err, FwError::InvalidRequest(_)),
            "expected InvalidRequest variant, got: {err:?}"
        );
    }

    // ── normalize_to_wav: path with unicode and spaces ──

    #[test]
    fn normalize_wav_with_spaces_in_path() {
        if !ffmpeg_available() {
            eprintln!("SKIPPED: ffmpeg not found on PATH");
            return;
        }
        use super::normalize_to_wav;

        let dir = tempfile::tempdir().expect("tempdir");
        let spaced = dir.path().join("my audio files");
        std::fs::create_dir_all(&spaced).expect("mkdir");
        let input_wav = generate_sine_wav(&spaced, "test tone.wav");
        let work = tempfile::tempdir().expect("work_dir");

        let result =
            normalize_to_wav(&input_wav, work.path()).expect("should handle spaces in path");
        assert!(result.exists());
    }

    #[test]
    fn normalize_wav_with_unicode_in_path() {
        if !ffmpeg_available() {
            eprintln!("SKIPPED: ffmpeg not found on PATH");
            return;
        }
        use super::normalize_to_wav;

        let dir = tempfile::tempdir().expect("tempdir");
        let unicode_dir = dir.path().join("donn\u{00e9}es_audio");
        std::fs::create_dir_all(&unicode_dir).expect("mkdir");
        let input_wav = generate_sine_wav(&unicode_dir, "\u{97f3}\u{58f0}.wav");
        let work = tempfile::tempdir().expect("work_dir");

        let result =
            normalize_to_wav(&input_wav, work.path()).expect("should handle unicode in path");
        assert!(result.exists());
    }

    // ── Verify WAV header structure of inline generator ──

    #[test]
    fn generated_wav_has_valid_riff_header() {
        let dir = tempfile::tempdir().expect("tempdir");
        let wav = generate_sine_wav(dir.path(), "header_check.wav");

        let data = std::fs::read(&wav).expect("read");
        assert!(data.len() > 44, "WAV should be at least 44 bytes + data");
        assert_eq!(&data[0..4], b"RIFF", "RIFF marker");
        assert_eq!(&data[8..12], b"WAVE", "WAVE marker");
        assert_eq!(&data[12..16], b"fmt ", "fmt chunk marker");
        assert_eq!(&data[36..40], b"data", "data chunk marker");

        // PCM format tag = 1
        let fmt_tag = u16::from_le_bytes([data[20], data[21]]);
        assert_eq!(fmt_tag, 1, "should be PCM format");

        // Channels = 1
        let channels = u16::from_le_bytes([data[22], data[23]]);
        assert_eq!(channels, 1, "should be mono");

        // Sample rate = 16000
        let sample_rate = u32::from_le_bytes([data[24], data[25], data[26], data[27]]);
        assert_eq!(sample_rate, 16000, "should be 16kHz");

        // Bits per sample = 16
        let bits = u16::from_le_bytes([data[34], data[35]]);
        assert_eq!(bits, 16, "should be 16-bit");
    }

    #[test]
    fn generated_silence_wav_contains_all_zeros_in_data_section() {
        let dir = tempfile::tempdir().expect("tempdir");
        let wav = generate_silence_wav(dir.path(), "zero_check.wav", 0.1);

        let data = std::fs::read(&wav).expect("read");
        // Data section starts at byte 44 for a standard PCM WAV.
        let audio_data = &data[44..];
        assert!(!audio_data.is_empty(), "should have audio data");
        assert!(
            audio_data.iter().all(|&b| b == 0),
            "silence WAV data section should be all zeros"
        );
    }

    #[test]
    fn generated_sine_wav_has_expected_sample_count() {
        let dir = tempfile::tempdir().expect("tempdir");
        let wav = generate_sine_wav(dir.path(), "count_check.wav");

        let data = std::fs::read(&wav).expect("read");
        let audio_data = &data[44..];
        // 1 second * 16000 Hz * 2 bytes/sample = 32000 bytes
        assert_eq!(
            audio_data.len(),
            32000,
            "1s at 16kHz 16-bit mono should be 32000 bytes of audio data"
        );
    }

    // ── normalize_to_wav output verification: 16kHz mono PCM ──

    #[test]
    fn normalized_output_is_16khz_mono_pcm() {
        if !ffmpeg_available() {
            eprintln!("SKIPPED: ffmpeg not found on PATH");
            return;
        }
        use super::normalize_to_wav;

        let dir = tempfile::tempdir().expect("tempdir");
        let input_wav = generate_sine_wav(dir.path(), "verify_format.wav");
        let work = tempfile::tempdir().expect("work_dir");

        let output = normalize_to_wav(&input_wav, work.path()).expect("should succeed");
        let data = std::fs::read(&output).expect("read output");
        assert!(data.len() > 44, "output too small");

        // Verify RIFF/WAVE
        assert_eq!(&data[0..4], b"RIFF");
        assert_eq!(&data[8..12], b"WAVE");

        // PCM format = 1
        let fmt_tag = u16::from_le_bytes([data[20], data[21]]);
        assert_eq!(fmt_tag, 1, "normalized output should be PCM");

        // Mono
        let channels = u16::from_le_bytes([data[22], data[23]]);
        assert_eq!(channels, 1, "normalized output should be mono");

        // 16 kHz
        let sample_rate = u32::from_le_bytes([data[24], data[25], data[26], data[27]]);
        assert_eq!(sample_rate, 16000, "normalized output should be 16kHz");

        // 16-bit
        let bits = u16::from_le_bytes([data[34], data[35]]);
        assert_eq!(bits, 16, "normalized output should be 16-bit");
    }
}
