use std::path::Path;
use std::process::{Command, Output, Stdio};
use std::thread;
use std::time::{Duration, Instant};

use crate::error::{FwError, FwResult};

#[must_use]
pub fn command_exists(program: &str) -> bool {
    which::which(program).is_ok()
}

pub fn run_command(program: &str, args: &[String], cwd: Option<&Path>) -> FwResult<Output> {
    run_command_with_timeout(program, args, cwd, None)
}

pub fn run_command_with_timeout(
    program: &str,
    args: &[String],
    cwd: Option<&Path>,
    timeout: Option<Duration>,
) -> FwResult<Output> {
    if !command_exists(program) {
        return Err(FwError::CommandMissing {
            command: program.to_owned(),
        });
    }

    let rendered = format!("{} {}", program, args.join(" "));
    let mut command = Command::new(program);
    command.args(args);
    command.stdout(Stdio::piped());
    command.stderr(Stdio::piped());
    if let Some(dir) = cwd {
        command.current_dir(dir);
    }

    if let Some(limit) = timeout {
        let mut child = command.spawn()?;
        let started_at = Instant::now();

        let mut stdout_pipe = child.stdout.take().expect("stdout piped");
        let mut stderr_pipe = child.stderr.take().expect("stderr piped");

        let (stdout_tx, stdout_rx) = std::sync::mpsc::channel();
        let (stderr_tx, stderr_rx) = std::sync::mpsc::channel();

        thread::spawn(move || {
            use std::io::Read;
            let mut buf = Vec::new();
            let _ = stdout_pipe.read_to_end(&mut buf);
            let _ = stdout_tx.send(buf);
        });

        thread::spawn(move || {
            use std::io::Read;
            let mut buf = Vec::new();
            let _ = stderr_pipe.read_to_end(&mut buf);
            let _ = stderr_tx.send(buf);
        });

        loop {
            if let Some(status) = child.try_wait()? {
                let stdout = stdout_rx
                    .recv_timeout(Duration::from_millis(100))
                    .unwrap_or_default();
                let stderr = stderr_rx
                    .recv_timeout(Duration::from_millis(100))
                    .unwrap_or_default();
                return validate_command_output(
                    &rendered,
                    Output {
                        status,
                        stdout,
                        stderr,
                    },
                );
            }

            if started_at.elapsed() >= limit {
                let _ = child.kill();
                let _ = child.wait();
                let stderr = stderr_rx
                    .recv_timeout(Duration::from_millis(100))
                    .unwrap_or_default();
                let stderr_str = String::from_utf8_lossy(&stderr).into_owned();
                return Err(FwError::from_command_timeout(
                    rendered,
                    saturating_duration_ms(limit),
                    stderr_str,
                ));
            }

            thread::sleep(Duration::from_millis(20));
        }
    }

    let output = command.output()?;
    validate_command_output(&rendered, output)
}

/// Run a subprocess with cancellation-aware polling.
///
/// Instead of a fixed timeout, this variant polls `token.checkpoint()` on every
/// iteration (50ms sleep). If the checkpoint returns `Err(Cancelled)`, the
/// child process is killed immediately and the error is propagated. An optional
/// hard timeout is still respected as a safety net.
pub(crate) fn run_command_cancellable(
    program: &str,
    args: &[String],
    cwd: Option<&Path>,
    token: &crate::orchestrator::CancellationToken,
    hard_timeout: Option<Duration>,
) -> FwResult<Output> {
    if !command_exists(program) {
        return Err(FwError::CommandMissing {
            command: program.to_owned(),
        });
    }

    let rendered = format!("{} {}", program, args.join(" "));
    let mut command = Command::new(program);
    command.args(args);
    command.stdout(Stdio::piped());
    command.stderr(Stdio::piped());
    if let Some(dir) = cwd {
        command.current_dir(dir);
    }

    let mut child = command.spawn()?;
    let started_at = Instant::now();

    let mut stdout_pipe = child.stdout.take().expect("stdout piped");
    let mut stderr_pipe = child.stderr.take().expect("stderr piped");

    let (stdout_tx, stdout_rx) = std::sync::mpsc::channel();
    let (stderr_tx, stderr_rx) = std::sync::mpsc::channel();

    thread::spawn(move || {
        use std::io::Read;
        let mut buf = Vec::new();
        let _ = stdout_pipe.read_to_end(&mut buf);
        let _ = stdout_tx.send(buf);
    });

    thread::spawn(move || {
        use std::io::Read;
        let mut buf = Vec::new();
        let _ = stderr_pipe.read_to_end(&mut buf);
        let _ = stderr_tx.send(buf);
    });

    loop {
        if let Some(status) = child.try_wait()? {
            let stdout = stdout_rx
                .recv_timeout(Duration::from_millis(100))
                .unwrap_or_default();
            let stderr = stderr_rx
                .recv_timeout(Duration::from_millis(100))
                .unwrap_or_default();
            return validate_command_output(
                &rendered,
                Output {
                    status,
                    stdout,
                    stderr,
                },
            );
        }

        // Check pipeline deadline via cancellation token.
        if let Err(err) = token.checkpoint() {
            let _ = child.kill();
            let _ = child.wait();
            return Err(err);
        }

        // Hard timeout safety net.
        if let Some(limit) = hard_timeout
            && started_at.elapsed() >= limit
        {
            let _ = child.kill();
            let _ = child.wait();
            let stderr = stderr_rx
                .recv_timeout(Duration::from_millis(100))
                .unwrap_or_default();
            let stderr_str = String::from_utf8_lossy(&stderr).into_owned();
            return Err(FwError::from_command_timeout(
                rendered,
                saturating_duration_ms(limit),
                stderr_str,
            ));
        }

        thread::sleep(Duration::from_millis(50));
    }
}

fn validate_command_output(rendered: &str, output: Output) -> FwResult<Output> {
    if output.status.success() {
        return Ok(output);
    }

    let status = output.status.code().unwrap_or(-1);
    let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
    Err(FwError::from_command_failure(
        rendered.to_owned(),
        status,
        stderr,
    ))
}

fn saturating_duration_ms(duration: Duration) -> u64 {
    duration.as_millis().try_into().unwrap_or(u64::MAX)
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use crate::orchestrator::CancellationToken;

    use super::run_command_cancellable;

    #[test]
    fn cancellable_completes_fast_command() {
        // A command that exits immediately should succeed with a far-future deadline.
        let token = CancellationToken::with_deadline_from_now(Duration::from_secs(60));
        let result =
            run_command_cancellable("true", &[], None, &token, Some(Duration::from_secs(10)));
        assert!(result.is_ok(), "true should succeed: {result:?}");
    }

    #[test]
    fn cancellable_kills_on_expired_deadline() {
        // Create a token whose deadline is already in the past.
        let token = CancellationToken::with_deadline_from_now(Duration::from_millis(0));
        // Tiny sleep to ensure we're past the deadline.
        std::thread::sleep(Duration::from_millis(10));

        let result = run_command_cancellable(
            "sleep",
            &["60".to_owned()],
            None,
            &token,
            Some(Duration::from_secs(120)),
        );

        assert!(result.is_err(), "should be cancelled");
        let err = result.unwrap_err();
        assert!(
            matches!(err, crate::error::FwError::Cancelled(_)),
            "expected Cancelled error, got: {err:?}"
        );
    }

    #[test]
    fn cancellable_hard_timeout_takes_effect() {
        // Token with no deadline (far future), but hard timeout is tiny.
        let token = CancellationToken::with_deadline_from_now(Duration::from_secs(600));
        let result = run_command_cancellable(
            "sleep",
            &["60".to_owned()],
            None,
            &token,
            Some(Duration::from_millis(100)),
        );

        assert!(result.is_err(), "should hit hard timeout");
        // Should NOT be Cancelled — should be a CommandTimeout.
        let err = result.unwrap_err();
        assert!(
            !matches!(err, crate::error::FwError::Cancelled(_)),
            "expected timeout error, not Cancelled: {err:?}"
        );
    }

    #[test]
    fn cancellable_no_deadline_still_works() {
        // Token with no deadline at all — should complete normally for fast commands.
        let token = CancellationToken::no_deadline();
        let result = run_command_cancellable("true", &[], None, &token, None);
        assert!(result.is_ok());
    }

    // -----------------------------------------------------------------------
    // run_command / run_command_with_timeout / validate_command_output tests
    // -----------------------------------------------------------------------

    use super::{run_command, run_command_with_timeout, saturating_duration_ms};

    #[test]
    fn run_command_succeeds_for_true() {
        let output = run_command("true", &[], None).expect("true should succeed");
        assert!(output.status.success());
    }

    #[test]
    fn run_command_missing_program_returns_command_missing() {
        let err = run_command("nonexistent_binary_xyz_12345", &[], None)
            .expect_err("nonexistent binary should fail");
        assert!(
            matches!(err, crate::error::FwError::CommandMissing { .. }),
            "expected CommandMissing, got: {err:?}"
        );
    }

    #[test]
    fn run_command_nonzero_exit_returns_command_failed() {
        let err = run_command("false", &[], None).expect_err("false should fail");
        let text = err.to_string();
        assert!(
            text.contains("command failed") || text.contains("status"),
            "expected command failure message, got: {text}"
        );
    }

    #[test]
    fn run_command_with_timeout_succeeds_when_fast() {
        let output = run_command_with_timeout("true", &[], None, Some(Duration::from_secs(5)))
            .expect("true should succeed within timeout");
        assert!(output.status.success());
    }

    #[test]
    fn run_command_with_timeout_kills_slow_command() {
        let err = run_command_with_timeout(
            "sleep",
            &["60".to_owned()],
            None,
            Some(Duration::from_millis(100)),
        )
        .expect_err("should timeout");
        let text = err.to_string();
        assert!(
            text.contains("timed out") || text.contains("timeout"),
            "expected timeout message, got: {text}"
        );
    }

    #[test]
    fn run_command_captures_stderr() {
        // `ls` on a nonexistent path writes to stderr and exits non-zero.
        let err = run_command("ls", &["/nonexistent_path_xyz_99999".to_owned()], None)
            .expect_err("ls on nonexistent should fail");
        let text = err.to_string();
        assert!(
            text.contains("nonexistent_path") || text.contains("No such file"),
            "expected stderr content, got: {text}"
        );
    }

    #[test]
    fn run_command_with_cwd() {
        let dir = tempfile::tempdir().expect("tempdir");
        let output = run_command("pwd", &[], Some(dir.path())).expect("pwd should succeed");
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(
            stdout.contains(dir.path().to_str().unwrap()),
            "expected cwd in stdout, got: {stdout}"
        );
    }

    #[test]
    fn saturating_duration_ms_normal_case() {
        assert_eq!(saturating_duration_ms(Duration::from_secs(5)), 5000);
        assert_eq!(saturating_duration_ms(Duration::from_millis(1234)), 1234);
    }

    #[test]
    fn saturating_duration_ms_max_does_not_panic() {
        let result = saturating_duration_ms(Duration::from_secs(u64::MAX));
        assert_eq!(result, u64::MAX);
    }

    // -----------------------------------------------------------------------
    // command_exists tests
    // -----------------------------------------------------------------------

    use super::command_exists;

    #[test]
    fn command_exists_true_for_known_binary() {
        // `ls` and `true` exist on all Unix-like systems.
        assert!(command_exists("ls"), "ls should exist");
        assert!(command_exists("true"), "true should exist");
    }

    #[test]
    fn command_exists_false_for_absent_binary() {
        assert!(
            !command_exists("definitely_not_a_real_binary_abc_xyz_99999"),
            "absent binary should not exist"
        );
    }

    // -----------------------------------------------------------------------
    // validate_command_output tests
    // -----------------------------------------------------------------------

    use super::validate_command_output;
    use std::os::unix::process::ExitStatusExt;
    use std::process::ExitStatus;

    fn fake_output(code: i32, stderr: &str) -> std::process::Output {
        std::process::Output {
            status: ExitStatus::from_raw(code << 8), // raw wait status: exit code in upper byte
            stdout: Vec::new(),
            stderr: stderr.as_bytes().to_vec(),
        }
    }

    #[test]
    fn validate_command_output_success_returns_ok() {
        let output = fake_output(0, "");
        let result = validate_command_output("test-cmd", output);
        assert!(result.is_ok());
    }

    #[test]
    fn validate_command_output_nonzero_exit_returns_error() {
        let output = fake_output(1, "something went wrong");
        let result = validate_command_output("test-cmd", output);
        assert!(result.is_err());
        let text = result.unwrap_err().to_string();
        assert!(
            text.contains("something went wrong"),
            "error should contain stderr, got: {text}"
        );
    }

    #[test]
    fn validate_command_output_preserves_exit_code_in_error() {
        let output = fake_output(42, "exit code 42");
        let err = validate_command_output("my-tool --flag", output).unwrap_err();
        let text = err.to_string();
        assert!(
            text.contains("42"),
            "error should mention exit code 42, got: {text}"
        );
    }

    #[test]
    fn validate_command_output_empty_stderr_still_fails_on_nonzero() {
        let output = fake_output(2, "");
        let result = validate_command_output("cmd", output);
        assert!(
            result.is_err(),
            "non-zero exit with empty stderr should still fail"
        );
    }

    // ── Additional edge case tests ──

    #[test]
    fn run_command_with_timeout_none_behaves_like_run_command() {
        let output = run_command_with_timeout("true", &[], None, None).expect("should succeed");
        assert!(output.status.success());
    }

    #[test]
    fn run_command_with_args() {
        let output = run_command("echo", &["hello".to_owned(), "world".to_owned()], None)
            .expect("echo should succeed");
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(
            stdout.contains("hello world"),
            "expected 'hello world', got: {stdout}"
        );
    }

    #[test]
    fn cancellable_missing_program_returns_command_missing() {
        let token = CancellationToken::no_deadline();
        let err = run_command_cancellable("nonexistent_binary_xyz_99999", &[], None, &token, None)
            .expect_err("should fail");
        assert!(
            matches!(err, crate::error::FwError::CommandMissing { .. }),
            "expected CommandMissing, got: {err:?}"
        );
    }

    #[test]
    fn cancellable_captures_output_from_successful_command() {
        let token = CancellationToken::no_deadline();
        let output =
            run_command_cancellable("echo", &["test_output".to_owned()], None, &token, None)
                .expect("echo should succeed");
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(
            stdout.contains("test_output"),
            "should capture stdout: {stdout}"
        );
    }

    #[test]
    fn cancellable_nonzero_exit_returns_error() {
        let token = CancellationToken::no_deadline();
        let err = run_command_cancellable("false", &[], None, &token, None)
            .expect_err("false should fail");
        assert!(
            !matches!(err, crate::error::FwError::Cancelled(_)),
            "should not be cancelled, should be command failure: {err:?}"
        );
    }

    #[test]
    fn saturating_duration_ms_zero() {
        assert_eq!(saturating_duration_ms(Duration::ZERO), 0);
    }

    #[test]
    fn saturating_duration_ms_subsecond() {
        assert_eq!(saturating_duration_ms(Duration::from_millis(500)), 500);
        assert_eq!(saturating_duration_ms(Duration::from_millis(1)), 1);
    }

    #[test]
    fn validate_command_output_includes_command_name_in_error() {
        let output = fake_output(1, "boom");
        let err = validate_command_output("my-special-cmd --flag", output).unwrap_err();
        let text = err.to_string();
        assert!(
            text.contains("my-special-cmd"),
            "error should mention command: {text}"
        );
    }

    #[test]
    fn cancellable_with_cwd_succeeds() {
        let dir = tempfile::tempdir().expect("tempdir");
        let token = CancellationToken::no_deadline();
        let output = run_command_cancellable("pwd", &[], Some(dir.path()), &token, None)
            .expect("pwd should succeed");
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(
            stdout.contains(dir.path().to_str().unwrap()),
            "expected cwd in stdout, got: {stdout}"
        );
    }

    #[test]
    fn run_command_empty_args_succeeds() {
        let output = run_command("true", &[], None).expect("true with no args");
        assert!(output.status.success());
    }

    #[test]
    fn cancellable_with_hard_timeout_none_and_no_deadline() {
        // Both safety nets disabled — should still work for fast commands.
        let token = CancellationToken::no_deadline();
        let output = run_command_cancellable("echo", &["ok".to_owned()], None, &token, None)
            .expect("should succeed");
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(stdout.contains("ok"));
    }

    #[test]
    fn validate_command_output_signal_terminated_uses_negative_one() {
        // When a process is killed by a signal, exit code may not be available.
        // On Unix, from_raw(9) represents SIGKILL (signal 9, no exit code).
        let output = std::process::Output {
            status: ExitStatus::from_raw(9), // signal 9 (SIGKILL), no exit code
            stdout: Vec::new(),
            stderr: b"killed".to_vec(),
        };
        let result = validate_command_output("signaled-cmd", output);
        assert!(result.is_err(), "signal-killed process should fail");
        let text = result.unwrap_err().to_string();
        // The code falls back to -1 when .code() returns None.
        assert!(
            text.contains("-1") || text.contains("killed"),
            "should mention -1 or killed: {text}"
        );
    }

    #[test]
    fn run_command_with_timeout_missing_program_returns_command_missing() {
        let err = run_command_with_timeout(
            "nonexistent_xyz_99",
            &[],
            None,
            Some(Duration::from_secs(5)),
        )
        .expect_err("should fail");
        assert!(matches!(err, crate::error::FwError::CommandMissing { .. }));
    }
}
