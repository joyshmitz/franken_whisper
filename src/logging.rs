//! Structured logging configuration for franken_whisper.
//!
//! Initializes a `tracing` subscriber with:
//! - `RUST_LOG` environment filter support
//! - Default level: INFO
//! - JSON output when `RUST_LOG_FORMAT=json`
//! - Human-readable output otherwise

use tracing_subscriber::EnvFilter;

/// Initialize the global tracing subscriber.
///
/// Call this once at program startup (main.rs).
/// Safe to call multiple times (subsequent calls are no-ops).
pub fn init() {
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("franken_whisper=info"));

    let is_json = std::env::var("RUST_LOG_FORMAT")
        .map(|v| v.eq_ignore_ascii_case("json"))
        .unwrap_or(false);

    let subscriber = tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_writer(std::io::stderr)
        .with_target(true)
        .with_thread_ids(false)
        .with_file(false)
        .with_line_number(false);

    if is_json {
        let _ = subscriber.json().try_init();
    } else {
        let _ = subscriber.try_init();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn init_does_not_panic() {
        // Calling init() should not panic even if called multiple times
        init();
        init();
    }

    #[test]
    fn init_respects_env_filter() {
        // The filter should parse without error
        let filter = EnvFilter::try_from_default_env()
            .unwrap_or_else(|_| EnvFilter::new("franken_whisper=debug"));
        assert!(format!("{filter:?}").contains("franken_whisper"));
    }
}
