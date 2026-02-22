//! Fixture generator — produces deterministic synthetic audio files in
//! `tests/fixtures/audio/`.
//!
//! Run with: `cargo test --test generate_fixtures -- --ignored`
//!
//! Files are only regenerated when missing. To force-regenerate, delete the
//! target file first.

mod helpers;

use std::path::PathBuf;

fn audio_fixtures_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("audio")
}

/// Generate a 1-second 440 Hz sine tone (standard A4 pitch).
/// Deterministic: same samples every time (pure math, no PRNG needed).
#[test]
#[ignore]
fn generate_test_1s_tone() {
    let dir = audio_fixtures_dir();
    let path = dir.join("test_1s_tone.wav");
    if path.exists() {
        return;
    }
    helpers::generate_test_wav(&dir, "test_1s_tone.wav", 1.0, 440.0);
}

/// Generate a 10-second multi-frequency "speech-like" pattern.
/// Uses three superimposed sine waves at 150, 300, and 600 Hz
/// (fundamental + harmonics typical of voiced speech).
/// Deterministic: pure trigonometric generation.
#[test]
#[ignore]
fn generate_test_10s_speech() {
    let dir = audio_fixtures_dir();
    let path = dir.join("test_10s_speech.wav");
    if path.exists() {
        return;
    }

    let sample_rate: u32 = 16000;
    let duration_secs: f32 = 10.0;
    let num_samples = (sample_rate as f32 * duration_secs) as usize;
    let mut samples = Vec::with_capacity(num_samples);

    for i in 0..num_samples {
        let t = i as f32 / sample_rate as f32;
        // Superposition of speech-like harmonics
        let f1 = (2.0 * core::f32::consts::PI * 150.0 * t).sin() * 0.5;
        let f2 = (2.0 * core::f32::consts::PI * 300.0 * t).sin() * 0.3;
        let f3 = (2.0 * core::f32::consts::PI * 600.0 * t).sin() * 0.2;
        let value = f1 + f2 + f3;
        let sample = (value * 32767.0) as i16;
        samples.push(sample);
    }

    write_raw_wav(&dir.join("test_10s_speech.wav"), &samples, sample_rate, 1);
}

/// Generate a 1-second silence file (all-zero samples).
/// Useful for testing edge cases: empty transcript, zero-energy input.
#[test]
#[ignore]
fn generate_test_silence() {
    let dir = audio_fixtures_dir();
    let path = dir.join("test_silence.wav");
    if path.exists() {
        return;
    }
    helpers::generate_silence_wav(&dir, "test_silence.wav", 1.0);
}

fn write_raw_wav(path: &std::path::Path, samples: &[i16], sample_rate: u32, channels: u16) {
    use std::io::Write;
    let data_size = (samples.len() * 2) as u32;
    let file_size = 36 + data_size;
    let byte_rate = sample_rate * channels as u32 * 2;
    let block_align = channels * 2;

    let mut file = std::fs::File::create(path).expect("failed to create WAV file");
    file.write_all(b"RIFF").unwrap();
    file.write_all(&file_size.to_le_bytes()).unwrap();
    file.write_all(b"WAVE").unwrap();
    file.write_all(b"fmt ").unwrap();
    file.write_all(&16u32.to_le_bytes()).unwrap();
    file.write_all(&1u16.to_le_bytes()).unwrap();
    file.write_all(&channels.to_le_bytes()).unwrap();
    file.write_all(&sample_rate.to_le_bytes()).unwrap();
    file.write_all(&byte_rate.to_le_bytes()).unwrap();
    file.write_all(&block_align.to_le_bytes()).unwrap();
    file.write_all(&16u16.to_le_bytes()).unwrap();
    file.write_all(b"data").unwrap();
    file.write_all(&data_size.to_le_bytes()).unwrap();
    for sample in samples {
        file.write_all(&sample.to_le_bytes()).unwrap();
    }
}
