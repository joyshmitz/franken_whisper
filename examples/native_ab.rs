//! End-to-end A/B + conformance harness for the native engine.
//!
//! Loads a whisper ggml model by short name, transcribes the jfk fixture
//! `runs` times, and prints per-run wall times plus a golden-comparable JSON
//! (`{transcript, segments:[[start,end,text]]}`). Used by the round-2 f16
//! optimization pass for:
//!   * interleaved wall A/B vs a pre-change REF binary (min/p25 of N runs), and
//!   * the exact-transcript conformance gate against
//!     `/tmp/fw_golden/{tiny.en,large-v3-turbo}.json`.
//!
//! It exercises the same `transcribe_samples` path production uses, so the
//! `FRANKEN_WHISPER_NATIVE_F16_COMPUTE` runtime switch is honored exactly as in
//! the real engine. Usage:
//!   native_ab <tiny.en|large-v3-turbo> [runs]
//! Audio is the in-repo `tests/fixtures/native/jfk.wav` (mono 16 kHz).

use std::time::Instant;

use franken_whisper::native_engine::decode::{DecodeParams, LoadedModel, transcribe_samples};
use franken_whisper::native_engine::find_model_file;
use franken_whisper::native_engine::ggml::GgmlModel;

fn load_jfk_samples() -> Vec<f32> {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/native/jfk.wav");
    let mut reader = hound::WavReader::open(path).expect("open jfk.wav");
    let spec = reader.spec();
    assert_eq!(spec.channels, 1, "jfk.wav must be mono");
    assert_eq!(spec.sample_rate, 16_000, "jfk.wav must be 16 kHz");
    reader
        .samples::<i16>()
        .map(|s| f32::from(s.expect("sample")) / 32768.0)
        .collect()
}

fn main() {
    let mut args = std::env::args().skip(1);
    let model_name = args.next().unwrap_or_else(|| "tiny.en".to_string());
    let runs: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(1);

    let path = find_model_file(&model_name)
        .unwrap_or_else(|| panic!("model {model_name} not found in search dirs"));
    let model =
        LoadedModel::from_ggml(GgmlModel::load(&path).expect("load ggml")).expect("from_ggml");
    let samples = load_jfk_samples();

    let params = DecodeParams {
        language: None,
        translate: false,
        timestamps: true,
        n_threads: 4,
        max_text_ctx: None,
        ..DecodeParams::default()
    };
    let noop = || Ok(());

    let mut last = None;
    for r in 0..runs {
        let t = Instant::now();
        let out = transcribe_samples(&model, &samples, &params, &noop).expect("transcribe");
        let ms = t.elapsed().as_secs_f64() * 1e3;
        eprintln!("RUN {r} wall_ms={ms:.2}");
        last = Some(out);
    }

    let out = last.expect("at least one run");
    let transcript: String = out
        .segments
        .iter()
        .map(|s| s.text.trim())
        .collect::<Vec<_>>()
        .join(" ");
    // Emit a golden-comparable JSON document on stdout (segments rounded to the
    // same 2-decimal precision the golden files use).
    let mut json = String::new();
    json.push_str("{\n \"transcript\": ");
    push_json_str(&mut json, &transcript);
    json.push_str(",\n \"segments\": [\n");
    for (i, s) in out.segments.iter().enumerate() {
        let start = round2(s.start_sec.unwrap_or(0.0));
        let end = round2(s.end_sec.unwrap_or(0.0));
        json.push_str("  [\n");
        json.push_str(&format!("   {start},\n   {end},\n   "));
        push_json_str(&mut json, s.text.trim());
        json.push_str("\n  ]");
        if i + 1 < out.segments.len() {
            json.push(',');
        }
        json.push('\n');
    }
    json.push_str(" ]\n}");
    println!("{json}");
}

fn round2(v: f64) -> f64 {
    (v * 100.0).round() / 100.0
}

fn push_json_str(out: &mut String, s: &str) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            _ => out.push(c),
        }
    }
    out.push('"');
}
