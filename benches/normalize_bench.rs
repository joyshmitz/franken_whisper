//! Performance benchmarks for the backend output normalization layer.
//!
//! Exercises `normalize_whisper_cpp`, `normalize_insanely_fast`, and
//! `to_transcription_result` with varying payload sizes.

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use serde_json::{Value, json};

use franken_whisper::backend::normalize::{
    NormalizedOutput, normalize_insanely_fast, normalize_whisper_cpp, to_transcription_result,
};
use franken_whisper::model::{BackendKind, TranscriptionSegment};

// ---------------------------------------------------------------------------
// Fixture helpers
// ---------------------------------------------------------------------------

/// Build a whisper.cpp-style JSON payload with `n` segments.
fn whisper_cpp_json(n: usize) -> Value {
    let segments: Vec<Value> = (0..n)
        .map(|i| {
            json!({
                "start": i as f64 * 0.5,
                "end": i as f64 * 0.5 + 0.5,
                "text": format!("segment number {i}"),
                "probability": 0.92,
            })
        })
        .collect();

    let text: String = (0..n)
        .map(|i| format!("segment number {i}"))
        .collect::<Vec<_>>()
        .join(" ");

    json!({
        "text": text,
        "segments": segments,
        "result": { "language": "en" },
    })
}

/// Build an insanely-fast-whisper-style JSON payload with `n` chunks.
fn insanely_fast_json(n: usize) -> Value {
    let chunks: Vec<Value> = (0..n)
        .map(|i| {
            json!({
                "text": format!("chunk number {i}"),
                "timestamp": [i as f64 * 0.5, i as f64 * 0.5 + 0.5],
            })
        })
        .collect();

    let text: String = (0..n)
        .map(|i| format!("chunk number {i}"))
        .collect::<Vec<_>>()
        .join(" ");

    json!({
        "text": text,
        "chunks": chunks,
        "language": "en",
    })
}

/// Build an insanely-fast-whisper-style JSON payload with word-level
/// timestamps inside each chunk (batch output scenario).
fn insanely_fast_batch_json(num_chunks: usize, words_per_chunk: usize) -> Value {
    let chunks: Vec<Value> = (0..num_chunks)
        .map(|i| {
            let words: Vec<Value> = (0..words_per_chunk)
                .map(|w| {
                    let start = i as f64 * 2.0 + w as f64 * 0.2;
                    json!({
                        "word": format!("w{w}"),
                        "start": start,
                        "end": start + 0.15,
                    })
                })
                .collect();

            let chunk_text: String = (0..words_per_chunk)
                .map(|w| format!("w{w}"))
                .collect::<Vec<_>>()
                .join(" ");

            json!({
                "text": chunk_text,
                "timestamp": [i as f64 * 2.0, (i + 1) as f64 * 2.0],
                "words": words,
            })
        })
        .collect();

    json!({
        "text": "batch output",
        "chunks": chunks,
        "language": "en",
    })
}

// ---------------------------------------------------------------------------
// Benchmarks: normalize_whisper_cpp
// ---------------------------------------------------------------------------

fn bench_normalize_whisper_cpp(c: &mut Criterion) {
    let mut group = c.benchmark_group("normalize/whisper_cpp");

    for n in [1, 10, 100, 500] {
        let payload = whisper_cpp_json(n);
        group.bench_with_input(BenchmarkId::new("segments", n), &payload, |b, data| {
            b.iter(|| {
                normalize_whisper_cpp(data).expect("normalization should succeed");
            });
        });
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Benchmarks: normalize_insanely_fast
// ---------------------------------------------------------------------------

fn bench_normalize_insanely_fast(c: &mut Criterion) {
    let mut group = c.benchmark_group("normalize/insanely_fast");

    for n in [1, 10, 100, 500] {
        let payload = insanely_fast_json(n);
        group.bench_with_input(BenchmarkId::new("chunks", n), &payload, |b, data| {
            b.iter(|| {
                normalize_insanely_fast(data).expect("normalization should succeed");
            });
        });
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Benchmarks: normalize_insanely_fast with batch (word-level) outputs
// ---------------------------------------------------------------------------

fn bench_normalize_insanely_fast_batch(c: &mut Criterion) {
    let mut group = c.benchmark_group("normalize/insanely_fast_batch");

    for (chunks, words) in [(10, 5), (50, 10), (100, 20)] {
        let payload = insanely_fast_batch_json(chunks, words);
        let label = format!("{chunks}x{words}");
        group.bench_with_input(
            BenchmarkId::new("chunks_x_words", &label),
            &payload,
            |b, data| {
                b.iter(|| {
                    normalize_insanely_fast(data).expect("normalization should succeed");
                });
            },
        );
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Benchmarks: to_transcription_result conversion
// ---------------------------------------------------------------------------

fn bench_to_transcription_result(c: &mut Criterion) {
    let mut group = c.benchmark_group("normalize/to_transcription_result");

    for n in [0, 10, 100] {
        let segments: Vec<TranscriptionSegment> = (0..n)
            .map(|i| TranscriptionSegment {
                start_sec: Some(i as f64),
                end_sec: Some(i as f64 + 0.5),
                text: format!("segment {i}"),
                speaker: None,
                confidence: Some(0.9),
            })
            .collect();

        let normalized = NormalizedOutput {
            transcript: "benchmark transcript".to_owned(),
            segments,
            language: Some("en".to_owned()),
            raw_output: json!({"text": "benchmark"}),
        };

        group.bench_with_input(BenchmarkId::new("segments", n), &normalized, |b, data| {
            b.iter(|| {
                let _ = to_transcription_result(data.clone(), BackendKind::WhisperCpp);
            });
        });
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Criterion harness
// ---------------------------------------------------------------------------

criterion_group!(
    benches,
    bench_normalize_whisper_cpp,
    bench_normalize_insanely_fast,
    bench_normalize_insanely_fast_batch,
    bench_to_transcription_result,
);
criterion_main!(benches);
