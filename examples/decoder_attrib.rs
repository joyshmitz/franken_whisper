//! Per-sub-part attribution harness for the decoder token step (pass 5).
//!
//! Loads a whisper ggml model by short name, builds a real encoder-derived
//! `DecoderState` from the jfk window, then runs `STEPS` incremental
//! `forward_step`s and prints the per-sub-part wall-time split that the
//! decoder's measurement-only instrumentation accumulates (drained via
//! `decoder::take_sub_ns`). This is the ATTRIBUTE deliverable: it answers
//! "where do the ~ms/token go?" against the real large weights, in-process,
//! without the host-load noise of an e2e wall A/B (each sub-part is summed
//! over many steps, so the per-step figure is the mean of `STEPS` samples).
//!
//! The instrumentation is gated by `FRANKEN_WHISPER_PERF_SPANS=1`; this harness
//! sets it programmatically is NOT possible (it is read once via OnceLock), so
//! run with the env var set:
//!
//! ```text
//! FRANKEN_WHISPER_PERF_SPANS=1 FRANKEN_WHISPER_MODEL_DIR=/path \
//!   cargo run --profile release-perf --example decoder_attrib -- large-v3-turbo 200
//! ```
//!
//! A second positional arg overrides the step count (default 200). The f16
//! compute switch (`FRANKEN_WHISPER_NATIVE_F16_COMPUTE`) is honored, so the
//! same harness attributes both the f16-ON (default) and f32 paths.

use franken_whisper::native_engine::decode::LoadedModel;
use franken_whisper::native_engine::decoder::{self, DecoderState, SUB_COUNT, SUB_LABELS};
use franken_whisper::native_engine::encoder;
use franken_whisper::native_engine::ggml::GgmlModel;
use franken_whisper::native_engine::mel::{self, FRAMES_PER_CHUNK};
use franken_whisper::native_engine::{Mat, find_model_file};

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
    let model_name = args.next().unwrap_or_else(|| "large-v3-turbo".to_string());
    let steps: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(200);

    if std::env::var("FRANKEN_WHISPER_PERF_SPANS").as_deref() != Ok("1") {
        eprintln!(
            "WARNING: FRANKEN_WHISPER_PERF_SPANS is not '1'; the per-sub-part \
             instrumentation is OFF and the table will be all zeros. Re-run with \
             FRANKEN_WHISPER_PERF_SPANS=1."
        );
    }

    let path = find_model_file(&model_name)
        .unwrap_or_else(|| panic!("model {model_name} not found in search dirs"));
    let model =
        LoadedModel::from_ggml(GgmlModel::load(&path).expect("load ggml")).expect("from_ggml");
    let samples = load_jfk_samples();

    // Real encoder output → real cross-K/V, exactly the decode path.
    let full = mel::log_mel(&samples, &model.filters, 8).expect("log_mel");
    let window = mel::chunk_frames(&full, 0, FRAMES_PER_CHUNK);
    let noop = || Ok(());
    let enc_out = encoder::forward(&model.encoder, &window, 8, &noop).expect("encoder forward");

    let w = &model.decoder;
    let sot = model.tokenizer.sot;
    let n_state = w.n_state();
    let n_head = w.n_head();
    let n_layer = w.n_layer();
    let n_vocab = w.n_vocab();
    let enc_frames = enc_out.rows;
    let f16 = std::env::var("FRANKEN_WHISPER_NATIVE_F16_COMPUTE").unwrap_or_default();

    let mut st = DecoderState::new(w, &enc_out).expect("decoder state");

    // Warm up (page-in weights, settle caches) without counting it: run a short
    // sequence, then drain+discard the accumulator.
    st.reset();
    for i in 0..8 {
        let tok = if i == 0 { sot } else { 1 + i };
        decoder::forward_step(w, &mut st, &[tok], &noop).expect("warmup step");
    }
    let _ = decoder::take_sub_ns();

    // Measured run: a fresh sequence of `steps` tokens from a reset cache. The
    // cache grows 0..steps, so the per-step figure is the mean marginal cost
    // over that depth range — the same quantity the criterion bench averages,
    // but here split by sub-part.
    st.reset();
    let _ = decoder::take_sub_ns();
    let t_all = std::time::Instant::now();
    for i in 0..steps {
        let tok = if i == 0 { sot } else { 1 + (i as i32 % 64) };
        decoder::forward_step(w, &mut st, &[tok], &noop).expect("measured step");
    }
    let total_ms = t_all.elapsed().as_secs_f64() * 1e3;
    let sub_ns = decoder::take_sub_ns();

    let sum_ns: u128 = sub_ns.iter().sum();
    let per_step_ms = total_ms / steps as f64;
    let attributed_ms = sum_ns as f64 / 1e6;
    let attributed_per_step_ms = attributed_ms / steps as f64;

    println!("# decoder token-step attribution");
    println!(
        "model={model_name} f16={f16:?} steps={steps} n_state={n_state} n_head={n_head} \
         n_layer={n_layer} n_vocab={n_vocab} enc_frames={enc_frames}"
    );
    println!(
        "wall total={total_ms:.1}ms  per-step={per_step_ms:.3}ms  \
         attributed/step={attributed_per_step_ms:.3}ms  \
         (unattributed/overhead={:.3}ms)",
        per_step_ms - attributed_per_step_ms
    );
    println!();
    println!("{:<20} {:>12} {:>9}", "sub-part", "ms/token", "% of attr");
    println!("{:-<20} {:->12} {:->9}", "", "", "");
    let mut rows: Vec<(usize, f64)> = (0..SUB_COUNT)
        .map(|i| (i, sub_ns[i] as f64 / 1e6 / steps as f64))
        .collect();
    rows.sort_by(|a, b| b.1.total_cmp(&a.1));
    for (i, ms) in rows {
        let pct = if attributed_ms > 0.0 {
            100.0 * sub_ns[i] as f64 / sum_ns as f64
        } else {
            0.0
        };
        println!("{:<20} {ms:>12.4} {pct:>8.1}%", SUB_LABELS[i]);
    }
    println!("{:-<20} {:->12} {:->9}", "", "", "");
    println!(
        "{:<20} {attributed_per_step_ms:>12.4} {:>8.1}%",
        "TOTAL (attr)", 100.0
    );

    // Black-box the state so nothing is optimized away.
    std::hint::black_box(&st);
    let _ = Mat::zeros(1, 1);
}
