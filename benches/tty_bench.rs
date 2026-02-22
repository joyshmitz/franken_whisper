//! Criterion benches for low-bandwidth TTY audio paths.
//!
//! Covers:
//! - frame encode throughput (`encode_to_writer`)
//! - frame decode throughput (`decode_frames_to_raw_with_policy`)
//! - control-frame serialization throughput (`emit_control_frame_to_writer`)

use std::io::Cursor;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};

use franken_whisper::tty_audio::{
    DecodeRecoveryPolicy, TtyControlFrame, decode_frames_to_raw_with_policy,
    emit_control_frame_to_writer, encode_to_writer,
};

const CHUNK_SIZES_MS: [u32; 2] = [20, 60];

fn ffmpeg_available() -> bool {
    Command::new("ffmpeg")
        .args(["-hide_banner", "-version"])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|status| status.success())
        .unwrap_or(false)
}

fn fixture_wav_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("audio")
        .join("test_1s_tone.wav")
}

fn encoded_fixture(chunk_ms: u32) -> Vec<u8> {
    let input = fixture_wav_path();
    let mut output = Vec::new();
    encode_to_writer(&input, chunk_ms, &mut output).expect("tty encode fixture should succeed");
    output
}

fn bench_tty_encode(c: &mut Criterion) {
    if !ffmpeg_available() {
        return;
    }

    let input = fixture_wav_path();
    let input_size = std::fs::metadata(&input).map(|m| m.len()).unwrap_or(0);
    let mut group = c.benchmark_group("tty/encode");
    group.throughput(Throughput::Bytes(input_size));

    for chunk_ms in CHUNK_SIZES_MS {
        group.bench_with_input(
            BenchmarkId::new("chunk_ms", chunk_ms),
            &chunk_ms,
            |b, &chunk| {
                b.iter(|| {
                    let mut output = Vec::new();
                    encode_to_writer(&input, chunk, &mut output)
                        .expect("tty encode should succeed");
                    output.len()
                });
            },
        );
    }

    group.finish();
}

fn bench_tty_decode(c: &mut Criterion) {
    if !ffmpeg_available() {
        return;
    }

    let mut group = c.benchmark_group("tty/decode");

    for chunk_ms in CHUNK_SIZES_MS {
        let encoded = encoded_fixture(chunk_ms);
        group.throughput(Throughput::Bytes(encoded.len() as u64));
        group.bench_with_input(
            BenchmarkId::new("chunk_ms", chunk_ms),
            &encoded,
            |b, payload| {
                b.iter(|| {
                    let mut reader = Cursor::new(payload.as_slice());
                    let (report, raw) = decode_frames_to_raw_with_policy(
                        &mut reader,
                        DecodeRecoveryPolicy::FailClosed,
                    )
                    .expect("tty decode should succeed");
                    assert!(report.frames_decoded > 0);
                    raw.len()
                });
            },
        );
    }

    group.finish();
}

fn bench_tty_control_emit(c: &mut Criterion) {
    let mut group = c.benchmark_group("tty/control_emit");

    let fixtures: [(&str, TtyControlFrame); 6] = [
        (
            "handshake",
            TtyControlFrame::Handshake {
                min_version: 1,
                max_version: 1,
                supported_codecs: vec!["mulaw+zlib+b64".to_owned()],
            },
        ),
        (
            "handshake_ack",
            TtyControlFrame::HandshakeAck {
                negotiated_version: 1,
                negotiated_codec: "mulaw+zlib+b64".to_owned(),
            },
        ),
        ("ack", TtyControlFrame::Ack { up_to_seq: 128 }),
        (
            "backpressure",
            TtyControlFrame::Backpressure {
                remaining_capacity: 32,
            },
        ),
        (
            "retransmit_request",
            TtyControlFrame::RetransmitRequest {
                sequences: vec![4, 5, 6, 99],
            },
        ),
        (
            "retransmit_response",
            TtyControlFrame::RetransmitResponse {
                sequences: vec![4, 5, 6, 99],
            },
        ),
    ];

    for (name, frame) in fixtures {
        group.bench_with_input(BenchmarkId::new("frame", name), &frame, |b, control| {
            b.iter(|| {
                let mut out = Vec::new();
                emit_control_frame_to_writer(&mut out, control)
                    .expect("control frame emit should succeed");
                out.len()
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_tty_encode,
    bench_tty_decode,
    bench_tty_control_emit
);
criterion_main!(benches);
