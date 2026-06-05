# Hotspot Table — native engine baseline (run 20260605T0218Z)

Scenario S2 = large-v3-turbo, jfk.wav 11s, 8 threads, sole-stage native, release-perf.
Scenario S1 = tiny.en, same. Baselines: S1 1.225s ± 0.029 (15 runs); S2 32.92s ± 0.16 (5 runs).
whisper-cli same host: S1 0.57s CPU; S2 9.63s CPU / 2.11s Metal.

| Rank | Location | Metric | Value | % of S2 | Category | Evidence |
|------|----------|--------|-------|---------|----------|----------|
| 1 | tail-window full-context encode (decode.rs window loop → encoder::forward) | wall | 9.28s for 0.24s of real audio (5 tokens) | 28% | CPU/waste | spans_large_timeline.txt at_ms 31079 |
| 2 | duplicate window-0 encode in resolve_language (decode.rs lang auto-detect) | wall gap | 8.83s (mel@2469 → enc1 start@11299) | 26% | CPU/waste | spans_large_timeline.txt gap |
| 3 | GEMM thread starvation: __psynch_cvwait 180,005 top-of-stack samples vs sgemm NEON 71k (~22%) | idle | ~3.6 effective cores of 14; serial glue (layer_norm, softmax, attention head loops, im2col) between fork-join matmuls | pervasive | lock/sched | sample_large.txt top-of-stack |
| 4 | version_tag sha256 over 1.6GB model at raw_output build | wall | 3.04s | 9% | CPU (serial) | spans_large_timeline.txt version_tag |
| 5 | model_weights f16→f32 dequant + pre-transpose (LoadedModel::from_ggml) | wall | 2.31-2.44s | 7% | CPU (serial) | spans_*.jsonl model_weights |
| 6 | decode_loop per-token cost: S1 23ms/token (616ms/27), S2 ~61ms/token | wall | S1: 50% of total | 50% S1 | CPU/alloc | spans_tiny.en.jsonl; per-call Vec alloc in ft matmul wrapper; serial GEMV |
| 7 | expf/tanhf (softmax/gelu transcendentals) | samples | 4.4k/322k (~1.4%) | 1.4% | CPU | sample_large.txt |

# Hypothesis Ledger

| hypothesis | verdict | evidence |
|---|---|---|
| H1 matmul kernels themselves are slow | rejects (mostly) — matrixmultiply NEON kernels are the top real-work symbol; the problem is FEEDING them (55% cvwait idle) | sample_large.txt |
| H2 decoder GEMV doesn't parallelize | supports — 23ms/token on tiny where ~2ms of MACs exist; rayon fork-join + per-call allocs dominate small shapes | spans_tiny.en.jsonl, sample |
| H3 f16→f32 at load is significant | supports — 2.4s serial (7%) on large; also doubles memory bandwidth during inference (f32 weights) | spans model_weights |
| H4 logits matmul per token is heavy | partially — included in decode_loop; [51866,1280]x[1280,1] ≈ 66M MACs/token ≈ real work; needs parallel GEMV not serial | arithmetic + spans |
| H5 encoder conv/im2col | minor — conv1d self-samples tiny (15 top-of-stack) | sample_large.txt |
| H6 mel FFT | rejects — 31ms total | spans |
| H7 hidden duplicate encode (lang detect) | supports — 8.8s timeline gap exactly matches one encoder_window | timeline |
| H8 sha256 version tag | supports — 3.0s span | timeline |
| H9 tail windows waste full encodes | supports — window 2: 9.3s for 5 tokens; whisper.cpp's audio_ctx truncation is the sanctioned fix | timeline |

Golden output constraint: S1 transcript must stay byte-identical to jfk_tiny_reference.json; S2 segments within existing conformance tolerances.
