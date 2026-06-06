# SCALE HOTSPOTS — Round 3 Pass 1 (measurement-only)

Host: Apple M4 Pro (10P+4E / 14 core), 64 GB, macOS 26.2, `release-perf`, git
`d29e0b7`. Fixtures: 2 h real call recording (`/tmp/gerard_call/call_16k.wav`,
7267 s) + 600 s / 60 s slices. Env: native sole-stage, `PERF_SPANS=1`.

**Discipline note (load-robustness):** the host carried a heavy agent swarm the
whole time (load avg 12 → 100+). The **2 h tiny run** and the first
large/insanely-fast batch ran *concurrently* (load 50–100), so their absolute
walls / ms-per-tok are inflated by oversubscription — only their within-run span
**ratios** and **drift ratios** are quoted as conclusions. The `clean_*` 600 s
pairs were run one-at-a-time (load 16–30) and are the trustworthy absolute
seq-vs-IF numbers. mel/encoder per-second-of-audio at low load (60 s & 600 s) is
the load-clean scaling signal.

## Ranked SCALE hotspot table

| # | Location | Metric | Value | % of run | Evidence file |
|---|----------|--------|-------|----------|---------------|
| 1 | `decode.rs` greedy `decode_loop` (token-by-token `decoder::forward_step`) | share of span sum, **tiny / long audio** | **444.75 s of 601.5 s** | **74 %** (tiny 2 h); 75 % (tiny 600 s) | `spans_tiny_7200_seq.log` |
| 2 | `encoder::forward` per window (`encoder_window`, GEMM) | share of span sum, **large model** | **262.8 s of 347.6 s** (contended) / clean 76 % | **76 %** (large 600 s) | `spans_large_600_seq.log` |
| 3 | `mel::log_mel` full-audio mel + `read_normalized_wav` f32 buffer | peak RSS driver vs duration | RSS 465 → 622 → **2048 MB** (60 s→600 s→2 h tiny); **5.4 GB** (large 600 s) | RSS, not wall | `spans_tiny_7200_seq.log`, `baseline_matrix.json` |
| 4 | `DecoderState::new` cross-K/V build per window (`cross_kv`) | share, both models | 2.8 % tiny / 4 % large | small | `spans_*_600_*.log` |
| 5 | `decoder_prefill` (prompt re-encode each window) | share + prompt-growth | 2.2 % tiny / 2.7 % large; **LINEAR in prompt len, capped** | small | `spans_tiny_7200_seq.log` |
| 6 | `mel::log_mel` compute | wall share | **~1.0–1.8 ms / s-of-audio, 2.1 % of run** | tiny only | `spans_tiny_7200_seq.log` |
| 7 | Post-`backend_run` tail (raw_output_json + RunReport + CLI serialize) | wall − backend_run end | **2.1 s of 604 s (0.35 %)** tiny 2 h; **0.39 s (0.25 %)** large 600 s | NOT material | `spans_tiny_7200_seq.log`, `spans_large_600_seq.log` |

### What dominates depends on (model, audio length):
- **tiny.en, long audio → decode_loop (token-bound).** The encoder is cheap on
  tiny; the bottleneck is the inherently *sequential* greedy per-token forward
  pass. ~5–6 ms/tok at low load × 25 k tokens for a 2 h call.
- **large-v3-turbo → encoder GEMM (compute-bound).** 76 % of the run regardless
  of length; decode is only 17 %. Matches Rounds 1–2: encoder is sgemm-bound.

## Scaling laws (tiny.en sequential)

| dur | wall_s | RTF | enc/s-audio | loop/s-audio | mel/s-audio | RSS_MB | win | tok | ms/tok |
|----:|-------:|----:|------------:|-------------:|------------:|-------:|----:|----:|-------:|
| 60   | 2.56   | 0.043 | 4.50 ms | 33.5 ms | 1.17 ms | 465  | 3   | 396   | 5.07 |
| 600  | 15.94  | 0.027 | 4.37 ms | 22.9 ms | 1.05 ms | 622  | 23  | 2298  | 5.97 |
| 7200 | 604.24 | 0.084 | 15.83 ms*| 61.8 ms* | 1.77 ms* | 2048 | 297 | 25348 | 17.55* |

\* 2 h row CONTENDED (load 50–100). Per-second-of-audio for enc/loop/mel is
~constant at low load (60/600 s) ⇒ **wall is LINEAR in duration; no super-linear
bend.** The 2 h inflation is host contention, not an algorithmic scaling wall.

- **mel:** linear, ~1 ms per second of audio, 2 % of run. The full mel is one
  `n_mel × n_frames × 4 B` buffer (tiny 80×727k×4B ≈ **232 MB** at 2 h; large
  128×727k×4B ≈ **372 MB**). `chunk_frames` copies one 30 s window (≈ 1 MB) per
  window — negligible. mel is NOT a scale hotspot.
- **RSS vs duration (sub-linear total, driven by two duration-linear buffers):**
  full-audio f32 sample buffer (`read_normalized_wav`: 2 h tiny ≈ 465 MB) +
  full mel (232 MB) + model (74 MB tiny / 1.5 GB large) + per-window
  encoder/decoder scratch churn. tiny 2 h peak = **2.05 GB**; large 600 s =
  **5.4 GB** (the large encoder activations + 1.5 GB model dominate). RSS/s-audio
  DROPS with length (fixed costs amortize) but absolute RSS grows linearly with
  the two big audio buffers.
- **prefill / prompt growth: NOT quadratic.** prefill ms rises with prompt
  length (18 ms @ <20 tok → 107 ms @ 120–139 tok) but is **capped** by
  `max_prompt_ctx = n_text_ctx/2` (mean prompt 39, max 131 tok), so it stays a
  bounded ~2–3 % of the run at any length. No O(n²) blowup over a 2 h run.

## Sequential vs insanely-fast (clean 600 s pairs, load 16–30)

| | tiny.en seq | tiny.en IF | large-v3-turbo seq | large-v3-turbo IF |
|--|---:|---:|---:|---:|
| wall_s | **15.94** | 22.85 | **154.08** | 167.49 |
| RSS_MB | **622** | 918 | **5407** | 6241 |
| user_s | 50.5 | 80.0 | 1059 | 1392 |

**VERDICT: insanely-fast (hard-window parallel) is SLOWER on this 14-core host at
both model sizes** — tiny ×1.43, large ×1.09 wall — and costs 1.15–1.45× RSS and
+31 %/+58 % user CPU. Cause: the sequential path already saturates all cores via
intra-op (rayon) parallelism on each encoder GEMM / decoder GEMV; IF then layers
`n_workers × threads_per_worker` *on top*, oversubscribing the 14 cores
(`plan_workers` gives ~7 workers × 2 threads = 14, but under ambient load that is
pure context-switch thrash). IF wins only when cores would otherwise be idle —
not on a busy host, and not when intra-op parallelism already fills the machine.

**Determinism / text-diff tradeoff (seq is the reference, byte-golden path):**
- large 600 s: **27.7 % word-diff rate** (seq 1660 w vs IF 1507 w).
- tiny 600 s: **67 % word-diff rate**, and IF emitted **1666 words vs 803** —
  ~2× over-generation. The hard 30 s window cuts mid-utterance and IF carries
  **no rolling prompt** (documented in `insanely_fast_native.rs`), so tiny IF
  hallucinates/repeats badly at window seams.
⇒ insanely-fast trades a large accuracy regression for a wall **regression** on
this host. It is not a scale win here.

## Hypothesis ledger (progress-file scale suspects → verdict with numbers)

| Suspect (from `.skill-loop-progress.md`) | Verdict | Evidence |
|---|---|---|
| `full_mel` memory + slicing at 720 k frames | **REJECT as hotspot.** mel = 2 % of run, ~1 ms/s linear; full mel buffer 232 MB (tiny) / 372 MB (large) is real but `chunk_frames` copy is ~1 MB/window. Not a wall or RSS bottleneck. | `spans_tiny_7200_seq.log` |
| per-window `prompt_past` growth → prefill cost | **REJECT (bounded, linear).** prefill capped by `max_prompt_ctx`; 18→107 ms across prompt buckets; 2–3 % of run; no O(n²). | prefill-bucket analysis |
| window-loop fixed overheads × 240 | **REJECT.** cross_kv + prefill + per-window glue = ~5 % combined; model load amortized to <0.1 % at 2 h. | `spans_tiny_7200_seq.log` |
| insanely-fast worker policy + memory at 240 windows (only tested at 2) | **CONFIRMED PROBLEMATIC.** IF slower (×1.09–1.43) + 1.15–1.45× RSS + accuracy regression on this host; oversubscription. | clean 600 s pairs |
| RunReport / segments JSON serialization at 1000+ segments | **REJECT.** post-backend tail = 0.35 % (2 h) / 0.25 % (large). In-backend wav-read+seg-build = 0.81 s of 604 s. | backend_run at_ms vs wall |
| events vec growth | **REJECT.** subsumed by the immaterial post-backend tail. | as above |
| DTW recording memory (word timestamps) | **NOT EXERCISED** (word_timestamps off in this matrix). Flag for a future pass if word-ts is a real use case. | n/a |
| sequential-seek vs hard-window parallel throughput at scale | **MEASURED:** seq wins on this host (see seq-vs-IF table). | clean pairs |
| tiny self_qkv spawn-bound (r2 leftover, 5.2 ms/tok floor) | **STILL THE FLOOR.** tiny decode_loop = 5.07–5.97 ms/tok at low load = the r2 floor; this is hotspot #1 for long tiny audio and the obvious pass-3 target. | `spans_tiny_60_seq.log`, `spans_tiny_600_seq.log` |

## Recommended pass-2 / pass-3 targets (no changes made here)
1. **tiny.en `decode_loop` per-token cost** (hotspot #1 for long audio): the 5–6
   ms/tok serial greedy loop × 25 k tokens IS the 2 h tiny wall. r2 already
   flagged self_qkv spawn overhead at the 5.2 ms/tok floor — the dominant scale
   lever for the real (tiny, long-call) use case.
2. **large encoder GEMM** (hotspot #2): unchanged from r1/r2 — compute-bound
   sgemm; out of in-crate reach per r2's ft-kernel rejection, but it is 76 % of
   any large run.
3. **Do NOT route long audio through insanely-fast on a busy multi-core host** —
   it regresses wall, RSS, and accuracy here.
