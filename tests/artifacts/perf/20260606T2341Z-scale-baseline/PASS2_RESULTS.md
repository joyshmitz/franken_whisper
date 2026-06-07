# Round 3 Pass 2 — tiny.en per-token decode floor (self_qkv spawn elimination)

Host: Apple M4 Pro (10P+4E), 64 GB, macOS 26.2, `release-perf`. HEAVY ambient
load throughout (load avg 13–30); criterion guard benches that share an unchanged
code path drifted with load, so the load-robust evidence is (a) the within-instant
serial-vs-`thread::scope` probe, (b) the within-run decoder_attrib sub-part split,
and (c) the e2e 600 s decode_loop ms/tok (deterministic token count).

## Root cause (confirmed)
`nn::gemv_f16` had `PAR_THRESHOLD = 1<<16` (65 536 MACs). Every tiny.en per-token
Linear crosses it, so each `[1,384]x[384,384]` GEMV (147 k MACs ≈ ~9 µs compute)
spawned an 8-way `std::thread::scope` — pure spawn/join overhead on µs of work.
self_qkv runs three of these (nested under `project_qkv`'s 2-thread split).

Measured serial-vs-parallel `gemv_f16` crossover (same binary, µs apart):
| shape (out,inp) | MACs | parallel | serial | winner |
|---|---|---|---|---|
| 384,384   |  147 k | 90 µs | 26 µs | **serial 3.5×** |
| 768,384   |  295 k | 101 µs | 53 µs | serial 1.9× |
| 1024,384  |  393 k | 104 µs | 70 µs | serial 1.5× |
| 1536,384  |  590 k | 109 µs | 115 µs | break-even |
| 384,1536  |  590 k | 113 µs | 122 µs | break-even |
| 1280,1280 | 1.64 M | 154 µs | 355 µs | parallel 2.3× |
| 51864,384 | 19.9 M | 932 µs | 3616 µs | parallel 3.9× |

## Lever 2 (LANDED): raise `gemv_f16` / `gemv_f16_batch` PAR_THRESHOLD 1<<16 → 1<<19
`524 288` sits in the break-even band: every tiny `[384,384]` per-token Linear
(self q/k/v, self_out, cross_q, cross_out) goes serial; large-model Linears and
the logits GEMV stay parallel. Pure scheduling knob (disjoint row bands,
band-independent `dot8` order) → bit-identical.

### tiny.en attribution (decoder_attrib, FRANKEN_WHISPER_PERF_SPANS=1, 400 steps, f16 ON)
| sub-part | before ms/tok | after ms/tok |
|---|---|---|
| self_qkv_proj | 1.38 (#1, 24.6%) | **0.32 (#5, 7.8%) −77%** |
| self_out_proj | 0.39 | 0.14 |
| cross_q_proj  | 0.39 | 0.14 |
| cross_out_proj| 0.39 | 0.14 |
| TOTAL/step    | **5.63** | **4.18  (−26%)** |
New top floor (all already-parallel compute): mlp_fc_gelu 1.00, logits 0.94,
cross_attn 0.84, self_attn 0.65.

### large-v3-turbo attribution (200 steps): behaviorally unchanged
self_qkv stays the parallel path (1.6 M-MAC shapes > threshold); 15.26 ms/tok at
load ~20 (load-inflated vs the ~10.3 ms/tok r2 low-load figure) — no regression.

### e2e corroboration (600 s tiny slice, native sole-stage, single pass each)
| | pre | post |
|---|---|---|
| decode_loop ms/tok | 6.83 | **3.83 (−44%)** |
| decode_loop total  | 15 687 ms | 8 809 ms |
| tokens (deterministic) | 2298 | 2298 |
| wall | 21.4 s | 14.7 s |
| peak RSS | 632 MB | 631 MB |
(pre ran at lower load than post, so the win is if anything understated; encoder —
unchanged code — moved only −6% vs decode_loop −44%.)

### criterion (--baseline round3-pre; load-contaminated, directional only)
decoder_token_step_tiny −30.8% (62.2→43.0 ms/8-step). The guard benches
(logits_gemv_large, f16_gemv_1280x1280, decoder_token_step_large) also "improved"
~17–28% on the SAME unchanged code path → that delta is load drift; the tiny win
above the drift floor is the real signal, corroborated by probe + attribution + e2e.

## Lever 3 (ABANDONED): persistent rayon pool for project_qkv
A `rayon::ThreadPool` + `rayon::scope` (persistent pool, no per-call spawn) was
probed against `std::thread::scope` for the project_qkv 2-task fan-out at the tiny
shape: **rayon 77–83 µs vs std 60–64 µs — rayon LOSES.** Work-stealing deque +
scope-join overhead exceeds plain std spawn for a fixed 2-way split of µs tasks.
Does not beat Lever 2; not landed; no rayon dependency added.
project_qkv's own 2-spawn split was separately probed and KEPT: even with serial
inner GEMVs, 2-spawn (60 µs) beats fully-serial 3× (82 µs).

## Lever 4 (NOT PURSUED): no Score≥2.0 target left
Post-lever-2 the remaining floor is all already-parallel compute (mlp/logits/
cross_attn/self_attn); the LN+clone/embed/final_ln sub-parts are each ≤0.1%.

## Gates
- Golden: tiny.en + large-v3-turbo jfk transcripts BYTE-IDENTICAL to /tmp/fw_golden.
- Tests: `cargo test --lib native_engine -- gated` 202/202; `native_engine::nn` 26/26.
- fmt --check clean; clippy --all-targets -D warnings clean; check --all-targets clean.

## Files changed
- src/native_engine/nn.rs — PAR_THRESHOLD 1<<16 → 1<<19 in gemv_f16 + gemv_f16_batch (+ docs).
- benches/native_engine_bench.rs — added f16_gemv_dequant_384x384 (tiny per-token shape instrument).
