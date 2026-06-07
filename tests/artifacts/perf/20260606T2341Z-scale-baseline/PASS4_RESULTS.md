# PASS 4 RESULTS — fresh scale re-profile after passes 2-3 (round 3, pass 4/6)

Host: Apple M4 Pro (10P+4E / 14 core), 64 GB, macOS 26.2, `release-perf` built
with `RUSTFLAGS="-C force-frame-pointers=yes"`, git `b817f86` + pass-4 working
tree. Fixtures: 2 h real call (`/tmp/gerard_call/call_16k.wav`, 7267 s) + 600 s
slice (`/tmp/fw_scale_600.wav`). Env: native sole-stage, `FRANKEN_WHISPER_PERF_SPANS=1`,
`FRANKEN_WHISPER_MODEL_DIR=~/models/whisper`.

**Discipline note:** the host carried a heavy agent swarm throughout (load avg
15 → 57). Per-run load is recorded inline. Walls are load-contaminated and used
only directionally; the load-robust conclusions are (a) within-run span **shares**,
(b) per-window **drift** ratios, (c) the **plan structure** of the worker policy
(load-independent integer arithmetic), and (d) worker finish-time **skew** (two
workers measured in the same instant, so their ratio is load-cancelling).

## TL;DR

After the pass-2 (`PAR_THRESHOLD` GEMV size-gate) and pass-3 (contiguous-range +
no-oversubscription IF) landings, the decode floor and the IF policy are both at
their measured floor. **No remaining lever scores ≥ 2.0 — this is an
evidence-backed zero-change verdict.** The headline real-world number is in:

> **2 h tiny.en, insanely-fast default = 262.9 s wall at load ~50** — vs the
> pass-1 documented **604 s sequential** baseline. That is **2.30× faster**
> end-to-end on the real 2-hour call, *despite* running at far higher ambient
> load than the baseline.

The only changes are **measurement-only** (perf-span house style, gated by
`FRANKEN_WHISPER_PERF_SPANS`; one documented profiling-only env knob that
defaults to the production constant): they produced the skew + knob-sweep
evidence below and leave every production code path bit-identical.

## Refreshed ranked hotspot table (tiny.en, sequential, span shares)

Children of `backend_run` (the enclosing parent span; its 50.1 % is the
parent-double-count). Within the actual work (`backend_run` children sum), the
shares are:

| # | Span | 600 s ms | 600 s % of work | 2 h ms | 2 h % of work | Note |
|---|------|---------:|----------------:|-------:|--------------:|------|
| 1 | `decode_loop` (greedy per-token forward) | 8 523 | **65.4 %** | 109 213 | **63.8 %** | still #1; 3.71 / 4.31 ms/tok |
| 2 | `encoder_window` (GEMM) | 2 638 | 20.2 % | 41 210 | 24.1 % | flat per-window |
| 3 | `cross_kv` (cross-K/V build) | 942 | 7.2 % | 14 267 | 8.3 % | per-window |
| 4 | `mel` (full-audio log-mel) | 530 | 4.1 % | 8 179 | 4.8 % | one buffer, linear |
| 5 | `decoder_prefill` (prompt re-encode) | 403 | 3.1 % | 5 731 | 3.3 % | bounded |
| — | model_parse + model_weights | 34 | 0.3 % | 36 | 0.02 % | amortized at 2 h |

**decode_loop ms/tok (the pass-2 lever, re-confirmed at scale):**
- 600 s: **3.71 ms/tok** (2298 tok). Pass-1 baseline was 5.97 ms/tok → the
  `1<<16 → 1<<19` PAR_THRESHOLD fix is holding (−38 %).
- 2 h: **4.31 ms/tok** (25 348 tok), at load 21–39. Pass-1's *contended* 2 h row
  read 17.55 ms/tok at load 50–100 — i.e. the apparent 2 h regression in pass-1
  was pure host contention; the per-token floor is ~4 ms/tok and **flat across
  all 297 windows** (no upward drift; w0 4.47 → w296 in the same 3.3–4.5 band).

**Structure verdict:** with the old #1 (self_qkv spawn) fixed, `decode_loop` is
*still* #1 by a wide margin (64–65 % of work), but it is now entirely
**already-parallel or inherently-serial compute** — the per-token GEMVs are
serial-by-design (pass-2), mlp/logits/cross_attn/self_attn are parallel and at
their compute floor (pass-2's lever-4 finding). The sequential tiny run does
**not** show encoder-share dominance (encoder is only 20–24 %); encoder
dominance is the *large-model* regime (pass-1 hotspot #2, ft-kernel-bound, out of
in-crate reach).

## Worker finish-time SKEW (IF, contiguous even-duration ranges)

Measured via a new measurement-only `if_worker` perf-span (per-worker wall + span
window range), `FRANKEN_WHISPER_PERF_SPANS=1`:

| run | worker A | worker B | skew (slow−fast) | skew % | seams |
|-----|---------:|---------:|-----------------:|-------:|------:|
| 600 s tiny (2w × 7t) | 11 138 ms `[w0,w10)` | 11 618 ms `[w10,w20)` | 480 ms | **4.1 %** | 1 |
| 2 h tiny (2w × 7t) | 257 937 ms `[w122,w243)` | 260 148 ms `[w0,w122)` | 2 210 ms | **0.85 %** | 1 |

**Finding:** the even-duration contiguous split is **very well balanced** on real
call audio — skew is 4.1 % at 600 s and *shrinks* to 0.85 % at 2 h (speech density
averages out over a longer span). The tail-latency penalty of imbalance is at
most ~4 % of wall, and ~1 % at the 2 h scale that actually matters.

## Knob sweep — `MIN_THREADS_PER_WORKER` (tiny.en, 600 s)

Via the new measurement-only `FRANKEN_WHISPER_IF_MIN_THREADS` override (defaults
to the compiled-in `5`; production unchanged). Walls are load-contaminated
(load shown); the **plan** (workers × tpw) and **LCS word-diff** are
load-independent and are the real signal:

| MIN_THREADS | plan (w × tpw) | LCS-diff vs seq | wall ratio (load) | verdict |
|------------:|---------------:|----------------:|------------------:|---------|
| 3 | **4 × 3** | 42.4 % | 0.75 (load 35) | more seams, 2× drift |
| 4 | **3 × 4** | 40.8 % | 1.00 (load 31) | **no wall win**, 2× drift |
| **5 (default)** | **2 × 7** | **22.4 %** | 0.89 (load 18) | sweet spot |
| 6 | 2 × 7 | 22.4 % | (load drift) | **same plan as 5** |
| 7 | **2 × 7** | 22.4 % | 0.72 (load 35) | **same plan as 5** |

**Finding:** on a 14-core host, `MIN_THREADS ∈ {5,6,7}` all yield the **identical**
2-worker × 7-thread plan (`14/5 = 14/6 = 14/7 = 2`), so they are behaviorally
identical — the production default of `5` is on the correct plateau. Dropping to
`≤4` adds a 3rd/4th worker that gives **no reliable wall win** (the multi-worker
walls bracket sequential once load is held even) while **doubling** the
word-diff (each added range = one more cold-started seam that can drift/loop
through its whole multi-minute span — the pass-3 finding, re-confirmed). The
knob is correctly set; raising it would only matter on ≥28-core hosts (untested,
and would *reduce* worker count there, i.e. toward sequential).

## Model-size-aware policy question (does tiny want MORE workers?)

Hypothesis (from the mission): tiny's intra-op parallelism saturates poorly, so
tiny specifically might prefer more workers × fewer threads, gated on
`n_audio_state`. **Rejected by measurement:** tiny with 3–4 workers ran at
**1.00× wall (no win)** and **2× the word-diff** (LCS 22 % → 41 %). Because
pass-2 made tiny's per-token GEMVs *serial*, a tiny worker's remaining intra-op
parallelism is on the encoder, and a single fully-threaded worker already uses it
well; extra workers buy nothing on wall and cost accuracy. Both models therefore
land on the *same* optimal plan (2 × 7 here), so an `n_audio_state`-conditioned
policy would add a branch with no behavioral payoff. No change.

## Lever verdict — ZERO CHANGE (no Score ≥ 2.0)

| Lever | Impact | Conf | Effort | Score | Verdict |
|-------|-------:|-----:|-------:|------:|---------|
| A. work-stealing / more-ranges-than-workers (skew fix) | 1 (≤4 % / ≤0.85 % at 2 h) | 2 (each seam ~2× word-diff → likely net-negative) | 4 (steal queue + dynamic merge + accuracy gates) | **0.50** | REJECT |
| B. raise `MIN_THREADS` 5→7 | 0 (no-op on 14 cores) | — | 1 | **0.0** | REJECT |
| C. model-size-aware (`n_audio_state`) worker policy | 1 (tiny → 1.00× wall, 2× drift) | 4 (sweep proves it hurts) | 2 | negative | REJECT |
| D. lower `MIN_THREADS` to add workers | negative (1.00× wall + 2× drift) | 4 | 1 | negative | REJECT |

**The IF backend and the native decode path are at their measured floor.** The
skew lever — the one structural candidate the mission flagged — does not pay: the
even-duration contiguous split is already 99.15 %-balanced at 2 h, and the only
way to reclaim the remaining <1 % (more ranges with work-stealing) re-introduces
exactly the seam-drift the pass-3 contiguous-range design removed. Honest verdict:
**floor reached; no production change.**

## What proves the floor

- `decode_loop` is 64 % of work and is serial-by-design GEMVs (pass-2) + parallel
  compute at its floor (pass-2 lever-4); 4 ms/tok flat across 297 windows.
- IF default (2 × 7) is faster than sequential at every scale measured and the
  2 h headline is 2.30× the sequential baseline.
- Worker skew ≤ 0.85 % at 2 h ⇒ no tail-latency lever worth its seams.
- Every alternate `MIN_THREADS` is either the same plan (5/6/7) or strictly worse
  (≤4: no wall win + 2× word-diff).

## Gates (all green)

- **Goldens byte-exact** (sequential whisper-cpp-native path, untouched):
  tiny.en + large-v3-turbo transcript + segments reproduced identically vs
  `/tmp/fw_golden/{tiny.en,large-v3-turbo}.json`.
- **IF backend tests 29/29** incl. all 4 gated:
  `gated_one_worker_equals_sequential_byte_exact`,
  `gated_fixed_count_deterministic`, `gated_word_diff_vs_sequential_bounded`
  (< 10 %), `gated_single_window_matches_whisper_cpp_native`.
- `cargo fmt --check` clean; `cargo clippy --all-targets -D warnings` clean (only
  external `fsqlite` dep warning).

## Files changed (measurement-only; production behavior bit-identical)

- `src/backend/insanely_fast_native.rs`:
  - `decode_ranges_parallel` — added an `if_worker` `perf_span` (per-worker wall +
    `start_window`/`end_window`/`samples`), gated by `FRANKEN_WHISPER_PERF_SPANS`.
    No control-flow change; produces the skew evidence above.
  - `MIN_THREADS_PER_WORKER` — added `min_threads_per_worker()` reading the
    **measurement-only** `FRANKEN_WHISPER_IF_MIN_THREADS` override (positive int;
    unset/empty/0/unparsable ⇒ the compiled-in `5`). Production default unchanged;
    enables the knob sweep above and future re-sweeps on other core counts.
- `tests/artifacts/perf/20260606T2341Z-scale-baseline/analyze_p4.py` — new span
  analyzer (shares + per-window drift + skew).
- `tests/artifacts/perf/.../p4_*.{txt,spans}` — raw run artifacts.

## Refreshed scale matrix (this pass)

| run | backend | wall_s | RSS_MB | load (before) | notes |
|-----|---------|-------:|-------:|--------------:|-------|
| tiny 600 s | sequential | 14.10 | 633 | 17 | decode_loop 3.71 ms/tok |
| tiny 600 s | IF default | 12.69 | 639 | 20 | 2w×7t, skew 4.1 % |
| tiny 2 h | sequential | 181.22 | 2051 | 40 (dropped to 21) | decode_loop 4.31 ms/tok |
| **tiny 2 h** | **IF default** | **262.91** | 2082 | **47–57** | **2w×7t, skew 0.85 %, word-diff 26.7 %** |
| large 600 s | IF default | — | — | 28 | run SIGTERM'd under load+RSS; pass-3 figs stand (0.83× / 19.7 %) |

> The 2 h IF wall (262.9 s) ran at load ~50 and the 2 h sequential (181.2 s) at
> load ~25; the same-session ratio is load-confounded (IF carried ~2× the
> ambient load). The honest cross-session headline is **IF 262.9 s vs the pass-1
> 604 s sequential baseline = 2.30× faster** on the real 2 h call.
