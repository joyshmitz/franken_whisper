# PASS 3 RESULTS — insanely-fast-native at-scale fix (round 3, pass 3/6)

Host: Apple M4 Pro (10P+4E / 14 core), 64 GB, macOS 26.2, `release-perf`.
Fixtures: `/tmp/fw_scale_600.wav` (600 s real call recording, 16 kHz mono) +
`/tmp/jfk3x.wav` (33 s). Harness: `examples/native_if_scale.rs` (runs the REAL
public `Engine::run` for both engines, reports wall + word-diff; the sequential
`whisper-cpp-native` path is the byte-golden reference). Measurements run
one-at-a-time; model warm in the process-wide cache for the IF run (so the IF
wall excludes model load — the honest backend-run comparison).

## TL;DR

The old insanely-fast-native used **hard per-window round-robin** with **no
rolling prompt**, oversubscribed the cores (`~7 workers × 2 threads` on 14
cores), and so ran **slower than sequential** *and* diverged badly. Pass 3
replaces it with **contiguous, 30 s-aligned ranges** (one per worker, each
decoded by the real sequential `transcribe_samples`) plus a **no-oversubscription
thread budget**. Result on the 600 s slice, both models:

| metric | OLD (baseline) | NEW (pass 3) |
|---|---|---|
| tiny wall ratio (IF / seq) | **1.43× (slower)** | **0.73× (27 % FASTER)** |
| large wall ratio (IF / seq) | **1.09× (slower)** | **0.83× (17 % FASTER)** |
| tiny word-diff vs seq (multiset) | 67 % | **33 %** |
| large word-diff vs seq (multiset) | 27.7 % | **19.7 %** |
| tiny LCS word-diff vs seq | (n/a) | **22 %** |
| large LCS word-diff vs seq | (n/a) | **13 %** |
| tiny over-generation | ~2× (1666 vs 803 w) | **none (802 vs 803 w)** |
| seams over 600 s | ~19 (one per window) | **1** (one per worker boundary) |
| oversubscription | yes (≈14 over 14 cores ×2) | **no** (workers × tpw ≤ total) |

## Lever 1 — thread budgeting (perf)

`plan_workers` was re-derived from measurement. The sequential path already
saturates the machine via **intra-op** rayon parallelism on every encoder GEMM /
decoder GEMV, so the old `n_workers × threads_per_worker` layered on top just
thrashed. New policy:

- `n_workers = min(n_windows, batch_size, total_threads / MIN_THREADS_PER_WORKER)`,
  floored at 1 — add a worker **only** when each can still hold
  `MIN_THREADS_PER_WORKER` intra-op threads.
- `threads_per_worker = total_threads / n_workers` → by construction
  `n_workers × threads_per_worker ≤ total_threads` (**never oversubscribes**;
  unit-tested invariant `plan_workers_no_oversubscription_invariant`).
- `total_threads < 2 × MIN_THREADS_PER_WORKER` ⇒ **1 worker = byte-exact
  sequential** (intra-op already fills the box).

`MIN_THREADS_PER_WORKER = 5` was chosen from the per-worker tradeoff (each extra
worker adds one cold-started seam). Forced-worker-count sweep, tiny.en, 600 s:

| workers | tpw | wall ratio (IF/seq) | LCS-diff vs seq |
|--------:|----:|--------------------:|----------------:|
| 1 | 14 | 0.96 (byte-exact) | **0.0 %** |
| 2 | 7 | 0.93 | 22 % |
| 3 | 4 | 0.87 | 41 % |

The 3rd worker's wall gain (0.93→0.87) is marginal but it **doubles** the tiny
word-diff (a cold range start can drift/loop through its whole 3-min span). `5`
makes the default `14/5 = 2` workers here — the sweet spot. It generalizes as
"fewer workers × fuller threads": 8 cores ⇒ 1 worker (= sequential); 32 cores ⇒
6 workers. Callers wanting max throughput raise `batch_size`; callers wanting
byte-exact output use the sequential engine (or `batch_size = 1`).

**Policy decision:** IF now defaults to **window-range parallelism only when
there is genuine thread headroom** (≥ 2× `MIN_THREADS_PER_WORKER`), and even then
biases to few, fully-threaded workers. On this 14-core host that is 2 workers,
which is *faster than sequential for both models* — IF is a real scale win again.

## Lever 2 — contiguous ranges + rolling prompt (accuracy at seams)

Instead of round-robin 30 s windows, the clip is partitioned into `n_workers`
**contiguous** 30 s-aligned ranges (`plan_ranges`), and each worker runs the real
sequential `transcribe_samples` over its **whole contiguous span**
(`decode_ranges_parallel`). This restores whisper's seek-continuation + rolling
prompt + timestamp-seek behavior **within** each range; the only discontinuities
are the `n_workers − 1` seams between ranges (≈30× fewer than before). Merge
(`merge_ranges`) offsets each range's timestamps by `start_window × 30 s` and
concatenates in range order (documented seam semantics: plain concatenation, no
de-dup).

Seam-accuracy before/after (600 s, multiset word-diff vs seq): **tiny 67 % →
33 %**, **large 27.7 % → 19.7 %**. The remaining divergence is concentrated at
the single seam (the cold-started 2nd range can drift); the clip *head* is
byte-identical to sequential because range 1 carries the rolling prompt.

### `transcribe_samples` reuse

No new decode API was needed — `transcribe_samples` already windows + seeks +
rolls the prompt internally over whatever sample slice it is given. A worker just
calls it on its contiguous slice and the merge adds the range base offset. So
`decode.rs` was **not modified** (public contract untouched; goldens safe).

## Final-state measurements (default policy)

```
tiny.en  /tmp/fw_scale_600.wav : seq 13145 ms, IF 9638 ms, ratio 0.733,
                                 seq 803 w / IF 802 w, multiset 33.3 %, LCS 22.4 %,
                                 2 workers × 7 threads, 2 ranges, 1 seam
large-v3 /tmp/fw_scale_600.wav : seq 192084 ms, IF 159332 ms, ratio 0.830,
                                 seq 1660 w / IF 1593 w, multiset 19.7 %, LCS 13.1 %,
                                 2 workers × 7 threads, 2 ranges, 1 seam
tiny.en  /tmp/jfk3x.wav (33 s) : multiset 2.9 %, LCS 2.9 %, 2 workers, 1 seam
                                 (wall 2.3× — too short to amortize; not the
                                  scale regime, IF is for long audio)
```

## Test-contract changes (new honest contract)

Old gated test `gated_determinism_two_workers_equals_one` (asserted 2-worker ==
1-worker text exactly) is **removed** — that property is false by design now
(output varies with worker count). Replaced by three gated tests:

- `gated_one_worker_equals_sequential_byte_exact` — 1-worker IF == direct
  `transcribe_samples` **byte-exact** (text + timestamps). The better property.
- `gated_fixed_count_deterministic` — 2-worker run repeated == itself (text +
  timestamps), monotonic timestamps across the seam.
- `gated_word_diff_vs_sequential_bounded` — 2-worker word-diff vs sequential
  **< 10 %** on the jfk3x fixture (measured 2.9 %).

Pure-fn tests updated for the new policy/API: `plan_workers_*` (new headroom
policy + `plan_workers_no_oversubscription_invariant`), `n_windows_*`,
`plan_ranges_*`, `range_samples_*`, `merge_ranges_offsets_by_start_window_*`,
`raw_output_carries_parallel_range_metadata` (asserts additive `ranges`+`seams`).

## raw_output schema

Stays `native-v2`-compatible. All legacy fields retained; **additive** fields:
`ranges` (`[{start_window,end_window,start_sec}]`) and `seams` (count of hard
cuts between ranges). `parallel_windows` now carries the 30 s-window count.

## Gates

- Goldens (whisper-cpp-native path): **byte-identical** for tiny.en AND
  large-v3-turbo vs `/tmp/fw_golden/*.json` (decode.rs untouched).
- Gated IF tests: 4/4 green (above).
- Full lib test suite: **3076 passed, 0 failed** (non-gated); backend module
  625/625.
- `cargo fmt --check` clean; `cargo clippy --all-targets -- -D warnings` clean
  (only external `fsqlite` dep warnings); `cargo check --all-targets` clean.

## Files changed

- `src/backend/insanely_fast_native.rs` — new `plan_workers` policy +
  `MIN_THREADS_PER_WORKER`; `plan_ranges` / `n_windows` / `range_samples`;
  `decode_ranges_parallel` + `RangeResult`; `merge_ranges`; `raw_output_json`
  (additive `ranges`/`seams`); module + fn rustdoc; tests rewritten to the new
  contract.
- `examples/native_if_scale.rs` — NEW scale A/B + word-diff harness.
- `src/native_engine/decode.rs` — **unchanged** (no window-range API needed;
  `transcribe_samples` already handles a sub-range contiguously).
