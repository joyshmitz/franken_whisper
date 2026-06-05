# RESULTS — Native Engine Extreme-Optimization Arc (run 20260605T0218Z)

Final landing summary for the profile-driven optimization loop on
`src/native_engine/`. Host: **Apple M4 Pro**, 14 cores (10P+4E), 64 GB,
macOS 26.2, `release-perf` profile (opt3, thin-LTO, codegen-units=1). All
"final" hyperfine numbers below were taken **interleaved** (REF/CAND runs
alternated back-to-back on the same host under the same concurrent agent
load) so they compare like-for-like rather than across drifting machine load.

Baseline (this run, git `0553cbb`, release-perf, warm page cache):
**tiny.en jfk 1.225 s ± 0.029** (15 runs); **large-v3-turbo jfk 32.92 s ± 0.16**
(5 runs). whisper-cli same host: tiny 0.57 s CPU; large 9.63 s CPU / 2.11 s Metal.

---

## 1. Levers Landed

Six commits, profile-justified, each kept only on measured wins
(bit-identical unless noted under DISC-004):

| Commit | Lever | Effect (measured, this host) |
|--------|-------|------------------------------|
| `2ed6471` | Background model **version-tag hash** at load | sha256 over the model moved off the run path: **3.0 s → 0** on the critical path (overlaps encode/decode; tag itself unchanged) |
| `0361bb2` | **Parallel f16→f32 dequant + tiled parallel weight transpose** at load (`LoadedModel::from_ggml`) | `model_weights` span **2.4 s → 0.47 s** |
| `bdbdd21` | **Language-detect encode reuse** — reuse window-0's encode instead of a hidden duplicate encoder pass | **−8.8 s** (eliminates the duplicate window-0 encoder window on large) |
| `ee36fd4` | **Parallel serial glue ops** (layer_norm / softmax / gelu / im2col / attention-head loops / logits GEMV / cross-K-V) | encoder **8.3 → 5.8 s**; tiny wall **0.92 → 0.56 s** |
| `0989a5a` | **Tail-window encoder-context truncation** (audio_ctx-style, DISC-004, default-on with `FRANKEN_WHISPER_NATIVE_TAIL_TRUNCATE=0` kill switch) | tail-window encode **4.2 s → 0.24 s** (~94%); large wall ~11.0 → ~7.0 s. Main transcript byte-identical; divergence confined to spurious trailing-silence hallucination on tail windows (see DISC-004) |
| `5bb778b` | **Decoder per-token path** — KV buffer reuse, in-place logits band reads, hoisted window-constant cross K^T/V transforms, threaded QKV, head-parallel cross-attn | **large 62.6 → 39.9 ms/tok**; **tiny 11.9 → 6.99 ms/tok**. Bit-identical (4 new bitwise nn tests); also drops ~490 MB cross-buffers on large |

---

## 2. Evidence-Backed Abandons (zero-change passes 2 & 4)

Levers that were implemented and **measured**, then dropped because the
evidence did not justify them — recorded so the next loop does not re-pay
the same investigation:

- **Fused QKV projection** — bit-identical *proven*, but **~16% slower**: one
  wide column-parallel sgemm has more thread-feeding overhead than 3 narrow
  calls at these shapes.
- **bmm-batched per-head attention** — bit-identical, **~4% slower** after a
  proper interleaved A/B (a transient low-load reading had briefly lied).
- **Parallel residual adds** — slower: spawn storm beats a memory-bound add.
- **Encoder scratch-reuse arena** (eliminate the 2× `x.clone()` per encoder
  block) — bit-identical, removes ~490 MB transient/window, but **wall- and
  RSS-neutral**: the encoder is sgemm-**compute**-bound and ft's output `Vec`
  is `vec![0.0; n]` = calloc with lazy first-touch pages the sgemm writes
  anyway. No wall-clock movement.
- **ft-side matmul arena (lever 6, quantified)** — fresh out-`Vec` per ft
  matmul call (1472 calls/large-encoder-window). Microbench: fresh calloc+write
  550 ms/window vs reuse-one-buffer 503 ms/window ⇒ **~9% theoretical**
  (~47 ms/large-window) of alloc-bookkeeping + first-touch fault overhead a
  reuse arena *could* reclaim. This is a **frankentorch** change, out of scope
  for this crate, and below the loop's score bar; flagged for the orchestrator.

---

## 3. Final Interleaved Numbers (same host, concurrent load, runs interleaved)

### tiny.en, jfk.wav
| Engine | Wall | Verdict |
|--------|------|---------|
| **native** | **475 ms ± 13** | **2.33× FASTER** than whisper-cli CPU |
| whisper-cli CPU | 1105 ms ± 21 | |

### large-v3-turbo, jfk.wav
| Engine | Wall | User CPU | Note |
|--------|------|----------|------|
| **native** | **9.731 s ± 0.272** | **53.8 s** | parity with whisper-cli CPU, *less* user CPU |
| whisper-cli CPU | 9.585 s ± 0.224 | 65.4 s | |
| whisper-cli Metal | 2.169 s | — | GPU reference (out of native CPU scope) |

### Cumulative vs session start (release profile)
| Model | Session start | Final | Speedup | RTF (final) |
|-------|---------------|-------|---------|-------------|
| tiny.en | 1.57 s | **0.475 s** | **3.3×** | **0.043** (faster than realtime) |
| large-v3-turbo | 44.6 s | **9.73 s** | **4.6×** | **0.88** (faster than realtime) |

(RTF = wall / 11 s of jfk audio.)

---

## 4. Release-profile discovery (follow-up, do NOT auto-apply)

The `release` profile ships `opt-level = "z"` (size-optimized). Building the
perf binary under plain `release` cost **~26% on large-v3-turbo** vs the
`release-perf` (opt3 + thin-LTO) numbers above. **Recommendation:** evaluate
promoting the `dist`/`release` profile to opt3 (or aligning it with
`release-perf`) as a follow-up. **Not changed here** — the `Cargo.toml`
release profile was deliberately left untouched; this is an orchestrator/owner
decision, not a perf-loop edit.

---

## 5. Cross-references

- Hotspot table + hypothesis ledger: `HOTSPOTS.md` (this dir).
- Spans timelines: `spans_large_timeline.txt`, `spans_*.jsonl` (this dir);
  enable with `FRANKEN_WHISPER_PERF_SPANS=1`.
- Tail-truncation accuracy/precision analysis + kill switch: `DISCREPANCIES.md`
  **DISC-004**.
- Promotion-criteria status: `docs/native_engine_contract.md` §Performance.
- Tracking bead: **bd-2th6** (criterion benches still outstanding — see report).
