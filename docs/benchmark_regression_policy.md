# Benchmark Regression Policy

This file records baseline comparator thresholds and acceptance criteria for
new `criterion` benches added in packet `P2.6`.

## Acceptance Criteria

- Throughput regressions larger than `20%` against the recorded baseline are
  considered failures and must be investigated before release.
- Improvements or neutral movement are accepted.
- If workload shape changes materially, regenerate baselines and update this
  document in the same change.

## Automated Guardrail Checker

Benchmark thresholds are encoded in `docs/benchmark_guardrails.json` and
validated by `src/bin/benchmark_guardrails.rs`.

Run checker directly:

```bash
cargo run --bin benchmark_guardrails -- --criterion-root target/criterion --policy docs/benchmark_guardrails.json
```

Or use the shell hook entrypoint (CI/local friendly):

```bash
scripts/check_benchmark_guardrails.sh target/criterion docs/benchmark_guardrails.json
```

Checker behavior:
- Uses criterion `new/estimates.json` per benchmark ID.
- Uses `slope.point_estimate` when present, otherwise `mean.point_estimate`.
- Fails hard when enforced benchmarks regress by more than `20%`.
- Emits warnings (not failures) for provisional sync benchmarks until upstream
  `frankensqlite` io_uring instability is resolved.

## Baseline Thresholds (Current Snapshot: 2026-02-22)

Measured with:

```bash
CARGO_TARGET_DIR=/data/projects/franken_whisper/target_local cargo bench --bench tty_bench -- --sample-size 10 --warm-up-time 0.1 --measurement-time 0.1
CARGO_TARGET_DIR=/data/projects/franken_whisper/target_local cargo bench --bench sync_bench -- --sample-size 10 --warm-up-time 0.1 --measurement-time 0.1
```

| Benchmark ID | Baseline | Guardrail |
| --- | --- | --- |
| `tty/encode/chunk_ms=20` | `~61 ms` (`~512 KiB/s`) | do not exceed `73 ms` (`+20%`) |
| `tty/encode/chunk_ms=60` | `~56 ms` (`~563 KiB/s`) | do not exceed `67 ms` (`+20%`) |
| `tty/decode/chunk_ms=20` | `~0.38 ms` (`~55 MiB/s`) | do not exceed `0.46 ms` (`+20%`) |
| `tty/decode/chunk_ms=60` | `~0.20 ms` (`~40 MiB/s`) | do not exceed `0.24 ms` (`+20%`) |
| `tty/control_emit/frame=*` | `~0.18-0.33 µs` | do not exceed `~0.40 µs` (`+20%` from worst baseline) |
| `sync/export/runs/10` | `~104.44 ms` | do not exceed `125.33 ms` (`+20%`) |
| `sync/export/runs/50` | `~561.75 ms` | do not exceed `674.10 ms` (`+20%`) |
| `sync/import/runs/10` | `~84.46 ms` (`~264.66 KiB/s`) | do not exceed `101.35 ms` (`+20%`) |
| `sync/import/runs/50` | `~78.69 ms` (`~1.36 MiB/s`) | do not exceed `94.43 ms` (`+20%`) |

Sync guardrails are currently **provisional** because upstream
`frankensqlite` `io_uring` runtime panics still occur under criterion stress.
Treat sync figures as diagnostic until that upstream panic is removed.

## Sync Probe Evidence (2026-02-22, blocked)

Probe command:

```bash
CARGO_TARGET_DIR=/data/tmp/franken_whisper-target cargo bench --bench sync_bench -- --sample-size 10 --warm-up-time 0.1 --measurement-time 0.1 > /tmp/franken_whisper_sync_bench_2026-02-22.log 2>&1 || true
```

Observed medians in this blocked probe:

| Benchmark ID | Observed median |
| --- | --- |
| `sync/export/runs/10` | `~85.58 ms` |
| `sync/export/runs/50` | `~510.10 ms` |
| `sync/import/runs/10` | `~53.89 ms` (`~414.78 KiB/s`) |
| `sync/import/runs/50` | `~86.34 ms` (`~1.239 MiB/s`) |

Observed panic signatures in the captured log:

- `uring-fs-1.4.0/src/lib.rs:359`: `Option::unwrap()` on `None`
- `/data/projects/frankensqlite/crates/fsqlite-vfs/src/uring.rs:190`:
  `io_uring runtime lock poisoned`
- panic-related line count: `1,253,267`

## Collection Command

```bash
cargo bench --bench tty_bench
cargo bench --bench sync_bench
```
Record observed medians in PR notes and compare against the thresholds above.
