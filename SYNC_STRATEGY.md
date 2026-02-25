# SYNC_STRATEGY.md

## Source of Truth

`frankensqlite` (`fsqlite`) database is canonical state.

JSONL is an adjunct audit/export medium for:
- git-friendly review,
- cross-machine transfer,
- recovery workflows.

## Scope

Tables:
- `runs`
- `segments`
- `events`

JSONL channels:
- `runs.jsonl`
- `segments.jsonl`
- `events.jsonl`

## One-Way Operations Only

Allowed operations:
1. `db -> jsonl` export snapshot
2. `jsonl -> db` import replay

Disallowed:
- implicit two-way merge
- concurrent export+import against same DB

## Locking Model

Before sync, create lock file under `.franken_whisper/locks/sync.lock`.

Rules:
- lock acquisition is mandatory.
- stale lock detection based on timestamp + PID verification.
- sync aborts if lock is active and not stale.

## Snapshot Semantics

### Export (`db -> jsonl`)
1. Acquire lock.
2. Write export manifest (`manifest.json`) with:
- schema version
- created timestamp
- source DB path
- row counts
- checksum placeholders
3. Stream each table to temp `*.jsonl.tmp`.
4. Flush + fsync temp files.
5. Atomic rename `*.tmp -> *.jsonl`.
6. Update manifest checksums.
7. Release lock.

### Import (`jsonl -> db`)
1. Acquire lock.
2. Validate manifest + schema compatibility.
3. Begin DB transaction.
4. Replay JSONL in deterministic order (`runs`, `segments`, `events`).
5. Apply conflict policy by stable keys (`reject` default, `overwrite`/`overwrite-strict` opt-in).
6. Enforce overwrite safety constraints in current runtime:
   - `runs` parent-row replacement is allowed.
   - `overwrite` is fail-closed for child-row `UPDATE`/`DELETE` on `segments`/`events`.
   - `overwrite-strict` performs verified child-row replacement for imported runs:
     - conflicting child rows are replaced via delete+insert,
     - stale child rows not present in import are pruned.
7. Commit transaction.
8. Release lock.

## Conflict Strategy

Default import mode: `reject` (`sync import-jsonl --conflict-policy reject`).

Rules:
- same primary key + same payload: no-op.
- same primary key + different payload: reject unless explicitly set to overwrite via
  `sync import-jsonl --conflict-policy overwrite` or strict overwrite via
  `sync import-jsonl --conflict-policy overwrite-strict`.
- `overwrite` does **not** imply unrestricted in-place mutation:
  - if resolving a conflict requires child-row `UPDATE` (`segments`/`events` same key, different payload), import fails closed.
  - if strict replacement requires deleting stale child rows not present in import, import fails closed.
- `overwrite-strict` is the explicit in-place strict replacement mode for imported runs.
- all conflicts logged to `sync_conflicts.jsonl`.

## Integrity

Integrity checks:
- per-file SHA-256 checksum in manifest.
- row-count reconciliation.
- referential checks: `segments.run_id` and `events.run_id` must reference existing `runs.id`.

## Versioning

Manifest carries:
- `schema_version`
- `export_format_version`

Import rules:
- exact major match required.
- minor mismatch allowed only with backward-compatible fields.

## Failure Recovery

Any failed sync operation must:
- preserve previous committed DB state,
- keep temp artifacts for forensic analysis,
- emit machine-readable error with operation stage.

See `RECOVERY_RUNBOOK.md` for exact procedures.
