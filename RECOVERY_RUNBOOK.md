# RECOVERY_RUNBOOK.md

## Purpose

Operational recovery for `franken_whisper` persistence and sync faults.

Canonical storage: `frankensqlite` (`fsqlite`) DB.

## Quick Triage Checklist

1. Identify failing operation:
- ingest
- backend run
- persist
- sync export
- sync import

2. Capture artifacts:
- stderr/stdout logs
- robot-mode event stream
- sync manifest/temp files

3. Validate DB accessibility:
- open DB with `fsqlite`
- run simple query (`SELECT 1`)

4. Determine scope:
- single run corruption
- table-level integrity issue
- sync artifact mismatch

## Scenario A: Failed Persist During Run

Symptoms:
- transcription succeeds but run missing from DB.

Procedure:
1. Locate final result envelope from robot-mode output.
2. Re-run persist-only command with saved result payload.
3. Verify rows in `runs`, `segments`, `events`.
4. Record remediation event linked to original `run_id`.

## Scenario B: Export Failed Mid-Write

Symptoms:
- `*.jsonl.tmp` exists, manifest incomplete.

Procedure:
1. Ensure sync lock is released or confirmed stale.
2. Remove stale temp files only after verification (non-destructive archival preferred).
3. Re-run export end-to-end.
4. Validate checksums + row counts.

## Scenario C: Import Conflict

Symptoms:
- duplicate key mismatch, import aborted.

Procedure:
1. Inspect `sync_conflicts.jsonl`.
2. Classify conflicts:
- benign duplicates
- divergent payloads
3. Choose explicit policy:
- default reject
- overwrite only with operator intent
4. Re-run import with selected conflict policy.
5. If error indicates child-row `UPDATE`/`DELETE` is unsupported:
- stop in-place overwrite attempts,
- create a fresh target DB,
- import snapshot into that empty DB,
- swap active DB pointer only after validation.

## Scenario D: Corrupted or Inaccessible DB

Symptoms:
- `fsqlite::Connection::open` or core query errors.

Procedure:
1. Create timestamped backup copy of DB file.
2. Attempt recovery by replaying latest valid JSONL snapshot into fresh DB.
3. Compare row counts + checksums against manifest.
4. Switch active DB pointer only after validation.

## Scenario F: Legacy Schema Migration Blocked

Symptoms:
- open/migration fails with a safe-legacy-migration error.

Procedure:
1. Preserve original DB as immutable evidence copy.
2. Attempt reopen once to allow rollback-safe snapshot/rebuild/swap migration to finish.
3. If failure persists, locate latest valid JSONL snapshot (`runs.jsonl`, `segments.jsonl`, `events.jsonl`, `manifest.json`).
4. Create a fresh DB path.
5. Import snapshot with reject policy:
- `cargo run -- sync import-jsonl --input <snapshot_dir> --conflict-policy reject`
6. Validate row counts/checksums and basic run queries.
7. Switch active DB pointer only after successful validation.

## Scenario E: Backend Process Crash/Hang

Symptoms:
- external backend command timeout or non-zero exit.

Procedure:
1. Capture command, args, exit status, stderr.
2. Emit error event with backend id and stage.
3. Fallback to next backend if policy allows.
4. If all fail, persist failed run shell with diagnostics.

## Post-Recovery Validation

Run:
- `cargo check --all-targets`
- `cargo test`
- targeted CLI smoke for persist/sync paths

Then verify:
- no orphan lock files
- run/event counts consistent
- robot mode outputs valid NDJSON
