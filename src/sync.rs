use std::collections::HashSet;
use std::fs;
use std::fs::OpenOptions;
use std::io::{BufRead, BufReader, Read as _, Write};
use std::path::{Path, PathBuf};

use chrono::Utc;
use flate2::Compression;
use flate2::read::GzDecoder;
use flate2::write::GzEncoder;
use fsqlite::Connection;
use fsqlite_types::value::SqliteValue;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::error::{FwError, FwResult};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const SCHEMA_VERSION: &str = "1.1";
const EXPORT_FORMAT_VERSION: &str = "1.0";
const LOCK_STALE_SECONDS: i64 = 300; // 5 minutes

// ---------------------------------------------------------------------------
// Manifest
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyncManifest {
    pub schema_version: String,
    pub export_format_version: String,
    pub created_at_rfc3339: String,
    pub source_db_path: String,
    pub row_counts: RowCounts,
    pub checksums: FileChecksums,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RowCounts {
    pub runs: u64,
    pub segments: u64,
    pub events: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FileChecksums {
    pub runs_jsonl_sha256: String,
    pub segments_jsonl_sha256: String,
    pub events_jsonl_sha256: String,
}

// ---------------------------------------------------------------------------
// Lock
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
struct LockInfo {
    pid: u32,
    created_at_rfc3339: String,
    operation: String,
}

pub struct SyncLock {
    path: PathBuf,
    released: bool,
}

impl SyncLock {
    pub fn acquire(state_root: &Path, operation: &str) -> FwResult<Self> {
        let locks_dir = state_root.join("locks");
        fs::create_dir_all(&locks_dir)?;
        let path = locks_dir.join("sync.lock");

        if path.exists() {
            let contents = fs::read_to_string(&path)?;
            if let Ok(info) = serde_json::from_str::<LockInfo>(&contents) {
                if is_lock_stale(&info) {
                    archive_stale_lock(&path, "stale")?;
                } else {
                    return Err(FwError::Storage(format!(
                        "sync lock held by pid {} since {}; \
                         remove {} if the process is dead",
                        info.pid,
                        info.created_at_rfc3339,
                        path.display()
                    )));
                }
            } else {
                // Corrupt lock file — archive and proceed.
                archive_stale_lock(&path, "corrupt")?;
            }
        }

        let info = LockInfo {
            pid: std::process::id(),
            created_at_rfc3339: Utc::now().to_rfc3339(),
            operation: operation.to_owned(),
        };
        let mut file = OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&path)
            .map_err(|error| {
                FwError::Storage(format!(
                    "failed to acquire sync lock {}: {error}",
                    path.display()
                ))
            })?;
        file.write_all(serde_json::to_string_pretty(&info)?.as_bytes())?;
        file.sync_all()?;

        Ok(Self {
            path,
            released: false,
        })
    }

    pub fn release(mut self) -> FwResult<()> {
        self.release_inner()
    }

    fn release_inner(&mut self) -> FwResult<()> {
        if self.released {
            return Ok(());
        }
        if self.path.exists() {
            fs::remove_file(&self.path)?;
        }
        self.released = true;
        Ok(())
    }
}

impl Drop for SyncLock {
    fn drop(&mut self) {
        if !self.released && self.path.exists() {
            let _ = fs::remove_file(&self.path);
            self.released = true;
        }
    }
}

fn is_lock_stale(info: &LockInfo) -> bool {
    // Check if PID is still alive
    let pid_alive = pid_is_alive(info.pid);
    if !pid_alive {
        return true;
    }

    // Check age
    if let Ok(created) = chrono::DateTime::parse_from_rfc3339(&info.created_at_rfc3339) {
        let age = Utc::now().signed_duration_since(created);
        if age.num_seconds() > LOCK_STALE_SECONDS {
            return true;
        }
    }

    false
}

fn archive_stale_lock(path: &Path, reason: &str) -> FwResult<()> {
    let timestamp = Utc::now().timestamp_millis();
    let archived = path.with_file_name(format!("sync.lock.{reason}.{timestamp}.json"));
    fs::rename(path, archived)?;
    Ok(())
}

#[cfg(target_os = "linux")]
fn pid_is_alive(pid: u32) -> bool {
    Path::new(&format!("/proc/{pid}")).exists()
}

#[cfg(not(target_os = "linux"))]
fn pid_is_alive(_pid: u32) -> bool {
    true
}

// ---------------------------------------------------------------------------
// Export (db -> jsonl)
// ---------------------------------------------------------------------------

pub fn export(db_path: &Path, output_dir: &Path, state_root: &Path) -> FwResult<SyncManifest> {
    let _lock = SyncLock::acquire(state_root, "export")?;
    export_inner(db_path, output_dir)
}

fn export_inner(db_path: &Path, output_dir: &Path) -> FwResult<SyncManifest> {
    fs::create_dir_all(output_dir)?;

    let connection = Connection::open(db_path.display().to_string())
        .map_err(|error| FwError::Storage(error.to_string()))?;
    verify_schema_exists(&connection)?;

    // Export runs
    let runs_tmp = output_dir.join("runs.jsonl.tmp");
    let runs_final = output_dir.join("runs.jsonl");
    let runs_count = export_table_runs(&connection, &runs_tmp)?;
    atomic_rename(&runs_tmp, &runs_final)?;

    // Export segments
    let segments_tmp = output_dir.join("segments.jsonl.tmp");
    let segments_final = output_dir.join("segments.jsonl");
    let segments_count = export_table_segments(&connection, &segments_tmp)?;
    atomic_rename(&segments_tmp, &segments_final)?;

    // Export events
    let events_tmp = output_dir.join("events.jsonl.tmp");
    let events_final = output_dir.join("events.jsonl");
    let events_count = export_table_events(&connection, &events_tmp)?;
    atomic_rename(&events_tmp, &events_final)?;

    // Compute checksums
    let checksums = FileChecksums {
        runs_jsonl_sha256: sha256_file(&runs_final)?,
        segments_jsonl_sha256: sha256_file(&segments_final)?,
        events_jsonl_sha256: sha256_file(&events_final)?,
    };

    let manifest = SyncManifest {
        schema_version: SCHEMA_VERSION.to_owned(),
        export_format_version: EXPORT_FORMAT_VERSION.to_owned(),
        created_at_rfc3339: Utc::now().to_rfc3339(),
        source_db_path: db_path.display().to_string(),
        row_counts: RowCounts {
            runs: runs_count,
            segments: segments_count,
            events: events_count,
        },
        checksums,
    };

    let manifest_path = output_dir.join("manifest.json");
    let manifest_json = serde_json::to_string_pretty(&manifest)?;
    atomic_write_bytes(&manifest_path, manifest_json.as_bytes())?;

    Ok(manifest)
}

fn export_table_runs(connection: &Connection, path: &Path) -> FwResult<u64> {
    let rows = connection
        .query(
            "SELECT id, started_at, finished_at, backend, input_path, \
             normalized_wav_path, request_json, result_json, warnings_json, transcript, replay_json \
             FROM runs ORDER BY started_at ASC",
        )
        .map_err(|error| FwError::Storage(error.to_string()))?;

    let mut file = fs::File::create(path)?;
    let mut count = 0u64;

    for row in rows {
        let obj = serde_json::json!({
            "id": value_to_json(row.get(0)),
            "started_at": value_to_json(row.get(1)),
            "finished_at": value_to_json(row.get(2)),
            "backend": value_to_json(row.get(3)),
            "input_path": value_to_json(row.get(4)),
            "normalized_wav_path": value_to_json(row.get(5)),
            "request_json": value_to_json(row.get(6)),
            "result_json": value_to_json(row.get(7)),
            "warnings_json": value_to_json(row.get(8)),
            "transcript": value_to_json(row.get(9)),
            "replay_json": value_to_json(row.get(10)),
        });
        writeln!(file, "{}", serde_json::to_string(&obj)?)?;
        count += 1;
    }
    file.flush()?;
    file.sync_all()?;

    Ok(count)
}

fn export_table_segments(connection: &Connection, path: &Path) -> FwResult<u64> {
    let rows = connection
        .query(
            "SELECT run_id, idx, start_sec, end_sec, speaker, text, confidence \
             FROM segments ORDER BY run_id ASC, idx ASC",
        )
        .map_err(|error| FwError::Storage(error.to_string()))?;

    let mut file = fs::File::create(path)?;
    let mut count = 0u64;

    for row in rows {
        let obj = serde_json::json!({
            "run_id": value_to_json(row.get(0)),
            "idx": value_to_json(row.get(1)),
            "start_sec": value_to_json(row.get(2)),
            "end_sec": value_to_json(row.get(3)),
            "speaker": value_to_json(row.get(4)),
            "text": value_to_json(row.get(5)),
            "confidence": value_to_json(row.get(6)),
        });
        writeln!(file, "{}", serde_json::to_string(&obj)?)?;
        count += 1;
    }
    file.flush()?;
    file.sync_all()?;

    Ok(count)
}

fn export_table_events(connection: &Connection, path: &Path) -> FwResult<u64> {
    let rows = connection
        .query(
            "SELECT run_id, seq, ts_rfc3339, stage, code, message, payload_json \
             FROM events ORDER BY run_id ASC, seq ASC",
        )
        .map_err(|error| FwError::Storage(error.to_string()))?;

    let mut file = fs::File::create(path)?;
    let mut count = 0u64;

    for row in rows {
        let obj = serde_json::json!({
            "run_id": value_to_json(row.get(0)),
            "seq": value_to_json(row.get(1)),
            "ts_rfc3339": value_to_json(row.get(2)),
            "stage": value_to_json(row.get(3)),
            "code": value_to_json(row.get(4)),
            "message": value_to_json(row.get(5)),
            "payload_json": value_to_json(row.get(6)),
        });
        writeln!(file, "{}", serde_json::to_string(&obj)?)?;
        count += 1;
    }
    file.flush()?;
    file.sync_all()?;

    Ok(count)
}

// ---------------------------------------------------------------------------
// Incremental Export (changed records only)
// ---------------------------------------------------------------------------

/// Tracks the position of the last incremental export so subsequent calls
/// only emit records that appeared after the cursor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyncCursor {
    /// RFC-3339 timestamp of the last export (inclusive upper bound).
    pub last_export_rfc3339: String,
    /// Number of runs exported in the last batch.
    pub last_run_count: u64,
}

/// Mode selector for the export function family.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExportMode {
    /// Export every record in the database.
    Full,
    /// Export only records whose `started_at` is strictly greater than the
    /// cursor timestamp, or all records when no cursor exists.
    Incremental,
}

/// Manifest for an incremental export.  It is a superset of [`SyncManifest`]
/// that additionally carries the cursor used to produce the snapshot.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IncrementalExportManifest {
    pub schema_version: String,
    pub export_format_version: String,
    pub export_mode: String,
    pub created_at_rfc3339: String,
    pub source_db_path: String,
    pub row_counts: RowCounts,
    pub checksums: FileChecksums,
    /// `None` when this is the first incremental export (no prior cursor).
    pub cursor_used: Option<SyncCursor>,
    /// Updated cursor that should be persisted and passed to the next call.
    pub cursor_after: SyncCursor,
}

const CURSOR_FILENAME: &str = "sync_cursor.json";

/// Perform an incremental export: only records whose `started_at` is
/// strictly greater than the cursor timestamp are emitted.  When no cursor
/// file exists inside `state_root`, all records are exported (equivalent to
/// a full export) and a new cursor file is written.
pub fn export_incremental(
    db_path: &Path,
    output_dir: &Path,
    state_root: &Path,
) -> FwResult<IncrementalExportManifest> {
    let _lock = SyncLock::acquire(state_root, "export_incremental")?;
    export_incremental_inner(db_path, output_dir, state_root)
}

fn export_incremental_inner(
    db_path: &Path,
    output_dir: &Path,
    state_root: &Path,
) -> FwResult<IncrementalExportManifest> {
    fs::create_dir_all(output_dir)?;

    let connection = Connection::open(db_path.display().to_string())
        .map_err(|error| FwError::Storage(error.to_string()))?;
    verify_schema_exists(&connection)?;

    let cursor_path = state_root.join(CURSOR_FILENAME);
    let cursor_used = load_cursor(&cursor_path)?;

    // Build the WHERE clause.  When there is no prior cursor we export everything.
    let after_ts = cursor_used.as_ref().map(|c| c.last_export_rfc3339.clone());

    // --- runs ---
    let runs_tmp = output_dir.join("runs.jsonl.tmp");
    let runs_final = output_dir.join("runs.jsonl");
    let runs_count = export_table_runs_incremental(&connection, &runs_tmp, after_ts.as_deref())?;
    atomic_rename(&runs_tmp, &runs_final)?;

    // Collect the run_ids that were exported so segments/events can be scoped.
    let run_ids = collect_exported_run_ids(&connection, after_ts.as_deref())?;

    // --- segments ---
    let segments_tmp = output_dir.join("segments.jsonl.tmp");
    let segments_final = output_dir.join("segments.jsonl");
    let segments_count = export_table_segments_for_runs(&connection, &segments_tmp, &run_ids)?;
    atomic_rename(&segments_tmp, &segments_final)?;

    // --- events ---
    let events_tmp = output_dir.join("events.jsonl.tmp");
    let events_final = output_dir.join("events.jsonl");
    let events_count = export_table_events_for_runs(&connection, &events_tmp, &run_ids)?;
    atomic_rename(&events_tmp, &events_final)?;

    // Compute checksums
    let checksums = FileChecksums {
        runs_jsonl_sha256: sha256_file(&runs_final)?,
        segments_jsonl_sha256: sha256_file(&segments_final)?,
        events_jsonl_sha256: sha256_file(&events_final)?,
    };

    // Determine the new cursor: the maximum started_at of the exported runs,
    // or the previous cursor if nothing was exported.
    let new_cursor_ts = max_started_at(&connection, after_ts.as_deref())?.unwrap_or_else(|| {
        cursor_used
            .as_ref()
            .map(|c| c.last_export_rfc3339.clone())
            .unwrap_or_else(|| Utc::now().to_rfc3339())
    });

    let cursor_after = SyncCursor {
        last_export_rfc3339: new_cursor_ts,
        last_run_count: runs_count,
    };

    // Persist the updated cursor.
    save_cursor(&cursor_path, &cursor_after)?;

    let manifest = IncrementalExportManifest {
        schema_version: SCHEMA_VERSION.to_owned(),
        export_format_version: EXPORT_FORMAT_VERSION.to_owned(),
        export_mode: "incremental".to_owned(),
        created_at_rfc3339: Utc::now().to_rfc3339(),
        source_db_path: db_path.display().to_string(),
        row_counts: RowCounts {
            runs: runs_count,
            segments: segments_count,
            events: events_count,
        },
        checksums,
        cursor_used: cursor_used.clone(),
        cursor_after,
    };

    let manifest_path = output_dir.join("manifest.json");
    let manifest_json = serde_json::to_string_pretty(&manifest)?;
    atomic_write_bytes(&manifest_path, manifest_json.as_bytes())?;

    Ok(manifest)
}

/// Load the cursor from `path`, returning `None` if the file does not exist.
fn load_cursor(path: &Path) -> FwResult<Option<SyncCursor>> {
    if !path.exists() {
        return Ok(None);
    }
    let contents = fs::read_to_string(path)?;
    let cursor: SyncCursor = serde_json::from_str(&contents)
        .map_err(|error| FwError::Storage(format!("invalid sync cursor: {error}")))?;
    Ok(Some(cursor))
}

/// Persist the cursor as a pretty-printed JSON file.
fn save_cursor(path: &Path, cursor: &SyncCursor) -> FwResult<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let json = serde_json::to_string_pretty(cursor)?;
    atomic_write_bytes(path, json.as_bytes())?;
    Ok(())
}

fn export_table_runs_incremental(
    connection: &Connection,
    path: &Path,
    after_ts: Option<&str>,
) -> FwResult<u64> {
    let (sql, params) = match after_ts {
        Some(ts) => (
            "SELECT id, started_at, finished_at, backend, input_path, \
             normalized_wav_path, request_json, result_json, warnings_json, transcript, replay_json \
             FROM runs WHERE started_at > ?1 ORDER BY started_at ASC"
                .to_owned(),
            vec![SqliteValue::Text(ts.to_owned())],
        ),
        None => (
            "SELECT id, started_at, finished_at, backend, input_path, \
             normalized_wav_path, request_json, result_json, warnings_json, transcript, replay_json \
             FROM runs ORDER BY started_at ASC"
                .to_owned(),
            vec![],
        ),
    };

    let rows = if params.is_empty() {
        connection
            .query(&sql)
            .map_err(|error| FwError::Storage(error.to_string()))?
    } else {
        connection
            .query_with_params(&sql, &params)
            .map_err(|error| FwError::Storage(error.to_string()))?
    };

    let mut file = fs::File::create(path)?;
    let mut count = 0u64;

    for row in rows {
        let obj = serde_json::json!({
            "id": value_to_json(row.get(0)),
            "started_at": value_to_json(row.get(1)),
            "finished_at": value_to_json(row.get(2)),
            "backend": value_to_json(row.get(3)),
            "input_path": value_to_json(row.get(4)),
            "normalized_wav_path": value_to_json(row.get(5)),
            "request_json": value_to_json(row.get(6)),
            "result_json": value_to_json(row.get(7)),
            "warnings_json": value_to_json(row.get(8)),
            "transcript": value_to_json(row.get(9)),
            "replay_json": value_to_json(row.get(10)),
        });
        writeln!(file, "{}", serde_json::to_string(&obj)?)?;
        count += 1;
    }
    file.flush()?;
    file.sync_all()?;

    Ok(count)
}

/// Collect all run IDs that match the incremental filter, so we can scope
/// segments and events to those runs.
fn collect_exported_run_ids(
    connection: &Connection,
    after_ts: Option<&str>,
) -> FwResult<Vec<String>> {
    let (sql, params) = match after_ts {
        Some(ts) => (
            "SELECT id FROM runs WHERE started_at > ?1 ORDER BY started_at ASC".to_owned(),
            vec![SqliteValue::Text(ts.to_owned())],
        ),
        None => (
            "SELECT id FROM runs ORDER BY started_at ASC".to_owned(),
            vec![],
        ),
    };

    let rows = if params.is_empty() {
        connection
            .query(&sql)
            .map_err(|error| FwError::Storage(error.to_string()))?
    } else {
        connection
            .query_with_params(&sql, &params)
            .map_err(|error| FwError::Storage(error.to_string()))?
    };

    Ok(rows
        .iter()
        .map(|row| value_to_string_sqlite(row.get(0)))
        .collect())
}

/// Export segments belonging to a specific set of run_ids.
fn export_table_segments_for_runs(
    connection: &Connection,
    path: &Path,
    run_ids: &[String],
) -> FwResult<u64> {
    let mut file = fs::File::create(path)?;
    let mut count = 0u64;

    for run_id in run_ids {
        let rows = connection
            .query_with_params(
                "SELECT run_id, idx, start_sec, end_sec, speaker, text, confidence \
                 FROM segments WHERE run_id = ?1 ORDER BY idx ASC",
                &[SqliteValue::Text(run_id.clone())],
            )
            .map_err(|error| FwError::Storage(error.to_string()))?;

        for row in rows {
            let obj = serde_json::json!({
                "run_id": value_to_json(row.get(0)),
                "idx": value_to_json(row.get(1)),
                "start_sec": value_to_json(row.get(2)),
                "end_sec": value_to_json(row.get(3)),
                "speaker": value_to_json(row.get(4)),
                "text": value_to_json(row.get(5)),
                "confidence": value_to_json(row.get(6)),
            });
            writeln!(file, "{}", serde_json::to_string(&obj)?)?;
            count += 1;
        }
    }
    file.flush()?;
    file.sync_all()?;

    Ok(count)
}

/// Export events belonging to a specific set of run_ids.
fn export_table_events_for_runs(
    connection: &Connection,
    path: &Path,
    run_ids: &[String],
) -> FwResult<u64> {
    let mut file = fs::File::create(path)?;
    let mut count = 0u64;

    for run_id in run_ids {
        let rows = connection
            .query_with_params(
                "SELECT run_id, seq, ts_rfc3339, stage, code, message, payload_json \
                 FROM events WHERE run_id = ?1 ORDER BY seq ASC",
                &[SqliteValue::Text(run_id.clone())],
            )
            .map_err(|error| FwError::Storage(error.to_string()))?;

        for row in rows {
            let obj = serde_json::json!({
                "run_id": value_to_json(row.get(0)),
                "seq": value_to_json(row.get(1)),
                "ts_rfc3339": value_to_json(row.get(2)),
                "stage": value_to_json(row.get(3)),
                "code": value_to_json(row.get(4)),
                "message": value_to_json(row.get(5)),
                "payload_json": value_to_json(row.get(6)),
            });
            writeln!(file, "{}", serde_json::to_string(&obj)?)?;
            count += 1;
        }
    }
    file.flush()?;
    file.sync_all()?;

    Ok(count)
}

/// Find the maximum `started_at` among runs matching the incremental filter.
fn max_started_at(connection: &Connection, after_ts: Option<&str>) -> FwResult<Option<String>> {
    let (sql, params) = match after_ts {
        Some(ts) => (
            "SELECT MAX(started_at) FROM runs WHERE started_at > ?1".to_owned(),
            vec![SqliteValue::Text(ts.to_owned())],
        ),
        None => ("SELECT MAX(started_at) FROM runs".to_owned(), vec![]),
    };

    let rows = if params.is_empty() {
        connection
            .query(&sql)
            .map_err(|error| FwError::Storage(error.to_string()))?
    } else {
        connection
            .query_with_params(&sql, &params)
            .map_err(|error| FwError::Storage(error.to_string()))?
    };

    match rows.first() {
        Some(row) => {
            let value = value_to_string_sqlite(row.get(0));
            if value.is_empty() {
                Ok(None)
            } else {
                Ok(Some(value))
            }
        }
        None => Ok(None),
    }
}

// ---------------------------------------------------------------------------
// Import (jsonl -> db)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum ConflictPolicy {
    Reject,
    Overwrite,
}

#[derive(Debug)]
pub struct ImportResult {
    pub runs_imported: u64,
    pub segments_imported: u64,
    pub events_imported: u64,
    pub conflicts: Vec<SyncConflict>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyncConflict {
    pub table: String,
    pub key: String,
    pub reason: String,
}

pub fn import(
    db_path: &Path,
    input_dir: &Path,
    state_root: &Path,
    conflict_policy: ConflictPolicy,
) -> FwResult<ImportResult> {
    let _lock = SyncLock::acquire(state_root, "import")?;
    import_inner(db_path, input_dir, conflict_policy)
}

fn import_inner(
    db_path: &Path,
    input_dir: &Path,
    conflict_policy: ConflictPolicy,
) -> FwResult<ImportResult> {
    // Validate manifest
    let manifest_path = input_dir.join("manifest.json");
    if !manifest_path.exists() {
        return Err(FwError::Storage(format!(
            "manifest.json not found in {}",
            input_dir.display()
        )));
    }
    let manifest_text = fs::read_to_string(&manifest_path)?;
    let manifest: SyncManifest = serde_json::from_str(&manifest_text)
        .map_err(|error| FwError::Storage(format!("invalid manifest: {error}")))?;

    // Validate schema version (exact major match)
    validate_schema_version(&manifest)?;

    // Validate checksums
    validate_checksums(&manifest, input_dir)?;

    // Open DB and ensure schema exists
    if let Some(parent) = db_path.parent() {
        fs::create_dir_all(parent)?;
    }
    let connection = Connection::open(db_path.display().to_string())
        .map_err(|error| FwError::Storage(error.to_string()))?;
    ensure_schema(&connection)?;

    // Begin transaction
    connection
        .execute("BEGIN;")
        .map_err(|error| FwError::Storage(error.to_string()))?;

    let result = import_tables(&connection, input_dir, &manifest, conflict_policy);

    match result {
        Ok(import_result) => {
            if !import_result.conflicts.is_empty() {
                write_conflicts_file(input_dir, &import_result.conflicts)?;
                if conflict_policy == ConflictPolicy::Reject {
                    let _ = connection.execute("ROLLBACK;");
                    return Err(FwError::Storage(format!(
                        "import rejected due to {} conflict(s); see sync_conflicts.jsonl",
                        import_result.conflicts.len()
                    )));
                }
            }

            connection
                .execute("COMMIT;")
                .map_err(|error| FwError::Storage(error.to_string()))?;

            Ok(import_result)
        }
        Err(error) => {
            let _ = connection.execute("ROLLBACK;");
            Err(error)
        }
    }
}

fn import_tables(
    connection: &Connection,
    input_dir: &Path,
    manifest: &SyncManifest,
    conflict_policy: ConflictPolicy,
) -> FwResult<ImportResult> {
    let mut conflicts = Vec::new();

    // Import runs first (referential parent)
    let runs_path = input_dir.join("runs.jsonl");
    let runs_imported = import_runs(connection, &runs_path, conflict_policy, &mut conflicts)?;

    // Validate row count
    if runs_imported != manifest.row_counts.runs && conflicts.is_empty() {
        return Err(FwError::Storage(format!(
            "row count mismatch for runs: expected {}, processed {}",
            manifest.row_counts.runs, runs_imported
        )));
    }

    // Import segments
    let segments_path = input_dir.join("segments.jsonl");
    let segments_imported =
        import_segments(connection, &segments_path, conflict_policy, &mut conflicts)?;

    if segments_imported != manifest.row_counts.segments && conflicts.is_empty() {
        return Err(FwError::Storage(format!(
            "row count mismatch for segments: expected {}, processed {}",
            manifest.row_counts.segments, segments_imported
        )));
    }

    // Import events
    let events_path = input_dir.join("events.jsonl");
    let events_imported = import_events(connection, &events_path, conflict_policy, &mut conflicts)?;

    if events_imported != manifest.row_counts.events && conflicts.is_empty() {
        return Err(FwError::Storage(format!(
            "row count mismatch for events: expected {}, processed {}",
            manifest.row_counts.events, events_imported
        )));
    }

    Ok(ImportResult {
        runs_imported,
        segments_imported,
        events_imported,
        conflicts,
    })
}

fn import_runs(
    connection: &Connection,
    path: &Path,
    conflict_policy: ConflictPolicy,
    conflicts: &mut Vec<SyncConflict>,
) -> FwResult<u64> {
    let file = fs::File::open(path)?;
    let reader = BufReader::new(file);
    let mut count = 0u64;

    for line in reader.lines() {
        let line = line?;
        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        let row: serde_json::Value = serde_json::from_str(line)?;
        let id = json_str(&row, "id")?;

        // Check for existing row
        let existing = connection
            .query_with_params(
                "SELECT id, started_at, finished_at, backend, input_path, normalized_wav_path, request_json, result_json, warnings_json, transcript, replay_json FROM runs WHERE id = ?1",
                &[SqliteValue::Text(id.clone())],
            )
            .map_err(|error| FwError::Storage(error.to_string()))?;

        if !existing.is_empty() {
            // Compare payload: if identical, noop; if different, apply conflict policy
            let existing_row = &existing[0];
            let identical = value_to_string_sqlite(existing_row.get(1))
                == json_str(&row, "started_at")?
                && value_to_string_sqlite(existing_row.get(2)) == json_str(&row, "finished_at")?
                && value_to_string_sqlite(existing_row.get(3)) == json_str(&row, "backend")?
                && value_to_string_sqlite(existing_row.get(4)) == json_str(&row, "input_path")?
                && value_to_string_sqlite(existing_row.get(5))
                    == json_str(&row, "normalized_wav_path")?
                && value_to_string_sqlite(existing_row.get(6)) == json_str(&row, "request_json")?
                && value_to_string_sqlite(existing_row.get(7)) == json_str(&row, "result_json")?
                && value_to_string_sqlite(existing_row.get(8)) == json_str(&row, "warnings_json")?
                && value_to_string_sqlite(existing_row.get(9)) == json_str(&row, "transcript")?
                && value_to_string_sqlite(existing_row.get(10))
                    == json_string_or_default(&row, "replay_json", "{}");

            if identical {
                count += 1;
                continue; // identical — noop
            }

            match conflict_policy {
                ConflictPolicy::Reject => {
                    conflicts.push(SyncConflict {
                        table: "runs".to_owned(),
                        key: id.clone(),
                        reason: "different payload for same id".to_owned(),
                    });
                    count += 1;
                    continue;
                }
                ConflictPolicy::Overwrite => {
                    connection
                        .execute_with_params(
                            "DELETE FROM runs WHERE id = ?1",
                            &[SqliteValue::Text(id.clone())],
                        )
                        .map_err(|error| FwError::Storage(error.to_string()))?;
                }
            }
        }

        connection
            .execute_with_params(
                "INSERT INTO runs (id, started_at, finished_at, backend, input_path, \
                 normalized_wav_path, request_json, result_json, warnings_json, transcript, replay_json) \
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)",
                &[
                    SqliteValue::Text(id),
                    SqliteValue::Text(json_str(&row, "started_at")?),
                    SqliteValue::Text(json_str(&row, "finished_at")?),
                    SqliteValue::Text(json_str(&row, "backend")?),
                    SqliteValue::Text(json_str(&row, "input_path")?),
                    SqliteValue::Text(json_str(&row, "normalized_wav_path")?),
                    SqliteValue::Text(json_str(&row, "request_json")?),
                    SqliteValue::Text(json_str(&row, "result_json")?),
                    SqliteValue::Text(json_str(&row, "warnings_json")?),
                    SqliteValue::Text(json_str(&row, "transcript")?),
                    SqliteValue::Text(json_string_or_default(&row, "replay_json", "{}")),
                ],
            )
            .map_err(|error| FwError::Storage(error.to_string()))?;

        count += 1;
    }

    Ok(count)
}

fn import_segments(
    connection: &Connection,
    path: &Path,
    conflict_policy: ConflictPolicy,
    conflicts: &mut Vec<SyncConflict>,
) -> FwResult<u64> {
    let file = fs::File::open(path)?;
    let reader = BufReader::new(file);
    let mut count = 0u64;

    for line in reader.lines() {
        let line = line?;
        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        let row: serde_json::Value = serde_json::from_str(line)?;
        let run_id = json_str(&row, "run_id")?;
        let idx = row
            .get("idx")
            .and_then(|value| value.as_i64())
            .ok_or_else(|| FwError::Storage("missing idx in segments row".to_owned()))?;

        let key = format!("{run_id}/{idx}");
        ensure_run_reference_exists(connection, &run_id, "segments", &key)?;

        // Check existing
        let existing = connection
            .query_with_params(
                "SELECT run_id, idx, start_sec, end_sec, speaker, text, confidence FROM segments WHERE run_id = ?1 AND idx = ?2",
                &[SqliteValue::Text(run_id.clone()), SqliteValue::Integer(idx)],
            )
            .map_err(|error| FwError::Storage(error.to_string()))?;

        if !existing.is_empty() {
            let existing_row = &existing[0];
            let identical = optional_floats_equal(
                sqlite_to_optional_f64(existing_row.get(2)),
                json_to_optional_f64(&row, "start_sec"),
            ) && optional_floats_equal(
                sqlite_to_optional_f64(existing_row.get(3)),
                json_to_optional_f64(&row, "end_sec"),
            ) && sqlite_to_optional_text(existing_row.get(4))
                == json_to_optional_text(&row, "speaker")
                && value_to_string_sqlite(existing_row.get(5)) == json_str(&row, "text")?
                && optional_floats_equal(
                    sqlite_to_optional_f64(existing_row.get(6)),
                    json_to_optional_f64(&row, "confidence"),
                );

            if identical {
                count += 1;
                continue;
            }

            match conflict_policy {
                ConflictPolicy::Reject => {
                    conflicts.push(SyncConflict {
                        table: "segments".to_owned(),
                        key: key.clone(),
                        reason: "duplicate composite key".to_owned(),
                    });
                    count += 1;
                    continue;
                }
                ConflictPolicy::Overwrite => {
                    connection
                        .execute_with_params(
                            "DELETE FROM segments WHERE run_id = ?1 AND idx = ?2",
                            &[SqliteValue::Text(run_id.clone()), SqliteValue::Integer(idx)],
                        )
                        .map_err(|error| FwError::Storage(error.to_string()))?;
                }
            }
        }

        connection
            .execute_with_params(
                "INSERT INTO segments (run_id, idx, start_sec, end_sec, speaker, text, confidence) \
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
                &[
                    SqliteValue::Text(run_id),
                    SqliteValue::Integer(idx),
                    json_optional_float(&row, "start_sec"),
                    json_optional_float(&row, "end_sec"),
                    json_optional_text(&row, "speaker"),
                    SqliteValue::Text(json_str(&row, "text")?),
                    json_optional_float(&row, "confidence"),
                ],
            )
            .map_err(|error| FwError::Storage(error.to_string()))?;

        count += 1;
    }

    Ok(count)
}

fn import_events(
    connection: &Connection,
    path: &Path,
    conflict_policy: ConflictPolicy,
    conflicts: &mut Vec<SyncConflict>,
) -> FwResult<u64> {
    let file = fs::File::open(path)?;
    let reader = BufReader::new(file);
    let mut count = 0u64;

    for line in reader.lines() {
        let line = line?;
        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        let row: serde_json::Value = serde_json::from_str(line)?;
        let run_id = json_str(&row, "run_id")?;
        let seq = row
            .get("seq")
            .and_then(|value| value.as_i64())
            .ok_or_else(|| FwError::Storage("missing seq in events row".to_owned()))?;

        let key = format!("{run_id}/{seq}");
        ensure_run_reference_exists(connection, &run_id, "events", &key)?;

        let existing = connection
            .query_with_params(
                "SELECT run_id, seq, ts_rfc3339, stage, code, message, payload_json FROM events WHERE run_id = ?1 AND seq = ?2",
                &[SqliteValue::Text(run_id.clone()), SqliteValue::Integer(seq)],
            )
            .map_err(|error| FwError::Storage(error.to_string()))?;

        if !existing.is_empty() {
            let existing_row = &existing[0];
            let identical = value_to_string_sqlite(existing_row.get(2))
                == json_str(&row, "ts_rfc3339")?
                && value_to_string_sqlite(existing_row.get(3)) == json_str(&row, "stage")?
                && value_to_string_sqlite(existing_row.get(4)) == json_str(&row, "code")?
                && value_to_string_sqlite(existing_row.get(5)) == json_str(&row, "message")?
                && value_to_string_sqlite(existing_row.get(6)) == json_str(&row, "payload_json")?;
            if identical {
                count += 1;
                continue;
            }

            match conflict_policy {
                ConflictPolicy::Reject => {
                    conflicts.push(SyncConflict {
                        table: "events".to_owned(),
                        key: key.clone(),
                        reason: "duplicate composite key".to_owned(),
                    });
                    count += 1;
                    continue;
                }
                ConflictPolicy::Overwrite => {
                    connection
                        .execute_with_params(
                            "DELETE FROM events WHERE run_id = ?1 AND seq = ?2",
                            &[SqliteValue::Text(run_id.clone()), SqliteValue::Integer(seq)],
                        )
                        .map_err(|error| FwError::Storage(error.to_string()))?;
                }
            }
        }

        connection
            .execute_with_params(
                "INSERT INTO events (run_id, seq, ts_rfc3339, stage, code, message, payload_json) \
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
                &[
                    SqliteValue::Text(run_id),
                    SqliteValue::Integer(seq),
                    SqliteValue::Text(json_str(&row, "ts_rfc3339")?),
                    SqliteValue::Text(json_str(&row, "stage")?),
                    SqliteValue::Text(json_str(&row, "code")?),
                    SqliteValue::Text(json_str(&row, "message")?),
                    SqliteValue::Text(json_str(&row, "payload_json")?),
                ],
            )
            .map_err(|error| FwError::Storage(error.to_string()))?;

        count += 1;
    }

    Ok(count)
}

fn ensure_run_reference_exists(
    connection: &Connection,
    run_id: &str,
    table: &str,
    key: &str,
) -> FwResult<()> {
    let parent = connection
        .query_with_params(
            "SELECT id FROM runs WHERE id = ?1 LIMIT 1",
            &[SqliteValue::Text(run_id.to_owned())],
        )
        .map_err(|error| FwError::Storage(error.to_string()))?;

    if parent.is_empty() {
        return Err(FwError::Storage(format!(
            "referential integrity violation: {table} row `{key}` references missing runs.id `{run_id}`"
        )));
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Validation helpers
// ---------------------------------------------------------------------------

fn validate_schema_version(manifest: &SyncManifest) -> FwResult<()> {
    let expected_major = SCHEMA_VERSION.split('.').next().unwrap_or("1");
    let actual_major = manifest.schema_version.split('.').next().unwrap_or("0");

    if expected_major != actual_major {
        return Err(FwError::Storage(format!(
            "schema version mismatch: expected major {expected_major}, got {}",
            manifest.schema_version
        )));
    }
    Ok(())
}

fn validate_checksums(manifest: &SyncManifest, input_dir: &Path) -> FwResult<()> {
    let checks = [
        ("runs.jsonl", &manifest.checksums.runs_jsonl_sha256),
        ("segments.jsonl", &manifest.checksums.segments_jsonl_sha256),
        ("events.jsonl", &manifest.checksums.events_jsonl_sha256),
    ];

    for (filename, expected) in checks {
        let path = input_dir.join(filename);
        if !path.exists() {
            return Err(FwError::Storage(format!(
                "missing export file: {}",
                path.display()
            )));
        }
        let actual = sha256_file(&path)?;
        if &actual != expected {
            return Err(FwError::Storage(format!(
                "checksum mismatch for {filename}: expected {expected}, got {actual}"
            )));
        }
    }

    Ok(())
}

fn ensure_schema(connection: &Connection) -> FwResult<()> {
    let sql = r#"
CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY,
    started_at TEXT NOT NULL,
    finished_at TEXT NOT NULL,
    backend TEXT NOT NULL,
    input_path TEXT NOT NULL,
    normalized_wav_path TEXT NOT NULL,
    request_json TEXT NOT NULL,
    result_json TEXT NOT NULL,
    warnings_json TEXT NOT NULL,
    transcript TEXT NOT NULL,
    replay_json TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS segments (
    run_id TEXT NOT NULL,
    idx INTEGER NOT NULL,
    start_sec REAL,
    end_sec REAL,
    speaker TEXT,
    text TEXT NOT NULL,
    confidence REAL,
    PRIMARY KEY (run_id, idx)
);

CREATE TABLE IF NOT EXISTS events (
    run_id TEXT NOT NULL,
    seq INTEGER NOT NULL,
    ts_rfc3339 TEXT NOT NULL,
    stage TEXT NOT NULL,
    code TEXT NOT NULL,
    message TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    PRIMARY KEY (run_id, seq)
);
"#;
    connection
        .execute(sql)
        .map_err(|error| FwError::Storage(error.to_string()))?;
    ensure_runs_replay_column(connection)?;
    Ok(())
}

/// Read-only schema check for export: verifies that required tables exist
/// without creating them.  Returns an error when the DB has no schema (e.g.
/// freshly created by `Connection::open`).
fn verify_schema_exists(connection: &Connection) -> FwResult<()> {
    for table in &["runs", "segments", "events"] {
        let sql = format!("SELECT 1 FROM {table} LIMIT 1");
        if connection.query(&sql).is_err() {
            return Err(FwError::Storage(format!(
                "export requires an existing database with schema; \
                 missing table `{table}`"
            )));
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Utility helpers
// ---------------------------------------------------------------------------

fn atomic_rename(from: &Path, to: &Path) -> FwResult<()> {
    fs::rename(from, to)?;
    sync_parent_dir(to)?;
    Ok(())
}

fn atomic_write_bytes(path: &Path, bytes: &[u8]) -> FwResult<()> {
    let tmp = path.with_extension("tmp");
    {
        let mut file = fs::File::create(&tmp)?;
        file.write_all(bytes)?;
        file.flush()?;
        file.sync_all()?;
    }
    fs::rename(tmp, path)?;
    sync_parent_dir(path)?;
    Ok(())
}

fn sync_parent_dir(path: &Path) -> FwResult<()> {
    #[cfg(unix)]
    {
        if let Some(parent) = path.parent() {
            let dir = fs::File::open(parent)?;
            dir.sync_all()?;
        }
    }
    Ok(())
}

fn sha256_file(path: &Path) -> FwResult<String> {
    let bytes = fs::read(path)?;
    let mut hasher = Sha256::new();
    hasher.update(&bytes);
    Ok(format!("{:x}", hasher.finalize()))
}

fn value_to_json(value: Option<&SqliteValue>) -> serde_json::Value {
    match value {
        Some(SqliteValue::Text(text)) => serde_json::Value::String(text.clone()),
        Some(SqliteValue::Integer(number)) => serde_json::json!(number),
        Some(SqliteValue::Float(number)) => serde_json::json!(number),
        Some(SqliteValue::Blob(blob)) => {
            serde_json::Value::String(format!("<blob:{}>", blob.len()))
        }
        Some(SqliteValue::Null) | None => serde_json::Value::Null,
    }
}

fn value_to_string_sqlite(value: Option<&SqliteValue>) -> String {
    match value {
        Some(SqliteValue::Text(text)) => text.clone(),
        Some(SqliteValue::Integer(number)) => number.to_string(),
        Some(SqliteValue::Float(number)) => number.to_string(),
        _ => String::new(),
    }
}

fn sqlite_to_optional_f64(value: Option<&SqliteValue>) -> Option<f64> {
    match value {
        Some(SqliteValue::Float(number)) => Some(*number),
        Some(SqliteValue::Integer(number)) => Some(*number as f64),
        _ => None,
    }
}

fn sqlite_to_optional_text(value: Option<&SqliteValue>) -> Option<String> {
    match value {
        Some(SqliteValue::Text(text)) => Some(text.clone()),
        _ => None,
    }
}

fn json_str(value: &serde_json::Value, key: &str) -> FwResult<String> {
    value
        .get(key)
        .and_then(|value| value.as_str())
        .map(ToOwned::to_owned)
        .ok_or_else(|| FwError::Storage(format!("missing string field `{key}` in JSONL row")))
}

fn json_string_or_default(value: &serde_json::Value, key: &str, fallback: &str) -> String {
    value
        .get(key)
        .and_then(serde_json::Value::as_str)
        .map(ToOwned::to_owned)
        .unwrap_or_else(|| fallback.to_owned())
}

fn ensure_runs_replay_column(connection: &Connection) -> FwResult<()> {
    let rows = connection
        .query("PRAGMA table_info(runs);")
        .map_err(|error| FwError::Storage(error.to_string()))?;
    let has_replay = rows
        .iter()
        .any(|row| value_to_string_sqlite(row.get(1)) == "replay_json");
    if has_replay {
        return Ok(());
    }

    connection
        .execute("ALTER TABLE runs ADD COLUMN replay_json TEXT NOT NULL DEFAULT '{}';")
        .map_err(|error| FwError::Storage(error.to_string()))?;
    Ok(())
}

fn json_optional_float(value: &serde_json::Value, key: &str) -> SqliteValue {
    match value.get(key) {
        Some(serde_json::Value::Number(number)) => match number.as_f64() {
            Some(float) => SqliteValue::Float(float),
            None => SqliteValue::Null,
        },
        _ => SqliteValue::Null,
    }
}

fn json_optional_text(value: &serde_json::Value, key: &str) -> SqliteValue {
    match value.get(key) {
        Some(serde_json::Value::String(text)) => SqliteValue::Text(text.clone()),
        _ => SqliteValue::Null,
    }
}

fn json_to_optional_f64(value: &serde_json::Value, key: &str) -> Option<f64> {
    value.get(key).and_then(serde_json::Value::as_f64)
}

fn json_to_optional_text(value: &serde_json::Value, key: &str) -> Option<String> {
    value
        .get(key)
        .and_then(serde_json::Value::as_str)
        .map(ToOwned::to_owned)
}

fn optional_floats_equal(left: Option<f64>, right: Option<f64>) -> bool {
    match (left, right) {
        (Some(a), Some(b)) => (a - b).abs() <= 1e-9,
        (None, None) => true,
        _ => false,
    }
}

fn write_conflicts_file(input_dir: &Path, conflicts: &[SyncConflict]) -> FwResult<()> {
    let conflicts_path = input_dir.join("sync_conflicts.jsonl");
    let mut file = fs::File::create(&conflicts_path)?;
    for conflict in conflicts {
        writeln!(file, "{}", serde_json::to_string(conflict)?)?;
    }
    file.flush()?;
    file.sync_all()?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Compression support
// ---------------------------------------------------------------------------

/// Compression mode for JSONL files.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompressionMode {
    /// No compression — plain JSONL.
    None,
    /// Gzip compression (.jsonl.gz).
    Gzip,
}

/// Gzip-compresses a JSONL file, writing the result to `output_path`.
///
/// The input file is read in its entirety, compressed, and written atomically.
pub fn compress_jsonl(input_path: &Path, output_path: &Path) -> FwResult<()> {
    let data = fs::read(input_path)?;
    let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
    encoder.write_all(&data)?;
    let compressed = encoder.finish()?;
    atomic_write_bytes(output_path, &compressed)?;
    Ok(())
}

/// Decompresses a gzip-compressed JSONL file, writing the result to `output_path`.
pub fn decompress_jsonl(input_path: &Path, output_path: &Path) -> FwResult<()> {
    let compressed = fs::read(input_path)?;
    let mut decoder = GzDecoder::new(&compressed[..]);
    let mut decompressed = Vec::new();
    decoder.read_to_end(&mut decompressed)?;
    atomic_write_bytes(output_path, &decompressed)?;
    Ok(())
}

/// Open a JSONL file for reading, transparently decompressing if the path
/// ends with `.gz`.
fn open_jsonl_reader(path: &Path) -> FwResult<Box<dyn BufRead>> {
    let file = fs::File::open(path)?;
    if path.extension().is_some_and(|ext| ext == "gz") {
        let decoder = GzDecoder::new(BufReader::new(file));
        Ok(Box::new(BufReader::new(decoder)))
    } else {
        Ok(Box::new(BufReader::new(file)))
    }
}

// ---------------------------------------------------------------------------
// Sync validation
// ---------------------------------------------------------------------------

/// Report produced by [`validate_sync`] comparing a SQLite database against
/// a JSONL export directory.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyncValidationReport {
    /// Number of runs in the SQLite database.
    pub db_run_count: u64,
    /// Number of runs in the JSONL export.
    pub jsonl_run_count: u64,
    /// Number of segments in the database.
    pub db_segment_count: u64,
    /// Number of segments in the JSONL export.
    pub jsonl_segment_count: u64,
    /// Number of events in the database.
    pub db_event_count: u64,
    /// Number of events in the JSONL export.
    pub jsonl_event_count: u64,
    /// Run IDs present in the database but absent from the JSONL export.
    pub missing_from_jsonl: Vec<String>,
    /// Run IDs present in the JSONL export but absent from the database.
    pub missing_from_db: Vec<String>,
    /// Run IDs where the record exists in both but data differs.
    pub mismatched_records: Vec<String>,
    /// `true` when the database and JSONL export are in perfect agreement.
    pub is_valid: bool,
}

/// Validate that a SQLite database and a JSONL export directory are in sync.
///
/// Opens the database, reads the runs/segments/events JSONL files from
/// `jsonl_dir`, and compares run counts, run IDs, segment counts, and event
/// counts. Supports reading both plain `.jsonl` and compressed `.jsonl.gz`
/// files (the `.gz` variant is preferred when both exist).
pub fn validate_sync(db_path: &Path, jsonl_dir: &Path) -> FwResult<SyncValidationReport> {
    let connection = Connection::open(db_path.display().to_string())
        .map_err(|error| FwError::Storage(error.to_string()))?;
    verify_schema_exists(&connection)?;

    // --- Collect DB run IDs ---
    let db_run_rows = connection
        .query("SELECT id FROM runs ORDER BY id ASC")
        .map_err(|error| FwError::Storage(error.to_string()))?;
    let db_run_ids: HashSet<String> = db_run_rows
        .iter()
        .map(|row| value_to_string_sqlite(row.get(0)))
        .collect();

    // --- Collect DB counts ---
    let db_segment_count = count_table(&connection, "segments")?;
    let db_event_count = count_table(&connection, "events")?;

    // --- Read JSONL run IDs ---
    let runs_path = resolve_jsonl_path(jsonl_dir, "runs");
    let jsonl_run_ids = collect_jsonl_ids(&runs_path, "id")?;

    // --- Count JSONL segments and events ---
    let segments_path = resolve_jsonl_path(jsonl_dir, "segments");
    let jsonl_segment_count = count_jsonl_lines(&segments_path)?;

    let events_path = resolve_jsonl_path(jsonl_dir, "events");
    let jsonl_event_count = count_jsonl_lines(&events_path)?;

    // --- Compare ---
    let mut missing_from_jsonl: Vec<String> =
        db_run_ids.difference(&jsonl_run_ids).cloned().collect();
    missing_from_jsonl.sort();

    let mut missing_from_db: Vec<String> = jsonl_run_ids.difference(&db_run_ids).cloned().collect();
    missing_from_db.sort();

    // --- Compare record content for shared run IDs ---
    let shared_ids: HashSet<&String> = db_run_ids.intersection(&jsonl_run_ids).collect();
    let jsonl_run_map = load_jsonl_run_map(&runs_path)?;
    let mut mismatched_records: Vec<String> = Vec::new();

    for id in &shared_ids {
        let db_rows = connection
            .query_with_params(
                "SELECT id, started_at, finished_at, backend, input_path, \
                 normalized_wav_path, request_json, result_json, warnings_json, \
                 transcript FROM runs WHERE id = ?1",
                &[SqliteValue::Text((*id).clone())],
            )
            .map_err(|error| FwError::Storage(error.to_string()))?;

        if let Some(db_row) = db_rows.first()
            && let Some(jsonl_value) = jsonl_run_map.get(*id)
        {
            let matches = value_to_string_sqlite(db_row.get(1))
                == json_str_or_empty(jsonl_value, "started_at")
                && value_to_string_sqlite(db_row.get(2))
                    == json_str_or_empty(jsonl_value, "finished_at")
                && value_to_string_sqlite(db_row.get(3))
                    == json_str_or_empty(jsonl_value, "backend")
                && value_to_string_sqlite(db_row.get(9))
                    == json_str_or_empty(jsonl_value, "transcript");
            if !matches {
                mismatched_records.push((*id).clone());
            }
        }
    }
    mismatched_records.sort();

    let is_valid = missing_from_jsonl.is_empty()
        && missing_from_db.is_empty()
        && mismatched_records.is_empty()
        && db_run_ids.len() as u64 == jsonl_run_ids.len() as u64
        && db_segment_count == jsonl_segment_count
        && db_event_count == jsonl_event_count;

    Ok(SyncValidationReport {
        db_run_count: db_run_ids.len() as u64,
        jsonl_run_count: jsonl_run_ids.len() as u64,
        db_segment_count,
        jsonl_segment_count,
        db_event_count,
        jsonl_event_count,
        missing_from_jsonl,
        missing_from_db,
        mismatched_records,
        is_valid,
    })
}

/// Resolve a JSONL file path, preferring `.jsonl.gz` over `.jsonl`.
fn resolve_jsonl_path(dir: &Path, stem: &str) -> PathBuf {
    let gz_path = dir.join(format!("{stem}.jsonl.gz"));
    if gz_path.exists() {
        gz_path
    } else {
        dir.join(format!("{stem}.jsonl"))
    }
}

/// Count the rows in a table via `SELECT COUNT(*)`.
fn count_table(connection: &Connection, table: &str) -> FwResult<u64> {
    let sql = format!("SELECT COUNT(*) FROM {table}");
    let rows = connection
        .query(&sql)
        .map_err(|error| FwError::Storage(error.to_string()))?;
    match rows.first() {
        Some(row) => match row.get(0) {
            Some(SqliteValue::Integer(n)) => Ok(*n as u64),
            _ => Ok(0),
        },
        None => Ok(0),
    }
}

/// Collect all values of a given string `key` from a JSONL file into a set.
fn collect_jsonl_ids(path: &Path, key: &str) -> FwResult<HashSet<String>> {
    if !path.exists() {
        return Ok(HashSet::new());
    }
    let reader = open_jsonl_reader(path)?;
    let mut ids = HashSet::new();
    for line in reader.lines() {
        let line = line?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let value: serde_json::Value = serde_json::from_str(trimmed)?;
        if let Some(id_str) = value.get(key).and_then(serde_json::Value::as_str) {
            ids.insert(id_str.to_owned());
        }
    }
    Ok(ids)
}

/// Count non-empty lines in a JSONL file.
fn count_jsonl_lines(path: &Path) -> FwResult<u64> {
    if !path.exists() {
        return Ok(0);
    }
    let reader = open_jsonl_reader(path)?;
    let mut count = 0u64;
    for line in reader.lines() {
        let line = line?;
        if !line.trim().is_empty() {
            count += 1;
        }
    }
    Ok(count)
}

/// Load all runs from a JSONL file into a map keyed by run ID.
fn load_jsonl_run_map(
    path: &Path,
) -> FwResult<std::collections::HashMap<String, serde_json::Value>> {
    let mut map = std::collections::HashMap::new();
    if !path.exists() {
        return Ok(map);
    }
    let reader = open_jsonl_reader(path)?;
    for line in reader.lines() {
        let line = line?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let value: serde_json::Value = serde_json::from_str(trimmed)?;
        if let Some(id_str) = value.get("id").and_then(serde_json::Value::as_str) {
            map.insert(id_str.to_owned(), value);
        }
    }
    Ok(map)
}

/// Extract a string field from a JSON value, returning an empty string on
/// missing or non-string values.
fn json_str_or_empty(value: &serde_json::Value, key: &str) -> String {
    value
        .get(key)
        .and_then(serde_json::Value::as_str)
        .unwrap_or("")
        .to_owned()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    #[cfg(unix)]
    use std::os::unix::fs::PermissionsExt;
    use std::path::PathBuf;

    use serde_json::json;
    use tempfile::tempdir;

    use crate::model::{
        BackendKind, BackendParams, InputSource, RunEvent, RunReport, TranscribeRequest,
        TranscriptionResult, TranscriptionSegment,
    };
    use crate::storage::RunStore;

    use super::*;

    fn write_jsonl_snapshot(
        output_dir: &Path,
        runs_rows: &[serde_json::Value],
        segments_rows: &[serde_json::Value],
        events_rows: &[serde_json::Value],
    ) {
        fs::create_dir_all(output_dir).expect("output dir should exist");

        let runs_path = output_dir.join("runs.jsonl");
        let segments_path = output_dir.join("segments.jsonl");
        let events_path = output_dir.join("events.jsonl");

        let runs_jsonl = runs_rows
            .iter()
            .map(|row| serde_json::to_string(row).expect("runs row should serialize"))
            .collect::<Vec<_>>()
            .join("\n");
        let segments_jsonl = segments_rows
            .iter()
            .map(|row| serde_json::to_string(row).expect("segments row should serialize"))
            .collect::<Vec<_>>()
            .join("\n");
        let events_jsonl = events_rows
            .iter()
            .map(|row| serde_json::to_string(row).expect("events row should serialize"))
            .collect::<Vec<_>>()
            .join("\n");

        fs::write(
            &runs_path,
            if runs_jsonl.is_empty() {
                String::new()
            } else {
                format!("{runs_jsonl}\n")
            },
        )
        .expect("runs jsonl should write");
        fs::write(
            &segments_path,
            if segments_jsonl.is_empty() {
                String::new()
            } else {
                format!("{segments_jsonl}\n")
            },
        )
        .expect("segments jsonl should write");
        fs::write(
            &events_path,
            if events_jsonl.is_empty() {
                String::new()
            } else {
                format!("{events_jsonl}\n")
            },
        )
        .expect("events jsonl should write");

        let manifest = SyncManifest {
            schema_version: SCHEMA_VERSION.to_owned(),
            export_format_version: EXPORT_FORMAT_VERSION.to_owned(),
            created_at_rfc3339: Utc::now().to_rfc3339(),
            source_db_path: "fixture.sqlite3".to_owned(),
            row_counts: RowCounts {
                runs: runs_rows.len() as u64,
                segments: segments_rows.len() as u64,
                events: events_rows.len() as u64,
            },
            checksums: FileChecksums {
                runs_jsonl_sha256: sha256_file(&runs_path).expect("runs checksum"),
                segments_jsonl_sha256: sha256_file(&segments_path).expect("segments checksum"),
                events_jsonl_sha256: sha256_file(&events_path).expect("events checksum"),
            },
        };

        fs::write(
            output_dir.join("manifest.json"),
            serde_json::to_string_pretty(&manifest).expect("manifest should serialize"),
        )
        .expect("manifest should write");
    }

    fn fixture_report(id: &str, db_path: &Path) -> RunReport {
        RunReport {
            run_id: id.to_owned(),
            trace_id: "00000000000000000000000000000000".to_owned(),
            started_at_rfc3339: "2026-01-01T00:00:00Z".to_owned(),
            finished_at_rfc3339: "2026-01-01T00:00:05Z".to_owned(),
            input_path: "test.wav".to_owned(),
            normalized_wav_path: "normalized.wav".to_owned(),
            request: TranscribeRequest {
                input: InputSource::File {
                    path: PathBuf::from("test.wav"),
                },
                backend: BackendKind::Auto,
                model: None,
                language: Some("en".to_owned()),
                translate: false,
                diarize: false,
                persist: true,
                db_path: db_path.to_path_buf(),
                timeout_ms: None,
                backend_params: BackendParams::default(),
            },
            result: TranscriptionResult {
                backend: BackendKind::WhisperCpp,
                transcript: "hello world from sync test".to_owned(),
                language: Some("en".to_owned()),
                segments: vec![
                    TranscriptionSegment {
                        start_sec: Some(0.0),
                        end_sec: Some(2.5),
                        text: "hello world".to_owned(),
                        speaker: Some("SPEAKER_00".to_owned()),
                        confidence: Some(0.95),
                    },
                    TranscriptionSegment {
                        start_sec: Some(2.5),
                        end_sec: Some(5.0),
                        text: "from sync test".to_owned(),
                        speaker: None,
                        confidence: None,
                    },
                ],
                acceleration: None,
                raw_output: json!({"test": true}),
                artifact_paths: vec!["out.json".to_owned()],
            },
            events: vec![
                RunEvent {
                    seq: 1,
                    ts_rfc3339: "2026-01-01T00:00:01Z".to_owned(),
                    stage: "ingest".to_owned(),
                    code: "ingest.ok".to_owned(),
                    message: "input materialized".to_owned(),
                    payload: json!({"path": "test.wav"}),
                },
                RunEvent {
                    seq: 2,
                    ts_rfc3339: "2026-01-01T00:00:03Z".to_owned(),
                    stage: "backend".to_owned(),
                    code: "backend.ok".to_owned(),
                    message: "backend completed".to_owned(),
                    payload: json!({"resolved_backend": "whisper_cpp"}),
                },
            ],
            warnings: vec!["test warning".to_owned()],
            evidence: vec![],
            replay: crate::model::ReplayEnvelope {
                input_content_hash: Some("sync-input-hash".to_owned()),
                backend_identity: Some("whisper-cli".to_owned()),
                backend_version: Some("whisper 1.2.3".to_owned()),
                output_payload_hash: Some("sync-output-hash".to_owned()),
            },
        }
    }

    #[test]
    fn export_empty_db_produces_valid_manifest() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("test.sqlite3");
        let export_dir = dir.path().join("export");
        let state_root = dir.path().join("state");

        // Open DB to create schema
        let _store = RunStore::open(&db_path).expect("store open");

        let manifest = export(&db_path, &export_dir, &state_root).expect("export should succeed");

        assert_eq!(manifest.schema_version, "1.1");
        assert_eq!(manifest.row_counts.runs, 0);
        assert_eq!(manifest.row_counts.segments, 0);
        assert_eq!(manifest.row_counts.events, 0);

        // Verify files exist
        assert!(export_dir.join("manifest.json").exists());
        assert!(export_dir.join("runs.jsonl").exists());
        assert!(export_dir.join("segments.jsonl").exists());
        assert!(export_dir.join("events.jsonl").exists());
    }

    #[test]
    fn export_import_round_trip() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("source.sqlite3");
        let export_dir = dir.path().join("export");
        let state_root = dir.path().join("state");

        // Persist a report to source DB
        let store = RunStore::open(&db_path).expect("store open");
        let report = fixture_report("run-sync-1", &db_path);
        store.persist_report(&report).expect("persist");

        // Export
        let manifest = export(&db_path, &export_dir, &state_root).expect("export");
        assert_eq!(manifest.row_counts.runs, 1);
        assert_eq!(manifest.row_counts.segments, 2);
        assert_eq!(manifest.row_counts.events, 2);

        // Import into a fresh DB
        let target_db = dir.path().join("target.sqlite3");
        let result =
            import(&target_db, &export_dir, &state_root, ConflictPolicy::Reject).expect("import");

        assert_eq!(result.runs_imported, 1);
        assert_eq!(result.segments_imported, 2);
        assert_eq!(result.events_imported, 2);
        assert!(result.conflicts.is_empty());

        // Verify data in target DB
        let target_store = RunStore::open(&target_db).expect("target store");
        let runs = target_store.list_recent_runs(10).expect("list runs");
        assert_eq!(runs.len(), 1);
        assert!(runs[0].transcript_preview.contains("hello world"));
        let details = target_store
            .load_run_details("run-sync-1")
            .expect("load")
            .expect("details");
        assert_eq!(
            details.replay.input_content_hash.as_deref(),
            Some("sync-input-hash")
        );
        assert_eq!(
            details.replay.backend_identity.as_deref(),
            Some("whisper-cli")
        );
        assert_eq!(
            details.replay.backend_version.as_deref(),
            Some("whisper 1.2.3")
        );
        assert_eq!(
            details.replay.output_payload_hash.as_deref(),
            Some("sync-output-hash")
        );
    }

    #[test]
    fn import_detects_checksum_mismatch() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("source.sqlite3");
        let export_dir = dir.path().join("export");
        let state_root = dir.path().join("state");

        let _store = RunStore::open(&db_path).expect("store open");
        export(&db_path, &export_dir, &state_root).expect("export");

        // Corrupt a file
        fs::write(export_dir.join("runs.jsonl"), "corrupted data\n").expect("write");

        let target_db = dir.path().join("target.sqlite3");
        let result = import(&target_db, &export_dir, &state_root, ConflictPolicy::Reject);

        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("checksum mismatch"));
    }

    #[test]
    fn import_duplicate_noop_on_identical_payload() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("source.sqlite3");
        let export_dir = dir.path().join("export");
        let state_root = dir.path().join("state");

        let store = RunStore::open(&db_path).expect("store open");
        let report = fixture_report("run-dup-1", &db_path);
        store.persist_report(&report).expect("persist");
        export(&db_path, &export_dir, &state_root).expect("export");

        // Import back into the same DB (all rows already exist, identical)
        let result = import(&db_path, &export_dir, &state_root, ConflictPolicy::Reject)
            .expect("import same data");

        assert_eq!(result.runs_imported, 1);
        assert!(
            result.conflicts.is_empty(),
            "expected no conflicts, got: {:?}",
            result.conflicts
        );
    }

    #[test]
    fn import_reject_policy_fails_on_conflicting_payload_and_writes_conflicts() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("source.sqlite3");
        let export_dir = dir.path().join("export");
        let state_root = dir.path().join("state");

        let store = RunStore::open(&db_path).expect("store open");
        let report = fixture_report("run-conflict-1", &db_path);
        store.persist_report(&report).expect("persist");
        export(&db_path, &export_dir, &state_root).expect("export");

        let runs_path = export_dir.join("runs.jsonl");
        let first_line = fs::read_to_string(&runs_path)
            .expect("runs.jsonl should be readable")
            .lines()
            .next()
            .expect("runs.jsonl should contain first line")
            .to_owned();
        let mut mutated: serde_json::Value =
            serde_json::from_str(&first_line).expect("valid runs line");
        mutated["transcript"] = serde_json::json!("mutated transcript payload");
        fs::write(
            &runs_path,
            format!("{}\n", serde_json::to_string(&mutated).expect("serialize")),
        )
        .expect("runs.jsonl overwrite should succeed");

        let manifest = export_dir.join("manifest.json");
        let mut manifest_value: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(&manifest).expect("manifest"))
                .expect("manifest json");
        manifest_value["checksums"]["runs_jsonl_sha256"] =
            serde_json::json!(sha256_file(&runs_path).expect("checksum"));
        fs::write(
            &manifest,
            serde_json::to_string_pretty(&manifest_value).expect("serialize manifest"),
        )
        .expect("manifest rewrite should succeed");

        let result = import(&db_path, &export_dir, &state_root, ConflictPolicy::Reject);
        assert!(result.is_err());
        let conflicts_path = export_dir.join("sync_conflicts.jsonl");
        assert!(conflicts_path.exists());
        let conflicts = fs::read_to_string(conflicts_path).expect("conflicts should be readable");
        assert!(conflicts.contains("\"table\":\"runs\""));
    }

    #[test]
    fn import_overwrite_policy_replaces_conflicting_payload() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("source.sqlite3");
        let export_dir = dir.path().join("export");
        let state_root = dir.path().join("state");

        let store = RunStore::open(&db_path).expect("store open");
        let report = fixture_report("run-overwrite-1", &db_path);
        store.persist_report(&report).expect("persist");
        export(&db_path, &export_dir, &state_root).expect("export");

        let runs_path = export_dir.join("runs.jsonl");
        let first_line = fs::read_to_string(&runs_path)
            .expect("runs.jsonl should be readable")
            .lines()
            .next()
            .expect("runs.jsonl should contain first line")
            .to_owned();
        let mut mutated: serde_json::Value =
            serde_json::from_str(&first_line).expect("valid runs line");
        mutated["transcript"] = serde_json::json!("overwrite transcript");
        let mut result_json: serde_json::Value = serde_json::from_str(
            mutated["result_json"]
                .as_str()
                .expect("result_json string should be present"),
        )
        .expect("result_json should parse");
        result_json["transcript"] = serde_json::json!("overwrite transcript");
        mutated["result_json"] =
            serde_json::json!(serde_json::to_string(&result_json).expect("serialize result json"));
        fs::write(
            &runs_path,
            format!("{}\n", serde_json::to_string(&mutated).expect("serialize")),
        )
        .expect("runs.jsonl overwrite should succeed");

        let manifest = export_dir.join("manifest.json");
        let mut manifest_value: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(&manifest).expect("manifest"))
                .expect("manifest json");
        manifest_value["checksums"]["runs_jsonl_sha256"] =
            serde_json::json!(sha256_file(&runs_path).expect("checksum"));
        fs::write(
            &manifest,
            serde_json::to_string_pretty(&manifest_value).expect("serialize manifest"),
        )
        .expect("manifest rewrite should succeed");

        let result = import(
            &db_path,
            &export_dir,
            &state_root,
            ConflictPolicy::Overwrite,
        )
        .expect("overwrite import should succeed");
        assert_eq!(result.runs_imported, 1);

        let store = RunStore::open(&db_path).expect("store open");
        let run = store
            .load_run_details("run-overwrite-1")
            .expect("load should succeed")
            .expect("run should exist");
        assert_eq!(run.transcript, "overwrite transcript");
    }

    #[test]
    fn lock_prevents_concurrent_sync() {
        let dir = tempdir().expect("tempdir");
        let state_root = dir.path().join("state");

        let lock = SyncLock::acquire(&state_root, "test").expect("first lock");

        // Second acquire should fail
        let second = SyncLock::acquire(&state_root, "test");
        assert!(second.is_err());

        lock.release().expect("release");

        // Now should succeed
        let third = SyncLock::acquire(&state_root, "test").expect("after release");
        third.release().expect("release");
    }

    #[test]
    fn export_error_releases_lock_via_drop_guard() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("missing-schema.sqlite3");
        let export_dir = dir.path().join("export");
        let state_root = dir.path().join("state");

        let result = export(&db_path, &export_dir, &state_root);
        assert!(result.is_err(), "export should fail when schema is absent");

        let lock_path = state_root.join("locks/sync.lock");
        assert!(
            !lock_path.exists(),
            "lock file should be removed even when export fails"
        );
    }

    #[test]
    fn corrupt_lock_is_archived_and_replaced() {
        let dir = tempdir().expect("tempdir");
        let state_root = dir.path().join("state");
        let locks_dir = state_root.join("locks");
        fs::create_dir_all(&locks_dir).expect("locks dir");
        fs::write(locks_dir.join("sync.lock"), "{not-json").expect("write corrupt lock");

        let lock = SyncLock::acquire(&state_root, "test").expect("acquire should recover");
        lock.release().expect("release");

        let archived = fs::read_dir(&locks_dir)
            .expect("read dir")
            .filter_map(Result::ok)
            .map(|entry| entry.file_name().to_string_lossy().to_string())
            .any(|name| name.starts_with("sync.lock.corrupt."));
        assert!(archived, "corrupt lock should be archived");
    }

    #[test]
    fn stale_lock_is_archived_before_new_lock_is_granted() {
        let dir = tempdir().expect("tempdir");
        let state_root = dir.path().join("state");
        let locks_dir = state_root.join("locks");
        fs::create_dir_all(&locks_dir).expect("locks dir");

        let stale = LockInfo {
            pid: u32::MAX - 1,
            created_at_rfc3339: "2000-01-01T00:00:00Z".to_owned(),
            operation: "export".to_owned(),
        };
        fs::write(
            locks_dir.join("sync.lock"),
            serde_json::to_string_pretty(&stale).expect("serialize"),
        )
        .expect("write stale lock");

        let lock = SyncLock::acquire(&state_root, "import").expect("acquire should recover");
        lock.release().expect("release");

        let archived = fs::read_dir(&locks_dir)
            .expect("read dir")
            .filter_map(Result::ok)
            .map(|entry| entry.file_name().to_string_lossy().to_string())
            .any(|name| name.starts_with("sync.lock.stale."));
        assert!(archived, "stale lock should be archived");
    }

    #[test]
    fn import_rejects_segment_row_with_missing_parent_run() {
        let dir = tempdir().expect("tempdir");
        let export_dir = dir.path().join("export");
        let state_root = dir.path().join("state");
        let target_db = dir.path().join("target.sqlite3");

        write_jsonl_snapshot(
            &export_dir,
            &[],
            &[json!({
                "run_id": "missing-run",
                "idx": 0,
                "start_sec": 0.0,
                "end_sec": 1.0,
                "speaker": serde_json::Value::Null,
                "text": "orphan segment",
                "confidence": 0.9,
            })],
            &[],
        );

        let error = import(&target_db, &export_dir, &state_root, ConflictPolicy::Reject)
            .expect_err("import should fail on orphan segment reference");
        let error_text = error.to_string();
        assert!(error_text.contains("referential integrity violation"));
        assert!(error_text.contains("segments"));
        assert!(error_text.contains("missing-run"));
    }

    #[test]
    fn import_rejects_event_row_with_missing_parent_run() {
        let dir = tempdir().expect("tempdir");
        let export_dir = dir.path().join("export");
        let state_root = dir.path().join("state");
        let target_db = dir.path().join("target.sqlite3");

        write_jsonl_snapshot(
            &export_dir,
            &[],
            &[],
            &[json!({
                "run_id": "missing-run",
                "seq": 1,
                "ts_rfc3339": "2026-01-01T00:00:01Z",
                "stage": "backend",
                "code": "backend.ok",
                "message": "completed",
                "payload_json": "{}",
            })],
        );

        let error = import(&target_db, &export_dir, &state_root, ConflictPolicy::Reject)
            .expect_err("import should fail on orphan event reference");
        let error_text = error.to_string();
        assert!(error_text.contains("referential integrity violation"));
        assert!(error_text.contains("events"));
        assert!(error_text.contains("missing-run"));
    }

    #[test]
    fn import_rejects_major_schema_version_mismatch() {
        let dir = tempdir().expect("tempdir");
        let export_dir = dir.path().join("export");
        let state_root = dir.path().join("state");
        let target_db = dir.path().join("target.sqlite3");

        write_jsonl_snapshot(&export_dir, &[], &[], &[]);

        let manifest_path = export_dir.join("manifest.json");
        let mut manifest_value: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(&manifest_path).expect("manifest read"))
                .expect("manifest json");
        manifest_value["schema_version"] = json!("2.0");
        fs::write(
            &manifest_path,
            serde_json::to_string_pretty(&manifest_value).expect("manifest serialize"),
        )
        .expect("manifest write");

        let error = import(&target_db, &export_dir, &state_root, ConflictPolicy::Reject)
            .expect_err("major schema mismatch must fail");
        assert!(error.to_string().contains("schema version mismatch"));
    }

    #[test]
    fn import_accepts_backward_compatible_minor_schema_version() {
        let dir = tempdir().expect("tempdir");
        let export_dir = dir.path().join("export");
        let state_root = dir.path().join("state");
        let target_db = dir.path().join("target.sqlite3");

        write_jsonl_snapshot(&export_dir, &[], &[], &[]);

        let manifest_path = export_dir.join("manifest.json");
        let mut manifest_value: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(&manifest_path).expect("manifest read"))
                .expect("manifest json");
        manifest_value["schema_version"] = json!("1.7");
        fs::write(
            &manifest_path,
            serde_json::to_string_pretty(&manifest_value).expect("manifest serialize"),
        )
        .expect("manifest write");

        let result = import(&target_db, &export_dir, &state_root, ConflictPolicy::Reject)
            .expect("minor version should be accepted");
        assert_eq!(result.runs_imported, 0);
        assert_eq!(result.segments_imported, 0);
        assert_eq!(result.events_imported, 0);
    }

    #[test]
    fn import_row_count_mismatch_errors_when_no_conflicts_present() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("source.sqlite3");
        let export_dir = dir.path().join("export");
        let state_root = dir.path().join("state");

        let store = RunStore::open(&db_path).expect("store open");
        let report = fixture_report("row-mismatch-1", &db_path);
        store.persist_report(&report).expect("persist");
        export(&db_path, &export_dir, &state_root).expect("export");

        let manifest_path = export_dir.join("manifest.json");
        let mut manifest_value: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(&manifest_path).expect("manifest read"))
                .expect("manifest json");
        manifest_value["row_counts"]["events"] = json!(9999);
        fs::write(
            &manifest_path,
            serde_json::to_string_pretty(&manifest_value).expect("manifest serialize"),
        )
        .expect("manifest write");

        // keep checksum valid by not mutating data files.
        let target_db = dir.path().join("target.sqlite3");
        let error = import(&target_db, &export_dir, &state_root, ConflictPolicy::Reject)
            .expect_err("row mismatch should fail");
        assert!(error.to_string().contains("row count mismatch for events"));
    }

    #[test]
    fn import_row_count_mismatch_is_deferred_when_conflicts_exist() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("source.sqlite3");
        let export_dir = dir.path().join("export");
        let state_root = dir.path().join("state");

        let store = RunStore::open(&db_path).expect("store open");
        let report = fixture_report("row-mismatch-conflict-1", &db_path);
        store.persist_report(&report).expect("persist");
        export(&db_path, &export_dir, &state_root).expect("export");

        let runs_path = export_dir.join("runs.jsonl");
        let first_line = fs::read_to_string(&runs_path)
            .expect("runs.jsonl should be readable")
            .lines()
            .next()
            .expect("runs.jsonl should contain first line")
            .to_owned();
        let mut mutated: serde_json::Value =
            serde_json::from_str(&first_line).expect("valid runs line");
        mutated["transcript"] = json!("mutated transcript payload");
        fs::write(
            &runs_path,
            format!("{}\n", serde_json::to_string(&mutated).expect("serialize")),
        )
        .expect("runs write");

        let manifest_path = export_dir.join("manifest.json");
        let mut manifest_value: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(&manifest_path).expect("manifest read"))
                .expect("manifest json");
        manifest_value["checksums"]["runs_jsonl_sha256"] =
            json!(sha256_file(&runs_path).expect("runs checksum"));
        manifest_value["row_counts"]["events"] = json!(9999);
        fs::write(
            &manifest_path,
            serde_json::to_string_pretty(&manifest_value).expect("manifest serialize"),
        )
        .expect("manifest write");

        let error = import(&db_path, &export_dir, &state_root, ConflictPolicy::Reject)
            .expect_err("conflict should fail before row-count mismatch");
        let text = error.to_string();
        assert!(text.contains("import rejected due to"));
        assert!(!text.contains("row count mismatch"));
    }

    #[cfg(unix)]
    #[test]
    fn import_surfaces_error_when_conflicts_file_cannot_be_written() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("source.sqlite3");
        let export_dir = dir.path().join("export");
        let state_root = dir.path().join("state");

        let store = RunStore::open(&db_path).expect("store open");
        let report = fixture_report("conflict-write-fail-1", &db_path);
        store.persist_report(&report).expect("persist");
        export(&db_path, &export_dir, &state_root).expect("export");

        let runs_path = export_dir.join("runs.jsonl");
        let first_line = fs::read_to_string(&runs_path)
            .expect("runs.jsonl should be readable")
            .lines()
            .next()
            .expect("runs.jsonl should contain first line")
            .to_owned();
        let mut mutated: serde_json::Value =
            serde_json::from_str(&first_line).expect("valid runs line");
        mutated["transcript"] = json!("mutated transcript payload");
        fs::write(
            &runs_path,
            format!("{}\n", serde_json::to_string(&mutated).expect("serialize")),
        )
        .expect("runs write");

        let manifest_path = export_dir.join("manifest.json");
        let mut manifest_value: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(&manifest_path).expect("manifest read"))
                .expect("manifest json");
        manifest_value["checksums"]["runs_jsonl_sha256"] =
            json!(sha256_file(&runs_path).expect("runs checksum"));
        fs::write(
            &manifest_path,
            serde_json::to_string_pretty(&manifest_value).expect("manifest serialize"),
        )
        .expect("manifest write");

        let original_permissions = fs::metadata(&export_dir).expect("metadata").permissions();
        let mut read_only = original_permissions.clone();
        read_only.set_mode(0o555);
        fs::set_permissions(&export_dir, read_only).expect("set readonly dir");

        let result = import(&db_path, &export_dir, &state_root, ConflictPolicy::Reject);

        let mut writable = original_permissions;
        writable.set_mode(0o755);
        fs::set_permissions(&export_dir, writable).expect("restore writable dir");

        let error = result.expect_err("conflict artifact write should fail");
        let text = error.to_string();
        assert!(
            text.contains("Permission denied") || text.contains("permission denied"),
            "unexpected error: {text}"
        );
    }

    #[test]
    fn stale_lock_archive_name_follows_deterministic_pattern() {
        let dir = tempdir().expect("tempdir");
        let state_root = dir.path().join("state");
        let locks_dir = state_root.join("locks");
        fs::create_dir_all(&locks_dir).expect("locks dir");

        let stale = LockInfo {
            pid: u32::MAX - 1,
            created_at_rfc3339: "2000-01-01T00:00:00Z".to_owned(),
            operation: "export".to_owned(),
        };
        fs::write(
            locks_dir.join("sync.lock"),
            serde_json::to_string_pretty(&stale).expect("serialize"),
        )
        .expect("write stale lock");

        let _lock = SyncLock::acquire(&state_root, "test").expect("acquire");

        let archived: Vec<String> = fs::read_dir(&locks_dir)
            .expect("read dir")
            .filter_map(Result::ok)
            .map(|entry| entry.file_name().to_string_lossy().to_string())
            .filter(|name| name.starts_with("sync.lock.stale."))
            .collect();

        assert_eq!(archived.len(), 1, "exactly one stale archive should exist");
        // Pattern: sync.lock.stale.{unix_timestamp}.json
        let name = &archived[0];
        assert!(name.ends_with(".json"), "archive should end with .json");
        let parts: Vec<&str> = name.split('.').collect();
        assert_eq!(parts.len(), 5, "expected sync.lock.stale.<ts>.json");
        assert!(
            parts[3].parse::<u64>().is_ok(),
            "timestamp part should be a valid u64"
        );
    }

    #[test]
    fn double_export_produces_consistent_manifest() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("source.sqlite3");
        let state_root = dir.path().join("state");

        let store = RunStore::open(&db_path).expect("store open");
        let report = fixture_report("double-export-1", &db_path);
        store.persist_report(&report).expect("persist");

        let export_dir_1 = dir.path().join("export1");
        let manifest_1 = export(&db_path, &export_dir_1, &state_root).expect("export 1");

        let export_dir_2 = dir.path().join("export2");
        let manifest_2 = export(&db_path, &export_dir_2, &state_root).expect("export 2");

        assert_eq!(manifest_1.row_counts, manifest_2.row_counts);
        assert_eq!(manifest_1.schema_version, manifest_2.schema_version);
        assert_eq!(manifest_1.checksums, manifest_2.checksums);
    }

    // -----------------------------------------------------------------------
    // Helper function unit tests
    // -----------------------------------------------------------------------

    #[test]
    fn is_lock_stale_returns_true_for_old_dead_pid() {
        let info = LockInfo {
            pid: u32::MAX - 1, // Almost certainly doesn't exist
            created_at_rfc3339: "2000-01-01T00:00:00Z".to_owned(),
            operation: "test".to_owned(),
        };
        assert!(is_lock_stale(&info));
    }

    #[test]
    fn is_lock_stale_returns_true_for_ancient_timestamp() {
        // Even if PID exists (current process), an ancient timestamp is stale.
        let info = LockInfo {
            pid: std::process::id(),
            created_at_rfc3339: "2000-01-01T00:00:00Z".to_owned(),
            operation: "test".to_owned(),
        };
        assert!(is_lock_stale(&info));
    }

    #[test]
    fn is_lock_stale_returns_false_for_current_pid_and_recent_time() {
        let info = LockInfo {
            pid: std::process::id(),
            created_at_rfc3339: Utc::now().to_rfc3339(),
            operation: "test".to_owned(),
        };
        assert!(!is_lock_stale(&info));
    }

    #[test]
    fn pid_is_alive_current_process_returns_true() {
        assert!(pid_is_alive(std::process::id()));
    }

    #[test]
    fn optional_floats_equal_both_none() {
        assert!(optional_floats_equal(None, None));
    }

    #[test]
    fn optional_floats_equal_both_same() {
        assert!(optional_floats_equal(Some(1.5), Some(1.5)));
    }

    #[test]
    fn optional_floats_equal_mismatched_some_none() {
        assert!(!optional_floats_equal(Some(1.0), None));
        assert!(!optional_floats_equal(None, Some(1.0)));
    }

    #[test]
    fn optional_floats_equal_within_epsilon() {
        assert!(optional_floats_equal(Some(1.0), Some(1.0 + 1e-10)));
    }

    #[test]
    fn optional_floats_equal_outside_epsilon() {
        assert!(!optional_floats_equal(Some(1.0), Some(1.0 + 1e-8)));
    }

    #[test]
    fn json_str_returns_error_on_missing_key() {
        let value = json!({"a": "hello"});
        let err = json_str(&value, "missing").expect_err("missing key");
        assert!(err.to_string().contains("missing string field"));
    }

    #[test]
    fn json_string_or_default_uses_fallback() {
        let value = json!({"a": "hello"});
        assert_eq!(json_string_or_default(&value, "a", "def"), "hello");
        assert_eq!(json_string_or_default(&value, "missing", "def"), "def");
    }

    #[test]
    fn value_to_json_all_variants() {
        use serde_json::Value as JV;
        assert_eq!(
            value_to_json(Some(&SqliteValue::Text("hi".to_owned()))),
            JV::String("hi".to_owned())
        );
        assert_eq!(value_to_json(Some(&SqliteValue::Integer(42))), json!(42));
        assert_eq!(value_to_json(Some(&SqliteValue::Float(1.5))), json!(1.5));
        assert!(matches!(
            value_to_json(Some(&SqliteValue::Blob(vec![1, 2]))),
            JV::String(_)
        ));
        assert_eq!(value_to_json(Some(&SqliteValue::Null)), JV::Null);
        assert_eq!(value_to_json(None), JV::Null);
    }

    #[test]
    fn validate_schema_version_rejects_major_mismatch() {
        let manifest = SyncManifest {
            schema_version: "2.0".to_owned(),
            export_format_version: "1.0".to_owned(),
            created_at_rfc3339: "2026-01-01T00:00:00Z".to_owned(),
            source_db_path: "db.sqlite3".to_owned(),
            row_counts: RowCounts {
                runs: 0,
                segments: 0,
                events: 0,
            },
            checksums: FileChecksums {
                runs_jsonl_sha256: String::new(),
                segments_jsonl_sha256: String::new(),
                events_jsonl_sha256: String::new(),
            },
        };
        let err = validate_schema_version(&manifest).expect_err("should reject");
        assert!(err.to_string().contains("schema version mismatch"));
    }

    #[test]
    fn validate_schema_version_accepts_minor_difference() {
        let manifest = SyncManifest {
            schema_version: "1.99".to_owned(),
            export_format_version: "1.0".to_owned(),
            created_at_rfc3339: "2026-01-01T00:00:00Z".to_owned(),
            source_db_path: "db.sqlite3".to_owned(),
            row_counts: RowCounts {
                runs: 0,
                segments: 0,
                events: 0,
            },
            checksums: FileChecksums {
                runs_jsonl_sha256: String::new(),
                segments_jsonl_sha256: String::new(),
                events_jsonl_sha256: String::new(),
            },
        };
        assert!(validate_schema_version(&manifest).is_ok());
    }

    // ── Additional helper-function unit tests ──

    #[test]
    fn value_to_string_sqlite_all_variants() {
        assert_eq!(
            value_to_string_sqlite(Some(&SqliteValue::Text("hello".to_owned()))),
            "hello"
        );
        assert_eq!(
            value_to_string_sqlite(Some(&SqliteValue::Integer(42))),
            "42"
        );
        assert_eq!(
            value_to_string_sqlite(Some(&SqliteValue::Float(3.5))),
            "3.5"
        );
        assert_eq!(
            value_to_string_sqlite(Some(&SqliteValue::Blob(vec![1, 2, 3]))),
            ""
        );
        assert_eq!(value_to_string_sqlite(Some(&SqliteValue::Null)), "");
        assert_eq!(value_to_string_sqlite(None), "");
    }

    #[test]
    fn sqlite_to_optional_f64_all_variants() {
        assert_eq!(
            sqlite_to_optional_f64(Some(&SqliteValue::Float(1.5))),
            Some(1.5)
        );
        assert_eq!(
            sqlite_to_optional_f64(Some(&SqliteValue::Integer(7))),
            Some(7.0)
        );
        assert_eq!(
            sqlite_to_optional_f64(Some(&SqliteValue::Text("nope".to_owned()))),
            None
        );
        assert_eq!(sqlite_to_optional_f64(Some(&SqliteValue::Null)), None);
        assert_eq!(sqlite_to_optional_f64(None), None);
    }

    #[test]
    fn sqlite_to_optional_text_all_variants() {
        assert_eq!(
            sqlite_to_optional_text(Some(&SqliteValue::Text("hi".to_owned()))),
            Some("hi".to_owned())
        );
        assert_eq!(
            sqlite_to_optional_text(Some(&SqliteValue::Integer(1))),
            None
        );
        assert_eq!(
            sqlite_to_optional_text(Some(&SqliteValue::Float(1.0))),
            None
        );
        assert_eq!(sqlite_to_optional_text(None), None);
    }

    #[test]
    fn json_optional_float_extracts_number_or_null() {
        let value = json!({"score": 42.5, "name": "test", "missing": null});
        match json_optional_float(&value, "score") {
            SqliteValue::Float(f) => assert!((f - 42.5).abs() < 1e-9),
            other => panic!("expected Float, got {other:?}"),
        }
        assert!(matches!(
            json_optional_float(&value, "name"),
            SqliteValue::Null
        ));
        assert!(matches!(
            json_optional_float(&value, "missing"),
            SqliteValue::Null
        ));
        assert!(matches!(
            json_optional_float(&value, "absent"),
            SqliteValue::Null
        ));
    }

    #[test]
    fn json_optional_text_extracts_string_or_null() {
        let value = json!({"name": "alice", "count": 5});
        match json_optional_text(&value, "name") {
            SqliteValue::Text(s) => assert_eq!(s, "alice"),
            other => panic!("expected Text, got {other:?}"),
        }
        assert!(matches!(
            json_optional_text(&value, "count"),
            SqliteValue::Null
        ));
        assert!(matches!(
            json_optional_text(&value, "absent"),
            SqliteValue::Null
        ));
    }

    #[test]
    fn json_to_optional_f64_extracts_or_returns_none() {
        let value = json!({"score": 42.5, "label": "test"});
        assert_eq!(json_to_optional_f64(&value, "score"), Some(42.5));
        assert_eq!(json_to_optional_f64(&value, "label"), None);
        assert_eq!(json_to_optional_f64(&value, "missing"), None);
    }

    #[test]
    fn json_to_optional_text_extracts_or_returns_none() {
        let value = json!({"label": "hello", "count": 5});
        assert_eq!(
            json_to_optional_text(&value, "label"),
            Some("hello".to_owned())
        );
        assert_eq!(json_to_optional_text(&value, "count"), None);
        assert_eq!(json_to_optional_text(&value, "missing"), None);
    }

    #[test]
    fn json_str_returns_ok_for_present_string() {
        let value = json!({"name": "alice"});
        assert_eq!(json_str(&value, "name").unwrap(), "alice");
    }

    #[test]
    fn json_str_returns_error_for_non_string_value() {
        let value = json!({"count": 42});
        let err = json_str(&value, "count").expect_err("non-string should fail");
        assert!(err.to_string().contains("missing string field"));
    }

    #[test]
    fn json_string_or_default_returns_value_for_non_string_type() {
        let value = json!({"count": 42});
        // Non-string value should fall back to default.
        assert_eq!(
            json_string_or_default(&value, "count", "fallback"),
            "fallback"
        );
    }

    #[test]
    fn optional_floats_equal_nan_is_not_equal() {
        // NaN != NaN in float arithmetic; (NaN - NaN).abs() is NaN, which fails <= 1e-9.
        assert!(!optional_floats_equal(Some(f64::NAN), Some(f64::NAN)));
    }

    #[test]
    fn sync_manifest_serde_round_trip() {
        let manifest = SyncManifest {
            schema_version: "1.1".to_owned(),
            export_format_version: "1.0".to_owned(),
            created_at_rfc3339: "2026-02-22T12:00:00Z".to_owned(),
            source_db_path: "/tmp/test.sqlite3".to_owned(),
            row_counts: RowCounts {
                runs: 10,
                segments: 50,
                events: 30,
            },
            checksums: FileChecksums {
                runs_jsonl_sha256: "abc123".to_owned(),
                segments_jsonl_sha256: "def456".to_owned(),
                events_jsonl_sha256: "ghi789".to_owned(),
            },
        };

        let json = serde_json::to_string_pretty(&manifest).expect("serialize");
        let deserialized: SyncManifest = serde_json::from_str(&json).expect("deserialize");

        assert_eq!(deserialized.schema_version, "1.1");
        assert_eq!(deserialized.export_format_version, "1.0");
        assert_eq!(deserialized.created_at_rfc3339, "2026-02-22T12:00:00Z");
        assert_eq!(deserialized.source_db_path, "/tmp/test.sqlite3");
        assert_eq!(deserialized.row_counts, manifest.row_counts);
        assert_eq!(deserialized.checksums, manifest.checksums);
    }

    #[test]
    fn row_counts_equality_and_inequality() {
        let a = RowCounts {
            runs: 1,
            segments: 2,
            events: 3,
        };
        let b = RowCounts {
            runs: 1,
            segments: 2,
            events: 3,
        };
        let c = RowCounts {
            runs: 1,
            segments: 2,
            events: 4,
        };
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn file_checksums_equality_and_inequality() {
        let a = FileChecksums {
            runs_jsonl_sha256: "aaa".to_owned(),
            segments_jsonl_sha256: "bbb".to_owned(),
            events_jsonl_sha256: "ccc".to_owned(),
        };
        let b = a.clone();
        let mut c = a.clone();
        c.events_jsonl_sha256 = "zzz".to_owned();
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn sha256_file_empty_file() {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("empty.txt");
        fs::write(&path, b"").expect("write empty");
        let hash = sha256_file(&path).expect("hash");
        // SHA-256 of empty input is a well-known constant.
        assert_eq!(
            hash,
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
    }

    #[test]
    fn sha256_file_known_content() {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("hello.txt");
        fs::write(&path, b"hello\n").expect("write");
        let hash = sha256_file(&path).expect("hash");
        // SHA-256 of "hello\n" — deterministic.
        assert_eq!(
            hash,
            "5891b5b522d5df086d0ff0b110fbd9d21bb4fc7163af34d08286a2e846f6be03"
        );
    }

    #[test]
    fn sha256_file_nonexistent_returns_error() {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("nonexistent.txt");
        assert!(sha256_file(&path).is_err());
    }

    #[test]
    fn atomic_rename_moves_file() {
        let dir = tempdir().expect("tempdir");
        let src = dir.path().join("src.txt");
        let dst = dir.path().join("dst.txt");
        fs::write(&src, b"payload").expect("write");
        atomic_rename(&src, &dst).expect("rename");
        assert!(!src.exists());
        assert!(dst.exists());
        assert_eq!(fs::read_to_string(&dst).expect("read"), "payload");
    }

    #[test]
    fn atomic_write_bytes_creates_file() {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("output.txt");
        atomic_write_bytes(&path, b"content here").expect("write");
        assert_eq!(fs::read_to_string(&path).expect("read"), "content here");
        // Temp file should not remain.
        let tmp = path.with_extension("tmp");
        assert!(!tmp.exists());
    }

    #[test]
    fn sync_lock_double_release_is_idempotent() {
        let dir = tempdir().expect("tempdir");
        let state_root = dir.path().join("state");
        let lock = SyncLock::acquire(&state_root, "test").expect("acquire");
        lock.release().expect("first release");
        // After explicit release, drop should be a no-op (no panic, no error).
    }

    #[test]
    fn sync_lock_drop_cleans_up_lock_file() {
        let dir = tempdir().expect("tempdir");
        let state_root = dir.path().join("state");
        let lock_path = state_root.join("locks/sync.lock");

        {
            let _lock = SyncLock::acquire(&state_root, "test").expect("acquire");
            assert!(lock_path.exists(), "lock file should exist while held");
        }
        // After drop, lock file should be removed.
        assert!(
            !lock_path.exists(),
            "lock file should be cleaned up on drop"
        );
    }

    #[test]
    fn sync_conflict_serde_round_trip() {
        let conflict = SyncConflict {
            table: "runs".to_owned(),
            key: "run-123".to_owned(),
            reason: "different payload for same id".to_owned(),
        };
        let json = serde_json::to_string(&conflict).expect("serialize");
        let deserialized: SyncConflict = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(deserialized.table, "runs");
        assert_eq!(deserialized.key, "run-123");
        assert_eq!(deserialized.reason, "different payload for same id");
    }

    #[test]
    fn is_lock_stale_with_invalid_rfc3339_returns_false_for_alive_pid() {
        // When timestamp is unparseable, only PID liveness is checked.
        // Current PID is alive, so should return false.
        let info = LockInfo {
            pid: std::process::id(),
            created_at_rfc3339: "not-a-timestamp".to_owned(),
            operation: "test".to_owned(),
        };
        assert!(
            !is_lock_stale(&info),
            "unparseable timestamp with alive PID should not be stale"
        );
    }

    #[test]
    fn is_lock_stale_with_invalid_rfc3339_dead_pid_returns_true() {
        let info = LockInfo {
            pid: u32::MAX - 1,
            created_at_rfc3339: "not-a-timestamp".to_owned(),
            operation: "test".to_owned(),
        };
        assert!(
            is_lock_stale(&info),
            "dead PID should be stale regardless of timestamp"
        );
    }

    #[test]
    fn validate_checksums_missing_file_returns_error() {
        let dir = tempdir().expect("tempdir");
        let input_dir = dir.path().join("export");
        fs::create_dir_all(&input_dir).expect("mkdir");
        // Only create runs.jsonl — segments and events missing.
        fs::write(input_dir.join("runs.jsonl"), b"").expect("write");

        let manifest = SyncManifest {
            schema_version: "1.1".to_owned(),
            export_format_version: "1.0".to_owned(),
            created_at_rfc3339: Utc::now().to_rfc3339(),
            source_db_path: "test.sqlite3".to_owned(),
            row_counts: RowCounts {
                runs: 0,
                segments: 0,
                events: 0,
            },
            checksums: FileChecksums {
                runs_jsonl_sha256: sha256_file(&input_dir.join("runs.jsonl")).expect("hash"),
                segments_jsonl_sha256: "fake".to_owned(),
                events_jsonl_sha256: "fake".to_owned(),
            },
        };

        let err = validate_checksums(&manifest, &input_dir).expect_err("missing file");
        assert!(
            err.to_string().contains("missing export file"),
            "got: {}",
            err
        );
    }

    #[test]
    fn validate_checksums_matching_checksums_succeeds() {
        let dir = tempdir().expect("tempdir");
        let input_dir = dir.path().join("export");
        fs::create_dir_all(&input_dir).expect("mkdir");

        for name in ["runs.jsonl", "segments.jsonl", "events.jsonl"] {
            fs::write(input_dir.join(name), b"").expect("write");
        }

        let manifest = SyncManifest {
            schema_version: "1.1".to_owned(),
            export_format_version: "1.0".to_owned(),
            created_at_rfc3339: Utc::now().to_rfc3339(),
            source_db_path: "test.sqlite3".to_owned(),
            row_counts: RowCounts {
                runs: 0,
                segments: 0,
                events: 0,
            },
            checksums: FileChecksums {
                runs_jsonl_sha256: sha256_file(&input_dir.join("runs.jsonl")).expect("hash"),
                segments_jsonl_sha256: sha256_file(&input_dir.join("segments.jsonl"))
                    .expect("hash"),
                events_jsonl_sha256: sha256_file(&input_dir.join("events.jsonl")).expect("hash"),
            },
        };

        assert!(validate_checksums(&manifest, &input_dir).is_ok());
    }

    #[test]
    fn value_to_json_blob_format_includes_length() {
        let blob = SqliteValue::Blob(vec![0xDE, 0xAD, 0xBE, 0xEF]);
        let result = value_to_json(Some(&blob));
        assert_eq!(result, serde_json::Value::String("<blob:4>".to_owned()));
    }

    #[test]
    fn import_missing_manifest_returns_error() {
        let dir = tempdir().expect("tempdir");
        let empty_dir = dir.path().join("no-manifest");
        fs::create_dir_all(&empty_dir).expect("mkdir");
        let state_root = dir.path().join("state");
        let target_db = dir.path().join("target.sqlite3");

        let err = import(&target_db, &empty_dir, &state_root, ConflictPolicy::Reject)
            .expect_err("missing manifest");
        assert!(
            err.to_string().contains("manifest.json not found"),
            "got: {}",
            err
        );
    }

    #[test]
    fn verify_schema_exists_rejects_empty_db() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("empty.sqlite3");
        // Open a connection without creating tables.
        let conn = Connection::open(db_path.display().to_string()).expect("connection should open");
        let err = verify_schema_exists(&conn).expect_err("empty db should fail");
        assert!(err.to_string().contains("missing table"), "got: {}", err);
    }

    #[test]
    fn ensure_schema_creates_all_tables() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("schema_test.sqlite3");
        let conn = Connection::open(db_path.display().to_string()).expect("connection should open");
        ensure_schema(&conn).expect("ensure_schema");
        // verify_schema_exists should now succeed.
        verify_schema_exists(&conn).expect("schema should now exist");
    }

    #[test]
    fn ensure_runs_replay_column_adds_column_to_old_schema() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("old_schema.sqlite3");
        let conn = Connection::open(db_path.display().to_string()).expect("connection should open");

        // Create table WITHOUT replay_json column.
        conn.execute(
            "CREATE TABLE runs (
                id TEXT PRIMARY KEY,
                started_at TEXT NOT NULL,
                finished_at TEXT NOT NULL,
                backend TEXT NOT NULL,
                input_path TEXT NOT NULL,
                normalized_wav_path TEXT NOT NULL,
                request_json TEXT NOT NULL,
                result_json TEXT NOT NULL,
                warnings_json TEXT NOT NULL,
                transcript TEXT NOT NULL
            );",
        )
        .expect("create old schema");

        // First call should add the column.
        ensure_runs_replay_column(&conn).expect("add column");

        // Second call should be a no-op.
        ensure_runs_replay_column(&conn).expect("idempotent");

        // Verify column exists.
        let rows = conn.query("PRAGMA table_info(runs);").expect("pragma");
        let has_replay = rows
            .iter()
            .any(|row| value_to_string_sqlite(row.get(1)) == "replay_json");
        assert!(
            has_replay,
            "replay_json column should exist after migration"
        );
    }

    #[test]
    fn export_import_multiple_runs() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("multi.sqlite3");
        let export_dir = dir.path().join("export");
        let state_root = dir.path().join("state");

        let store = RunStore::open(&db_path).expect("store");
        for i in 0..3 {
            store
                .persist_report(&fixture_report(&format!("multi-{i}"), &db_path))
                .expect("persist");
        }
        drop(store);
        let conn = Connection::open(db_path.display().to_string()).expect("open");
        ensure_schema(&conn).expect("schema");
        drop(conn);
        let sanity_store = RunStore::open(&db_path).expect("reopen store");
        assert_eq!(sanity_store.list_recent_runs(10).expect("list").len(), 3);
        drop(sanity_store);

        let manifest = export(&db_path, &export_dir, &state_root).expect("export");
        assert_eq!(manifest.row_counts.runs, 3);
        assert_eq!(manifest.row_counts.segments, 6); // 2 per run
        assert_eq!(manifest.row_counts.events, 6); // 2 per run

        let target_db = dir.path().join("target.sqlite3");
        let result =
            import(&target_db, &export_dir, &state_root, ConflictPolicy::Reject).expect("import");
        assert_eq!(result.runs_imported, 3);
        assert_eq!(result.segments_imported, 6);
        assert_eq!(result.events_imported, 6);
        assert!(result.conflicts.is_empty());
    }

    #[test]
    fn validate_checksums_detects_hash_mismatch() {
        let dir = tempdir().expect("tempdir");
        let input_dir = dir.path().join("export");
        fs::create_dir_all(&input_dir).expect("mkdir");

        for name in ["runs.jsonl", "segments.jsonl", "events.jsonl"] {
            fs::write(input_dir.join(name), b"").expect("write");
        }

        let empty_hash = sha256_file(&input_dir.join("runs.jsonl")).expect("hash");
        let manifest = SyncManifest {
            schema_version: "1.1".to_owned(),
            export_format_version: "1.0".to_owned(),
            created_at_rfc3339: Utc::now().to_rfc3339(),
            source_db_path: "test.sqlite3".to_owned(),
            row_counts: RowCounts {
                runs: 0,
                segments: 0,
                events: 0,
            },
            checksums: FileChecksums {
                runs_jsonl_sha256: empty_hash.clone(),
                segments_jsonl_sha256: empty_hash,
                events_jsonl_sha256: "wrong_hash_value".to_owned(),
            },
        };

        let err = validate_checksums(&manifest, &input_dir).expect_err("hash mismatch");
        assert!(
            err.to_string().contains("checksum mismatch"),
            "got: {}",
            err
        );
    }

    #[test]
    fn json_str_returns_error_for_null_value() {
        let value = json!({"key": null});
        let err = json_str(&value, "key").expect_err("null should fail");
        assert!(err.to_string().contains("missing string field"));
    }

    #[test]
    fn export_import_unicode_content() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("unicode.sqlite3");
        let export_dir = dir.path().join("export");
        let state_root = dir.path().join("state");

        let store = RunStore::open(&db_path).expect("store");
        let mut report = fixture_report("unicode-1", &db_path);
        report.result.transcript = "\u{1F600} \u{4E16}\u{754C} caf\u{00E9}".to_owned();
        report.result.segments[0].text = "\u{1F600} \u{4E16}\u{754C}".to_owned();
        report.result.segments[1].text = "caf\u{00E9}".to_owned();
        store.persist_report(&report).expect("persist");

        let manifest = export(&db_path, &export_dir, &state_root).expect("export");
        assert_eq!(manifest.row_counts.runs, 1);

        let target_db = dir.path().join("target.sqlite3");
        let result =
            import(&target_db, &export_dir, &state_root, ConflictPolicy::Reject).expect("import");
        assert!(result.conflicts.is_empty());

        let target_store = RunStore::open(&target_db).expect("target store");
        let details = target_store
            .load_run_details("unicode-1")
            .expect("load")
            .expect("details");
        assert!(details.transcript.contains('\u{1F600}'));
        assert!(details.transcript.contains("caf\u{00E9}"));
    }

    #[test]
    fn write_conflicts_file_creates_valid_jsonl() {
        let dir = tempdir().expect("tempdir");
        let conflicts = vec![
            SyncConflict {
                table: "runs".to_owned(),
                key: "run-1".to_owned(),
                reason: "payload mismatch".to_owned(),
            },
            SyncConflict {
                table: "segments".to_owned(),
                key: "run-1/0".to_owned(),
                reason: "duplicate composite key".to_owned(),
            },
        ];

        write_conflicts_file(dir.path(), &conflicts).expect("write");

        let content = fs::read_to_string(dir.path().join("sync_conflicts.jsonl")).expect("read");
        let lines: Vec<&str> = content.lines().collect();
        assert_eq!(lines.len(), 2);

        let first: SyncConflict = serde_json::from_str(lines[0]).expect("parse line 0");
        assert_eq!(first.table, "runs");
        let second: SyncConflict = serde_json::from_str(lines[1]).expect("parse line 1");
        assert_eq!(second.key, "run-1/0");
    }

    #[test]
    fn import_overwrite_replaces_conflicting_segment() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("source.sqlite3");
        let export_dir = dir.path().join("export");
        let state_root = dir.path().join("state");

        let store = RunStore::open(&db_path).expect("store");
        let report = fixture_report("seg-conflict-1", &db_path);
        store.persist_report(&report).expect("persist");
        export(&db_path, &export_dir, &state_root).expect("export");

        // Mutate a segment's text in the JSONL.
        let seg_path = export_dir.join("segments.jsonl");
        let original = fs::read_to_string(&seg_path).expect("read segments");
        let mutated = original.replace("hello world", "replaced text");
        fs::write(&seg_path, &mutated).expect("write segments");

        // Update manifest checksum.
        let manifest_path = export_dir.join("manifest.json");
        let mut manifest_value: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(&manifest_path).expect("manifest"))
                .expect("manifest json");
        manifest_value["checksums"]["segments_jsonl_sha256"] =
            json!(sha256_file(&seg_path).expect("checksum"));
        fs::write(
            &manifest_path,
            serde_json::to_string_pretty(&manifest_value).expect("serialize"),
        )
        .expect("manifest write");

        let result = import(
            &db_path,
            &export_dir,
            &state_root,
            ConflictPolicy::Overwrite,
        )
        .expect("overwrite import");
        assert_eq!(result.segments_imported, 2);

        // Verify the segment was replaced in the segments table directly.
        // (load_run_details reads from result_json, not the segments table.)
        let conn = Connection::open(db_path.display().to_string()).expect("conn");
        let rows = conn
            .query_with_params(
                "SELECT text FROM segments WHERE run_id = ?1 AND idx = 0",
                &[SqliteValue::Text("seg-conflict-1".to_owned())],
            )
            .expect("query");
        let text = value_to_string_sqlite(rows[0].get(0));
        assert_eq!(text, "replaced text");
    }

    #[test]
    fn import_overwrite_replaces_conflicting_event() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("source.sqlite3");
        let export_dir = dir.path().join("export");
        let state_root = dir.path().join("state");

        let store = RunStore::open(&db_path).expect("store");
        let report = fixture_report("evt-conflict-1", &db_path);
        store.persist_report(&report).expect("persist");
        export(&db_path, &export_dir, &state_root).expect("export");

        // Mutate an event's message in the JSONL.
        let evt_path = export_dir.join("events.jsonl");
        let original = fs::read_to_string(&evt_path).expect("read events");
        let mutated = original.replace("input materialized", "replaced event message");
        fs::write(&evt_path, &mutated).expect("write events");

        // Update manifest checksum.
        let manifest_path = export_dir.join("manifest.json");
        let mut manifest_value: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(&manifest_path).expect("manifest"))
                .expect("manifest json");
        manifest_value["checksums"]["events_jsonl_sha256"] =
            json!(sha256_file(&evt_path).expect("checksum"));
        fs::write(
            &manifest_path,
            serde_json::to_string_pretty(&manifest_value).expect("serialize"),
        )
        .expect("manifest write");

        let result = import(
            &db_path,
            &export_dir,
            &state_root,
            ConflictPolicy::Overwrite,
        )
        .expect("overwrite import");
        assert_eq!(result.events_imported, 2);

        // Verify the event was replaced in the events table directly.
        let conn = Connection::open(db_path.display().to_string()).expect("conn");
        let rows = conn
            .query_with_params(
                "SELECT message FROM events WHERE run_id = ?1 AND seq = 1",
                &[SqliteValue::Text("evt-conflict-1".to_owned())],
            )
            .expect("query");
        let message = value_to_string_sqlite(rows[0].get(0));
        assert_eq!(message, "replaced event message");
    }

    #[test]
    fn import_idempotent_on_identical_data() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("source.sqlite3");
        let export_dir = dir.path().join("export");
        let state_root = dir.path().join("state");
        let target_db = dir.path().join("target.sqlite3");

        let store = RunStore::open(&db_path).expect("store open");
        let report = fixture_report("idempotent-1", &db_path);
        store.persist_report(&report).expect("persist");
        export(&db_path, &export_dir, &state_root).expect("export");

        // First import
        let result_1 =
            import(&target_db, &export_dir, &state_root, ConflictPolicy::Reject).expect("import 1");
        assert_eq!(result_1.runs_imported, 1);
        assert!(result_1.conflicts.is_empty());

        // Second import of same data → no conflicts (idempotent upsert)
        let result_2 = import(&target_db, &export_dir, &state_root, ConflictPolicy::Reject)
            .expect("import 2 should succeed (idempotent)");
        assert!(
            result_2.conflicts.is_empty(),
            "re-import of identical data should produce no conflicts"
        );
    }

    #[test]
    fn lock_info_serde_round_trip() {
        let info = LockInfo {
            pid: 12345,
            created_at_rfc3339: "2026-01-01T00:00:00Z".to_owned(),
            operation: "export".to_owned(),
        };
        let json = serde_json::to_string(&info).expect("serialize");
        let parsed: LockInfo = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(parsed.pid, 12345);
        assert_eq!(parsed.operation, "export");
        assert_eq!(parsed.created_at_rfc3339, "2026-01-01T00:00:00Z");
    }

    #[test]
    fn conflict_policy_all_variants_are_distinct() {
        assert_ne!(ConflictPolicy::Reject, ConflictPolicy::Overwrite);
        // Verify both can be used without panic.
        let _ = format!("{:?}", ConflictPolicy::Reject);
        let _ = format!("{:?}", ConflictPolicy::Overwrite);
    }

    #[test]
    fn import_malformed_manifest_json_returns_error() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("target.sqlite3");
        let input_dir = dir.path().join("bad_manifest");
        let state_root = dir.path().join("state");

        fs::create_dir_all(&input_dir).expect("mkdir");
        fs::write(input_dir.join("manifest.json"), "not valid json {{")
            .expect("write bad manifest");

        let result = import(&db_path, &input_dir, &state_root, ConflictPolicy::Reject);
        assert!(result.is_err());
        let err_text = result.unwrap_err().to_string();
        assert!(
            err_text.contains("manifest") || err_text.contains("invalid"),
            "expected manifest error, got: {err_text}"
        );
    }

    #[test]
    fn sha256_file_deterministic_for_same_content() {
        let dir = tempdir().expect("tempdir");
        let file_a = dir.path().join("a.txt");
        let file_b = dir.path().join("b.txt");
        let content = "deterministic content\n".repeat(100);
        fs::write(&file_a, &content).expect("write a");
        fs::write(&file_b, &content).expect("write b");

        let hash_a = sha256_file(&file_a).expect("hash a");
        let hash_b = sha256_file(&file_b).expect("hash b");
        assert_eq!(hash_a, hash_b, "same content should produce same hash");
        assert_eq!(hash_a.len(), 64, "SHA-256 hex should be 64 chars");
    }

    #[test]
    fn atomic_write_bytes_large_content() {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("large.bin");
        let content: Vec<u8> = (0..100_000).map(|i| (i % 256) as u8).collect();
        atomic_write_bytes(&path, &content).expect("write");
        let read_back = fs::read(&path).expect("read");
        assert_eq!(read_back.len(), 100_000);
        assert_eq!(read_back, content);
    }

    #[test]
    fn atomic_rename_moves_content() {
        let dir = tempdir().expect("tempdir");
        let src = dir.path().join("source.txt");
        let dst = dir.path().join("destination.txt");
        fs::write(&src, b"rename test content").expect("write");
        atomic_rename(&src, &dst).expect("rename");
        assert!(!src.exists());
        assert_eq!(
            fs::read_to_string(&dst).expect("read"),
            "rename test content"
        );
    }

    #[test]
    fn export_manifest_contains_expected_version_fields() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("version_check.sqlite3");
        let export_dir = dir.path().join("export");
        let state_root = dir.path().join("state");

        let _store = RunStore::open(&db_path).expect("store open");
        let manifest = export(&db_path, &export_dir, &state_root).expect("export");

        assert_eq!(manifest.schema_version, SCHEMA_VERSION);
        assert_eq!(manifest.export_format_version, EXPORT_FORMAT_VERSION);
        assert!(!manifest.created_at_rfc3339.is_empty());
        assert!(manifest.source_db_path.contains("version_check.sqlite3"));
    }

    #[test]
    fn export_with_data_includes_segments_and_events() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("full_data.sqlite3");
        let export_dir = dir.path().join("export");
        let state_root = dir.path().join("state");

        let store = RunStore::open(&db_path).expect("store open");
        let report = fixture_report("full-data-1", &db_path);
        store.persist_report(&report).expect("persist");
        let manifest = export(&db_path, &export_dir, &state_root).expect("export");

        // fixture_report has 1 run, 2 segments, 2 events
        assert_eq!(manifest.row_counts.runs, 1);
        assert_eq!(manifest.row_counts.segments, 2);
        assert_eq!(manifest.row_counts.events, 2);

        // JSONL files should exist and be non-empty
        let runs_content = fs::read_to_string(export_dir.join("runs.jsonl")).expect("read");
        assert!(!runs_content.trim().is_empty());
        let segs_content = fs::read_to_string(export_dir.join("segments.jsonl")).expect("read");
        assert!(!segs_content.trim().is_empty());
        let events_content = fs::read_to_string(export_dir.join("events.jsonl")).expect("read");
        assert!(!events_content.trim().is_empty());
    }

    #[test]
    fn write_conflicts_file_empty_array_creates_empty_file() {
        let dir = tempdir().expect("tempdir");
        write_conflicts_file(dir.path(), &[]).expect("write empty");

        let content = fs::read_to_string(dir.path().join("sync_conflicts.jsonl")).expect("read");
        assert!(
            content.trim().is_empty(),
            "empty conflict list should produce empty file"
        );
    }

    #[test]
    fn export_import_preserves_segment_optional_fields() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("optional_fields.sqlite3");
        let export_dir = dir.path().join("export");
        let state_root = dir.path().join("state");
        let target_db = dir.path().join("target.sqlite3");

        let store = RunStore::open(&db_path).expect("store open");
        let mut report = fixture_report("optional-1", &db_path);
        // One segment with all fields, one with all optional fields None
        report.result.segments = vec![
            TranscriptionSegment {
                start_sec: Some(0.0),
                end_sec: Some(1.0),
                text: "with fields".to_owned(),
                speaker: Some("SPEAKER_00".to_owned()),
                confidence: Some(0.95),
            },
            TranscriptionSegment {
                start_sec: None,
                end_sec: None,
                text: "no optional fields".to_owned(),
                speaker: None,
                confidence: None,
            },
        ];
        store.persist_report(&report).expect("persist");

        export(&db_path, &export_dir, &state_root).expect("export");
        let result =
            import(&target_db, &export_dir, &state_root, ConflictPolicy::Reject).expect("import");
        assert_eq!(result.segments_imported, 2);

        let target_store = RunStore::open(&target_db).expect("target store");
        let details = target_store
            .load_run_details("optional-1")
            .expect("query")
            .expect("exists");
        assert_eq!(details.segments.len(), 2);
        assert_eq!(details.segments[0].speaker.as_deref(), Some("SPEAKER_00"));
        assert_eq!(details.segments[0].confidence, Some(0.95));
        assert!(details.segments[1].start_sec.is_none());
        assert!(details.segments[1].speaker.is_none());
        assert!(details.segments[1].confidence.is_none());
    }

    #[test]
    fn export_import_multiple_runs_preserves_all_data() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("multi.sqlite3");
        let export_dir = dir.path().join("export");
        let state_root = dir.path().join("state");
        let target_db = dir.path().join("target.sqlite3");

        let store = RunStore::open(&db_path).expect("store open");
        for i in 0..3 {
            let mut report = fixture_report(&format!("multi-{i}"), &db_path);
            report.started_at_rfc3339 = format!("2026-01-0{i}T00:00:00Z");
            store.persist_report(&report).expect("persist");
        }
        drop(store);
        let conn = Connection::open(db_path.display().to_string()).expect("open");
        ensure_schema(&conn).expect("schema");
        drop(conn);

        export(&db_path, &export_dir, &state_root).expect("export");
        let result =
            import(&target_db, &export_dir, &state_root, ConflictPolicy::Reject).expect("import");
        assert_eq!(result.runs_imported, 3);
        // fixture_report has 2 segments and 2 events each
        assert_eq!(result.segments_imported, 6);
        assert_eq!(result.events_imported, 6);
        assert!(result.conflicts.is_empty());

        let target_store = RunStore::open(&target_db).expect("target store");
        let runs = target_store.list_recent_runs(10).expect("list");
        assert_eq!(runs.len(), 3);
    }

    #[test]
    fn sync_conflict_with_unicode_fields_round_trips() {
        let conflict = SyncConflict {
            table: "résultats".to_owned(),
            key: "clé-\u{1F600}".to_owned(),
            reason: "données incompatibles".to_owned(),
        };
        let json = serde_json::to_string(&conflict).expect("serialize");
        let parsed: SyncConflict = serde_json::from_str(&json).expect("parse");
        assert_eq!(parsed.table, "résultats");
        assert!(parsed.key.contains('\u{1F600}'));
        assert_eq!(parsed.reason, "données incompatibles");
    }

    #[test]
    fn import_overwrite_counts_include_replaced_rows() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("source.sqlite3");
        let export_dir = dir.path().join("export");
        let state_root = dir.path().join("state");
        let target_db = dir.path().join("target.sqlite3");

        let store = RunStore::open(&db_path).expect("store open");
        let report = fixture_report("overwrite-count-1", &db_path);
        store.persist_report(&report).expect("persist");

        export(&db_path, &export_dir, &state_root).expect("export");

        // First import
        import(
            &target_db,
            &export_dir,
            &state_root,
            ConflictPolicy::Overwrite,
        )
        .expect("import 1");
        // Second import — same data, should overwrite
        let result = import(
            &target_db,
            &export_dir,
            &state_root,
            ConflictPolicy::Overwrite,
        )
        .expect("import 2");
        // Overwrite should process all rows
        assert_eq!(result.runs_imported, 1);
        assert_eq!(result.segments_imported, 2);
        assert_eq!(result.events_imported, 2);
    }

    #[test]
    fn export_empty_db_produces_zero_row_counts() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("empty_counts.sqlite3");
        let export_dir = dir.path().join("export");
        let state_root = dir.path().join("state");

        let _store = RunStore::open(&db_path).expect("store open");
        let manifest = export(&db_path, &export_dir, &state_root).expect("export");

        assert_eq!(manifest.row_counts.runs, 0);
        assert_eq!(manifest.row_counts.segments, 0);
        assert_eq!(manifest.row_counts.events, 0);
    }

    #[test]
    fn import_jsonl_with_blank_lines_skips_them() {
        let dir = tempdir().expect("tempdir");
        let export_dir = dir.path().join("export");
        let state_root = dir.path().join("state");
        let target_db = dir.path().join("target.sqlite3");

        // Create a valid export, then inject blank lines into the JSONL files.
        let db_path = dir.path().join("source.sqlite3");
        let store = RunStore::open(&db_path).expect("store open");
        let report = fixture_report("blank-lines-1", &db_path);
        store.persist_report(&report).expect("persist");
        export(&db_path, &export_dir, &state_root).expect("export");

        // Inject blank and whitespace-only lines into each JSONL file.
        for filename in ["runs.jsonl", "segments.jsonl", "events.jsonl"] {
            let path = export_dir.join(filename);
            let original = fs::read_to_string(&path).expect("read");
            let with_blanks = format!("\n  \n\t\n{original}\n  \n");
            fs::write(&path, &with_blanks).expect("write");
        }

        // Update manifest checksums to match modified files.
        let manifest_path = export_dir.join("manifest.json");
        let mut manifest_value: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(&manifest_path).expect("read")).expect("json");
        manifest_value["checksums"]["runs_jsonl_sha256"] =
            json!(sha256_file(&export_dir.join("runs.jsonl")).expect("checksum"));
        manifest_value["checksums"]["segments_jsonl_sha256"] =
            json!(sha256_file(&export_dir.join("segments.jsonl")).expect("checksum"));
        manifest_value["checksums"]["events_jsonl_sha256"] =
            json!(sha256_file(&export_dir.join("events.jsonl")).expect("checksum"));
        fs::write(
            &manifest_path,
            serde_json::to_string_pretty(&manifest_value).expect("serialize"),
        )
        .expect("write manifest");

        let result =
            import(&target_db, &export_dir, &state_root, ConflictPolicy::Reject).expect("import");
        assert_eq!(result.runs_imported, 1);
        assert_eq!(result.segments_imported, 2);
        assert_eq!(result.events_imported, 2);
    }

    #[test]
    fn import_malformed_runs_jsonl_returns_error() {
        let dir = tempdir().expect("tempdir");
        let export_dir = dir.path().join("export");
        let state_root = dir.path().join("state");
        let target_db = dir.path().join("target.sqlite3");

        // Create a valid export first.
        let db_path = dir.path().join("source.sqlite3");
        let store = RunStore::open(&db_path).expect("store open");
        let report = fixture_report("malformed-1", &db_path);
        store.persist_report(&report).expect("persist");
        export(&db_path, &export_dir, &state_root).expect("export");

        // Replace runs.jsonl with malformed JSON.
        let runs_path = export_dir.join("runs.jsonl");
        fs::write(&runs_path, "{not valid json\n").expect("write malformed");

        // Update checksum.
        let manifest_path = export_dir.join("manifest.json");
        let mut mval: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(&manifest_path).expect("read")).expect("json");
        mval["checksums"]["runs_jsonl_sha256"] = json!(sha256_file(&runs_path).expect("checksum"));
        fs::write(
            &manifest_path,
            serde_json::to_string_pretty(&mval).expect("serialize"),
        )
        .expect("write manifest");

        let err = import(&target_db, &export_dir, &state_root, ConflictPolicy::Reject)
            .expect_err("malformed JSONL should fail");
        let text = err.to_string();
        assert!(
            text.contains("key must be") || text.contains("json") || text.contains("JSON"),
            "error should mention JSON parse failure: {text}"
        );
    }

    #[test]
    fn optional_floats_equal_infinities_are_not_equal() {
        // INF - INF = NaN, so optional_floats_equal returns false for infinities.
        // This documents the current behavior.
        assert!(!optional_floats_equal(
            Some(f64::INFINITY),
            Some(f64::INFINITY)
        ));
        assert!(!optional_floats_equal(
            Some(f64::NEG_INFINITY),
            Some(f64::NEG_INFINITY)
        ));
        assert!(!optional_floats_equal(
            Some(f64::INFINITY),
            Some(f64::NEG_INFINITY)
        ));
    }

    #[test]
    fn json_string_or_default_with_null_value_returns_fallback() {
        let value = json!({"key": null});
        assert_eq!(
            json_string_or_default(&value, "key", "fallback"),
            "fallback",
            "null value should trigger fallback"
        );
    }

    #[test]
    fn sync_lock_drop_after_release_does_not_panic() {
        let dir = tempdir().expect("tempdir");
        let state_root = dir.path().join("state");
        let lock = SyncLock::acquire(&state_root, "test").expect("acquire");
        lock.release().expect("explicit release");
        // Drop happens here — should not panic even though already released.
    }

    #[test]
    fn import_result_counts_reflect_actual_data() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("source.sqlite3");
        let export_dir = dir.path().join("export");
        let state_root = dir.path().join("state");
        let target_db = dir.path().join("target.sqlite3");

        let store = RunStore::open(&db_path).expect("store open");
        let report = fixture_report("counts-1", &db_path);
        store.persist_report(&report).expect("persist");
        export(&db_path, &export_dir, &state_root).expect("export");

        let result =
            import(&target_db, &export_dir, &state_root, ConflictPolicy::Reject).expect("import");
        assert_eq!(result.runs_imported, 1);
        assert_eq!(result.segments_imported, 2);
        assert_eq!(result.events_imported, 2);
        assert!(result.conflicts.is_empty());
    }

    #[test]
    fn release_after_external_lock_file_deletion_succeeds() {
        let dir = tempdir().expect("tempdir");
        let state_root = dir.path().join("state");
        let lock = SyncLock::acquire(&state_root, "test").expect("acquire");
        let lock_path = state_root.join("locks").join("sync.lock");
        assert!(lock_path.exists(), "lock file should exist after acquire");
        // Simulate external deletion (e.g. operator cleanup).
        fs::remove_file(&lock_path).expect("manual remove");
        assert!(!lock_path.exists());
        // release() should succeed even though the file is gone.
        lock.release()
            .expect("release after external deletion should succeed");
    }

    #[test]
    fn pid_is_alive_returns_false_for_nonexistent_pid() {
        // PID u32::MAX - 1 should not correspond to a running process.
        assert!(
            !pid_is_alive(u32::MAX - 1),
            "pid_is_alive should return false for a nonexistent PID"
        );
    }

    #[test]
    fn validate_schema_version_no_dot_same_major_accepts() {
        // A version string with no dot uses the unwrap_or fallback.
        // SCHEMA_VERSION is "1.1", so expected_major is "1".
        // A manifest with schema_version "1" (no dot) → actual_major is "1" → match.
        let manifest = SyncManifest {
            schema_version: "1".to_owned(),
            export_format_version: "1".to_owned(),
            created_at_rfc3339: "2026-01-01T00:00:00Z".to_owned(),
            source_db_path: "test.db".to_owned(),
            row_counts: RowCounts {
                runs: 0,
                segments: 0,
                events: 0,
            },
            checksums: FileChecksums {
                runs_jsonl_sha256: String::new(),
                segments_jsonl_sha256: String::new(),
                events_jsonl_sha256: String::new(),
            },
        };
        assert!(
            validate_schema_version(&manifest).is_ok(),
            "schema_version '1' (no dot) should match major '1'"
        );
    }

    #[test]
    fn validate_schema_version_no_dot_wrong_major_rejects() {
        let manifest = SyncManifest {
            schema_version: "2".to_owned(),
            export_format_version: "1".to_owned(),
            created_at_rfc3339: "2026-01-01T00:00:00Z".to_owned(),
            source_db_path: "test.db".to_owned(),
            row_counts: RowCounts {
                runs: 0,
                segments: 0,
                events: 0,
            },
            checksums: FileChecksums {
                runs_jsonl_sha256: String::new(),
                segments_jsonl_sha256: String::new(),
                events_jsonl_sha256: String::new(),
            },
        };
        let err = validate_schema_version(&manifest).expect_err("major 2 should be rejected");
        let text = err.to_string();
        assert!(text.contains("schema version mismatch"), "error: {text}");
        assert!(
            text.contains("2"),
            "error should mention the bad version: {text}"
        );
    }

    #[test]
    fn import_row_count_mismatch_for_runs_errors() {
        // Existing test only covers events mismatch (line 468).
        // This exercises the runs mismatch path at line 444.
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("source.sqlite3");
        let export_dir = dir.path().join("export");
        let state_root = dir.path().join("state");

        let store = RunStore::open(&db_path).expect("store open");
        let report = fixture_report("runs-mismatch-1", &db_path);
        store.persist_report(&report).expect("persist");
        export(&db_path, &export_dir, &state_root).expect("export");

        let manifest_path = export_dir.join("manifest.json");
        let mut manifest_value: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(&manifest_path).expect("read"))
                .expect("parse");
        manifest_value["row_counts"]["runs"] = json!(9999);
        fs::write(
            &manifest_path,
            serde_json::to_string_pretty(&manifest_value).expect("serialize"),
        )
        .expect("write");

        let target_db = dir.path().join("target.sqlite3");
        let error = import(&target_db, &export_dir, &state_root, ConflictPolicy::Reject)
            .expect_err("runs row count mismatch should fail");
        assert!(
            error.to_string().contains("row count mismatch for runs"),
            "error should mention runs mismatch: {}",
            error
        );
    }

    #[test]
    fn lock_acquire_error_message_contains_pid_and_path() {
        let dir = tempdir().expect("tempdir");
        let state_root = dir.path().join("state");
        let _lock = SyncLock::acquire(&state_root, "test").expect("first acquire");
        let err = match SyncLock::acquire(&state_root, "test") {
            Err(e) => e,
            Ok(_) => panic!("second acquire should fail"),
        };
        let text = err.to_string();
        assert!(
            text.contains("sync lock held by pid"),
            "error should mention pid: {text}"
        );
        assert!(
            text.contains("sync.lock"),
            "error should mention lock file path: {text}"
        );
    }

    #[test]
    fn import_row_count_mismatch_for_segments_errors() {
        // Existing test covers events mismatch (line 468) and runs mismatch.
        // This exercises the segments mismatch path at line 456.
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("source.sqlite3");
        let export_dir = dir.path().join("export");
        let state_root = dir.path().join("state");

        let store = RunStore::open(&db_path).expect("store open");
        let report = fixture_report("seg-mismatch-1", &db_path);
        store.persist_report(&report).expect("persist");
        export(&db_path, &export_dir, &state_root).expect("export");

        let manifest_path = export_dir.join("manifest.json");
        let mut manifest_value: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(&manifest_path).expect("read"))
                .expect("parse");
        manifest_value["row_counts"]["segments"] = json!(9999);
        fs::write(
            &manifest_path,
            serde_json::to_string_pretty(&manifest_value).expect("serialize"),
        )
        .expect("write");

        let target_db = dir.path().join("target.sqlite3");
        let error = import(&target_db, &export_dir, &state_root, ConflictPolicy::Reject)
            .expect_err("segments mismatch should fail");
        assert!(
            error
                .to_string()
                .contains("row count mismatch for segments"),
            "error should mention segments mismatch: {}",
            error
        );
    }

    #[test]
    fn import_reject_conflict_on_mutated_segment() {
        // The Reject path inside import_segments (line 637) is never exercised
        // by existing tests. This creates a segment conflict specifically.
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("source.sqlite3");
        let export_dir = dir.path().join("export");
        let state_root = dir.path().join("state");
        let target_db = dir.path().join("target.sqlite3");

        let store = RunStore::open(&db_path).expect("store open");
        let report = fixture_report("seg-conflict-1", &db_path);
        store.persist_report(&report).expect("persist");
        export(&db_path, &export_dir, &state_root).expect("export");

        // First import — establishes the data.
        import(&target_db, &export_dir, &state_root, ConflictPolicy::Reject).expect("first import");

        // Mutate a segment in segments.jsonl.
        let seg_path = export_dir.join("segments.jsonl");
        let seg_content = fs::read_to_string(&seg_path).expect("read");
        let first_line = seg_content.lines().next().expect("has at least one line");
        let mut mutated: serde_json::Value =
            serde_json::from_str(first_line).expect("valid segment");
        mutated["text"] = json!("MUTATED SEGMENT TEXT");
        let new_content = format!(
            "{}\n{}",
            serde_json::to_string(&mutated).expect("serialize"),
            seg_content.lines().skip(1).collect::<Vec<_>>().join("\n")
        );
        fs::write(&seg_path, format!("{new_content}\n")).expect("write");

        // Update manifest checksums.
        let manifest_path = export_dir.join("manifest.json");
        let mut manifest_value: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(&manifest_path).expect("read"))
                .expect("parse");
        manifest_value["checksums"]["segments_jsonl_sha256"] =
            json!(sha256_file(&seg_path).expect("checksum"));
        fs::write(
            &manifest_path,
            serde_json::to_string_pretty(&manifest_value).expect("serialize"),
        )
        .expect("write");

        // Second import with Reject should fail with conflicts.
        let error = import(&target_db, &export_dir, &state_root, ConflictPolicy::Reject)
            .expect_err("mutated segment should cause conflict");
        let text = error.to_string();
        assert!(text.contains("import rejected due to"), "error: {text}");
    }

    #[test]
    fn export_fails_on_partial_schema_missing_events_table() {
        // verify_schema_exists loops over "runs", "segments", "events".
        // Existing test only covers fully missing schema (empty DB).
        // This hits the third iteration: DB has runs + segments but not events.
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("partial.sqlite3");
        let export_dir = dir.path().join("export");
        let state_root = dir.path().join("state");

        let conn = fsqlite::Connection::open(db_path.display().to_string()).expect("conn");
        conn.execute(
            "CREATE TABLE runs (id TEXT PRIMARY KEY, started_at TEXT NOT NULL, \
             finished_at TEXT NOT NULL, backend TEXT NOT NULL, input_path TEXT NOT NULL, \
             normalized_wav_path TEXT NOT NULL, request_json TEXT NOT NULL, \
             result_json TEXT NOT NULL, warnings_json TEXT NOT NULL, transcript TEXT NOT NULL);",
        )
        .expect("create runs");
        conn.execute(
            "CREATE TABLE segments (run_id TEXT NOT NULL, idx INTEGER NOT NULL, \
             start_sec REAL, end_sec REAL, speaker TEXT, text TEXT NOT NULL, \
             confidence REAL, PRIMARY KEY (run_id, idx));",
        )
        .expect("create segments");
        drop(conn);

        let error = export(&db_path, &export_dir, &state_root)
            .expect_err("missing events table should fail");
        let text = error.to_string();
        assert!(
            text.contains("missing table `events`"),
            "error should mention events table: {text}"
        );
    }

    #[test]
    fn import_rollback_on_mid_import_error_preserves_db() {
        // When import_tables returns Err, the ROLLBACK branch (line 424)
        // should leave the target DB clean.
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("source.sqlite3");
        let export_dir = dir.path().join("export");
        let state_root = dir.path().join("state");
        let target_db = dir.path().join("target.sqlite3");

        let store = RunStore::open(&db_path).expect("store open");
        let report = fixture_report("rollback-1", &db_path);
        store.persist_report(&report).expect("persist");
        export(&db_path, &export_dir, &state_root).expect("export");

        // Corrupt segments.jsonl: valid JSON but missing required idx field.
        let seg_path = export_dir.join("segments.jsonl");
        fs::write(&seg_path, r#"{"run_id":"rollback-1","text":"bad"}"#)
            .expect("write corrupt segments");

        // Update checksums for the modified file.
        let manifest_path = export_dir.join("manifest.json");
        let mut manifest_value: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(&manifest_path).expect("read"))
                .expect("parse");
        manifest_value["checksums"]["segments_jsonl_sha256"] =
            json!(sha256_file(&seg_path).expect("checksum"));
        manifest_value["row_counts"]["segments"] = json!(1);
        fs::write(
            &manifest_path,
            serde_json::to_string_pretty(&manifest_value).expect("serialize"),
        )
        .expect("write");

        // Import should fail (missing idx).
        let err = import(&target_db, &export_dir, &state_root, ConflictPolicy::Reject);
        assert!(err.is_err(), "import should fail due to missing idx");

        // Target DB should exist but have no runs (rollback cleaned up).
        let target_store = RunStore::open(&target_db).expect("target store");
        let runs = target_store.list_recent_runs(10).expect("list");
        assert!(
            runs.is_empty(),
            "rollback should leave 0 runs in target DB, found {}",
            runs.len()
        );
    }

    #[test]
    fn import_malformed_segments_jsonl_returns_error() {
        // Exercises the serde_json::from_str error path at line 597
        // (malformed JSON in segments.jsonl).
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("source.sqlite3");
        let export_dir = dir.path().join("export");
        let state_root = dir.path().join("state");
        let target_db = dir.path().join("target.sqlite3");

        let store = RunStore::open(&db_path).expect("store open");
        let report = fixture_report("bad-seg-json-1", &db_path);
        store.persist_report(&report).expect("persist");
        export(&db_path, &export_dir, &state_root).expect("export");

        // Replace segments.jsonl with invalid JSON.
        let seg_path = export_dir.join("segments.jsonl");
        fs::write(&seg_path, "{not valid json at all!!!}\n").expect("write");

        // Update checksum.
        let manifest_path = export_dir.join("manifest.json");
        let mut manifest_value: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(&manifest_path).expect("read"))
                .expect("parse");
        manifest_value["checksums"]["segments_jsonl_sha256"] =
            json!(sha256_file(&seg_path).expect("checksum"));
        manifest_value["row_counts"]["segments"] = json!(1);
        fs::write(
            &manifest_path,
            serde_json::to_string_pretty(&manifest_value).expect("serialize"),
        )
        .expect("write");

        let err = import(&target_db, &export_dir, &state_root, ConflictPolicy::Reject)
            .expect_err("malformed segments should fail");
        let text = err.to_string();
        assert!(
            text.contains("key must be") || text.contains("json") || text.contains("JSON"),
            "error should mention JSON parse failure: {text}"
        );
    }

    #[test]
    fn import_reject_conflict_on_mutated_event() {
        // Exercises the Reject path inside import_events (line 727).
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("source.sqlite3");
        let export_dir = dir.path().join("export");
        let state_root = dir.path().join("state");
        let target_db = dir.path().join("target.sqlite3");

        let store = RunStore::open(&db_path).expect("store open");
        let mut report = fixture_report("evt-conflict-1", &db_path);
        report.events.push(RunEvent {
            seq: 99,
            ts_rfc3339: "2026-01-01T00:00:09Z".to_owned(),
            stage: "test".to_owned(),
            code: "test.ok".to_owned(),
            message: "original".to_owned(),
            payload: json!({}),
        });
        store.persist_report(&report).expect("persist");
        export(&db_path, &export_dir, &state_root).expect("export");

        // First import — establishes the data.
        import(&target_db, &export_dir, &state_root, ConflictPolicy::Reject).expect("first import");

        // Mutate an event in events.jsonl.
        let evt_path = export_dir.join("events.jsonl");
        let evt_content = fs::read_to_string(&evt_path).expect("read");
        let first_line = evt_content.lines().next().expect("has line");
        let mut mutated: serde_json::Value = serde_json::from_str(first_line).expect("valid event");
        mutated["message"] = json!("MUTATED MESSAGE");
        let new_content = format!(
            "{}\n{}",
            serde_json::to_string(&mutated).expect("serialize"),
            evt_content.lines().skip(1).collect::<Vec<_>>().join("\n")
        );
        fs::write(&evt_path, format!("{new_content}\n")).expect("write");

        // Update checksums.
        let manifest_path = export_dir.join("manifest.json");
        let mut manifest_value: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(&manifest_path).expect("read"))
                .expect("parse");
        manifest_value["checksums"]["events_jsonl_sha256"] =
            json!(sha256_file(&evt_path).expect("checksum"));
        fs::write(
            &manifest_path,
            serde_json::to_string_pretty(&manifest_value).expect("serialize"),
        )
        .expect("write");

        let error = import(&target_db, &export_dir, &state_root, ConflictPolicy::Reject)
            .expect_err("mutated event should cause conflict");
        let text = error.to_string();
        assert!(text.contains("import rejected due to"), "error: {text}");
    }

    #[test]
    fn import_empty_export_dir_with_valid_manifest_but_missing_jsonl_errors() {
        // Manifest references JSONL files that don't exist on disk.
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("source.sqlite3");
        let export_dir = dir.path().join("export");
        let state_root = dir.path().join("state");
        let target_db = dir.path().join("target.sqlite3");

        // Create a real export, then delete the data files but keep manifest.
        let store = RunStore::open(&db_path).expect("store open");
        let report = fixture_report("ghost-files", &db_path);
        store.persist_report(&report).expect("persist");
        export(&db_path, &export_dir, &state_root).expect("export");

        // Remove the actual data files.
        let _ = fs::remove_file(export_dir.join("runs.jsonl"));
        let _ = fs::remove_file(export_dir.join("segments.jsonl"));
        let _ = fs::remove_file(export_dir.join("events.jsonl"));

        let error = import(&target_db, &export_dir, &state_root, ConflictPolicy::Reject)
            .expect_err("missing JSONL files should fail");
        let text = error.to_string();
        // The error should mention the missing file or a read failure.
        assert!(
            text.contains("runs.jsonl")
                || text.contains("No such file")
                || text.contains("not found"),
            "error should reference missing file: {text}"
        );
    }

    #[test]
    fn import_corrupt_manifest_json_returns_error() {
        let dir = tempdir().expect("tempdir");
        let export_dir = dir.path().join("bad_manifest");
        let state_root = dir.path().join("state");
        let target_db = dir.path().join("target.sqlite3");
        fs::create_dir_all(&export_dir).expect("mkdir");
        fs::write(export_dir.join("manifest.json"), "{not valid json!!!").expect("write");

        let error = import(&target_db, &export_dir, &state_root, ConflictPolicy::Reject)
            .expect_err("corrupt manifest should fail");
        let text = error.to_string();
        assert!(
            text.contains("invalid manifest"),
            "error should mention invalid manifest: {text}"
        );
    }

    #[test]
    fn import_checksum_mismatch_returns_error() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("source.sqlite3");
        let export_dir = dir.path().join("export");
        let state_root = dir.path().join("state");
        let target_db = dir.path().join("target.sqlite3");

        let store = RunStore::open(&db_path).expect("store open");
        let report = fixture_report("checksum-1", &db_path);
        store.persist_report(&report).expect("persist");
        export(&db_path, &export_dir, &state_root).expect("export");

        // Corrupt runs.jsonl content to invalidate checksum.
        let runs_path = export_dir.join("runs.jsonl");
        fs::write(&runs_path, "extra garbage line\n").expect("write");

        let error = import(&target_db, &export_dir, &state_root, ConflictPolicy::Reject)
            .expect_err("checksum mismatch should fail");
        let text = error.to_string();
        assert!(
            text.contains("checksum mismatch"),
            "error should mention checksum: {text}"
        );
    }

    #[test]
    fn overwrite_replaces_run_data_on_second_import() {
        // When Overwrite policy encounters a different payload for the same run_id,
        // it deletes and re-inserts. Verify the new data is present afterwards.
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("source.sqlite3");
        let export_dir = dir.path().join("export");
        let state_root = dir.path().join("state");
        let target_db = dir.path().join("target.sqlite3");

        let store = RunStore::open(&db_path).expect("store open");
        let report = fixture_report("ow-replace-1", &db_path);
        store.persist_report(&report).expect("persist");
        export(&db_path, &export_dir, &state_root).expect("export");

        // First import.
        import(
            &target_db,
            &export_dir,
            &state_root,
            ConflictPolicy::Overwrite,
        )
        .expect("first import");

        // Mutate the transcript in runs.jsonl.
        let runs_path = export_dir.join("runs.jsonl");
        let content = fs::read_to_string(&runs_path).expect("read");
        let first_line = content.lines().next().expect("has line");
        let mut mutated: serde_json::Value = serde_json::from_str(first_line).expect("valid run");
        mutated["transcript"] = json!("OVERWRITTEN transcript text");
        fs::write(
            &runs_path,
            format!("{}\n", serde_json::to_string(&mutated).expect("s")),
        )
        .expect("write");

        let manifest_path = export_dir.join("manifest.json");
        let mut manifest_value: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(&manifest_path).expect("read"))
                .expect("parse");
        manifest_value["checksums"]["runs_jsonl_sha256"] =
            json!(sha256_file(&runs_path).expect("checksum"));
        // Update row_counts.runs to 1 since we only have 1 line now.
        manifest_value["row_counts"]["runs"] = json!(1);
        fs::write(
            &manifest_path,
            serde_json::to_string_pretty(&manifest_value).expect("serialize"),
        )
        .expect("write");

        // Second import with Overwrite — should succeed, replacing the run data.
        let result = import(
            &target_db,
            &export_dir,
            &state_root,
            ConflictPolicy::Overwrite,
        )
        .expect("overwrite import");
        // Verify the target DB now has the overwritten transcript column.
        let conn = Connection::open(target_db.display().to_string()).expect("open target db");
        let rows = conn
            .query_with_params(
                "SELECT transcript FROM runs WHERE id = ?1",
                &[SqliteValue::Text("ow-replace-1".to_owned())],
            )
            .expect("query");
        assert!(!rows.is_empty(), "run should exist after overwrite");
        let transcript = value_to_string_sqlite(rows[0].get(0));
        assert_eq!(
            transcript, "OVERWRITTEN transcript text",
            "overwrite should replace old transcript column"
        );
        assert_eq!(result.runs_imported, 1, "should report 1 run imported");
    }

    #[test]
    fn value_to_json_all_sqlite_variants() {
        use super::value_to_json;
        assert_eq!(
            value_to_json(Some(&SqliteValue::Text("hello".to_owned()))),
            json!("hello")
        );
        assert_eq!(value_to_json(Some(&SqliteValue::Integer(42))), json!(42));
        assert_eq!(value_to_json(Some(&SqliteValue::Float(2.75))), json!(2.75));
        assert_eq!(
            value_to_json(Some(&SqliteValue::Blob(vec![1, 2, 3]))),
            json!("<blob:3>")
        );
        assert_eq!(
            value_to_json(Some(&SqliteValue::Blob(vec![]))),
            json!("<blob:0>")
        );
        assert_eq!(value_to_json(Some(&SqliteValue::Null)), json!(null));
        assert_eq!(value_to_json(None), json!(null));
    }

    #[test]
    fn export_import_with_special_json_characters_in_transcript() {
        // Verify that transcripts containing JSON-special characters
        // (quotes, backslashes, newlines) survive the export/import round-trip.
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("special_chars.sqlite3");
        let export_dir = dir.path().join("export");
        let state_root = dir.path().join("state");
        let target_db = dir.path().join("target.sqlite3");

        let store = RunStore::open(&db_path).expect("store open");
        let mut report = fixture_report("special-chars", &db_path);
        report.result.transcript = "He said \"hello\"\nand\\backslash\ttab\0null".to_owned();
        report.result.segments[0].text = "line1\nline2".to_owned();
        store.persist_report(&report).expect("persist");

        export(&db_path, &export_dir, &state_root).expect("export");
        import(&target_db, &export_dir, &state_root, ConflictPolicy::Reject).expect("import");

        let target_store = RunStore::open(&target_db).expect("open target");
        let details = target_store
            .load_run_details("special-chars")
            .expect("load")
            .expect("exists");
        assert!(
            details.transcript.contains("\\"),
            "backslash should survive round-trip"
        );
        assert_eq!(
            details.segments[0].text, "line1\nline2",
            "newline in segment text should survive"
        );
    }

    #[test]
    fn import_identical_event_twice_is_noop() {
        // When import encounters the same event with identical fields,
        // it is a noop (line 721-723) — no conflict, no error.
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("idempotent_event.sqlite3");
        let export_dir = dir.path().join("export");
        let state_root = dir.path().join("state");
        let target_db = dir.path().join("target.sqlite3");

        let store = RunStore::open(&db_path).expect("store open");
        let report = fixture_report("idem-evt", &db_path);
        store.persist_report(&report).expect("persist");
        export(&db_path, &export_dir, &state_root).expect("export");

        // Import twice — second import should be idempotent.
        import(&target_db, &export_dir, &state_root, ConflictPolicy::Reject).expect("first import");
        let result = import(&target_db, &export_dir, &state_root, ConflictPolicy::Reject)
            .expect("second import should succeed (identical data)");
        assert!(
            result.conflicts.is_empty(),
            "identical re-import should have no conflicts"
        );
    }

    #[test]
    fn ensure_schema_is_idempotent() {
        // Calling ensure_schema twice on the same DB should not error.
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("schema_idem.sqlite3");
        let conn = Connection::open(db_path.display().to_string()).expect("open");
        ensure_schema(&conn).expect("first schema create");
        ensure_schema(&conn).expect("second schema create should be idempotent");

        // Tables should all exist.
        for table in ["runs", "segments", "events"] {
            let sql = format!("SELECT 1 FROM {table} LIMIT 1");
            conn.query(&sql)
                .unwrap_or_else(|_| panic!("table {table} should exist"));
        }
    }

    #[test]
    fn export_with_replay_json_field_round_trips() {
        // Verify that the replay_json column is included in exports
        // and correctly imported via ensure_runs_replay_column.
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("replay_export.sqlite3");
        let export_dir = dir.path().join("export");
        let state_root = dir.path().join("state");
        let target_db = dir.path().join("target.sqlite3");

        let store = RunStore::open(&db_path).expect("store open");
        let mut report = fixture_report("replay-rt", &db_path);
        report.replay.input_content_hash = Some("abc123".to_owned());
        report.replay.backend_identity = Some("whisper-cli".to_owned());
        store.persist_report(&report).expect("persist");

        export(&db_path, &export_dir, &state_root).expect("export");

        // Verify runs.jsonl contains replay_json field.
        let runs_content = fs::read_to_string(export_dir.join("runs.jsonl")).expect("read");
        assert!(
            runs_content.contains("replay_json"),
            "runs.jsonl should contain replay_json field"
        );
        assert!(
            runs_content.contains("abc123"),
            "runs.jsonl should contain replay hash"
        );

        // Import and verify replay data round-trips.
        import(&target_db, &export_dir, &state_root, ConflictPolicy::Reject).expect("import");
        let target_store = RunStore::open(&target_db).expect("open target");
        let details = target_store
            .load_run_details("replay-rt")
            .expect("load")
            .expect("exists");
        assert_eq!(details.replay.input_content_hash.as_deref(), Some("abc123"));
        assert_eq!(
            details.replay.backend_identity.as_deref(),
            Some("whisper-cli")
        );
    }

    // ── Ninth-pass edge case tests ──

    #[test]
    fn is_lock_stale_returns_true_for_alive_pid_with_aged_timestamp() {
        // PID is alive (current process) but timestamp is >300 seconds old.
        // The lock should be considered stale due to age.
        let old_time = Utc::now() - chrono::Duration::seconds(LOCK_STALE_SECONDS + 60);
        let info = LockInfo {
            pid: std::process::id(),
            created_at_rfc3339: old_time.to_rfc3339(),
            operation: "export".to_owned(),
        };
        assert!(
            is_lock_stale(&info),
            "should be stale: alive PID but old timestamp"
        );
    }

    #[test]
    fn verify_schema_exists_rejects_missing_segments_table() {
        // DB has runs + events tables but NO segments → should fail on iteration 2.
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("no_segments.sqlite3");
        let conn = fsqlite::Connection::open(db_path.display().to_string()).expect("conn");
        conn.execute(
            "CREATE TABLE runs (id TEXT PRIMARY KEY, started_at TEXT NOT NULL, \
             finished_at TEXT NOT NULL, backend TEXT NOT NULL, input_path TEXT NOT NULL, \
             normalized_wav_path TEXT NOT NULL, request_json TEXT NOT NULL, \
             result_json TEXT NOT NULL, warnings_json TEXT NOT NULL, transcript TEXT NOT NULL);",
        )
        .expect("create runs");
        conn.execute(
            "CREATE TABLE events (run_id TEXT NOT NULL, seq INTEGER NOT NULL, \
             ts_rfc3339 TEXT NOT NULL, stage TEXT NOT NULL, code TEXT NOT NULL, \
             message TEXT NOT NULL, payload_json TEXT NOT NULL, PRIMARY KEY (run_id, seq));",
        )
        .expect("create events");

        let err = verify_schema_exists(&conn).expect_err("missing segments should fail");
        let text = err.to_string();
        assert!(
            text.contains("missing table `segments`"),
            "should mention segments: {text}"
        );
    }

    #[test]
    fn sync_lock_file_records_operation_name() {
        // The lock file should contain the operation name as JSON.
        let dir = tempdir().expect("tempdir");
        let state_root = dir.path().join("state");

        let lock = SyncLock::acquire(&state_root, "test_op").expect("acquire");
        let lock_path = state_root.join("locks").join("sync.lock");
        assert!(lock_path.exists(), "lock file should exist");

        let contents = fs::read_to_string(&lock_path).expect("read lock");
        let info: LockInfo = serde_json::from_str(&contents).expect("parse lock JSON");
        assert_eq!(info.operation, "test_op");
        assert_eq!(info.pid, std::process::id());
        assert!(!info.created_at_rfc3339.is_empty());

        drop(lock); // Release cleans up
        assert!(!lock_path.exists(), "lock should be removed after drop");
    }

    #[test]
    fn export_import_row_counts_preserved_in_round_trip() {
        // Export a DB with specific counts, then verify the manifest row counts
        // match, and after import into a fresh DB the counts still hold.
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("counts.sqlite3");
        let export_dir = dir.path().join("export");
        let state_root = dir.path().join("state");

        let store = RunStore::open(&db_path).expect("store");
        for i in 0..3 {
            store
                .persist_report(&fixture_report(&format!("run-{i}"), &db_path))
                .expect("persist");
        }
        drop(store);
        let conn = Connection::open(db_path.display().to_string()).expect("open");
        ensure_schema(&conn).expect("schema");
        drop(conn);

        let manifest = export(&db_path, &export_dir, &state_root).expect("export");
        assert_eq!(manifest.row_counts.runs, 3);
        // fixture_report creates 2 segments and 2 events per run
        assert_eq!(manifest.row_counts.segments, 6);
        assert_eq!(manifest.row_counts.events, 6);

        let target_db = dir.path().join("target.sqlite3");
        let result =
            import(&target_db, &export_dir, &state_root, ConflictPolicy::Reject).expect("import");
        assert_eq!(result.runs_imported, 3);
        assert_eq!(result.segments_imported, 6);
        assert_eq!(result.events_imported, 6);
        assert!(result.conflicts.is_empty());
    }

    #[test]
    fn validate_checksums_fails_on_first_missing_file() {
        // validate_checksums iterates ["runs.jsonl", "segments.jsonl", "events.jsonl"]
        // and should fail on the first missing file (runs.jsonl).
        let dir = tempdir().expect("tempdir");
        let input_dir = dir.path().join("partial");
        fs::create_dir_all(&input_dir).expect("mkdir");

        // Create only segments.jsonl and events.jsonl but NOT runs.jsonl.
        fs::write(input_dir.join("segments.jsonl"), "").expect("write");
        fs::write(input_dir.join("events.jsonl"), "").expect("write");

        let manifest = SyncManifest {
            schema_version: SCHEMA_VERSION.to_owned(),
            export_format_version: EXPORT_FORMAT_VERSION.to_owned(),
            created_at_rfc3339: Utc::now().to_rfc3339(),
            source_db_path: "test.sqlite3".to_owned(),
            row_counts: RowCounts {
                runs: 0,
                segments: 0,
                events: 0,
            },
            checksums: FileChecksums {
                runs_jsonl_sha256: "fake".to_owned(),
                segments_jsonl_sha256: "fake".to_owned(),
                events_jsonl_sha256: "fake".to_owned(),
            },
        };

        let err = validate_checksums(&manifest, &input_dir).expect_err("missing runs.jsonl");
        let text = err.to_string();
        assert!(
            text.contains("missing export file") && text.contains("runs.jsonl"),
            "should fail on runs.jsonl specifically: {text}"
        );
    }

    #[test]
    fn import_events_with_seq_as_string_returns_missing_seq_error() {
        // When seq is present but a string (not integer), as_i64() returns None
        // and import_events should return "missing seq in events row" error.
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("source.sqlite3");
        let export_dir = dir.path().join("export");
        let state_root = dir.path().join("state");
        let target_db = dir.path().join("target.sqlite3");

        let store = RunStore::open(&db_path).expect("store");
        store
            .persist_report(&fixture_report("string-seq", &db_path))
            .expect("persist");
        export(&db_path, &export_dir, &state_root).expect("export");

        // Overwrite events.jsonl with seq as a string instead of integer.
        let events_path = export_dir.join("events.jsonl");
        fs::write(
            &events_path,
            r#"{"run_id":"string-seq","seq":"1","ts_rfc3339":"2026-01-01T00:00:01Z","stage":"ingest","code":"ingest.ok","message":"test","payload_json":"{}"}"#,
        )
        .expect("write");

        // Update manifest checksums and row count for the modified file.
        let manifest_path = export_dir.join("manifest.json");
        let mut manifest_val: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(&manifest_path).expect("read"))
                .expect("parse");
        manifest_val["checksums"]["events_jsonl_sha256"] =
            json!(sha256_file(&events_path).expect("hash"));
        manifest_val["row_counts"]["events"] = json!(1);
        fs::write(
            &manifest_path,
            serde_json::to_string_pretty(&manifest_val).expect("ser"),
        )
        .expect("write manifest");

        let result = import(&target_db, &export_dir, &state_root, ConflictPolicy::Reject);
        let err = result.expect_err("string seq should fail import");
        assert!(
            err.to_string().contains("missing seq"),
            "error should mention missing seq: {}",
            err
        );
    }

    #[test]
    fn import_segments_with_idx_as_float_returns_missing_idx_error() {
        // idx as 0.5 (float, not integer): as_i64() returns None → "missing idx" error.
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("source.sqlite3");
        let export_dir = dir.path().join("export");
        let state_root = dir.path().join("state");
        let target_db = dir.path().join("target.sqlite3");

        let store = RunStore::open(&db_path).expect("store");
        store
            .persist_report(&fixture_report("float-idx", &db_path))
            .expect("persist");
        export(&db_path, &export_dir, &state_root).expect("export");

        let seg_path = export_dir.join("segments.jsonl");
        fs::write(
            &seg_path,
            r#"{"run_id":"float-idx","idx":0.5,"text":"bad","start_sec":null,"end_sec":null,"speaker":null,"confidence":null}"#,
        )
        .expect("write");

        let manifest_path = export_dir.join("manifest.json");
        let mut manifest_val: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(&manifest_path).expect("read"))
                .expect("parse");
        manifest_val["checksums"]["segments_jsonl_sha256"] =
            json!(sha256_file(&seg_path).expect("hash"));
        manifest_val["row_counts"]["segments"] = json!(1);
        fs::write(
            &manifest_path,
            serde_json::to_string_pretty(&manifest_val).expect("ser"),
        )
        .expect("write manifest");

        let result = import(&target_db, &export_dir, &state_root, ConflictPolicy::Reject);
        let err = result.expect_err("float idx should fail import");
        assert!(
            err.to_string().contains("missing idx"),
            "error should mention missing idx: {}",
            err
        );
    }

    #[test]
    fn lock_acquire_fails_when_state_root_path_is_invalid() {
        // Attempting to create locks dir under /dev/null (not a directory) should fail.
        let state_root = PathBuf::from("/dev/null");
        let result = SyncLock::acquire(&state_root, "test_op");
        assert!(result.is_err(), "lock acquire under /dev/null should fail");
    }

    #[test]
    fn import_tolerates_extra_unknown_fields_in_jsonl_rows() {
        // JSONL rows may contain extra fields beyond what import expects.
        // Since parsing uses serde_json::Value (not strict struct deserialization),
        // extra fields should be silently ignored.
        let dir = tempdir().expect("tempdir");
        let export_dir = dir.path().join("export");
        let state_root = dir.path().join("state");
        let target_db = dir.path().join("target.sqlite3");

        write_jsonl_snapshot(
            &export_dir,
            &[json!({
                "id": "extra-fields-run",
                "started_at": "2026-01-01T00:00:00Z",
                "finished_at": "2026-01-01T00:00:05Z",
                "backend": "whisper_cpp",
                "input_path": "test.wav",
                "normalized_wav_path": "norm.wav",
                "request_json": "{}",
                "result_json": "{\"backend\":\"whisper_cpp\",\"transcript\":\"hello\",\"language\":null,\"segments\":[],\"acceleration\":null,\"raw_output\":{},\"artifact_paths\":[]}",
                "warnings_json": "[]",
                "transcript": "hello",
                "replay_json": "{}",
                "extra_field_1": "should be ignored",
                "extra_field_2": 42,
                "metadata": {"nested": true},
            })],
            &[json!({
                "run_id": "extra-fields-run",
                "idx": 0,
                "text": "hello",
                "start_sec": 0.0,
                "end_sec": 1.0,
                "speaker": null,
                "confidence": null,
                "bonus_field": "ignored",
            })],
            &[json!({
                "run_id": "extra-fields-run",
                "seq": 1,
                "ts_rfc3339": "2026-01-01T00:00:01Z",
                "stage": "backend",
                "code": "backend.ok",
                "message": "done",
                "payload_json": "{}",
                "unknown_col": [1, 2, 3],
            })],
        );

        let result = import(&target_db, &export_dir, &state_root, ConflictPolicy::Reject)
            .expect("import should succeed despite extra fields");
        assert_eq!(result.runs_imported, 1);
        assert_eq!(result.segments_imported, 1);
        assert_eq!(result.events_imported, 1);
        assert!(result.conflicts.is_empty());
    }

    #[test]
    fn archive_stale_lock_creates_file_with_json_extension() {
        let dir = tempdir().expect("tempdir");
        let state_root = dir.path().join("state");
        let locks_dir = state_root.join("locks");
        fs::create_dir_all(&locks_dir).expect("locks dir");

        // Create a stale lock (old timestamp, current pid).
        let old_time = Utc::now() - chrono::Duration::seconds(LOCK_STALE_SECONDS + 120);
        let stale_info = LockInfo {
            pid: std::process::id(),
            created_at_rfc3339: old_time.to_rfc3339(),
            operation: "export".to_owned(),
        };
        fs::write(
            locks_dir.join("sync.lock"),
            serde_json::to_string(&stale_info).expect("ser"),
        )
        .expect("write stale lock");

        // Acquire should archive the stale lock and succeed.
        let lock = SyncLock::acquire(&state_root, "new_op").expect("acquire");
        lock.release().expect("release");

        // Verify the archived file has the expected naming pattern: sync.lock.stale.{timestamp}.json
        let archived_names: Vec<String> = fs::read_dir(&locks_dir)
            .expect("read dir")
            .filter_map(Result::ok)
            .map(|entry| entry.file_name().to_string_lossy().to_string())
            .filter(|name| name.starts_with("sync.lock.stale."))
            .collect();
        assert_eq!(archived_names.len(), 1, "one stale archive expected");
        assert!(
            archived_names[0].ends_with(".json"),
            "archived filename should end with .json: {}",
            archived_names[0]
        );
    }

    #[test]
    fn validate_schema_version_empty_string_rejects() {
        let manifest = SyncManifest {
            schema_version: String::new(),
            export_format_version: EXPORT_FORMAT_VERSION.to_owned(),
            created_at_rfc3339: Utc::now().to_rfc3339(),
            source_db_path: "test.sqlite3".to_owned(),
            row_counts: RowCounts {
                runs: 0,
                segments: 0,
                events: 0,
            },
            checksums: FileChecksums {
                runs_jsonl_sha256: String::new(),
                segments_jsonl_sha256: String::new(),
                events_jsonl_sha256: String::new(),
            },
        };
        let err = validate_schema_version(&manifest).expect_err("empty version should fail");
        assert!(
            err.to_string().contains("schema version mismatch"),
            "error: {}",
            err
        );
    }

    #[test]
    fn exported_runs_jsonl_has_all_expected_keys() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("source.sqlite3");
        let export_dir = dir.path().join("export");
        let state_root = dir.path().join("state");

        let store = RunStore::open(&db_path).expect("store");
        store
            .persist_report(&fixture_report("keys-run", &db_path))
            .expect("persist");
        export(&db_path, &export_dir, &state_root).expect("export");

        let content = fs::read_to_string(export_dir.join("runs.jsonl")).expect("read");
        let line = content
            .lines()
            .next()
            .expect("should have at least one line");
        let row: serde_json::Value = serde_json::from_str(line).expect("parse");
        let obj = row.as_object().expect("should be object");

        let expected_keys = [
            "id",
            "started_at",
            "finished_at",
            "backend",
            "input_path",
            "normalized_wav_path",
            "request_json",
            "result_json",
            "warnings_json",
            "transcript",
            "replay_json",
        ];
        for key in &expected_keys {
            assert!(
                obj.contains_key(*key),
                "runs JSONL should contain key `{key}`"
            );
        }
        assert_eq!(
            obj.len(),
            expected_keys.len(),
            "should have exactly {} keys, got {}",
            expected_keys.len(),
            obj.len()
        );
    }

    #[test]
    fn exported_segments_preserve_null_optional_fields() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("source.sqlite3");
        let export_dir = dir.path().join("export");
        let state_root = dir.path().join("state");

        let mut report = fixture_report("null-opts", &db_path);
        report.result.segments = vec![crate::model::TranscriptionSegment {
            start_sec: None,
            end_sec: None,
            text: "bare text".to_owned(),
            speaker: None,
            confidence: None,
        }];
        let store = RunStore::open(&db_path).expect("store");
        store.persist_report(&report).expect("persist");
        export(&db_path, &export_dir, &state_root).expect("export");

        let content = fs::read_to_string(export_dir.join("segments.jsonl")).expect("read");
        let line = content.lines().next().expect("should have segment line");
        let row: serde_json::Value = serde_json::from_str(line).expect("parse");

        assert!(row["start_sec"].is_null(), "None start_sec → null");
        assert!(row["end_sec"].is_null(), "None end_sec → null");
        assert!(row["speaker"].is_null(), "None speaker → null");
        assert!(row["confidence"].is_null(), "None confidence → null");
        assert_eq!(row["text"], "bare text");
    }

    #[test]
    fn validate_checksums_succeeds_when_all_match() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("source.sqlite3");
        let export_dir = dir.path().join("export");
        let state_root = dir.path().join("state");

        let store = RunStore::open(&db_path).expect("store");
        store
            .persist_report(&fixture_report("chk-ok", &db_path))
            .expect("persist");
        let manifest = export(&db_path, &export_dir, &state_root).expect("export");

        validate_checksums(&manifest, &export_dir).expect("checksums should match");
    }

    #[test]
    fn import_runs_without_replay_json_field_uses_default() {
        let dir = tempdir().expect("tempdir");
        let export_dir = dir.path().join("export");
        let state_root = dir.path().join("state");
        let target_db = dir.path().join("target.sqlite3");

        write_jsonl_snapshot(
            &export_dir,
            &[json!({
                "id": "no-replay",
                "started_at": "2026-01-01T00:00:00Z",
                "finished_at": "2026-01-01T00:00:05Z",
                "backend": "whisper_cpp",
                "input_path": "test.wav",
                "normalized_wav_path": "norm.wav",
                "request_json": "{}",
                "result_json": "{\"backend\":\"whisper_cpp\",\"transcript\":\"hello\",\"language\":null,\"segments\":[],\"acceleration\":null,\"raw_output\":{},\"artifact_paths\":[]}",
                "warnings_json": "[]",
                "transcript": "hello",
            })],
            &[],
            &[],
        );

        let result = import(&target_db, &export_dir, &state_root, ConflictPolicy::Reject)
            .expect("import should succeed without replay_json");
        assert_eq!(result.runs_imported, 1);

        let store = RunStore::open(&target_db).expect("store");
        let details = store
            .load_run_details("no-replay")
            .expect("query")
            .expect("exists");
        assert!(
            details.replay.input_content_hash.is_none(),
            "missing replay_json should default to empty envelope"
        );
    }

    #[test]
    fn import_row_count_mismatch_for_events_errors() {
        // Runs and segments mismatch are tested; verify events mismatch too.
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("source.sqlite3");
        let export_dir = dir.path().join("export");
        let state_root = dir.path().join("state");
        let target_db = dir.path().join("target.sqlite3");

        let store = RunStore::open(&db_path).expect("store");
        store
            .persist_report(&fixture_report("evt-mis", &db_path))
            .expect("persist");
        export(&db_path, &export_dir, &state_root).expect("export");

        // Inflate the manifest's events row count so it doesn't match actual data.
        let manifest_path = export_dir.join("manifest.json");
        let mut manifest_value: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(&manifest_path).expect("read")).expect("json");
        manifest_value["row_counts"]["events"] = json!(999);
        fs::write(
            &manifest_path,
            serde_json::to_string_pretty(&manifest_value).expect("ser"),
        )
        .expect("write");

        let err = import(&target_db, &export_dir, &state_root, ConflictPolicy::Reject)
            .expect_err("events row count mismatch should error");
        let msg = err.to_string();
        assert!(
            msg.contains("row count mismatch") && msg.contains("events"),
            "error should mention events mismatch, got: {msg}"
        );
    }

    #[test]
    fn exported_events_jsonl_has_all_expected_keys() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("evt_keys.sqlite3");
        let export_dir = dir.path().join("export");
        let state_root = dir.path().join("state");

        let store = RunStore::open(&db_path).expect("store");
        store
            .persist_report(&fixture_report("evt-keys", &db_path))
            .expect("persist");
        export(&db_path, &export_dir, &state_root).expect("export");

        let events_text =
            fs::read_to_string(export_dir.join("events.jsonl")).expect("events.jsonl");
        let first_line = events_text.lines().next().expect("at least one event line");
        let obj: serde_json::Value = serde_json::from_str(first_line).expect("parse");
        let map = obj.as_object().expect("should be object");
        let expected_keys = [
            "run_id",
            "seq",
            "ts_rfc3339",
            "stage",
            "code",
            "message",
            "payload_json",
        ];
        for key in &expected_keys {
            assert!(map.contains_key(*key), "events JSONL missing key: {key}");
        }
        assert_eq!(
            map.len(),
            expected_keys.len(),
            "events JSONL should have exactly {} keys",
            expected_keys.len()
        );
    }

    #[test]
    fn import_identical_segment_twice_is_noop() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("seg_noop.sqlite3");
        let export_dir = dir.path().join("export");
        let state_root = dir.path().join("state");

        let store = RunStore::open(&db_path).expect("store");
        store
            .persist_report(&fixture_report("seg-noop", &db_path))
            .expect("persist");
        export(&db_path, &export_dir, &state_root).expect("export");

        // Import into same DB — segments already exist with identical data.
        let result = import(&db_path, &export_dir, &state_root, ConflictPolicy::Reject)
            .expect("identical re-import should succeed");
        assert!(
            result.conflicts.is_empty(),
            "identical segment data should produce no conflicts"
        );
        assert!(
            result.segments_imported > 0,
            "segments should still be counted"
        );
    }

    #[test]
    fn is_lock_stale_returns_false_for_alive_fresh_lock() {
        let info = LockInfo {
            pid: std::process::id(),
            created_at_rfc3339: Utc::now().to_rfc3339(),
            operation: "test".to_owned(),
        };
        assert!(
            !is_lock_stale(&info),
            "lock with current pid and fresh timestamp should not be stale"
        );
    }

    #[test]
    fn json_str_returns_error_for_integer_value() {
        let value = json!({"id": 42});
        let err = json_str(&value, "id").expect_err("integer should fail as_str");
        assert!(
            err.to_string().contains("missing string field"),
            "error should mention missing string field, got: {}",
            err
        );
    }

    // ── Comprehensive edge-case tests (added) ──

    #[test]
    fn export_empty_db_produces_valid_jsonl_files() {
        // Verify that exporting an empty DB yields syntactically valid (empty)
        // JSONL files that can be successfully re-imported.
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("empty_jsonl.sqlite3");
        let export_dir = dir.path().join("export");
        let state_root = dir.path().join("state");
        let target_db = dir.path().join("target.sqlite3");

        let _store = RunStore::open(&db_path).expect("store open");
        let manifest = export(&db_path, &export_dir, &state_root).expect("export");

        // All JSONL files should exist and be empty.
        for filename in ["runs.jsonl", "segments.jsonl", "events.jsonl"] {
            let path = export_dir.join(filename);
            assert!(path.exists(), "{filename} should exist");
            let content = fs::read_to_string(&path).expect("read");
            assert!(
                content.trim().is_empty(),
                "{filename} should be empty for an empty DB, got: {content:?}"
            );
        }

        // Checksums should be for empty files.
        assert_eq!(
            manifest.checksums.runs_jsonl_sha256, manifest.checksums.segments_jsonl_sha256,
            "all checksums should be identical for empty files"
        );

        // Re-importing should succeed with zero rows.
        let result = import(&target_db, &export_dir, &state_root, ConflictPolicy::Reject)
            .expect("import of empty export");
        assert_eq!(result.runs_imported, 0);
        assert_eq!(result.segments_imported, 0);
        assert_eq!(result.events_imported, 0);
        assert!(result.conflicts.is_empty());
    }

    #[test]
    fn import_with_reject_policy_fails_on_different_duplicates() {
        // Import data, then import the same run IDs with different payloads
        // under Reject policy. The import should fail and produce conflicts.
        let dir = tempdir().expect("tempdir");
        let export_dir = dir.path().join("export");
        let state_root = dir.path().join("state");
        let target_db = dir.path().join("target.sqlite3");

        let run_json = json!({
            "id": "reject-dup-1",
            "started_at": "2026-01-01T00:00:00Z",
            "finished_at": "2026-01-01T00:00:05Z",
            "backend": "whisper_cpp",
            "input_path": "test.wav",
            "normalized_wav_path": "norm.wav",
            "request_json": "{}",
            "result_json": "{\"backend\":\"whisper_cpp\",\"transcript\":\"original\",\"language\":null,\"segments\":[],\"acceleration\":null,\"raw_output\":{},\"artifact_paths\":[]}",
            "warnings_json": "[]",
            "transcript": "original transcript",
            "replay_json": "{}",
        });

        // First import.
        write_jsonl_snapshot(&export_dir, std::slice::from_ref(&run_json), &[], &[]);
        import(&target_db, &export_dir, &state_root, ConflictPolicy::Reject)
            .expect("first import should succeed");

        // Second import with different transcript.
        let export_dir_2 = dir.path().join("export2");
        let mut mutated = run_json.clone();
        mutated["transcript"] = json!("DIFFERENT transcript");
        write_jsonl_snapshot(&export_dir_2, &[mutated], &[], &[]);

        let err = import(
            &target_db,
            &export_dir_2,
            &state_root,
            ConflictPolicy::Reject,
        )
        .expect_err("reject policy should fail on different duplicate");
        let text = err.to_string();
        assert!(
            text.contains("import rejected due to"),
            "error should mention rejection: {text}"
        );
    }

    #[test]
    fn import_identical_data_twice_with_reject_is_noop() {
        // Importing identical data twice under Reject policy should succeed
        // (no conflicts because the data is the same -- effectively a "skip").
        let dir = tempdir().expect("tempdir");
        let export_dir = dir.path().join("export");
        let state_root = dir.path().join("state");
        let target_db = dir.path().join("target.sqlite3");

        let run_json = json!({
            "id": "skip-dup-1",
            "started_at": "2026-01-01T00:00:00Z",
            "finished_at": "2026-01-01T00:00:05Z",
            "backend": "whisper_cpp",
            "input_path": "test.wav",
            "normalized_wav_path": "norm.wav",
            "request_json": "{}",
            "result_json": "{\"backend\":\"whisper_cpp\",\"transcript\":\"hello\",\"language\":null,\"segments\":[],\"acceleration\":null,\"raw_output\":{},\"artifact_paths\":[]}",
            "warnings_json": "[]",
            "transcript": "hello",
            "replay_json": "{}",
        });
        let seg_json = json!({
            "run_id": "skip-dup-1",
            "idx": 0,
            "start_sec": 0.0,
            "end_sec": 1.0,
            "speaker": null,
            "text": "hello",
            "confidence": 0.9,
        });
        let evt_json = json!({
            "run_id": "skip-dup-1",
            "seq": 1,
            "ts_rfc3339": "2026-01-01T00:00:01Z",
            "stage": "backend",
            "code": "backend.ok",
            "message": "done",
            "payload_json": "{}",
        });

        write_jsonl_snapshot(
            &export_dir,
            std::slice::from_ref(&run_json),
            std::slice::from_ref(&seg_json),
            std::slice::from_ref(&evt_json),
        );

        // First import.
        let r1 = import(&target_db, &export_dir, &state_root, ConflictPolicy::Reject)
            .expect("first import");
        assert_eq!(r1.runs_imported, 1);
        assert_eq!(r1.segments_imported, 1);
        assert_eq!(r1.events_imported, 1);
        assert!(r1.conflicts.is_empty());

        // Second import of identical data.
        let r2 = import(&target_db, &export_dir, &state_root, ConflictPolicy::Reject)
            .expect("second import of identical data should succeed");
        assert_eq!(r2.runs_imported, 1);
        assert_eq!(r2.segments_imported, 1);
        assert_eq!(r2.events_imported, 1);
        assert!(
            r2.conflicts.is_empty(),
            "identical re-import should produce zero conflicts (skip semantics)"
        );

        // Verify DB still has exactly 1 run, 1 segment, 1 event.
        let conn = Connection::open(target_db.display().to_string()).expect("open");
        let rows = conn.query("SELECT count(*) FROM runs").expect("query");
        let count = match rows[0].get(0) {
            Some(SqliteValue::Integer(n)) => *n,
            _ => panic!("expected integer"),
        };
        assert_eq!(count, 1, "should still have exactly 1 run after 2 imports");
    }

    #[test]
    fn checksum_validation_catches_post_export_corruption() {
        // Export from a real DB, then modify a JSONL file, then attempt
        // import. The checksum validation should catch the corruption.
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("source.sqlite3");
        let export_dir = dir.path().join("export");
        let state_root = dir.path().join("state");
        let target_db = dir.path().join("target.sqlite3");

        let store = RunStore::open(&db_path).expect("store");
        store
            .persist_report(&fixture_report("corrupt-check", &db_path))
            .expect("persist");
        export(&db_path, &export_dir, &state_root).expect("export");

        // Corrupt segments.jsonl by appending extra data (simulating file damage).
        let seg_path = export_dir.join("segments.jsonl");
        let mut seg_file = OpenOptions::new()
            .append(true)
            .open(&seg_path)
            .expect("open segments for append");
        seg_file
            .write_all(b"CORRUPTED DATA APPENDED\n")
            .expect("append corruption");
        drop(seg_file);

        // Do NOT update manifest -- the whole point is the checksum won't match.
        let err = import(&target_db, &export_dir, &state_root, ConflictPolicy::Reject)
            .expect_err("corrupted file should fail checksum validation");
        let text = err.to_string();
        assert!(
            text.contains("checksum mismatch") && text.contains("segments.jsonl"),
            "error should mention segments.jsonl checksum mismatch: {text}"
        );
    }

    #[test]
    fn export_creates_manifest_with_correct_checksums() {
        // Verify the manifest file exists, is valid JSON, and its checksums
        // match the actual SHA-256 of the exported JSONL files.
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("manifest_check.sqlite3");
        let export_dir = dir.path().join("export");
        let state_root = dir.path().join("state");

        let store = RunStore::open(&db_path).expect("store");
        store
            .persist_report(&fixture_report("manifest-chk", &db_path))
            .expect("persist");
        let manifest = export(&db_path, &export_dir, &state_root).expect("export");

        // Verify manifest.json exists on disk.
        let manifest_path = export_dir.join("manifest.json");
        assert!(manifest_path.exists(), "manifest.json should exist on disk");

        // Parse it back from disk and compare.
        let disk_manifest: SyncManifest =
            serde_json::from_str(&fs::read_to_string(&manifest_path).expect("read manifest"))
                .expect("parse manifest from disk");

        assert_eq!(disk_manifest.schema_version, manifest.schema_version);
        assert_eq!(disk_manifest.row_counts, manifest.row_counts);
        assert_eq!(disk_manifest.checksums, manifest.checksums);

        // Independently compute checksums and verify they match.
        let runs_hash = sha256_file(&export_dir.join("runs.jsonl")).expect("hash runs");
        let segs_hash = sha256_file(&export_dir.join("segments.jsonl")).expect("hash segments");
        let evts_hash = sha256_file(&export_dir.join("events.jsonl")).expect("hash events");

        assert_eq!(manifest.checksums.runs_jsonl_sha256, runs_hash);
        assert_eq!(manifest.checksums.segments_jsonl_sha256, segs_hash);
        assert_eq!(manifest.checksums.events_jsonl_sha256, evts_hash);
    }

    #[test]
    fn import_validates_manifest_checksums_before_modifying_db() {
        // When checksums don't match, the DB should remain completely untouched.
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("source.sqlite3");
        let export_dir = dir.path().join("export");
        let state_root = dir.path().join("state");
        let target_db = dir.path().join("target.sqlite3");

        let store = RunStore::open(&db_path).expect("store");
        store
            .persist_report(&fixture_report("validate-first", &db_path))
            .expect("persist");
        export(&db_path, &export_dir, &state_root).expect("export");

        // Corrupt events.jsonl.
        let evt_path = export_dir.join("events.jsonl");
        fs::write(&evt_path, "this is not valid jsonl content\n").expect("corrupt events");

        // Attempt import -- should fail at checksum validation, before any DB writes.
        let err = import(&target_db, &export_dir, &state_root, ConflictPolicy::Reject)
            .expect_err("checksum mismatch should prevent import");
        assert!(
            err.to_string().contains("checksum mismatch"),
            "should fail on checksum: {}",
            err
        );

        // Target DB should either not exist or have zero rows.
        if target_db.exists() {
            let conn = Connection::open(target_db.display().to_string()).expect("open");
            // The DB might have been created by the import attempt but schema check
            // happens after validation, so tables might not exist. That's fine.
            let tables = conn
                .query("SELECT count(*) FROM sqlite_master WHERE type='table' AND name='runs'")
                .unwrap_or_default();
            let table_count = tables
                .first()
                .and_then(|r| match r.get(0) {
                    Some(SqliteValue::Integer(n)) => Some(*n),
                    _ => None,
                })
                .unwrap_or(0);
            if table_count > 0 {
                let rows = conn.query("SELECT count(*) FROM runs").expect("query runs");
                let run_count = match rows[0].get(0) {
                    Some(SqliteValue::Integer(n)) => *n,
                    _ => 0,
                };
                assert_eq!(
                    run_count, 0,
                    "DB should have zero runs after failed checksum validation"
                );
            }
        }
    }

    #[test]
    fn export_import_very_long_transcript() {
        // Test with a transcript exceeding 10KB to verify no truncation
        // or buffer issues in the JSONL export/import pipeline.
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("long_transcript.sqlite3");
        let export_dir = dir.path().join("export");
        let state_root = dir.path().join("state");
        let target_db = dir.path().join("target.sqlite3");

        let store = RunStore::open(&db_path).expect("store");
        let mut report = fixture_report("long-text-1", &db_path);

        // Create a >10KB transcript.
        let long_text = "The quick brown fox jumps over the lazy dog. ".repeat(300);
        assert!(
            long_text.len() > 10_000,
            "transcript should be >10KB, got {} bytes",
            long_text.len()
        );
        report.result.transcript = long_text.clone();
        report.result.segments[0].text = long_text[..5000].to_owned();
        report.result.segments[1].text = long_text[5000..].to_owned();
        store.persist_report(&report).expect("persist");

        let manifest = export(&db_path, &export_dir, &state_root).expect("export");
        assert_eq!(manifest.row_counts.runs, 1);
        assert_eq!(manifest.row_counts.segments, 2);

        let result = import(&target_db, &export_dir, &state_root, ConflictPolicy::Reject)
            .expect("import long transcript");
        assert_eq!(result.runs_imported, 1);
        assert!(result.conflicts.is_empty());

        // Verify full text survived the round trip.
        let target_store = RunStore::open(&target_db).expect("target store");
        let details = target_store
            .load_run_details("long-text-1")
            .expect("load")
            .expect("exists");
        assert_eq!(
            details.transcript.len(),
            long_text.len(),
            "transcript length should be preserved"
        );
        assert_eq!(details.transcript, long_text);
    }

    #[test]
    fn export_import_unicode_in_all_fields() {
        // Verify Unicode survives in segments (speaker, text), events (message,
        // stage, code), and run-level fields (transcript, input_path).
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("full_unicode.sqlite3");
        let export_dir = dir.path().join("export");
        let state_root = dir.path().join("state");
        let target_db = dir.path().join("target.sqlite3");

        let store = RunStore::open(&db_path).expect("store");
        let mut report = fixture_report("uni-all-1", &db_path);

        // Unicode in transcript
        report.result.transcript =
            "\u{1F3B5} \u{65E5}\u{672C}\u{8A9E} \u{D55C}\u{AD6D}\u{C5B4} \u{0410}\u{0411}\u{0412} caf\u{00E9}"
                .to_owned();

        // Unicode in segment text and speaker
        report.result.segments = vec![
            TranscriptionSegment {
                start_sec: Some(0.0),
                end_sec: Some(1.5),
                text: "\u{4F60}\u{597D}\u{4E16}\u{754C}".to_owned(), // Chinese: "Hello World"
                speaker: Some("\u{8A71}\u{8005}_00".to_owned()),     // Japanese-style speaker label
                confidence: Some(0.92),
            },
            TranscriptionSegment {
                start_sec: Some(1.5),
                end_sec: Some(3.0),
                text: "\u{041F}\u{0440}\u{0438}\u{0432}\u{0435}\u{0442} \u{043C}\u{0438}\u{0440}"
                    .to_owned(), // Russian: "Hello world"
                speaker: Some(
                    "\u{0413}\u{043E}\u{0432}\u{043E}\u{0440}\u{044F}\u{0449}\u{0438}\u{0439}_01"
                        .to_owned(),
                ),
                confidence: None,
            },
        ];

        // Unicode in events
        report.events = vec![RunEvent {
            seq: 1,
            ts_rfc3339: "2026-01-01T00:00:01Z".to_owned(),
            stage: "einlesen".to_owned(),                  // German
            code: "\u{00E9}v\u{00E9}nement.ok".to_owned(), // French: "evenement.ok"
            message: "\u{6210}\u{529F}".to_owned(),        // Chinese: "success"
            payload: json!({"\u{30C7}\u{30FC}\u{30BF}": "\u{5024}"}), // Japanese keys/values
        }];

        store
            .persist_report(&report)
            .expect("persist unicode report");

        let manifest = export(&db_path, &export_dir, &state_root).expect("export unicode");
        assert_eq!(manifest.row_counts.runs, 1);
        assert_eq!(manifest.row_counts.segments, 2);
        assert_eq!(manifest.row_counts.events, 1);

        let result = import(&target_db, &export_dir, &state_root, ConflictPolicy::Reject)
            .expect("import unicode");
        assert_eq!(result.runs_imported, 1);
        assert_eq!(result.segments_imported, 2);
        assert_eq!(result.events_imported, 1);
        assert!(result.conflicts.is_empty());

        // Verify round-tripped data.
        let target_store = RunStore::open(&target_db).expect("target store");
        let details = target_store
            .load_run_details("uni-all-1")
            .expect("load")
            .expect("exists");
        assert!(details.transcript.contains('\u{1F3B5}'));
        assert!(details.transcript.contains("caf\u{00E9}"));
        assert_eq!(details.segments.len(), 2);
        assert!(details.segments[0].text.contains('\u{4F60}'));
        assert_eq!(
            details.segments[0].speaker.as_deref(),
            Some("\u{8A71}\u{8005}_00")
        );

        // Verify event data via direct SQL query.
        let conn = Connection::open(target_db.display().to_string()).expect("open");
        let rows = conn
            .query_with_params(
                "SELECT code, message FROM events WHERE run_id = ?1 AND seq = 1",
                &[SqliteValue::Text("uni-all-1".to_owned())],
            )
            .expect("query events");
        let code = value_to_string_sqlite(rows[0].get(0));
        let message = value_to_string_sqlite(rows[0].get(1));
        assert!(
            code.contains('\u{00E9}'),
            "event code should contain accented char"
        );
        assert_eq!(message, "\u{6210}\u{529F}");
    }

    #[test]
    fn export_import_empty_segments_list() {
        // A run with zero segments should export and import cleanly.
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("no_segments.sqlite3");
        let export_dir = dir.path().join("export");
        let state_root = dir.path().join("state");
        let target_db = dir.path().join("target.sqlite3");

        let store = RunStore::open(&db_path).expect("store");
        let mut report = fixture_report("no-segs-1", &db_path);
        report.result.segments = vec![]; // Empty segments
        store.persist_report(&report).expect("persist");

        let manifest = export(&db_path, &export_dir, &state_root).expect("export");
        assert_eq!(manifest.row_counts.runs, 1);
        assert_eq!(manifest.row_counts.segments, 0);
        assert_eq!(manifest.row_counts.events, 2); // fixture_report has 2 events

        // Verify segments.jsonl is empty.
        let seg_content = fs::read_to_string(export_dir.join("segments.jsonl")).expect("read");
        assert!(
            seg_content.trim().is_empty(),
            "segments.jsonl should be empty for a run with no segments"
        );

        let result =
            import(&target_db, &export_dir, &state_root, ConflictPolicy::Reject).expect("import");
        assert_eq!(result.runs_imported, 1);
        assert_eq!(result.segments_imported, 0);
        assert_eq!(result.events_imported, 2);
        assert!(result.conflicts.is_empty());

        // Verify target DB has the run but zero segments.
        let target_store = RunStore::open(&target_db).expect("target store");
        let details = target_store
            .load_run_details("no-segs-1")
            .expect("load")
            .expect("exists");
        assert!(
            details.segments.is_empty(),
            "imported run should have zero segments"
        );
    }

    #[test]
    fn export_import_many_events() {
        // Verify that 100+ events for a single run export and import correctly.
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("many_events.sqlite3");
        let export_dir = dir.path().join("export");
        let state_root = dir.path().join("state");
        let target_db = dir.path().join("target.sqlite3");

        let store = RunStore::open(&db_path).expect("store");
        let mut report = fixture_report("many-evt-1", &db_path);

        // Generate 150 events.
        report.events = (1..=150)
            .map(|seq| RunEvent {
                seq,
                ts_rfc3339: format!("2026-01-01T00:00:{:02}Z", seq % 60),
                stage: format!("stage_{}", seq % 5),
                code: format!("code_{seq}.ok"),
                message: format!("event number {seq}"),
                payload: json!({"seq": seq, "data": "x".repeat(seq as usize % 50)}),
            })
            .collect();

        store.persist_report(&report).expect("persist");

        let manifest = export(&db_path, &export_dir, &state_root).expect("export");
        assert_eq!(manifest.row_counts.runs, 1);
        assert_eq!(manifest.row_counts.events, 150);

        // Verify events.jsonl has 150 lines.
        let events_content =
            fs::read_to_string(export_dir.join("events.jsonl")).expect("read events");
        let event_lines: Vec<&str> = events_content
            .lines()
            .filter(|l| !l.trim().is_empty())
            .collect();
        assert_eq!(
            event_lines.len(),
            150,
            "should have 150 event lines in JSONL"
        );

        let result = import(&target_db, &export_dir, &state_root, ConflictPolicy::Reject)
            .expect("import many events");
        assert_eq!(result.events_imported, 150);
        assert!(result.conflicts.is_empty());

        // Spot-check a few events via direct SQL.
        let conn = Connection::open(target_db.display().to_string()).expect("open");
        let rows = conn
            .query_with_params(
                "SELECT count(*) FROM events WHERE run_id = ?1",
                &[SqliteValue::Text("many-evt-1".to_owned())],
            )
            .expect("query");
        let count = match rows[0].get(0) {
            Some(SqliteValue::Integer(n)) => *n,
            _ => panic!("expected integer count"),
        };
        assert_eq!(count, 150, "DB should contain 150 events for this run");

        // Verify first and last events.
        let first = conn
            .query_with_params(
                "SELECT message FROM events WHERE run_id = ?1 AND seq = 1",
                &[SqliteValue::Text("many-evt-1".to_owned())],
            )
            .expect("query first");
        assert_eq!(value_to_string_sqlite(first[0].get(0)), "event number 1");

        let last = conn
            .query_with_params(
                "SELECT message FROM events WHERE run_id = ?1 AND seq = 150",
                &[SqliteValue::Text("many-evt-1".to_owned())],
            )
            .expect("query last");
        assert_eq!(value_to_string_sqlite(last[0].get(0)), "event number 150");
    }

    #[test]
    fn checksum_corruption_in_events_file_detected() {
        // Export, then modify only events.jsonl (leaving runs and segments intact).
        // The checksum validation should catch this specific file's corruption.
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("source.sqlite3");
        let export_dir = dir.path().join("export");
        let state_root = dir.path().join("state");
        let target_db = dir.path().join("target.sqlite3");

        let store = RunStore::open(&db_path).expect("store");
        store
            .persist_report(&fixture_report("evt-corrupt", &db_path))
            .expect("persist");
        export(&db_path, &export_dir, &state_root).expect("export");

        // Modify events.jsonl slightly (change a single character).
        let evt_path = export_dir.join("events.jsonl");
        let original = fs::read_to_string(&evt_path).expect("read");
        let modified = original.replacen("ingest", "Ingest", 1); // Capitalize one word
        assert_ne!(original, modified, "modification should change content");
        fs::write(&evt_path, &modified).expect("write modified events");

        let err = import(&target_db, &export_dir, &state_root, ConflictPolicy::Reject)
            .expect_err("modified events should fail checksum");
        let text = err.to_string();
        assert!(
            text.contains("checksum mismatch") && text.contains("events.jsonl"),
            "error should specifically mention events.jsonl: {text}"
        );
    }

    #[test]
    fn export_import_run_with_no_events() {
        // A run with zero events should export and import cleanly.
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("no_events.sqlite3");
        let export_dir = dir.path().join("export");
        let state_root = dir.path().join("state");
        let target_db = dir.path().join("target.sqlite3");

        let store = RunStore::open(&db_path).expect("store");
        let mut report = fixture_report("no-evt-1", &db_path);
        report.events = vec![]; // Empty events
        store.persist_report(&report).expect("persist");

        let manifest = export(&db_path, &export_dir, &state_root).expect("export");
        assert_eq!(manifest.row_counts.runs, 1);
        assert_eq!(manifest.row_counts.segments, 2);
        assert_eq!(manifest.row_counts.events, 0);

        let result =
            import(&target_db, &export_dir, &state_root, ConflictPolicy::Reject).expect("import");
        assert_eq!(result.runs_imported, 1);
        assert_eq!(result.segments_imported, 2);
        assert_eq!(result.events_imported, 0);
        assert!(result.conflicts.is_empty());
    }

    #[test]
    fn export_import_run_with_no_segments_and_no_events() {
        // A completely bare run (no segments, no events) round-trips.
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("bare_run.sqlite3");
        let export_dir = dir.path().join("export");
        let state_root = dir.path().join("state");
        let target_db = dir.path().join("target.sqlite3");

        let store = RunStore::open(&db_path).expect("store");
        let mut report = fixture_report("bare-1", &db_path);
        report.result.segments = vec![];
        report.events = vec![];
        store.persist_report(&report).expect("persist");

        let manifest = export(&db_path, &export_dir, &state_root).expect("export");
        assert_eq!(manifest.row_counts.runs, 1);
        assert_eq!(manifest.row_counts.segments, 0);
        assert_eq!(manifest.row_counts.events, 0);

        let result = import(&target_db, &export_dir, &state_root, ConflictPolicy::Reject)
            .expect("import bare run");
        assert_eq!(result.runs_imported, 1);
        assert_eq!(result.segments_imported, 0);
        assert_eq!(result.events_imported, 0);
        assert!(result.conflicts.is_empty());

        let target_store = RunStore::open(&target_db).expect("target store");
        let details = target_store
            .load_run_details("bare-1")
            .expect("load")
            .expect("exists");
        assert!(details.segments.is_empty());
    }

    #[test]
    fn sync_lock_concurrent_acquire_error_preserves_first_lock() {
        // Acquiring a second lock should fail, and the first lock should
        // remain intact and releasable.
        let dir = tempdir().expect("tempdir");
        let state_root = dir.path().join("state");

        let lock1 = SyncLock::acquire(&state_root, "export").expect("first lock");
        let lock_path = state_root.join("locks/sync.lock");
        assert!(lock_path.exists(), "lock file should exist");

        // Read the lock info to verify it belongs to the first operation.
        let info: LockInfo =
            serde_json::from_str(&fs::read_to_string(&lock_path).expect("read lock"))
                .expect("parse lock");
        assert_eq!(info.operation, "export");

        // Second acquire should fail.
        let err = SyncLock::acquire(&state_root, "import");
        assert!(err.is_err(), "second acquire should fail");

        // First lock's file should still be there and unchanged.
        assert!(
            lock_path.exists(),
            "first lock file should still exist after failed second acquire"
        );
        let info_after: LockInfo =
            serde_json::from_str(&fs::read_to_string(&lock_path).expect("read lock again"))
                .expect("parse lock again");
        assert_eq!(
            info_after.operation, "export",
            "lock file should still reflect first operation"
        );
        assert_eq!(info_after.pid, info.pid);

        // Release the first lock.
        lock1.release().expect("release first lock");
        assert!(
            !lock_path.exists(),
            "lock file should be removed after release"
        );
    }

    #[test]
    fn export_import_many_segments() {
        // Verify that 50+ segments for a single run export and import correctly.
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("many_segments.sqlite3");
        let export_dir = dir.path().join("export");
        let state_root = dir.path().join("state");
        let target_db = dir.path().join("target.sqlite3");

        let store = RunStore::open(&db_path).expect("store");
        let mut report = fixture_report("many-seg-1", &db_path);

        // Generate 60 segments.
        report.result.segments = (0..60)
            .map(|i| TranscriptionSegment {
                start_sec: Some(i as f64 * 2.0),
                end_sec: Some((i as f64 * 2.0) + 1.9),
                text: format!("segment number {i} with some text content"),
                speaker: if i % 3 == 0 {
                    Some(format!("SPEAKER_{:02}", i % 5))
                } else {
                    None
                },
                confidence: if i % 2 == 0 {
                    Some(0.8 + (i as f64 * 0.001))
                } else {
                    None
                },
            })
            .collect();

        store.persist_report(&report).expect("persist");

        let manifest = export(&db_path, &export_dir, &state_root).expect("export");
        assert_eq!(manifest.row_counts.segments, 60);

        let result = import(&target_db, &export_dir, &state_root, ConflictPolicy::Reject)
            .expect("import many segments");
        assert_eq!(result.segments_imported, 60);
        assert!(result.conflicts.is_empty());

        // Verify count in target DB.
        let conn = Connection::open(target_db.display().to_string()).expect("open");
        let rows = conn
            .query_with_params(
                "SELECT count(*) FROM segments WHERE run_id = ?1",
                &[SqliteValue::Text("many-seg-1".to_owned())],
            )
            .expect("query");
        let count = match rows[0].get(0) {
            Some(SqliteValue::Integer(n)) => *n,
            _ => panic!("expected integer count"),
        };
        assert_eq!(count, 60, "DB should contain 60 segments");

        // Spot-check last segment.
        let last = conn
            .query_with_params(
                "SELECT text FROM segments WHERE run_id = ?1 AND idx = 59",
                &[SqliteValue::Text("many-seg-1".to_owned())],
            )
            .expect("query last segment");
        assert_eq!(
            value_to_string_sqlite(last[0].get(0)),
            "segment number 59 with some text content"
        );
    }

    #[test]
    fn overwrite_policy_on_identical_data_is_still_noop() {
        // Overwrite policy with identical data should succeed without errors
        // (identical data = noop, just like Reject).
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("source.sqlite3");
        let export_dir = dir.path().join("export");
        let state_root = dir.path().join("state");
        let target_db = dir.path().join("target.sqlite3");

        let store = RunStore::open(&db_path).expect("store");
        store
            .persist_report(&fixture_report("ow-noop-1", &db_path))
            .expect("persist");
        export(&db_path, &export_dir, &state_root).expect("export");

        // First import.
        import(
            &target_db,
            &export_dir,
            &state_root,
            ConflictPolicy::Overwrite,
        )
        .expect("first import");

        // Second import with Overwrite -- identical data should be a noop.
        let result = import(
            &target_db,
            &export_dir,
            &state_root,
            ConflictPolicy::Overwrite,
        )
        .expect("second import should succeed");
        assert_eq!(result.runs_imported, 1);
        assert!(
            result.conflicts.is_empty(),
            "identical data under Overwrite should produce no conflicts"
        );

        // DB should still have exactly 1 run.
        let conn = Connection::open(target_db.display().to_string()).expect("open");
        let rows = conn.query("SELECT count(*) FROM runs").expect("query");
        let count = match rows[0].get(0) {
            Some(SqliteValue::Integer(n)) => *n,
            _ => 0,
        };
        assert_eq!(count, 1, "should have exactly 1 run");
    }

    #[test]
    fn checksum_corruption_in_runs_file_detected() {
        // Corrupt only the runs.jsonl file; segments and events remain valid.
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("source.sqlite3");
        let export_dir = dir.path().join("export");
        let state_root = dir.path().join("state");
        let target_db = dir.path().join("target.sqlite3");

        let store = RunStore::open(&db_path).expect("store");
        store
            .persist_report(&fixture_report("runs-corrupt", &db_path))
            .expect("persist");
        export(&db_path, &export_dir, &state_root).expect("export");

        // Append a byte to runs.jsonl to invalidate its checksum.
        let runs_path = export_dir.join("runs.jsonl");
        let mut f = OpenOptions::new()
            .append(true)
            .open(&runs_path)
            .expect("open");
        f.write_all(b" ").expect("append space");
        drop(f);

        let err = import(&target_db, &export_dir, &state_root, ConflictPolicy::Reject)
            .expect_err("corrupted runs.jsonl should fail checksum");
        let text = err.to_string();
        assert!(
            text.contains("checksum mismatch") && text.contains("runs.jsonl"),
            "error should mention runs.jsonl: {text}"
        );
    }

    #[test]
    fn export_import_transcript_with_embedded_json() {
        // Transcript containing valid JSON strings, braces, brackets -- should
        // survive JSONL encoding without corruption.
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("json_transcript.sqlite3");
        let export_dir = dir.path().join("export");
        let state_root = dir.path().join("state");
        let target_db = dir.path().join("target.sqlite3");

        let store = RunStore::open(&db_path).expect("store");
        let mut report = fixture_report("json-in-text", &db_path);
        report.result.transcript =
            r#"The output was {"key": "value", "array": [1,2,3]} and that's it."#.to_owned();
        report.result.segments[0].text = r#"{"nested": {"deep": true}}"#.to_owned();
        store.persist_report(&report).expect("persist");

        export(&db_path, &export_dir, &state_root).expect("export");
        let result =
            import(&target_db, &export_dir, &state_root, ConflictPolicy::Reject).expect("import");
        assert!(result.conflicts.is_empty());

        let target_store = RunStore::open(&target_db).expect("target store");
        let details = target_store
            .load_run_details("json-in-text")
            .expect("load")
            .expect("exists");
        assert!(details.transcript.contains(r#""key": "value""#));
        assert!(details.segments[0].text.contains(r#""deep": true"#));
    }

    #[test]
    fn import_creates_parent_directories_for_target_db() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("source.sqlite3");
        let export_dir = dir.path().join("export");
        let state_root = dir.path().join("state");

        let store = RunStore::open(&db_path).expect("store");
        store
            .persist_report(&fixture_report("parent-dir", &db_path))
            .expect("persist");
        export(&db_path, &export_dir, &state_root).expect("export");

        // Target DB in nested non-existent directory.
        let target_db = dir
            .path()
            .join("a")
            .join("b")
            .join("c")
            .join("target.sqlite3");
        assert!(!target_db.parent().unwrap().exists());

        let result =
            import(&target_db, &export_dir, &state_root, ConflictPolicy::Reject).expect("import");
        assert_eq!(result.runs_imported, 1);
        assert!(target_db.exists(), "target DB should have been created");
    }

    #[test]
    fn exported_runs_jsonl_ordered_ascending_by_started_at() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("order.sqlite3");
        let export_dir = dir.path().join("export");
        let state_root = dir.path().join("state");

        let store = RunStore::open(&db_path).expect("store");
        // Insert runs in non-chronological order.
        for (id, ts) in [
            ("run-c", "2026-03-01T00:00:00Z"),
            ("run-a", "2026-01-01T00:00:00Z"),
            ("run-b", "2026-02-01T00:00:00Z"),
        ] {
            let mut report = fixture_report(id, &db_path);
            report.started_at_rfc3339 = ts.to_owned();
            store.persist_report(&report).expect("persist");
        }

        export(&db_path, &export_dir, &state_root).expect("export");

        let runs_text = fs::read_to_string(export_dir.join("runs.jsonl")).expect("runs.jsonl");
        let ids: Vec<String> = runs_text
            .lines()
            .filter(|l| !l.is_empty())
            .map(|l| {
                let v: serde_json::Value = serde_json::from_str(l).expect("parse");
                v["id"].as_str().unwrap().to_owned()
            })
            .collect();
        // ORDER BY started_at ASC: a (Jan) < b (Feb) < c (Mar).
        assert_eq!(ids, vec!["run-a", "run-b", "run-c"]);
    }

    #[test]
    fn json_to_optional_f64_converts_integer_to_f64() {
        // JSON integer values should be converted to f64 via as_f64().
        let value = json!({"count": 42, "score": 2.5, "label": "text"});
        assert_eq!(
            json_to_optional_f64(&value, "count"),
            Some(42.0),
            "integer JSON value should convert to f64"
        );
        assert_eq!(
            json_to_optional_f64(&value, "score"),
            Some(2.5),
            "float JSON value should pass through"
        );
        assert_eq!(
            json_to_optional_f64(&value, "label"),
            None,
            "string JSON value should be None"
        );
    }

    #[test]
    fn atomic_write_bytes_overwrites_existing_file() {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("overwrite_me.txt");

        atomic_write_bytes(&path, b"original content").expect("first write");
        assert_eq!(fs::read_to_string(&path).expect("read"), "original content");

        atomic_write_bytes(&path, b"new content").expect("overwrite");
        assert_eq!(fs::read_to_string(&path).expect("read"), "new content");
    }

    #[test]
    fn value_to_json_float_nan_serializes_as_null() {
        // serde_json::json!(f64::NAN) produces null.
        let result = value_to_json(Some(&SqliteValue::Float(f64::NAN)));
        assert!(
            result.is_null(),
            "NaN float should serialize as null via json!, got: {result}"
        );
    }

    #[test]
    fn json_optional_float_with_integer_value_returns_float() {
        // json!(42) is a Number whose as_f64() returns Some(42.0).
        // Existing tests only use float values (42.5); this tests the integer path.
        let value = json!({"count": 42, "neg": -7});
        match json_optional_float(&value, "count") {
            SqliteValue::Float(f) => assert!((f - 42.0).abs() < f64::EPSILON),
            other => panic!("expected Float(42.0), got {other:?}"),
        }
        match json_optional_float(&value, "neg") {
            SqliteValue::Float(f) => assert!((f - (-7.0)).abs() < f64::EPSILON),
            other => panic!("expected Float(-7.0), got {other:?}"),
        }
    }

    #[test]
    fn exported_segments_jsonl_ordered_by_run_id_and_idx() {
        // export_table_segments uses ORDER BY run_id ASC, idx ASC.
        // Persist two runs with segments, then verify export order.
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("order_seg.sqlite3");
        let store = RunStore::open(&db_path).expect("open");

        // Persist run "bbb" first, then "aaa" — export should reorder by run_id ASC.
        let mut report_b = fixture_report("bbb-run", &db_path);
        report_b.result.segments = vec![
            TranscriptionSegment {
                start_sec: Some(0.0),
                end_sec: Some(1.0),
                text: "seg-b-0".to_owned(),
                speaker: None,
                confidence: None,
            },
            TranscriptionSegment {
                start_sec: Some(1.0),
                end_sec: Some(2.0),
                text: "seg-b-1".to_owned(),
                speaker: None,
                confidence: None,
            },
        ];
        store.persist_report(&report_b).expect("persist b");

        let mut report_a = fixture_report("aaa-run", &db_path);
        report_a.result.segments = vec![TranscriptionSegment {
            start_sec: Some(0.0),
            end_sec: Some(1.0),
            text: "seg-a-0".to_owned(),
            speaker: None,
            confidence: None,
        }];
        store.persist_report(&report_a).expect("persist a");

        let export_dir = dir.path().join("export");
        let seg_path = export_dir.join("segments.jsonl");
        let manifest = export_inner(&db_path, &export_dir).expect("export");
        assert_eq!(manifest.row_counts.segments, 3);

        let lines: Vec<serde_json::Value> = fs::read_to_string(&seg_path)
            .expect("read")
            .lines()
            .filter(|l| !l.trim().is_empty())
            .map(|l| serde_json::from_str(l).expect("parse"))
            .collect();
        assert_eq!(lines.len(), 3);
        // First should be aaa-run (ASC by run_id)
        assert_eq!(lines[0]["run_id"], "aaa-run");
        assert_eq!(lines[0]["text"], "seg-a-0");
        // Then bbb-run idx=0, idx=1
        assert_eq!(lines[1]["run_id"], "bbb-run");
        assert_eq!(lines[1]["text"], "seg-b-0");
        assert_eq!(lines[2]["run_id"], "bbb-run");
        assert_eq!(lines[2]["text"], "seg-b-1");
    }

    #[test]
    fn exported_events_jsonl_ordered_by_run_id_and_seq() {
        // export_table_events uses ORDER BY run_id ASC, seq ASC.
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("order_evt.sqlite3");
        let store = RunStore::open(&db_path).expect("open");

        // Persist "zzz" first, "aaa" second — export should reorder by run_id ASC.
        let report_z = fixture_report("zzz-run", &db_path);
        store.persist_report(&report_z).expect("persist z");

        let report_a = fixture_report("aaa-run", &db_path);
        store.persist_report(&report_a).expect("persist a");

        let export_dir = dir.path().join("export");
        let evt_path = export_dir.join("events.jsonl");
        let _manifest = export_inner(&db_path, &export_dir).expect("export");

        let lines: Vec<serde_json::Value> = fs::read_to_string(&evt_path)
            .expect("read")
            .lines()
            .filter(|l| !l.trim().is_empty())
            .map(|l| serde_json::from_str(l).expect("parse"))
            .collect();
        // Events from aaa-run should come before zzz-run events.
        let first_run_id = lines[0]["run_id"].as_str().expect("run_id");
        let last_run_id = lines.last().expect("last")["run_id"]
            .as_str()
            .expect("run_id");
        assert_eq!(first_run_id, "aaa-run");
        assert_eq!(last_run_id, "zzz-run");
        // Within each run, seq should be ascending.
        let aaa_seqs: Vec<u64> = lines
            .iter()
            .filter(|l| l["run_id"] == "aaa-run")
            .map(|l| l["seq"].as_u64().expect("seq"))
            .collect();
        for window in aaa_seqs.windows(2) {
            assert!(window[0] <= window[1], "seqs should be ascending");
        }
    }

    #[test]
    fn import_segments_missing_text_field_returns_error() {
        // import_segments calls json_str(&row, "text")? which errors on missing field.
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("missing_text.sqlite3");
        let store = RunStore::open(&db_path).expect("open");

        let report = fixture_report("run-1", &db_path);
        store.persist_report(&report).expect("persist");

        let export_dir = dir.path().join("export");
        export_inner(&db_path, &export_dir).expect("export");

        // Overwrite segments.jsonl with a row missing the "text" field.
        let seg_path = export_dir.join("segments.jsonl");
        let bad_segment = json!({
            "run_id": "run-1",
            "idx": 0,
            "start_sec": 0.0,
            "end_sec": 1.0,
            "speaker": null,
            "confidence": null
            // "text" field intentionally omitted
        });
        fs::write(
            &seg_path,
            format!("{}\n", serde_json::to_string(&bad_segment).expect("ser")),
        )
        .expect("write");

        // Update manifest checksums and row counts.
        let manifest_path = export_dir.join("manifest.json");
        let mut manifest_value: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(&manifest_path).expect("read"))
                .expect("parse");
        manifest_value["checksums"]["segments_jsonl_sha256"] =
            json!(sha256_file(&seg_path).expect("checksum"));
        manifest_value["row_counts"]["segments"] = json!(1);
        fs::write(
            &manifest_path,
            serde_json::to_string_pretty(&manifest_value).expect("ser"),
        )
        .expect("write manifest");

        // Import into a fresh DB — should fail with "missing string field `text`".
        let import_db = dir.path().join("import.sqlite3");
        let err = import_inner(&import_db, &export_dir, ConflictPolicy::Reject)
            .expect_err("should fail on missing text");
        assert!(
            err.to_string().contains("missing string field") && err.to_string().contains("text"),
            "error should mention missing text field: {}",
            err
        );
    }

    #[test]
    fn import_events_missing_message_field_returns_error() {
        // import_events calls json_str(&row, "message")? which errors on missing field.
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("missing_msg.sqlite3");
        let store = RunStore::open(&db_path).expect("open");

        let report = fixture_report("run-1", &db_path);
        store.persist_report(&report).expect("persist");

        let export_dir = dir.path().join("export");
        export_inner(&db_path, &export_dir).expect("export");

        // Overwrite events.jsonl with a row missing "message".
        let evt_path = export_dir.join("events.jsonl");
        let bad_event = json!({
            "run_id": "run-1",
            "seq": 1,
            "ts_rfc3339": "2026-01-01T00:00:01Z",
            "stage": "ingest",
            "code": "ingest.ok",
            "payload_json": "{}"
            // "message" intentionally omitted
        });
        fs::write(
            &evt_path,
            format!("{}\n", serde_json::to_string(&bad_event).expect("ser")),
        )
        .expect("write");

        // Update manifest checksums and row counts.
        let manifest_path = export_dir.join("manifest.json");
        let mut manifest_value: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(&manifest_path).expect("read"))
                .expect("parse");
        manifest_value["checksums"]["events_jsonl_sha256"] =
            json!(sha256_file(&evt_path).expect("checksum"));
        manifest_value["row_counts"]["events"] = json!(1);
        fs::write(
            &manifest_path,
            serde_json::to_string_pretty(&manifest_value).expect("ser"),
        )
        .expect("write manifest");

        // Import into a fresh DB — should fail.
        let import_db = dir.path().join("import.sqlite3");
        let err = import_inner(&import_db, &export_dir, ConflictPolicy::Reject)
            .expect_err("should fail on missing message");
        assert!(
            err.to_string().contains("missing string field") && err.to_string().contains("message"),
            "error should mention missing message field: {}",
            err
        );
    }

    // -----------------------------------------------------------------------
    // bd-246.1: Incremental export tests
    // -----------------------------------------------------------------------

    #[test]
    fn incremental_export_first_run_exports_all_records() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("incr_first.sqlite3");
        let export_dir = dir.path().join("export");
        let state_root = dir.path().join("state");

        let store = RunStore::open(&db_path).expect("store open");
        let report = fixture_report("incr-1", &db_path);
        store.persist_report(&report).expect("persist");

        let manifest =
            export_incremental(&db_path, &export_dir, &state_root).expect("incremental export");

        assert_eq!(manifest.export_mode, "incremental");
        assert_eq!(manifest.row_counts.runs, 1);
        assert_eq!(manifest.row_counts.segments, 2);
        assert_eq!(manifest.row_counts.events, 2);
        assert!(
            manifest.cursor_used.is_none(),
            "first export has no prior cursor"
        );
        assert!(!manifest.cursor_after.last_export_rfc3339.is_empty());
        assert_eq!(manifest.cursor_after.last_run_count, 1);
    }

    #[test]
    fn incremental_export_second_run_exports_only_new_records() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("incr_second.sqlite3");
        let state_root = dir.path().join("state");

        let store = RunStore::open(&db_path).expect("store open");

        // First run: timestamp T1
        let mut report1 = fixture_report("incr-old", &db_path);
        report1.started_at_rfc3339 = "2026-01-01T00:00:00Z".to_owned();
        store.persist_report(&report1).expect("persist 1");

        // First incremental export (captures everything)
        let export_dir_1 = dir.path().join("export1");
        let manifest1 =
            export_incremental(&db_path, &export_dir_1, &state_root).expect("first export");
        assert_eq!(manifest1.row_counts.runs, 1);

        // Second run: timestamp T2 > T1
        let mut report2 = fixture_report("incr-new", &db_path);
        report2.started_at_rfc3339 = "2026-06-15T12:00:00Z".to_owned();
        store.persist_report(&report2).expect("persist 2");

        // Second incremental export (should only capture the new run)
        let export_dir_2 = dir.path().join("export2");
        let manifest2 =
            export_incremental(&db_path, &export_dir_2, &state_root).expect("second export");

        assert_eq!(manifest2.row_counts.runs, 1, "only the new run");
        assert_eq!(
            manifest2.row_counts.segments, 2,
            "segments for new run only"
        );
        assert_eq!(manifest2.row_counts.events, 2, "events for new run only");
        assert!(manifest2.cursor_used.is_some(), "should carry prior cursor");
        assert_eq!(
            manifest2.cursor_used.as_ref().unwrap().last_export_rfc3339,
            manifest1.cursor_after.last_export_rfc3339,
        );

        // Verify that the exported runs.jsonl contains only the new run
        let runs_content = fs::read_to_string(export_dir_2.join("runs.jsonl")).expect("read runs");
        let lines: Vec<&str> = runs_content
            .lines()
            .filter(|l| !l.trim().is_empty())
            .collect();
        assert_eq!(lines.len(), 1);
        let run_obj: serde_json::Value = serde_json::from_str(lines[0]).expect("parse run line");
        assert_eq!(run_obj["id"].as_str().unwrap(), "incr-new");
    }

    #[test]
    fn incremental_export_no_new_records_produces_zero_counts() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("incr_noop.sqlite3");
        let state_root = dir.path().join("state");

        let store = RunStore::open(&db_path).expect("store open");
        let mut report = fixture_report("incr-only", &db_path);
        report.started_at_rfc3339 = "2026-01-01T00:00:00Z".to_owned();
        store.persist_report(&report).expect("persist");

        // First export: captures everything
        let export_dir_1 = dir.path().join("export1");
        export_incremental(&db_path, &export_dir_1, &state_root).expect("first export");

        // Second export: nothing new
        let export_dir_2 = dir.path().join("export2");
        let manifest =
            export_incremental(&db_path, &export_dir_2, &state_root).expect("second export");

        assert_eq!(manifest.row_counts.runs, 0);
        assert_eq!(manifest.row_counts.segments, 0);
        assert_eq!(manifest.row_counts.events, 0);
    }

    #[test]
    fn incremental_export_empty_db_produces_zero_counts_and_creates_cursor() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("incr_empty.sqlite3");
        let export_dir = dir.path().join("export");
        let state_root = dir.path().join("state");

        let _store = RunStore::open(&db_path).expect("store open");
        let manifest = export_incremental(&db_path, &export_dir, &state_root).expect("export");

        assert_eq!(manifest.row_counts.runs, 0);
        assert_eq!(manifest.row_counts.segments, 0);
        assert_eq!(manifest.row_counts.events, 0);
        assert!(manifest.cursor_used.is_none());
        // Cursor file should exist after export.
        assert!(state_root.join(CURSOR_FILENAME).exists());
    }

    #[test]
    fn incremental_export_cursor_persists_between_calls() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("incr_cursor.sqlite3");
        let state_root = dir.path().join("state");

        let store = RunStore::open(&db_path).expect("store open");

        // Insert three runs with ascending timestamps.
        for (i, ts) in [
            "2026-01-01T00:00:00Z",
            "2026-02-01T00:00:00Z",
            "2026-03-01T00:00:00Z",
        ]
        .iter()
        .enumerate()
        {
            let mut report = fixture_report(&format!("cursor-{i}"), &db_path);
            report.started_at_rfc3339 = ts.to_string();
            store.persist_report(&report).expect("persist");
        }

        // First export: all 3 runs.
        let export_dir_1 = dir.path().join("export1");
        let m1 = export_incremental(&db_path, &export_dir_1, &state_root).expect("export 1");
        assert_eq!(m1.row_counts.runs, 3);

        // Second export: no new runs.
        let export_dir_2 = dir.path().join("export2");
        let m2 = export_incremental(&db_path, &export_dir_2, &state_root).expect("export 2");
        assert_eq!(m2.row_counts.runs, 0);
        assert_eq!(
            m2.cursor_used.as_ref().unwrap().last_export_rfc3339,
            m1.cursor_after.last_export_rfc3339,
        );

        // Insert a fourth run.
        let mut report4 = fixture_report("cursor-3", &db_path);
        report4.started_at_rfc3339 = "2026-04-01T00:00:00Z".to_owned();
        store.persist_report(&report4).expect("persist 4");

        // Third export: only the new run.
        let export_dir_3 = dir.path().join("export3");
        let m3 = export_incremental(&db_path, &export_dir_3, &state_root).expect("export 3");
        assert_eq!(m3.row_counts.runs, 1);
        assert_eq!(m3.row_counts.segments, 2);
        assert_eq!(m3.row_counts.events, 2);
    }

    #[test]
    fn incremental_export_result_can_be_imported() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("incr_import.sqlite3");
        let state_root = dir.path().join("state");

        let store = RunStore::open(&db_path).expect("store open");
        let mut report = fixture_report("incr-importable", &db_path);
        report.started_at_rfc3339 = "2026-01-01T00:00:00Z".to_owned();
        store.persist_report(&report).expect("persist");

        let export_dir = dir.path().join("export");
        let incr_manifest =
            export_incremental(&db_path, &export_dir, &state_root).expect("incremental export");

        // Write a standard SyncManifest so import() can consume it.
        let sync_manifest = SyncManifest {
            schema_version: incr_manifest.schema_version.clone(),
            export_format_version: incr_manifest.export_format_version.clone(),
            created_at_rfc3339: incr_manifest.created_at_rfc3339.clone(),
            source_db_path: incr_manifest.source_db_path.clone(),
            row_counts: incr_manifest.row_counts.clone(),
            checksums: incr_manifest.checksums.clone(),
        };
        let manifest_path = export_dir.join("manifest.json");
        fs::write(
            &manifest_path,
            serde_json::to_string_pretty(&sync_manifest).expect("ser"),
        )
        .expect("write manifest");

        let target_db = dir.path().join("target.sqlite3");
        let result = import(&target_db, &export_dir, &state_root, ConflictPolicy::Reject)
            .expect("import incremental output");
        assert_eq!(result.runs_imported, 1);
        assert_eq!(result.segments_imported, 2);
        assert_eq!(result.events_imported, 2);
        assert!(result.conflicts.is_empty());
    }

    #[test]
    fn incremental_export_manifest_has_valid_checksums() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("incr_checksum.sqlite3");
        let export_dir = dir.path().join("export");
        let state_root = dir.path().join("state");

        let store = RunStore::open(&db_path).expect("store open");
        store
            .persist_report(&fixture_report("cksum-1", &db_path))
            .expect("persist");

        let manifest = export_incremental(&db_path, &export_dir, &state_root).expect("export");

        // Re-compute checksums and compare.
        let actual_runs = sha256_file(&export_dir.join("runs.jsonl")).expect("hash");
        let actual_segments = sha256_file(&export_dir.join("segments.jsonl")).expect("hash");
        let actual_events = sha256_file(&export_dir.join("events.jsonl")).expect("hash");

        assert_eq!(manifest.checksums.runs_jsonl_sha256, actual_runs);
        assert_eq!(manifest.checksums.segments_jsonl_sha256, actual_segments);
        assert_eq!(manifest.checksums.events_jsonl_sha256, actual_events);
    }

    #[test]
    fn sync_cursor_serde_round_trip() {
        let cursor = SyncCursor {
            last_export_rfc3339: "2026-06-15T12:00:00Z".to_owned(),
            last_run_count: 42,
        };
        let json = serde_json::to_string_pretty(&cursor).expect("serialize");
        let parsed: SyncCursor = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(parsed.last_export_rfc3339, "2026-06-15T12:00:00Z");
        assert_eq!(parsed.last_run_count, 42);
    }

    #[test]
    fn incremental_export_manifest_serde_round_trip() {
        let manifest = IncrementalExportManifest {
            schema_version: "1.1".to_owned(),
            export_format_version: "1.0".to_owned(),
            export_mode: "incremental".to_owned(),
            created_at_rfc3339: "2026-01-01T00:00:00Z".to_owned(),
            source_db_path: "test.sqlite3".to_owned(),
            row_counts: RowCounts {
                runs: 1,
                segments: 2,
                events: 3,
            },
            checksums: FileChecksums {
                runs_jsonl_sha256: "aaa".to_owned(),
                segments_jsonl_sha256: "bbb".to_owned(),
                events_jsonl_sha256: "ccc".to_owned(),
            },
            cursor_used: Some(SyncCursor {
                last_export_rfc3339: "2026-01-01T00:00:00Z".to_owned(),
                last_run_count: 5,
            }),
            cursor_after: SyncCursor {
                last_export_rfc3339: "2026-06-15T12:00:00Z".to_owned(),
                last_run_count: 1,
            },
        };

        let json = serde_json::to_string_pretty(&manifest).expect("serialize");
        let parsed: IncrementalExportManifest = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(parsed.export_mode, "incremental");
        assert_eq!(parsed.row_counts.runs, 1);
        assert!(parsed.cursor_used.is_some());
        assert_eq!(parsed.cursor_after.last_run_count, 1);
    }

    #[test]
    fn load_cursor_returns_none_when_file_missing() {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("nonexistent_cursor.json");
        let result = load_cursor(&path).expect("should succeed with None");
        assert!(result.is_none());
    }

    #[test]
    fn save_and_load_cursor_round_trip() {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("cursor_rt.json");
        let cursor = SyncCursor {
            last_export_rfc3339: "2026-03-15T08:30:00Z".to_owned(),
            last_run_count: 7,
        };
        save_cursor(&path, &cursor).expect("save");
        let loaded = load_cursor(&path).expect("load").expect("should exist");
        assert_eq!(loaded.last_export_rfc3339, "2026-03-15T08:30:00Z");
        assert_eq!(loaded.last_run_count, 7);
    }

    #[test]
    fn load_cursor_rejects_corrupt_json() {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("corrupt_cursor.json");
        fs::write(&path, "{not valid json}").expect("write");
        let result = load_cursor(&path);
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("invalid sync cursor")
        );
    }

    #[test]
    fn incremental_export_multiple_runs_same_timestamp_exports_all() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("incr_same_ts.sqlite3");
        let state_root = dir.path().join("state");

        let store = RunStore::open(&db_path).expect("store open");

        // Insert two runs with the same timestamp.
        for i in 0..2 {
            let mut report = fixture_report(&format!("same-ts-{i}"), &db_path);
            report.started_at_rfc3339 = "2026-01-01T00:00:00Z".to_owned();
            store.persist_report(&report).expect("persist");
        }

        // First export: both runs.
        let export_dir = dir.path().join("export1");
        let m1 = export_incremental(&db_path, &export_dir, &state_root).expect("export");
        assert_eq!(m1.row_counts.runs, 2);

        // Second export: no new runs (cursor_after == the shared timestamp).
        let export_dir_2 = dir.path().join("export2");
        let m2 = export_incremental(&db_path, &export_dir_2, &state_root).expect("export 2");
        assert_eq!(m2.row_counts.runs, 0);
    }

    #[test]
    fn export_mode_enum_variants() {
        // Verify both variants are distinct.
        assert_ne!(ExportMode::Full, ExportMode::Incremental);
        let _ = format!("{:?}", ExportMode::Full);
        let _ = format!("{:?}", ExportMode::Incremental);
    }

    #[test]
    fn max_started_at_empty_db_returns_none() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("empty.sqlite3");
        let store = RunStore::open(&db_path).expect("open");
        // Access the raw connection via the store's public API indirectly:
        // Open a separate connection for this test.
        let conn =
            fsqlite::Connection::open(db_path.display().to_string()).expect("open connection");

        // No runs → MAX(started_at) returns NULL → empty string → None
        let result = max_started_at(&conn, None).expect("should succeed");
        assert_eq!(result, None, "empty db should return None");
        drop(store);
    }

    #[test]
    fn max_started_at_with_filter_beyond_all_timestamps_returns_none() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("ts_filter.sqlite3");
        let store = RunStore::open(&db_path).expect("open");
        let report = fixture_report("ts-run", &db_path);
        store.persist_report(&report).expect("persist");

        let conn =
            fsqlite::Connection::open(db_path.display().to_string()).expect("open connection");

        // With after_ts far in the future, no runs match → None.
        let result = max_started_at(&conn, Some("2099-12-31T23:59:59Z")).expect("should succeed");
        assert_eq!(result, None, "no runs after far-future timestamp");

        // With no filter, should return the run's timestamp.
        let result = max_started_at(&conn, None).expect("should succeed");
        assert!(result.is_some(), "should find a run with no filter");
    }

    #[test]
    fn ensure_run_reference_exists_missing_run_returns_formatted_error() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("ref_check.sqlite3");
        let store = RunStore::open(&db_path).expect("open");
        drop(store);

        let conn =
            fsqlite::Connection::open(db_path.display().to_string()).expect("open connection");
        ensure_schema(&conn).expect("schema");

        let err = ensure_run_reference_exists(&conn, "ghost-run-id", "segments", "ghost-run-id/0")
            .expect_err("should fail for missing run");
        let msg = err.to_string();
        assert!(
            msg.contains("referential integrity violation"),
            "should mention referential integrity: {msg}"
        );
        assert!(msg.contains("ghost-run-id"), "should mention run_id: {msg}");
        assert!(msg.contains("segments"), "should mention table name: {msg}");
    }

    #[test]
    fn export_table_segments_for_runs_with_empty_run_ids_produces_empty_file() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("empty_ids.sqlite3");
        let store = RunStore::open(&db_path).expect("open");
        // Persist a run with segments — but we won't export it (empty run_ids).
        let report = fixture_report("has-segments", &db_path);
        store.persist_report(&report).expect("persist");

        let conn =
            fsqlite::Connection::open(db_path.display().to_string()).expect("open connection");
        let output_path = dir.path().join("segments.jsonl");
        let count =
            export_table_segments_for_runs(&conn, &output_path, &[]).expect("should succeed");
        assert_eq!(count, 0, "empty run_ids should produce 0 rows");

        let content = fs::read_to_string(&output_path).expect("read");
        assert!(
            content.is_empty(),
            "output file should be empty but got: {content}"
        );
    }

    #[test]
    fn save_cursor_creates_nested_parent_directories() {
        let dir = tempdir().expect("tempdir");
        let nested = dir.path().join("a").join("b").join("c").join("cursor.json");
        assert!(
            !nested.parent().unwrap().exists(),
            "dirs should not exist yet"
        );

        let cursor = SyncCursor {
            last_export_rfc3339: "2026-01-01T00:00:00Z".to_owned(),
            last_run_count: 7,
        };
        save_cursor(&nested, &cursor).expect("save should create dirs");

        assert!(nested.exists(), "cursor file should exist");
        let loaded = load_cursor(&nested)
            .expect("load")
            .expect("should have content");
        assert_eq!(loaded.last_export_rfc3339, "2026-01-01T00:00:00Z");
        assert_eq!(loaded.last_run_count, 7);
    }

    // ── bd-246.2: Validation tests ──

    #[test]
    fn validate_sync_matching_db_and_jsonl() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("validate.sqlite3");
        let export_dir = dir.path().join("export");
        let state_root = dir.path().join("state");

        let store = RunStore::open(&db_path).expect("store open");
        let report = fixture_report("val-1", &db_path);
        store.persist_report(&report).expect("persist");

        export(&db_path, &export_dir, &state_root).expect("export");

        let validation = validate_sync(&db_path, &export_dir).expect("validate");
        assert!(
            validation.is_valid,
            "matching DB and JSONL should be valid: {validation:?}"
        );
        assert_eq!(validation.db_run_count, 1);
        assert_eq!(validation.jsonl_run_count, 1);
        assert_eq!(validation.db_segment_count, 2);
        assert_eq!(validation.jsonl_segment_count, 2);
        assert_eq!(validation.db_event_count, 2);
        assert_eq!(validation.jsonl_event_count, 2);
        assert!(validation.missing_from_db.is_empty());
        assert!(validation.missing_from_jsonl.is_empty());
        assert!(validation.mismatched_records.is_empty());
    }

    #[test]
    fn validate_sync_multiple_runs_matching() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("val_multi.sqlite3");
        let export_dir = dir.path().join("export");
        let state_root = dir.path().join("state");

        let store = RunStore::open(&db_path).expect("store open");
        for i in 0..3 {
            store
                .persist_report(&fixture_report(&format!("val-multi-{i}"), &db_path))
                .expect("persist");
        }
        drop(store);
        let conn = Connection::open(db_path.display().to_string()).expect("open");
        ensure_schema(&conn).expect("schema");
        drop(conn);

        export(&db_path, &export_dir, &state_root).expect("export");

        let validation = validate_sync(&db_path, &export_dir).expect("validate");
        assert!(validation.is_valid, "should be valid: {validation:?}");
        assert_eq!(validation.db_run_count, 3);
        assert_eq!(validation.jsonl_run_count, 3);
        assert_eq!(validation.db_segment_count, 6);
        assert_eq!(validation.jsonl_segment_count, 6);
        assert_eq!(validation.db_event_count, 6);
        assert_eq!(validation.jsonl_event_count, 6);
    }

    #[test]
    fn validate_sync_detects_missing_runs_from_jsonl() {
        // DB has a run that is not in the JSONL export.
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("val_missing_jsonl.sqlite3");
        let export_dir = dir.path().join("export");
        let state_root = dir.path().join("state");

        let store = RunStore::open(&db_path).expect("store open");
        store
            .persist_report(&fixture_report("run-a", &db_path))
            .expect("persist");

        // Export with only run-a
        export(&db_path, &export_dir, &state_root).expect("export");

        // Add another run to DB after export
        store
            .persist_report(&fixture_report("run-b", &db_path))
            .expect("persist");
        drop(store);
        let conn = Connection::open(db_path.display().to_string()).expect("open");
        ensure_schema(&conn).expect("schema");
        drop(conn);

        let validation = validate_sync(&db_path, &export_dir).expect("validate");
        assert!(!validation.is_valid, "should not be valid");
        assert_eq!(validation.db_run_count, 2);
        assert_eq!(validation.jsonl_run_count, 1);
        assert!(
            validation.missing_from_jsonl.contains(&"run-b".to_owned()),
            "run-b should be missing from JSONL: {:?}",
            validation.missing_from_jsonl
        );
    }

    #[test]
    fn validate_sync_detects_missing_runs_from_db() {
        // JSONL has a run that is not in the DB (e.g. DB was wiped).
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("val_missing_db.sqlite3");
        let export_dir = dir.path().join("export");
        let state_root = dir.path().join("state");

        let store = RunStore::open(&db_path).expect("store open");
        store
            .persist_report(&fixture_report("run-x", &db_path))
            .expect("persist run-x");
        store
            .persist_report(&fixture_report("run-y", &db_path))
            .expect("persist run-y");

        // Export with both runs
        export(&db_path, &export_dir, &state_root).expect("export");
        drop(store);

        // Create a fresh DB with only run-x
        let fresh_db = dir.path().join("fresh.sqlite3");
        let fresh_store = RunStore::open(&fresh_db).expect("fresh store");
        fresh_store
            .persist_report(&fixture_report("run-x", &fresh_db))
            .expect("persist run-x");
        drop(fresh_store);
        let conn = Connection::open(fresh_db.display().to_string()).expect("open");
        ensure_schema(&conn).expect("schema");
        drop(conn);

        let validation = validate_sync(&fresh_db, &export_dir).expect("validate");
        assert!(!validation.is_valid, "should not be valid");
        assert!(
            validation.missing_from_db.contains(&"run-y".to_owned()),
            "run-y should be missing from DB: {:?}",
            validation.missing_from_db
        );
    }

    #[test]
    fn validate_sync_detects_mismatched_data() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("val_mismatch.sqlite3");
        let export_dir = dir.path().join("export");
        let state_root = dir.path().join("state");

        let store = RunStore::open(&db_path).expect("store open");
        store
            .persist_report(&fixture_report("mismatch-1", &db_path))
            .expect("persist");

        export(&db_path, &export_dir, &state_root).expect("export");

        // Mutate the transcript in the runs.jsonl
        let runs_path = export_dir.join("runs.jsonl");
        let content = fs::read_to_string(&runs_path).expect("read");
        let first_line = content.lines().next().expect("has line");
        let mut mutated: serde_json::Value = serde_json::from_str(first_line).expect("valid run");
        mutated["transcript"] = json!("DIFFERENT TRANSCRIPT");
        fs::write(
            &runs_path,
            format!("{}\n", serde_json::to_string(&mutated).expect("serialize")),
        )
        .expect("write");

        let validation = validate_sync(&db_path, &export_dir).expect("validate");
        assert!(!validation.is_valid, "should not be valid");
        assert!(
            validation
                .mismatched_records
                .contains(&"mismatch-1".to_owned()),
            "mismatch-1 should be in mismatched_records: {:?}",
            validation.mismatched_records
        );
    }

    #[test]
    fn validate_sync_empty_db_and_empty_jsonl() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("val_empty.sqlite3");
        let export_dir = dir.path().join("export");
        let state_root = dir.path().join("state");

        let _store = RunStore::open(&db_path).expect("store open");
        export(&db_path, &export_dir, &state_root).expect("export");

        let validation = validate_sync(&db_path, &export_dir).expect("validate");
        assert!(validation.is_valid, "empty DB and JSONL should be valid");
        assert_eq!(validation.db_run_count, 0);
        assert_eq!(validation.jsonl_run_count, 0);
        assert_eq!(validation.db_segment_count, 0);
        assert_eq!(validation.jsonl_segment_count, 0);
        assert_eq!(validation.db_event_count, 0);
        assert_eq!(validation.jsonl_event_count, 0);
    }

    #[test]
    fn validate_sync_report_serde_round_trip() {
        let report = SyncValidationReport {
            db_run_count: 5,
            jsonl_run_count: 4,
            db_segment_count: 10,
            jsonl_segment_count: 8,
            db_event_count: 15,
            jsonl_event_count: 12,
            missing_from_jsonl: vec!["run-5".to_owned()],
            missing_from_db: vec![],
            mismatched_records: vec!["run-2".to_owned()],
            is_valid: false,
        };
        let json_text = serde_json::to_string_pretty(&report).expect("serialize");
        let deserialized: SyncValidationReport =
            serde_json::from_str(&json_text).expect("deserialize");
        assert_eq!(deserialized.db_run_count, 5);
        assert_eq!(deserialized.jsonl_run_count, 4);
        assert!(!deserialized.is_valid);
        assert_eq!(deserialized.missing_from_jsonl, vec!["run-5"]);
        assert_eq!(deserialized.mismatched_records, vec!["run-2"]);
    }

    // ── bd-246.2: Compression tests ──

    #[test]
    fn compression_round_trip() {
        let dir = tempdir().expect("tempdir");
        let original = dir.path().join("data.jsonl");
        let compressed = dir.path().join("data.jsonl.gz");
        let decompressed = dir.path().join("data_back.jsonl");

        let content = "{\"id\":\"run-1\",\"text\":\"hello world\"}\n\
                        {\"id\":\"run-2\",\"text\":\"second line\"}\n";
        fs::write(&original, content).expect("write original");

        compress_jsonl(&original, &compressed).expect("compress");
        assert!(compressed.exists(), "compressed file should exist");

        // Compressed file should be different from original (not plain text)
        let compressed_bytes = fs::read(&compressed).expect("read compressed");
        assert_ne!(
            compressed_bytes,
            content.as_bytes(),
            "compressed should differ from original"
        );

        decompress_jsonl(&compressed, &decompressed).expect("decompress");
        let recovered = fs::read_to_string(&decompressed).expect("read decompressed");
        assert_eq!(recovered, content, "round-trip should preserve content");
    }

    #[test]
    fn compression_empty_file() {
        let dir = tempdir().expect("tempdir");
        let original = dir.path().join("empty.jsonl");
        let compressed = dir.path().join("empty.jsonl.gz");
        let decompressed = dir.path().join("empty_back.jsonl");

        fs::write(&original, "").expect("write empty");
        compress_jsonl(&original, &compressed).expect("compress empty");
        decompress_jsonl(&compressed, &decompressed).expect("decompress empty");

        let recovered = fs::read_to_string(&decompressed).expect("read");
        assert_eq!(recovered, "", "empty file round-trip");
    }

    #[test]
    fn compression_large_content() {
        let dir = tempdir().expect("tempdir");
        let original = dir.path().join("large.jsonl");
        let compressed = dir.path().join("large.jsonl.gz");
        let decompressed = dir.path().join("large_back.jsonl");

        let mut content = String::new();
        for i in 0..1000 {
            content.push_str(&format!(
                "{{\"id\":\"run-{i}\",\"text\":\"line number {i} with some repeated data aaabbbccc\"}}\n"
            ));
        }
        fs::write(&original, &content).expect("write large");

        compress_jsonl(&original, &compressed).expect("compress");

        // Compressed should be smaller than original for repetitive data
        let original_size = fs::metadata(&original).expect("metadata").len();
        let compressed_size = fs::metadata(&compressed).expect("metadata").len();
        assert!(
            compressed_size < original_size,
            "compressed ({compressed_size}) should be smaller than original ({original_size})"
        );

        decompress_jsonl(&compressed, &decompressed).expect("decompress");
        let recovered = fs::read_to_string(&decompressed).expect("read");
        assert_eq!(recovered, content, "large content round-trip");
    }

    #[test]
    fn compression_mode_enum_equality() {
        assert_eq!(CompressionMode::None, CompressionMode::None);
        assert_eq!(CompressionMode::Gzip, CompressionMode::Gzip);
        assert_ne!(CompressionMode::None, CompressionMode::Gzip);
        let _ = format!("{:?}", CompressionMode::None);
        let _ = format!("{:?}", CompressionMode::Gzip);
    }

    #[test]
    fn open_jsonl_reader_plain_file() {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("data.jsonl");
        fs::write(&path, "{\"id\":\"a\"}\n{\"id\":\"b\"}\n").expect("write");

        let reader = open_jsonl_reader(&path).expect("open");
        let lines: Vec<String> = reader
            .lines()
            .collect::<Result<Vec<_>, _>>()
            .expect("read lines");
        assert_eq!(lines.len(), 2);
        assert!(lines[0].contains("\"a\""));
        assert!(lines[1].contains("\"b\""));
    }

    #[test]
    fn open_jsonl_reader_compressed_file() {
        let dir = tempdir().expect("tempdir");
        let plain = dir.path().join("data.jsonl");
        let gz = dir.path().join("data.jsonl.gz");

        let content = "{\"id\":\"c\"}\n{\"id\":\"d\"}\n";
        fs::write(&plain, content).expect("write");
        compress_jsonl(&plain, &gz).expect("compress");

        let reader = open_jsonl_reader(&gz).expect("open gz");
        let lines: Vec<String> = reader
            .lines()
            .collect::<Result<Vec<_>, _>>()
            .expect("read lines");
        assert_eq!(lines.len(), 2);
        assert!(lines[0].contains("\"c\""));
        assert!(lines[1].contains("\"d\""));
    }

    #[test]
    fn validate_sync_with_compressed_jsonl() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("val_gz.sqlite3");
        let export_dir = dir.path().join("export");
        let state_root = dir.path().join("state");

        let store = RunStore::open(&db_path).expect("store open");
        store
            .persist_report(&fixture_report("gz-val-1", &db_path))
            .expect("persist");

        export(&db_path, &export_dir, &state_root).expect("export");

        // Compress all JSONL files and remove the originals
        for stem in ["runs", "segments", "events"] {
            let plain = export_dir.join(format!("{stem}.jsonl"));
            let gz = export_dir.join(format!("{stem}.jsonl.gz"));
            compress_jsonl(&plain, &gz).expect("compress");
            fs::remove_file(&plain).expect("remove plain");
        }

        let validation = validate_sync(&db_path, &export_dir).expect("validate gz");
        assert!(
            validation.is_valid,
            "should be valid with compressed JSONL: {validation:?}"
        );
        assert_eq!(validation.db_run_count, 1);
        assert_eq!(validation.jsonl_run_count, 1);
        assert_eq!(validation.db_segment_count, 2);
        assert_eq!(validation.jsonl_segment_count, 2);
        assert_eq!(validation.db_event_count, 2);
        assert_eq!(validation.jsonl_event_count, 2);
    }

    #[test]
    fn validate_sync_detects_segment_count_mismatch() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("val_seg_mismatch.sqlite3");
        let export_dir = dir.path().join("export");
        let state_root = dir.path().join("state");

        let store = RunStore::open(&db_path).expect("store open");
        store
            .persist_report(&fixture_report("seg-val", &db_path))
            .expect("persist");

        export(&db_path, &export_dir, &state_root).expect("export");

        // Append an extra line to segments.jsonl
        let seg_path = export_dir.join("segments.jsonl");
        let mut seg_file = OpenOptions::new()
            .append(true)
            .open(&seg_path)
            .expect("open segments for append");
        writeln!(
            seg_file,
            "{{\"run_id\":\"seg-val\",\"idx\":99,\"start_sec\":0.0,\"end_sec\":1.0,\"speaker\":null,\"text\":\"extra\",\"confidence\":0.5}}"
        )
        .expect("append");

        let validation = validate_sync(&db_path, &export_dir).expect("validate");
        assert!(!validation.is_valid, "should detect segment count mismatch");
        assert_eq!(validation.db_segment_count, 2);
        assert_eq!(validation.jsonl_segment_count, 3);
    }

    #[test]
    fn resolve_jsonl_path_prefers_gz() {
        let dir = tempdir().expect("tempdir");
        let base = dir.path();

        // Only plain file exists
        fs::write(base.join("runs.jsonl"), "plain").expect("write");
        let resolved = resolve_jsonl_path(base, "runs");
        assert_eq!(
            resolved,
            base.join("runs.jsonl"),
            "should resolve to plain when no gz"
        );

        // Create gz file — should now prefer it
        fs::write(base.join("runs.jsonl.gz"), "gzipped").expect("write gz");
        let resolved_gz = resolve_jsonl_path(base, "runs");
        assert_eq!(
            resolved_gz,
            base.join("runs.jsonl.gz"),
            "should prefer .gz when both exist"
        );
    }

    #[test]
    fn count_jsonl_lines_nonexistent_returns_zero() {
        let dir = tempdir().expect("tempdir");
        let missing = dir.path().join("does_not_exist.jsonl");
        let count = count_jsonl_lines(&missing).expect("count");
        assert_eq!(count, 0, "nonexistent file should return 0 lines");
    }

    #[test]
    fn collect_jsonl_ids_nonexistent_returns_empty() {
        let dir = tempdir().expect("tempdir");
        let missing = dir.path().join("does_not_exist.jsonl");
        let ids = collect_jsonl_ids(&missing, "id").expect("collect");
        assert!(ids.is_empty(), "nonexistent file should return empty set");
    }

    #[test]
    fn compress_jsonl_nonexistent_source_returns_error() {
        let dir = tempdir().expect("tempdir");
        let missing = dir.path().join("no_such_file.jsonl");
        let output = dir.path().join("output.jsonl.gz");
        let result = compress_jsonl(&missing, &output);
        assert!(result.is_err(), "compressing nonexistent file should fail");
    }

    #[test]
    fn decompress_jsonl_nonexistent_source_returns_error() {
        let dir = tempdir().expect("tempdir");
        let missing = dir.path().join("no_such_file.jsonl.gz");
        let output = dir.path().join("output.jsonl");
        let result = decompress_jsonl(&missing, &output);
        assert!(
            result.is_err(),
            "decompressing nonexistent file should fail"
        );
    }

    #[test]
    fn decompress_jsonl_invalid_gz_returns_error() {
        let dir = tempdir().expect("tempdir");
        let bad_gz = dir.path().join("bad.jsonl.gz");
        let output = dir.path().join("output.jsonl");
        fs::write(&bad_gz, "this is not gzip data").expect("write");
        let result = decompress_jsonl(&bad_gz, &output);
        assert!(result.is_err(), "decompressing invalid gz should fail");
    }

    #[test]
    fn compression_unicode_content() {
        let dir = tempdir().expect("tempdir");
        let original = dir.path().join("unicode.jsonl");
        let compressed = dir.path().join("unicode.jsonl.gz");
        let decompressed = dir.path().join("unicode_back.jsonl");

        let content = "{\"id\":\"u-1\",\"text\":\"\u{1F600} \u{4E16}\u{754C} caf\u{00E9}\"}\n";
        fs::write(&original, content).expect("write");
        compress_jsonl(&original, &compressed).expect("compress");
        decompress_jsonl(&compressed, &decompressed).expect("decompress");

        let recovered = fs::read_to_string(&decompressed).expect("read");
        assert_eq!(recovered, content, "unicode content round-trip");
    }

    #[test]
    fn json_str_or_empty_returns_empty_for_non_string_types() {
        let obj = json!({
            "name": "alice",
            "age": 42,
            "active": true,
            "scores": [1, 2, 3],
            "meta": {"nested": "object"},
            "gone": null
        });
        assert_eq!(json_str_or_empty(&obj, "name"), "alice");
        assert_eq!(json_str_or_empty(&obj, "age"), "");
        assert_eq!(json_str_or_empty(&obj, "active"), "");
        assert_eq!(json_str_or_empty(&obj, "scores"), "");
        assert_eq!(json_str_or_empty(&obj, "meta"), "");
        assert_eq!(json_str_or_empty(&obj, "gone"), "");
        assert_eq!(json_str_or_empty(&obj, "missing_key"), "");
    }

    #[test]
    fn collect_jsonl_ids_skips_blank_lines_and_missing_keys() {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("mixed.jsonl");
        // Mix of: valid id, blank line, line missing "id" key, another valid id
        let content = r#"{"id":"run-1","status":"ok"}

{"status":"orphan"}
{"id":"run-2","status":"done"}
{"id":42}
"#;
        fs::write(&path, content).expect("write");
        let ids = collect_jsonl_ids(&path, "id").expect("collect");
        assert_eq!(ids.len(), 2, "should find exactly 2 string ids");
        assert!(ids.contains("run-1"));
        assert!(ids.contains("run-2"));
        // "42" should be excluded since it's an integer, not a string
        assert!(!ids.contains("42"));
    }

    #[test]
    fn load_jsonl_run_map_indexes_by_id() {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("runs.jsonl");
        let content = r#"{"id":"r-a","backend":"whisper-cpp","started_at":"2025-01-01T00:00:00Z"}
{"id":"r-b","backend":"insanely-fast","started_at":"2025-01-02T00:00:00Z"}
"#;
        fs::write(&path, content).expect("write");
        let map = load_jsonl_run_map(&path).expect("load");
        assert_eq!(map.len(), 2);
        assert_eq!(
            map["r-a"].get("backend").and_then(|v| v.as_str()),
            Some("whisper-cpp")
        );
        assert_eq!(
            map["r-b"].get("backend").and_then(|v| v.as_str()),
            Some("insanely-fast")
        );
    }

    #[test]
    fn count_table_on_empty_table_returns_zero() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("empty.sqlite3");
        let _store = RunStore::open(&db_path).expect("open");
        // After RunStore::open the tables exist but are empty.
        let conn = Connection::open(db_path.display().to_string()).expect("open sqlite conn");
        let run_count = count_table(&conn, "runs").expect("count runs");
        let seg_count = count_table(&conn, "segments").expect("count segments");
        let evt_count = count_table(&conn, "events").expect("count events");
        assert_eq!(run_count, 0);
        assert_eq!(seg_count, 0);
        assert_eq!(evt_count, 0);
    }

    #[test]
    fn count_jsonl_lines_skips_whitespace_only_lines() {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("sparse.jsonl");
        // 3 data lines + 2 blank lines + 1 whitespace-only line
        let content = "{\"id\":\"a\"}\n\n{\"id\":\"b\"}\n   \n\n{\"id\":\"c\"}\n";
        fs::write(&path, content).expect("write");
        let count = count_jsonl_lines(&path).expect("count");
        assert_eq!(count, 3, "should count only non-empty/non-whitespace lines");
    }

    // -- seventeenth-pass edge case tests --

    #[test]
    fn max_started_at_with_partial_filter_returns_matching_maximum() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("partial_filter.sqlite3");
        let store = RunStore::open(&db_path).expect("open");

        // Persist two reports with different timestamps.
        let mut early = fixture_report("run-early", &db_path);
        early.started_at_rfc3339 = "2026-01-01T00:00:00Z".to_owned();
        store.persist_report(&early).expect("persist early");

        let mut late = fixture_report("run-late", &db_path);
        late.started_at_rfc3339 = "2026-06-15T12:00:00Z".to_owned();
        store.persist_report(&late).expect("persist late");

        let conn =
            fsqlite::Connection::open(db_path.display().to_string()).expect("open connection");

        // Filter after early → should return late's timestamp.
        let result = max_started_at(&conn, Some("2026-01-01T00:00:00Z")).expect("query");
        assert_eq!(
            result,
            Some("2026-06-15T12:00:00Z".to_owned()),
            "should return the max among filtered runs"
        );

        // Filter after late → no matching runs → None.
        let result = max_started_at(&conn, Some("2026-06-15T12:00:00Z")).expect("query");
        assert_eq!(result, None, "no runs strictly after late timestamp");
    }

    #[test]
    fn export_table_events_for_runs_empty_ids_produces_empty_file() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("evt_empty.sqlite3");
        let store = RunStore::open(&db_path).expect("open");
        store
            .persist_report(&fixture_report("evt-run", &db_path))
            .expect("persist");

        let conn =
            fsqlite::Connection::open(db_path.display().to_string()).expect("open connection");

        let out_path = dir.path().join("events_empty.jsonl");
        let count = export_table_events_for_runs(&conn, &out_path, &[]).expect("export");
        assert_eq!(count, 0, "empty run_ids should produce zero rows");
        let content = fs::read_to_string(&out_path).expect("read");
        assert!(content.is_empty(), "file should be empty");
    }

    #[test]
    fn collect_exported_run_ids_with_after_ts_filters_correctly() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("collect_ids.sqlite3");
        let store = RunStore::open(&db_path).expect("open");

        let mut r1 = fixture_report("run-a", &db_path);
        r1.started_at_rfc3339 = "2026-01-01T00:00:00Z".to_owned();
        store.persist_report(&r1).expect("persist r1");

        let mut r2 = fixture_report("run-b", &db_path);
        r2.started_at_rfc3339 = "2026-03-01T00:00:00Z".to_owned();
        store.persist_report(&r2).expect("persist r2");

        let mut r3 = fixture_report("run-c", &db_path);
        r3.started_at_rfc3339 = "2026-06-01T00:00:00Z".to_owned();
        store.persist_report(&r3).expect("persist r3");

        let conn =
            fsqlite::Connection::open(db_path.display().to_string()).expect("open connection");

        // None filter → all 3 IDs
        let all = collect_exported_run_ids(&conn, None).expect("all");
        assert_eq!(all.len(), 3, "no filter should return all runs");

        // After Jan → should return run-b and run-c (strictly after)
        let filtered =
            collect_exported_run_ids(&conn, Some("2026-01-01T00:00:00Z")).expect("filtered");
        assert_eq!(filtered.len(), 2, "should exclude run-a");
        assert!(
            !filtered.contains(&"run-a".to_owned()),
            "run-a should be excluded"
        );
    }

    #[test]
    fn validate_sync_detects_event_count_mismatch() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("val_evt_mismatch.sqlite3");
        let export_dir = dir.path().join("export");
        let state_root = dir.path().join("state");

        let store = RunStore::open(&db_path).expect("store open");
        store
            .persist_report(&fixture_report("evt-val", &db_path))
            .expect("persist");

        export(&db_path, &export_dir, &state_root).expect("export");

        // Append an extra line to events.jsonl
        let evt_path = export_dir.join("events.jsonl");
        let mut evt_file = OpenOptions::new()
            .append(true)
            .open(&evt_path)
            .expect("open events for append");
        writeln!(
            evt_file,
            "{{\"run_id\":\"evt-val\",\"seq\":99,\"ts_rfc3339\":\"2026-01-01T00:00:09Z\",\"stage\":\"extra\",\"code\":\"extra.ok\",\"message\":\"extra event\",\"payload_json\":\"{{}}\"}}"
        )
        .expect("append");

        let validation = validate_sync(&db_path, &export_dir).expect("validate");
        assert!(!validation.is_valid, "should detect event count mismatch");
        assert_eq!(validation.db_event_count, 2);
        assert_eq!(validation.jsonl_event_count, 3);
    }

    #[test]
    fn export_table_runs_incremental_filters_by_after_ts() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("inc_runs.sqlite3");
        let store = RunStore::open(&db_path).expect("open");

        let mut r1 = fixture_report("inc-a", &db_path);
        r1.started_at_rfc3339 = "2026-01-15T00:00:00Z".to_owned();
        store.persist_report(&r1).expect("persist r1");

        let mut r2 = fixture_report("inc-b", &db_path);
        r2.started_at_rfc3339 = "2026-05-20T00:00:00Z".to_owned();
        store.persist_report(&r2).expect("persist r2");

        let conn =
            fsqlite::Connection::open(db_path.display().to_string()).expect("open connection");

        // No filter → both runs exported.
        let all_path = dir.path().join("all_runs.jsonl");
        let all_count = export_table_runs_incremental(&conn, &all_path, None).expect("all");
        assert_eq!(all_count, 2, "no filter should export both runs");

        // After Jan → only inc-b (strictly after).
        let filtered_path = dir.path().join("filtered_runs.jsonl");
        let filtered_count =
            export_table_runs_incremental(&conn, &filtered_path, Some("2026-01-15T00:00:00Z"))
                .expect("filtered");
        assert_eq!(filtered_count, 1, "filter should export only inc-b");

        // Read and verify it's inc-b
        let content = fs::read_to_string(&filtered_path).expect("read");
        let line: serde_json::Value = serde_json::from_str(content.trim()).expect("parse");
        assert_eq!(line["id"], "inc-b", "exported run should be inc-b");
    }

    #[test]
    fn import_runs_missing_id_field_returns_error() {
        // Exercises json_str(&row, "id")? in import_runs when "id" key is absent.
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("source.sqlite3");
        let export_dir = dir.path().join("export");

        let store = RunStore::open(&db_path).expect("open");
        store
            .persist_report(&fixture_report("run-noid", &db_path))
            .expect("persist");
        export_inner(&db_path, &export_dir).expect("export");

        // Overwrite runs.jsonl with a valid JSON row that has no "id" key.
        let runs_path = export_dir.join("runs.jsonl");
        let bad_row = json!({
            "started_at": "2026-01-01T00:00:00Z",
            "finished_at": "2026-01-01T00:00:05Z",
            "backend": "whisper_cpp",
            "input_path": "test.wav",
            "normalized_wav_path": "normalized.wav",
            "request_json": "{}",
            "result_json": "{}",
            "warnings_json": "[]",
            "transcript": "hello"
        });
        fs::write(
            &runs_path,
            format!("{}\n", serde_json::to_string(&bad_row).expect("ser")),
        )
        .expect("write");

        // Update manifest checksum.
        let manifest_path = export_dir.join("manifest.json");
        let mut mval: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(&manifest_path).expect("read"))
                .expect("parse");
        mval["checksums"]["runs_jsonl_sha256"] = json!(sha256_file(&runs_path).expect("checksum"));
        mval["row_counts"]["runs"] = json!(1);
        fs::write(
            &manifest_path,
            serde_json::to_string_pretty(&mval).expect("ser"),
        )
        .expect("write manifest");

        let import_db = dir.path().join("import.sqlite3");
        let err = import_inner(&import_db, &export_dir, ConflictPolicy::Reject)
            .expect_err("missing id should fail");
        let text = err.to_string();
        assert!(
            text.contains("missing string field") && text.contains("id"),
            "error should mention missing 'id' field: {text}"
        );
    }

    #[test]
    fn import_segments_missing_run_id_field_returns_error() {
        // Exercises json_str(&row, "run_id")? in import_segments when "run_id" is absent.
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("source.sqlite3");
        let export_dir = dir.path().join("export");

        let store = RunStore::open(&db_path).expect("open");
        store
            .persist_report(&fixture_report("run-seg", &db_path))
            .expect("persist");
        export_inner(&db_path, &export_dir).expect("export");

        // Overwrite segments.jsonl with a row missing "run_id".
        let seg_path = export_dir.join("segments.jsonl");
        let bad_seg = json!({
            "idx": 0,
            "start_sec": 0.0,
            "end_sec": 1.0,
            "text": "hello",
            "speaker": null,
            "confidence": null
        });
        fs::write(
            &seg_path,
            format!("{}\n", serde_json::to_string(&bad_seg).expect("ser")),
        )
        .expect("write");

        // Update manifest checksums.
        let manifest_path = export_dir.join("manifest.json");
        let mut mval: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(&manifest_path).expect("read"))
                .expect("parse");
        mval["checksums"]["segments_jsonl_sha256"] =
            json!(sha256_file(&seg_path).expect("checksum"));
        mval["row_counts"]["segments"] = json!(1);
        fs::write(
            &manifest_path,
            serde_json::to_string_pretty(&mval).expect("ser"),
        )
        .expect("write manifest");

        let import_db = dir.path().join("import.sqlite3");
        let err = import_inner(&import_db, &export_dir, ConflictPolicy::Reject)
            .expect_err("missing run_id should fail");
        let text = err.to_string();
        assert!(
            text.contains("missing string field") && text.contains("run_id"),
            "error should mention missing 'run_id' field: {text}"
        );
    }

    #[test]
    fn import_malformed_events_jsonl_returns_error() {
        // Exercises serde_json::from_str error path in import_events (line 1062).
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("source.sqlite3");
        let export_dir = dir.path().join("export");

        let store = RunStore::open(&db_path).expect("open");
        store
            .persist_report(&fixture_report("bad-evt", &db_path))
            .expect("persist");
        export_inner(&db_path, &export_dir).expect("export");

        // Replace events.jsonl with malformed JSON.
        let evt_path = export_dir.join("events.jsonl");
        fs::write(&evt_path, "{not valid json!!!}\n").expect("write");

        // Update manifest checksum.
        let manifest_path = export_dir.join("manifest.json");
        let mut mval: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(&manifest_path).expect("read"))
                .expect("parse");
        mval["checksums"]["events_jsonl_sha256"] = json!(sha256_file(&evt_path).expect("checksum"));
        mval["row_counts"]["events"] = json!(1);
        fs::write(
            &manifest_path,
            serde_json::to_string_pretty(&mval).expect("ser"),
        )
        .expect("write manifest");

        let import_db = dir.path().join("import.sqlite3");
        let err = import_inner(&import_db, &export_dir, ConflictPolicy::Reject)
            .expect_err("malformed events should fail");
        let text = err.to_string();
        assert!(
            text.contains("key must be") || text.contains("json") || text.contains("JSON"),
            "error should mention JSON parse failure: {text}"
        );
    }

    #[test]
    fn load_jsonl_run_map_nonexistent_returns_empty() {
        let dir = tempdir().expect("tempdir");
        let nonexistent = dir.path().join("does_not_exist.jsonl");
        let map = load_jsonl_run_map(&nonexistent).expect("should succeed");
        assert!(
            map.is_empty(),
            "nonexistent file should produce empty map, got {} entries",
            map.len()
        );
    }

    #[test]
    fn validate_sync_detects_mismatched_backend_field() {
        // Exercises the backend comparison branch at line 1560-1561 in validate_sync,
        // independent of the transcript comparison (which is tested separately).
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("val_backend.sqlite3");
        let export_dir = dir.path().join("export");
        let state_root = dir.path().join("state");

        let store = RunStore::open(&db_path).expect("open");
        store
            .persist_report(&fixture_report("bk-mismatch", &db_path))
            .expect("persist");
        export(&db_path, &export_dir, &state_root).expect("export");

        // Mutate only the backend field in runs.jsonl (leave transcript unchanged).
        let runs_path = export_dir.join("runs.jsonl");
        let content = fs::read_to_string(&runs_path).expect("read");
        let first_line = content.lines().next().expect("has line");
        let mut mutated: serde_json::Value = serde_json::from_str(first_line).expect("valid run");
        mutated["backend"] = json!("totally_different_backend");
        fs::write(
            &runs_path,
            format!("{}\n", serde_json::to_string(&mutated).expect("ser")),
        )
        .expect("write");

        let validation = validate_sync(&db_path, &export_dir).expect("validate");
        assert!(!validation.is_valid, "should detect backend field mismatch");
        assert!(
            validation
                .mismatched_records
                .contains(&"bk-mismatch".to_owned()),
            "bk-mismatch should be in mismatched_records: {:?}",
            validation.mismatched_records
        );
    }

    #[test]
    fn optional_floats_equal_zero_and_negative_values() {
        // Zero vs zero.
        assert!(optional_floats_equal(Some(0.0), Some(0.0)));
        // Negative matching values.
        assert!(optional_floats_equal(Some(-1.5), Some(-1.5)));
        // Negative divergent values.
        assert!(!optional_floats_equal(Some(-1.0), Some(-2.0)));
        // Sign mismatch.
        assert!(!optional_floats_equal(Some(-1.0), Some(1.0)));
    }

    #[test]
    fn collect_jsonl_ids_deduplicates_repeated_ids() {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("dedup.jsonl");
        let content = [
            r#"{"id": "run-1", "data": 1}"#,
            r#"{"id": "run-1", "data": 2}"#,
            r#"{"id": "run-1", "data": 3}"#,
            r#"{"id": "run-2", "data": 4}"#,
        ]
        .join("\n");
        fs::write(&path, content).expect("write");

        let ids = collect_jsonl_ids(&path, "id").expect("collect");
        assert_eq!(ids.len(), 2);
        assert!(ids.contains("run-1"));
        assert!(ids.contains("run-2"));
    }

    #[test]
    fn load_jsonl_run_map_skips_non_string_and_missing_ids() {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("mixed.jsonl");
        let content = [
            r#"{"id": "valid-1", "text": "ok"}"#,
            r#"{"id": 42, "text": "integer id"}"#,
            r#"{"id": null, "text": "null id"}"#,
            r#"{"text": "no id field"}"#,
            "",
            r#"{"id": "valid-2", "text": "also ok"}"#,
        ]
        .join("\n");
        fs::write(&path, content).expect("write");

        let map = load_jsonl_run_map(&path).expect("load");
        assert_eq!(map.len(), 2);
        assert!(map.contains_key("valid-1"));
        assert!(map.contains_key("valid-2"));
    }

    #[test]
    fn count_jsonl_lines_skips_blanks() {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("sparse.jsonl");
        let content = [r#"{"id": "a"}"#, "", "   ", r#"{"id": "b"}"#, ""].join("\n");
        fs::write(&path, content).expect("write");

        let count = count_jsonl_lines(&path).expect("count");
        assert_eq!(count, 2, "should skip blank/whitespace-only lines");
    }

    #[test]
    fn count_jsonl_lines_missing_file_is_zero() {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("does_not_exist.jsonl");
        let count = count_jsonl_lines(&path).expect("count");
        assert_eq!(count, 0);
    }

    // ── Task #211 — sync pass 2 edge-case tests ─────────────────────

    #[test]
    fn resolve_jsonl_path_neither_file_exists_returns_plain_jsonl() {
        let dir = tempdir().expect("tempdir");
        // Neither runs.jsonl nor runs.jsonl.gz exists.
        let path = resolve_jsonl_path(dir.path(), "runs");
        assert!(
            path.ends_with("runs.jsonl"),
            "should return .jsonl variant as fallback: {}",
            path.display()
        );
        assert!(
            !path.exists(),
            "returned path should not exist on disk: {}",
            path.display()
        );
    }

    #[test]
    fn load_cursor_valid_json_wrong_schema_returns_error() {
        let dir = tempdir().expect("tempdir");
        let cursor_path = dir.path().join("cursor.json");
        // Valid JSON but missing required fields (last_export_rfc3339, last_run_count).
        fs::write(&cursor_path, r#"{"wrong_field": "value", "another": 0}"#).expect("write");

        let result = load_cursor(&cursor_path);
        assert!(result.is_err(), "should fail on wrong schema");
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("invalid sync cursor"),
            "error should mention invalid sync cursor: {msg}"
        );
    }

    #[test]
    fn sync_lock_release_inner_double_call_is_idempotent() {
        let dir = tempdir().expect("tempdir");
        let mut lock = SyncLock::acquire(dir.path(), "test-double-release").unwrap();

        // First release.
        lock.release_inner().unwrap();
        assert!(!dir.path().join("sync.lock").exists(), "lock file should be removed");

        // Second release — hits the `if self.released { return Ok(()) }` guard.
        lock.release_inner().unwrap();

        // Prevent Drop from running file removal again (already released).
        lock.released = true;
    }

    #[test]
    fn collect_jsonl_ids_empty_file_returns_empty_set() {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("empty.jsonl");
        fs::write(&path, "").expect("write");

        let ids = collect_jsonl_ids(&path, "id").expect("collect");
        assert!(ids.is_empty(), "empty file should produce empty id set");
    }

    #[test]
    fn load_jsonl_run_map_valid_entries_preserves_all_fields() {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("runs.jsonl");
        let line = serde_json::json!({
            "id": "run-123",
            "started_at": "2026-01-01T00:00:00Z",
            "extra_field": "preserved"
        });
        fs::write(&path, serde_json::to_string(&line).unwrap()).expect("write");

        let map = load_jsonl_run_map(&path).expect("load");
        assert_eq!(map.len(), 1);
        let entry = map.get("run-123").expect("should contain run-123");
        assert_eq!(entry["started_at"].as_str(), Some("2026-01-01T00:00:00Z"));
        assert_eq!(
            entry["extra_field"].as_str(),
            Some("preserved"),
            "extra fields should be preserved in the map"
        );
    }
}
