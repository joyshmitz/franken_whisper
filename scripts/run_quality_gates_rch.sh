#!/usr/bin/env bash
set -euo pipefail

# Run the mandatory quality gates through rch with remote-only enforcement.
# If rch falls back to local execution or known transient worker errors appear,
# the gate is retried on a fresh remote target dir.

MAX_ATTEMPTS="${MAX_ATTEMPTS:-4}"
RUN_TEST_GATE="${RUN_TEST_GATE:-1}"
PROJECT_TAG="${PROJECT_TAG:-franken_whisper}"
DRAIN_WORKERS="${DRAIN_WORKERS:-}"
BLOCK_WORKERS="${BLOCK_WORKERS:-${DRAIN_WORKERS}}"
LOCK_FILE="${LOCK_FILE:-/tmp/${PROJECT_TAG}_rch_quality_gates.lock}"

retryable_pattern='No space left on device|Dependency planner fail-open|primary-root-only sync|Remote toolchain failure|toolchain missing|Project sync failed: rsync failed|Permission denied \(13\)|Connection refused|no workers with Rust installed|/data/projects/asupersync/src/runtime/scheduler/three_lane.rs|\[RCH\] local'
declare -a blocked_workers=()

exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
    echo "[routing] another quality-gate run is already active (${LOCK_FILE})"
    exit 1
fi

restore_workers() {
    local exit_code=$?
    for worker in "${blocked_workers[@]}"; do
        echo "[routing] restoring worker ${worker}"
        rch workers enable "${worker}" >/dev/null 2>&1 || true
    done
    exit "${exit_code}"
}

trap restore_workers EXIT

disable_worker_for_retry() {
    local worker="$1"
    if [[ -z "${worker}" ]]; then
        return 0
    fi

    local existing
    for existing in "${blocked_workers[@]}"; do
        if [[ "${existing}" == "${worker}" ]]; then
            return 0
        fi
    done

    echo "[routing] disabling worker ${worker} after retryable failure"
    if rch workers disable "${worker}" --reason "temporary quality-gate retry quarantine" -y >/dev/null; then
        blocked_workers+=("${worker}")
    fi
}

preblock_unreachable_workers() {
    local probe_json
    probe_json="$(rch workers probe --all --json 2>/dev/null)" || return 0

    local unreachable_workers=""
    if command -v jq >/dev/null 2>&1; then
        unreachable_workers="$(
            jq -r '.data[] | select(.status != "ok") | .id' <<<"${probe_json}" 2>/dev/null
        )"
    elif command -v python3 >/dev/null 2>&1; then
        unreachable_workers="$(
            python3 -c '
import json, sys
data = json.load(sys.stdin)
for worker in data.get("data", []):
    if worker.get("status") != "ok":
        print(worker.get("id", ""))
' <<<"${probe_json}" 2>/dev/null
        )"
    fi

    if [[ -z "${unreachable_workers}" ]]; then
        return 0
    fi

    while IFS= read -r worker; do
        worker="${worker//[[:space:]]/}"
        if [[ -z "${worker}" ]]; then
            continue
        fi
        echo "[routing] preblocking unreachable worker ${worker}"
        disable_worker_for_retry "${worker}"
    done <<<"${unreachable_workers}"
}

run_gate() {
    local gate="$1"
    shift

    local attempt=1
    while (( attempt <= MAX_ATTEMPTS )); do
        local target_dir="/tmp/rch_target_${PROJECT_TAG}_${gate}_${attempt}_$$"
        local log_file
        local selected_worker=""
        log_file="$(mktemp)"

        echo "[gate:${gate}] attempt ${attempt}/${MAX_ATTEMPTS}"
        set +e
        rch exec -- env \
            CARGO_TARGET_DIR="${target_dir}" \
            CARGO_INCREMENTAL=0 \
            RUSTFLAGS='-Cdebuginfo=0' \
            "$@" 2>&1 | tee "${log_file}"
        local status=${PIPESTATUS[0]}
        set -e

        local fell_back_local=0
        if grep -Fq "[RCH] local" "${log_file}"; then
            fell_back_local=1
            echo "[gate:${gate}] detected local fallback; treating as failure"
        fi

        local dependency_sync_invalid=0
        if grep -Eq 'Dependency planner fail-open|primary-root-only sync' "${log_file}"; then
            dependency_sync_invalid=1
            echo "[gate:${gate}] detected degraded dependency sync; refusing stale-sibling validation"
        fi

        if (( status == 0 && fell_back_local == 0 && dependency_sync_invalid == 0 )); then
            rm -f "${log_file}"
            echo "[gate:${gate}] success"
            return 0
        fi

        local retryable=0
        if grep -Eq "${retryable_pattern}" "${log_file}"; then
            retryable=1
        fi

        selected_worker="$(sed -n 's/.*Selected worker: \([^ ]*\) .*/\1/p' "${log_file}" | tail -n 1)"

        if (( attempt < MAX_ATTEMPTS && retryable == 1 )); then
            disable_worker_for_retry "${selected_worker}"
            echo "[gate:${gate}] retrying after transient/worker issue"
            rm -f "${log_file}"
            sleep 2
            attempt=$((attempt + 1))
            continue
        fi

        echo "[gate:${gate}] failed"
        echo "[gate:${gate}] log kept at ${log_file}"
        return 1
    done

    return 1
}

if ! rch check; then
    echo "[routing] rch check reported degraded worker state; continuing with retry-based routing"
fi

preblock_unreachable_workers

if [[ -n "${BLOCK_WORKERS}" ]]; then
    IFS=',' read -r -a selected_workers <<<"${BLOCK_WORKERS}"
    for worker in "${selected_workers[@]}"; do
        worker="${worker//[[:space:]]/}"
        if [[ -z "${worker}" ]]; then
            continue
        fi
        echo "[routing] disabling worker ${worker}"
        if rch workers disable "${worker}" --reason "temporary quality-gate routing override" -y; then
            blocked_workers+=("${worker}")
        fi
    done
fi

run_gate fmt cargo fmt --check
run_gate check cargo check --all-targets
run_gate clippy cargo clippy --all-targets -- -D warnings

if [[ "${RUN_TEST_GATE}" == "1" ]]; then
    run_gate test cargo test
else
    echo "[gate:test] skipped (RUN_TEST_GATE=${RUN_TEST_GATE})"
fi

echo "All requested rch quality gates completed successfully."
