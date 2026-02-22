#!/usr/bin/env bash
set -euo pipefail

CRITERION_ROOT="${1:-target/criterion}"
POLICY_PATH="${2:-docs/benchmark_guardrails.json}"

cargo run --quiet --bin benchmark_guardrails -- \
  --criterion-root "${CRITERION_ROOT}" \
  --policy "${POLICY_PATH}"
