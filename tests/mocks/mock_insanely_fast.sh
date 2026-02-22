#!/usr/bin/env bash
# Mock insanely-fast-whisper backend for testing.
# Writes golden JSON output to stdout (insanely-fast-whisper prints to stdout).
#
# Usage: mock_insanely_fast.sh [options...]
#
# This mock logs its invocation for debugging, then emits a golden JSON
# response without requiring a real insanely-fast-whisper binary.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GOLDEN_DIR="$SCRIPT_DIR/../fixtures/golden"

# Log invocation for debugging
echo "Mock insanely_fast called with args: $*" >&2

# Parse args to find output file.
# Current adapter uses --transcript-path; keep --output-file for compatibility.
OUTPUT_FILE=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --transcript-path)
            OUTPUT_FILE="$2"
            shift 2
            ;;
        --output-file)
            OUTPUT_FILE="$2"
            shift 2
            ;;
        *)
            shift
            ;;
    esac
done

if [[ -z "$OUTPUT_FILE" ]]; then
    cat "$GOLDEN_DIR/insanely_fast_output.json"
else
    cp "$GOLDEN_DIR/insanely_fast_output.json" "$OUTPUT_FILE"
fi

exit 0
