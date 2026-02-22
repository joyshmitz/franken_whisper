#!/usr/bin/env bash
# Mock whisper-cli backend for testing.
# Writes golden JSON output to the path specified by the last positional arg
# (which is the expected output file path from whisper.cpp CLI convention).
#
# Usage: mock_whisper_cpp.sh [options...] -of <output_file>
#
# This mock logs its invocation for debugging, then emits a golden JSON
# response without requiring a real whisper.cpp binary.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GOLDEN_DIR="$SCRIPT_DIR/../fixtures/golden"

# Log invocation for debugging
echo "Mock whisper_cpp called with args: $*" >&2

# Parse args to find output prefix (whisper-cli uses -of <prefix> and writes
# <prefix>.json when JSON output is enabled).
OUTPUT_FILE=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        -of)
            OUTPUT_FILE="$2"
            shift 2
            ;;
        *)
            shift
            ;;
    esac
done

# If no output file specified, write to stdout
if [[ -z "$OUTPUT_FILE" ]]; then
    cat "$GOLDEN_DIR/whisper_cpp_output.json"
else
    # whisper.cpp commonly uses -of prefix and emits <prefix>.json.
    case "$OUTPUT_FILE" in
        *.json) OUTPUT_JSON="$OUTPUT_FILE" ;;
        *) OUTPUT_JSON="${OUTPUT_FILE}.json" ;;
    esac
    cp "$GOLDEN_DIR/whisper_cpp_output.json" "$OUTPUT_JSON"
fi

exit 0
