#!/usr/bin/env python3
"""Mock diarization backend for testing.

Writes golden SRT and TXT output to the expected locations without requiring
the real whisper-diarization Python pipeline.

Usage: python3 mock_diarization.py [options...]

This mock logs its invocation for debugging, then emits golden diarization
output files.
"""

import os
import shutil
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
GOLDEN_DIR = os.path.join(SCRIPT_DIR, "..", "fixtures", "golden")

# Log invocation for debugging
print(f"Mock diarization called with args: {sys.argv[1:]}", file=sys.stderr)

# Parse args to find output directory (--output-dir <path>)
output_dir = None
i = 1
while i < len(sys.argv):
    if sys.argv[i] == "--output-dir" and i + 1 < len(sys.argv):
        output_dir = sys.argv[i + 1]
        i += 2
    else:
        i += 1

if output_dir is None:
    output_dir = "."

os.makedirs(output_dir, exist_ok=True)

shutil.copy2(
    os.path.join(GOLDEN_DIR, "diarization_output.srt"),
    os.path.join(output_dir, "output.srt"),
)
shutil.copy2(
    os.path.join(GOLDEN_DIR, "diarization_output.txt"),
    os.path.join(output_dir, "output.txt"),
)

# Print transcript to stdout (as the real diarization tool does)
with open(os.path.join(GOLDEN_DIR, "diarization_output.txt")) as f:
    print(f.read(), end="")
