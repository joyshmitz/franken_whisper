#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FW_BIN="${FW_BIN:-$ROOT_DIR/target/debug/franken_whisper}"
FFMPEG_BUNDLE_DIR="${HOME}/.local/state/franken_whisper/tools/ffmpeg/bin"
DEMO_MODE="${DEMO_MODE:-interactive}"

if [[ ! -x "$FW_BIN" ]]; then
    echo "error: expected executable at $FW_BIN"
    echo "set FW_BIN=/path/to/franken_whisper before running this demo"
    exit 1
fi

if [[ -d "$FFMPEG_BUNDLE_DIR" ]]; then
    export PATH="$FFMPEG_BUNDLE_DIR:$PATH"
fi

if ! command -v ffmpeg >/dev/null 2>&1; then
    echo "error: ffmpeg not found on PATH"
    echo "tip: export PATH=\"$FFMPEG_BUNDLE_DIR:\$PATH\""
    exit 1
fi

if ! command -v jq >/dev/null 2>&1; then
    echo "error: jq is required for this demo"
    exit 1
fi

TMP_DIR="$(mktemp -d)"

HAS_GUM=0
if command -v gum >/dev/null 2>&1; then
    HAS_GUM=1
fi

if [[ -t 1 ]]; then
    BOLD=$'\033[1m'
    DIM=$'\033[2m'
    GREEN=$'\033[38;5;42m'
    YELLOW=$'\033[38;5;220m'
    CYAN=$'\033[38;5;45m'
    RESET=$'\033[0m'
else
    BOLD=""
    DIM=""
    GREEN=""
    YELLOW=""
    CYAN=""
    RESET=""
fi

style_block() {
    local border="${1:-rounded}"
    local color="${2:-45}"
    shift 2
    local text
    text="$(printf '%s\n' "$@")"
    if (( HAS_GUM == 1 )); then
        gum style --border "$border" --border-foreground "$color" --padding "1 2" "$text"
    else
        printf '%s\n' "$text"
    fi
}

hero() {
    style_block double 45 \
        "franken_whisper tty-audio demo" \
        "" \
        "A real, end-to-end demonstration of audio transported over terminal-safe NDJSON."
}

step() {
    local num="$1"
    local title="$2"
    if (( HAS_GUM == 1 )); then
        printf '\n'
        gum style --foreground 212 --bold "Step ${num}"
        printf ' '
        gum style --foreground 255 --bold "$title"
    else
        printf '\n%sStep %s%s %s\n' "$BOLD" "$num" "$RESET" "$title"
    fi
}

note() {
    local text="$1"
    if (( HAS_GUM == 1 )); then
        gum style --foreground 246 "$text"
    else
        printf '%s%s%s\n' "$DIM" "$text" "$RESET"
    fi
}

good() {
    local text="$1"
    if (( HAS_GUM == 1 )); then
        gum style --foreground 42 --bold "$text"
    else
        printf '%s%s%s\n' "$GREEN" "$text" "$RESET"
    fi
}

warn() {
    local text="$1"
    if (( HAS_GUM == 1 )); then
        gum style --foreground 220 --bold "$text"
    else
        printf '%s%s%s\n' "$YELLOW" "$text" "$RESET"
    fi
}

pause_demo() {
    if [[ "$DEMO_MODE" == "auto" ]]; then
        return 0
    fi
    if [[ -t 0 && -t 1 ]]; then
        if (( HAS_GUM == 1 )); then
            gum confirm "Continue to the next step?" >/dev/null || exit 0
        else
            printf '\nPress Enter to continue... '
            read -r _
        fi
    fi
}

show_command() {
    local cmd="$1"
    if (( HAS_GUM == 1 )); then
        gum style --foreground 51 --border normal --padding "0 1" "$cmd"
    else
        printf '%s$ %s%s\n' "$CYAN" "$cmd" "$RESET"
    fi
}

show_json_lines() {
    local file="$1"
    local count="$2"
    sed -n "1,${count}p" "$file" | while IFS= read -r line; do
        jq . <<<"$line"
    done
}

summarize_wav() {
    python3 - <<'PY' "$1"
import json
import sys
import wave

with wave.open(sys.argv[1], "rb") as wav:
    print(json.dumps({
        "path": sys.argv[1],
        "channels": wav.getnchannels(),
        "sample_rate_hz": wav.getframerate(),
        "frames": wav.getnframes(),
        "seconds": round(wav.getnframes() / wav.getframerate(), 3),
    }, indent=2))
PY
}

python3 - <<'PY' "$TMP_DIR/demo.wav"
import math
import struct
import sys
import wave

path = sys.argv[1]
rate = 16000
seconds = 1.0
freq_a = 440.0
freq_b = 660.0
frames = int(rate * seconds)

with wave.open(path, "wb") as wav:
    wav.setnchannels(1)
    wav.setsampwidth(2)
    wav.setframerate(rate)
    data = bytearray()
    for i in range(frames):
        blend = (
            0.22 * math.sin(2 * math.pi * freq_a * i / rate)
            + 0.12 * math.sin(2 * math.pi * freq_b * i / rate)
        )
        sample = int(32767 * blend)
        data += struct.pack("<h", sample)
    wav.writeframes(data)
PY

hero
note "Workspace: $ROOT_DIR"
note "Binary:    $FW_BIN"
note "ffmpeg:    $(command -v ffmpeg)"
note "Temp dir:  $TMP_DIR"
note "Artifacts are preserved after exit for inspection."

if (( HAS_GUM == 1 )); then
    printf '\n'
    gum join --vertical \
        "$(gum style --foreground 212 --bold 'What this demo proves')" \
        "1. audio can be converted into NDJSON frames" \
        "2. a handshake/control channel shares the same stream" \
        "3. strict decode fails on missing frames" \
        "4. retransmit planning is deterministic" \
        "5. skip-missing mode salvages partial audio"
fi

pause_demo

step 1 "Create a tiny, reproducible audio source"
show_command "python3 ... > $TMP_DIR/demo.wav"
summarize_wav "$TMP_DIR/demo.wav"
note "The source is synthesized locally, so the demo is self-contained and repeatable."

pause_demo

step 2 "Encode the WAV into terminal-safe NDJSON"
show_command "$FW_BIN tty-audio encode --input $TMP_DIR/demo.wav --chunk-ms 100 > $TMP_DIR/frames.ndjson"
"$FW_BIN" tty-audio encode --input "$TMP_DIR/demo.wav" --chunk-ms 100 >"$TMP_DIR/frames.ndjson"
printf 'line_count=%s\n' "$(wc -l <"$TMP_DIR/frames.ndjson")"
note "Line 1 is a handshake control frame. The rest are audio payload frames."
show_json_lines "$TMP_DIR/frames.ndjson" 3

pause_demo

step 3 "Decode the clean stream back into audio"
show_command "$FW_BIN tty-audio decode --output $TMP_DIR/restored.wav < $TMP_DIR/frames.ndjson"
"$FW_BIN" tty-audio decode --output "$TMP_DIR/restored.wav" <"$TMP_DIR/frames.ndjson"
summarize_wav "$TMP_DIR/restored.wav"
good "Clean round-trip succeeded."

pause_demo

step 4 "Simulate packet loss on a terminal link"
show_command "jq 'drop the data frame where seq == 3' $TMP_DIR/frames.ndjson > $TMP_DIR/lossy.ndjson"
jq -c 'select((has("seq") | not) or .seq != 3)' \
    "$TMP_DIR/frames.ndjson" >"$TMP_DIR/lossy.ndjson"
ORIGINAL_LINES="$(wc -l <"$TMP_DIR/frames.ndjson")"
LOSSY_LINES="$(wc -l <"$TMP_DIR/lossy.ndjson")"
printf 'original_lines=%s\n' "$ORIGINAL_LINES"
printf 'lossy_lines=%s\n' "$LOSSY_LINES"
if [[ $((ORIGINAL_LINES - LOSSY_LINES)) -ne 1 ]]; then
    echo "error: expected lossy stream to remove exactly one frame" >&2
    exit 1
fi
note "We removed exactly one audio frame but kept the handshake intact."

pause_demo

step 5 "Strict mode: fail_closed refuses damaged streams"
show_command "$FW_BIN tty-audio decode --output $TMP_DIR/fail_closed.wav < $TMP_DIR/lossy.ndjson"
set +e
"$FW_BIN" tty-audio decode --output "$TMP_DIR/fail_closed.wav" <"$TMP_DIR/lossy.ndjson" \
    >"$TMP_DIR/fail_closed.out" 2>"$TMP_DIR/fail_closed.err"
FAIL_CLOSED_RC=$?
set -e
printf 'exit_code=%s\n' "$FAIL_CLOSED_RC"
cat "$TMP_DIR/fail_closed.err"
if [[ "$FAIL_CLOSED_RC" -eq 0 ]]; then
    echo "error: fail_closed unexpectedly accepted a damaged stream" >&2
    exit 1
fi
warn "Any missing or corrupt frame aborts decode in fail_closed mode."

pause_demo

step 6 "Deterministically compute the retransmit plan"
show_command "$FW_BIN tty-audio retransmit-plan --recovery skip_missing < $TMP_DIR/lossy.ndjson"
"$FW_BIN" tty-audio retransmit-plan --recovery skip_missing <"$TMP_DIR/lossy.ndjson" | jq .
note "The plan identifies the missing sequence without guesswork."

pause_demo

step 7 "Emit the control traffic for retransmission"
show_command "$FW_BIN tty-audio control retransmit-loop --rounds 2 < $TMP_DIR/lossy.ndjson"
"$FW_BIN" tty-audio control retransmit-loop --rounds 2 <"$TMP_DIR/lossy.ndjson" \
    | while IFS= read -r line; do
        jq . <<<"$line"
    done
note "Two identical retransmit requests, then a retransmit response summary."

pause_demo

step 8 "Best-effort recovery: skip_missing salvages partial audio"
show_command "$FW_BIN tty-audio decode --output $TMP_DIR/skip_missing.wav --recovery skip_missing < $TMP_DIR/lossy.ndjson"
"$FW_BIN" tty-audio decode --output "$TMP_DIR/skip_missing.wav" --recovery skip_missing \
    <"$TMP_DIR/lossy.ndjson"
summarize_wav "$TMP_DIR/skip_missing.wav"
warn "Recovered audio is shorter because the missing frame was dropped instead of reconstructed."

pause_demo

step 9 "Control frames are first-class citizens in the same stream"
show_command "$FW_BIN tty-audio control ack --up-to-seq 2"
"$FW_BIN" tty-audio control ack --up-to-seq 2 | jq .
show_command "$FW_BIN tty-audio control backpressure --remaining-capacity 64"
"$FW_BIN" tty-audio control backpressure --remaining-capacity 64 | jq .
show_command "$FW_BIN tty-audio send-control eof"
"$FW_BIN" tty-audio send-control eof | jq .

pause_demo

step 10 "The mental model"
style_block rounded 212 \
'Audio over terminal works here because franken_whisper does not try to push raw PCM through the PTY.' \
'' \
'Instead it sends newline-delimited JSON frames:' \
'- a handshake frame to establish protocol and codec' \
'- numbered audio frames carrying compressed mu-law payloads' \
'- optional control frames for ack, backpressure, retransmit, and session close' \
'' \
'That gives you deterministic framing, explicit integrity boundaries, and machine-readable recovery behavior.'

printf '\n%s\n' "Key takeaways"
printf '%s\n' \
    '• `fail_closed` is the strict, audit-safe mode.' \
    '• `skip_missing` is the best-effort recovery mode.' \
    '• `retransmit-plan` tells you exactly which frames are missing.' \
    '• `control retransmit-loop` shows the protocol traffic a decoder would emit.' \
    "• Everything remains line-oriented NDJSON, so it survives PTY/TTY transport."

printf '\n%s\n' "Artifacts"
printf '%s\n' \
    "$TMP_DIR/demo.wav" \
    "$TMP_DIR/frames.ndjson" \
    "$TMP_DIR/lossy.ndjson" \
    "$TMP_DIR/restored.wav" \
    "$TMP_DIR/skip_missing.wav"
note "It is preserved after exit for inspection; no cleanup is performed."
