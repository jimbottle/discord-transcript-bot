#!/usr/bin/env bash
# Verify the artifacts produced by a live /scribe → /stop session.
# Run after performing the manual voice test against your personal server.
#
# Optionally pass a sentinel phrase as the first arg — the script will
# search the latest session transcript for it.
#
#   scripts/post_session_verify.sh "the quick brown fox"
set -euo pipefail

cd "$(dirname "$0")/.."

SENTINEL="${1:-}"
EXIT=0

echo "=== Latest session transcript ==="
LATEST_TXT=$(ls -t transcripts/*.txt 2>/dev/null | head -1 || true)
if [ -z "$LATEST_TXT" ]; then
    echo "FAIL: no transcript files in transcripts/"
    EXIT=1
else
    echo "  file: $LATEST_TXT"
    SIZE=$(stat -f%z "$LATEST_TXT" 2>/dev/null || stat -c%s "$LATEST_TXT")
    if [ "$SIZE" -eq 0 ]; then
        echo "  FAIL: session file is empty (no transcription was written)"
        EXIT=1
    else
        LINES=$(wc -l < "$LATEST_TXT")
        echo "  size: ${SIZE}B, lines: ${LINES}"
        echo "  --- first 5 lines ---"
        head -5 "$LATEST_TXT" | sed 's/^/    /'
    fi
fi

echo
echo "=== Latest JSON transcription log ==="
LATEST_LOG=$(ls -t .logs/transcripts/*.log 2>/dev/null | head -1 || true)
if [ -z "$LATEST_LOG" ]; then
    echo "WARN: no .logs/transcripts/*.log files"
else
    LOG_LINES=$(wc -l < "$LATEST_LOG")
    echo "  file: $LATEST_LOG ($LOG_LINES lines)"
    if [ "$LOG_LINES" -eq 0 ]; then
        echo "  WARN: log file empty (may be stale from an earlier session)"
    fi
fi

if [ -n "$SENTINEL" ] && [ -n "$LATEST_TXT" ]; then
    echo
    echo "=== Sentinel phrase check: \"$SENTINEL\" ==="
    if grep -qi "$SENTINEL" "$LATEST_TXT"; then
        echo "  OK: found in session transcript"
    else
        echo "  WARN: not found verbatim (whisper may have rephrased)"
        echo "  Inspect $LATEST_TXT manually if you spoke this exact phrase"
    fi
fi

echo
echo "=== Per-speaker separation check ==="
if [ -n "$LATEST_TXT" ]; then
    UNIQUE_USERS=$(awk -F'[][]' '{print $4}' "$LATEST_TXT" | sort -u | grep -c . || true)
    echo "  unique user IDs in transcript: $UNIQUE_USERS"
    if [ "$UNIQUE_USERS" -eq 0 ]; then
        echo "  WARN: no user IDs parsed — format may have changed"
    fi
fi

echo
echo "=== Open transcript file handles on running bot ==="
BOT_PID=$(pgrep -f "python main.py" || true)
if [ -n "$BOT_PID" ]; then
    LEAKED=$(lsof -p "$BOT_PID" 2>/dev/null | grep -c "transcripts/" || true)
    echo "  bot pid $BOT_PID has $LEAKED transcript fd(s) open"
    if [ "$LEAKED" -gt 0 ]; then
        echo "  (expected 0 after /stop — investigate if non-zero)"
    fi
else
    echo "  bot not running (skipped)"
fi

echo
if [ "$EXIT" -eq 0 ]; then
    echo "=== Post-session verify: PASS ==="
else
    echo "=== Post-session verify: FAIL ==="
fi
exit $EXIT
