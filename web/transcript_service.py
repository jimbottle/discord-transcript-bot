import json
import re
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRANSCRIPTS_DIR = PROJECT_ROOT / "transcripts"
LOGS_DIR = PROJECT_ROOT / ".logs" / "transcripts"

# A session .txt modified within this many seconds is treated as "live"
# (the bot writes a line per finished utterance, so a brief silence gap
# is normal — keep this comfortably above the ~1.5s speech-gap + Whisper
# latency without claiming a long-idle session is still recording).
LIVE_FRESH_SECONDS = 30

# Per-utterance line written by WhisperSink.write_transcription_log:
#   [HH:MM:SS] Player (Character) [user_id]: text
_LINE_RE = re.compile(
    r"^\[(\d{2}:\d{2}:\d{2})\]\s+(.*?)\s+\(([^)]*)\)\s+\[([^\]]+)\]:\s?(.*)$"
)


def parse_transcript_line(line):
    """Parse one session-transcript line into a structured entry.

    Lines that don't match the expected shape (older formats, manual
    edits) are returned as a raw-text entry rather than dropped, so the
    live view never silently loses content. Blank lines → None.
    """
    line = line.rstrip("\n")
    if not line.strip():
        return None
    m = _LINE_RE.match(line)
    if not m:
        return {
            "time": None,
            "player": None,
            "character": None,
            "user_id": None,
            "text": line,
        }
    t, player, character, user_id, text = m.groups()
    return {
        "time": t,
        "player": player,
        "character": character,
        "user_id": user_id,
        "text": text,
    }


def get_live_session():
    """The newest session .txt is the current/most-recent session.

    Returns its filename, parsed entries, mtime, age in seconds, and a
    `live` flag (modified within LIVE_FRESH_SECONDS). Decoupled from the
    bot process on purpose — works whether the bot was launched by the
    dashboard or externally.
    """
    empty = {
        "filename": None,
        "entries": [],
        "modified": None,
        "age_seconds": None,
        "live": False,
    }
    if not TRANSCRIPTS_DIR.is_dir():
        return empty
    files = sorted(TRANSCRIPTS_DIR.glob("*.txt"), reverse=True)
    if not files:
        return empty
    f = files[0]
    try:
        raw = f.read_text(encoding="utf-8", errors="replace")
        mtime = f.stat().st_mtime
    except OSError:
        return empty
    entries = [e for e in (parse_transcript_line(ln) for ln in raw.splitlines()) if e]
    age = time.time() - mtime
    return {
        "filename": f.name,
        "entries": entries,
        "modified": mtime,
        "age_seconds": age,
        "live": age <= LIVE_FRESH_SECONDS,
    }


def list_transcripts():
    """Return transcript .txt files sorted newest first."""
    if not TRANSCRIPTS_DIR.is_dir():
        return []
    files = sorted(TRANSCRIPTS_DIR.glob("*.txt"), reverse=True)
    results = []
    for f in files:
        results.append(
            {
                "filename": f.name,
                "size": f.stat().st_size,
                "modified": f.stat().st_mtime,
            }
        )
    return results


def get_transcript(filename):
    """Read a single transcript file. Returns None if not found or path traversal."""
    if "/" in filename or "\\" in filename or ".." in filename:
        return None
    path = TRANSCRIPTS_DIR / filename
    if not path.is_file() or path.resolve().parent != TRANSCRIPTS_DIR.resolve():
        return None
    return path.read_text(encoding="utf-8", errors="replace")


def list_logs():
    """Return .log JSON files sorted newest first."""
    if not LOGS_DIR.is_dir():
        return []
    files = sorted(LOGS_DIR.glob("*.log"), reverse=True)
    return [{"filename": f.name, "size": f.stat().st_size} for f in files]


def search_logs(query):
    """Search all JSON log files for a case-insensitive substring in the data field."""
    if not query or not LOGS_DIR.is_dir():
        return []
    query_lower = query.lower()
    results = []
    for log_file in sorted(LOGS_DIR.glob("*.log"), reverse=True):
        for line in log_file.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            data = entry.get("data", "")
            if query_lower in data.lower():
                results.append(
                    {
                        "source": log_file.name,
                        "player": entry.get("player"),
                        "character": entry.get("character"),
                        "date": entry.get("date"),
                        "time": entry.get("begin"),
                        "text": data,
                    }
                )
    return results
