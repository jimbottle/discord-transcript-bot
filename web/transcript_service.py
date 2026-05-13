import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRANSCRIPTS_DIR = PROJECT_ROOT / "transcripts"
LOGS_DIR = PROJECT_ROOT / ".logs" / "transcripts"


def list_transcripts():
    """Return transcript .txt files sorted newest first."""
    if not TRANSCRIPTS_DIR.is_dir():
        return []
    files = sorted(TRANSCRIPTS_DIR.glob("*.txt"), reverse=True)
    results = []
    for f in files:
        results.append({
            "filename": f.name,
            "size": f.stat().st_size,
            "modified": f.stat().st_mtime,
        })
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
                results.append({
                    "source": log_file.name,
                    "player": entry.get("player"),
                    "character": entry.get("character"),
                    "date": entry.get("date"),
                    "time": entry.get("begin"),
                    "text": data,
                })
    return results
