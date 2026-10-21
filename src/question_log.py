"""Question history (discord-transcript-bot-jmj).

Every /ask, /ask-public and dashboard question is appended to
``history/questions.jsonl``: one JSON object per line with who asked,
where, the question, the answer or error, cost, the notes the model
read, latency, and the session it was asked about (group, game, campaign,
session day, the recordings it covered).

Append-only and shared by two processes (bot and dashboard), so each write
takes an exclusive ``flock`` and writes one whole line. Recording must
never break /ask: every failure here is logged and swallowed. The file is
local-only (gitignored): it holds answers that may quote GM notes.
"""

import fcntl
import json
import logging
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_FILE = PROJECT_ROOT / "history" / "questions.jsonl"

# status values
ANSWERED = "answered"  # the model answered
FAILED = "failed"  # the API call failed or the model gave no usable answer
REJECTED = "rejected"  # never sent: no transcript yet, no API key, ...

SOURCES = ("discord", "discord-public", "dashboard")


def history_file() -> Path:
    override = os.getenv("QUESTION_LOG_FILE", "").strip()
    return Path(override) if override else DEFAULT_FILE


def _session_fields(transcript_path) -> dict:
    if not transcript_path:
        return {"session_key": None, "files": []}
    from src import library, sessions

    meta = sessions.read_meta(transcript_path)
    files = sessions.session_files(transcript_path)
    d = sessions.describe(library.load(), meta, len(files))
    return {
        "session_key": d.get("key"),
        "group": d.get("group_id"),
        "game": d.get("game_id"),
        "campaign": d.get("campaign_id"),
        "session_day": d.get("session_day"),
        "files": [Path(f).name for f in files],
    }


def record(
    *,
    source: str,
    question: str,
    status: str,
    answer: Optional[str] = None,
    error: Optional[str] = None,
    result=None,
    asker: Optional[str] = None,
    asker_id=None,
    guild_id=None,
    transcript_path=None,
    latency_s: Optional[float] = None,
) -> Optional[dict]:
    """Append one question to the history. Returns the entry, or None if it
    could not be written (logged, never raised)."""
    try:
        entry = {
            "id": uuid.uuid4().hex[:12],
            "ts": datetime.now().isoformat(timespec="seconds"),
            "source": source,
            "asker": asker,
            "asker_id": str(asker_id) if asker_id is not None else None,
            "guild_id": str(guild_id) if guild_id is not None else None,
            "question": question,
            "status": status,
            "answer": answer,
            "error": error,
            "model": getattr(result, "model", None),
            "input_tokens": getattr(result, "input_tokens", None),
            "cached_tokens": getattr(result, "cached_tokens", None),
            "output_tokens": getattr(result, "output_tokens", None),
            "rounds": getattr(result, "rounds", None),
            "notes_used": list(getattr(result, "notes_used", None) or []),
            "latency_s": round(latency_s, 2) if latency_s is not None else None,
        }
        try:
            entry.update(_session_fields(transcript_path))
        except Exception as e:  # noqa: BLE001 - tags are best-effort
            logger.warning("question log: could not read session tags: %s", e)
        path = history_file()
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(entry, ensure_ascii=False) + "\n"
        with open(path, "a", encoding="utf-8") as fh:
            fcntl.flock(fh, fcntl.LOCK_EX)
            try:
                fh.write(line)
                fh.flush()
            finally:
                fcntl.flock(fh, fcntl.LOCK_UN)
        return entry
    except Exception as e:  # noqa: BLE001 - history must never break /ask
        logger.warning("question log: could not record question: %s", e)
        return None


def read(
    *,
    session_key: Optional[str] = None,
    campaign: Optional[str] = None,
    source: Optional[str] = None,
    status: Optional[str] = None,
    limit: Optional[int] = None,
) -> list:
    """History entries, newest first, optionally filtered. Unreadable lines
    are skipped."""
    path = history_file()
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except FileNotFoundError:
        return []
    except OSError as e:
        logger.warning("question log: cannot read %s: %s", path, e)
        return []
    out = []
    for line in reversed(lines):
        try:
            e = json.loads(line)
        except ValueError:
            continue
        if not isinstance(e, dict):
            continue
        if session_key and e.get("session_key") != session_key:
            continue
        if campaign and e.get("campaign") != campaign:
            continue
        if source and e.get("source") != source:
            continue
        if status and e.get("status") != status:
            continue
        out.append(e)
        if limit and len(out) >= limit:
            break
    return out
