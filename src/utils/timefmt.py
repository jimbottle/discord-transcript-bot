"""12-hour clock times for anything people read.

Transcript files and logs keep their 24-hour stamps (the parsers and the
session tagging depend on them); everything shown to players, in Discord,
on the dashboard, in a PDF or in text the model quotes back, goes through
here as "8:26 PM".
"""

import re
from datetime import datetime

_HMS = re.compile(r"^(\d{1,2}):(\d{2})(?::(\d{2})(?:\.\d+)?)?$")
# "[20:26:09] " at the start of a transcript line, and the restart marker.
_LINE_STAMP = re.compile(r"(?m)^\[(\d{2}:\d{2}(?::\d{2})?)\]")
_RESTART = re.compile(r"(--- recording restarted at )(\d{2}:\d{2}(?::\d{2})?)( ---)")


def clock(value) -> str:
    """``"20:26:09"`` / ``"20:26"`` / ``"19:38:48.181"`` -> ``"8:26 PM"``.
    Anything unparseable is returned unchanged (never raises)."""
    if value is None:
        return ""
    m = _HMS.match(str(value).strip())
    if not m:
        return str(value)
    hour, minute = int(m.group(1)), int(m.group(2))
    if hour > 23 or minute > 59:
        return str(value)
    suffix = "AM" if hour < 12 else "PM"
    return f"{hour % 12 or 12}:{minute:02d} {suffix}"


def when(iso) -> str:
    """ISO timestamp ``"2026-10-20T21:23:02"`` -> ``"Oct 20, 9:23 PM"``."""
    if not iso:
        return ""
    try:
        dt = datetime.fromisoformat(str(iso))
    except ValueError:
        return str(iso)
    return f"{dt.strftime('%b')} {dt.day}, {clock(dt.strftime('%H:%M'))}"


def for_model(transcript: str) -> str:
    """Rewrite a transcript's line stamps and restart markers to 12-hour
    time, so the model reads (and quotes back) "8:26 PM", not "20:26:09"."""
    text = _LINE_STAMP.sub(lambda m: f"[{clock(m.group(1))}]", transcript)
    return _RESTART.sub(lambda m: m.group(1) + clock(m.group(2)) + m.group(3), text)
