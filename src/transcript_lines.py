"""Parse the per-session transcript files (``transcripts/<ts>.txt``).

Shared by the bot (``/generate_pdf``) and the dashboard
(``web/transcript_service`` re-exports these), so both read the file the
same way.
"""

import re

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


def parse_transcript_text(raw):
    """Parse a whole .txt body into entries (blank lines dropped).

    Single place the line-parsing happens so the file viewer, the live
    view, get_transcript_entries and the PDF export all behave identically
    and each only reads the file once.
    """
    return [e for e in (parse_transcript_line(ln) for ln in raw.splitlines()) if e]
