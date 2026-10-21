"""Reference notes for `/ask` (discord-transcript-bot-kda, -157).

A tiny retrieval layer: Markdown notes with YAML front matter, gathered
from the session's game, group and campaign folders (src/library.py).
Only the *index* (one line per note) rides along with every question; the
body of a note is sent only when the model asks for it by id through the
``read_campaign_notes`` tool. See library/README.md.

Pure file reading, no network. Notes are re-read on every call: they are a
few kilobytes and the GM edits them between sessions.
"""

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

import yaml

logger = logging.getLogger(__name__)

# Notes marked ``audience: gm`` hold spoilers. Players use /ask, so they are
# left out of the index unless the operator opts in on the bot host.
GM_NOTES_ENV = "CAMPAIGN_GM_NOTES"
AUDIENCES = ("table", "gm")

TOOL_NAME = "read_campaign_notes"
TOOL = {
    "name": TOOL_NAME,
    "description": (
        "Read campaign reference notes by id. The CAMPAIGN NOTES INDEX in the "
        "system prompt lists every note with a one-line summary and keywords. "
        "Call this only when the question needs background the transcript does "
        "not contain (who a named person, place, gang or corporation is; what "
        "happened in an earlier session) and request only the notes whose index "
        "line is relevant. Do not call it for questions the transcript answers."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "ids": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1,
                "description": 'Note ids from the index, e.g. ["npcs"].',
            }
        },
        "required": ["ids"],
        "additionalProperties": False,
    },
    "strict": True,
}


@dataclass
class Note:
    id: str
    title: str
    summary: str
    body: str
    keywords: list = field(default_factory=list)
    audience: str = "table"
    path: Optional[Path] = None

    def index_line(self) -> str:
        kw = f" [{', '.join(self.keywords)}]" if self.keywords else ""
        return f"- {self.id} — {self.title}: {self.summary}{kw}"


class NoteError(ValueError):
    """A note file is malformed (bad front matter)."""


def gm_notes_enabled() -> bool:
    return os.getenv(GM_NOTES_ENV, "").strip().lower() in ("1", "true", "yes", "on")


def parse_note(text: str, path: Optional[Path] = None) -> Note:
    """Split ``---`` front matter from the body and validate the header."""
    where = str(path) if path else "<text>"
    if not text.startswith("---"):
        raise NoteError(f"{where}: missing front matter")
    try:
        _, header, body = text.split("---", 2)
    except ValueError as e:
        raise NoteError(f"{where}: unterminated front matter") from e
    try:
        meta = yaml.safe_load(header) or {}
    except yaml.YAMLError as e:
        raise NoteError(f"{where}: bad front matter YAML: {e}") from e
    if not isinstance(meta, dict):
        raise NoteError(f"{where}: front matter is not a mapping")
    for key in ("id", "title", "summary"):
        if not str(meta.get(key) or "").strip():
            raise NoteError(f"{where}: front matter needs a non-empty '{key}'")
    note_id = str(meta["id"]).strip()
    if path is not None and path.stem != note_id:
        raise NoteError(f"{where}: id '{note_id}' must match the file name")
    audience = str(meta.get("audience") or "table").strip().lower()
    if audience not in AUDIENCES:
        raise NoteError(f"{where}: audience must be one of {AUDIENCES}")
    keywords = meta.get("keywords") or []
    if isinstance(keywords, str):
        keywords = [k.strip() for k in keywords.split(",") if k.strip()]
    return Note(
        id=note_id,
        title=str(meta["title"]).strip(),
        summary=str(meta["summary"]).strip(),
        body=body.strip(),
        keywords=[str(k) for k in keywords],
        audience=audience,
        path=path,
    )


def load_notes(dirs: Iterable = (), include_gm: Optional[bool] = None) -> list:
    """Every parseable note in ``dirs`` (general to specific), sorted by id.
    A note in a later folder replaces an earlier one with the same id, so a
    campaign can override a game-level note. A broken note is logged and
    skipped so one bad header never disables /ask. ``include_gm`` defaults
    to the CAMPAIGN_GM_NOTES setting."""
    if include_gm is None:
        include_gm = gm_notes_enabled()
    by_id = {}
    for directory in dirs:
        directory = Path(directory)
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob("*.md")):
            try:
                note = parse_note(path.read_text(encoding="utf-8"), path)
            except (NoteError, OSError) as e:
                logger.warning("campaign note skipped: %s", e)
                continue
            if note.audience == "gm" and not include_gm:
                continue
            if note.id in by_id:
                logger.debug("note %s overridden by %s", by_id[note.id].path, path)
            by_id[note.id] = note
    return [by_id[k] for k in sorted(by_id)]


def index_text(notes: Iterable[Note]) -> str:
    """The compact listing that goes in the system prompt; '' when empty."""
    lines = [n.index_line() for n in notes]
    if not lines:
        return ""
    return (
        "CAMPAIGN NOTES INDEX (background reference; read a note with "
        f"{TOOL_NAME} only when the question needs it):\n" + "\n".join(lines)
    )


def read_notes(ids: Iterable[str], notes: Iterable[Note]):
    """Tool implementation. Returns ``(text, is_error)``: the requested
    notes' bodies, or an error naming the unknown ids. A partially valid
    request still returns what exists, so one typo does not waste a round."""
    by_id = {n.id: n for n in notes}
    wanted = [str(i).strip() for i in ids if str(i).strip()]
    found = [by_id[i] for i in wanted if i in by_id]
    missing = [i for i in wanted if i not in by_id]
    parts = [f"## {n.title} (id: {n.id})\n{n.body}" for n in found]
    if missing:
        parts.append(
            "Unknown note id(s): "
            + ", ".join(missing)
            + ". Available: "
            + ", ".join(sorted(by_id))
            + "."
        )
    return "\n\n".join(parts), (not found)
