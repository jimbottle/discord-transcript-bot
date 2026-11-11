"""Cached recaps of finished sessions (discord-transcript-bot-fkj).

`/ask` offers each earlier session of the campaign as a whole transcript
(``sessions.earlier_session_notes``), tens of thousands of tokens a night,
and its index line says only the date, so the model has to read a night to
learn what happened in it. A recap is written once per session, kept beside
the transcripts, and offered as a short note (``recap-YYYY-MM-DD``) whose
index line lists the night's topics; the transcript stays available as the
second step for exact detail.

Storage: ``transcripts/<first recording's stamp>.recap.json`` with the
recap, its topics line, the model that wrote it and a fingerprint (name and
size of every recording of the session). A recap whose fingerprint no
longer matches (a recording was added, deleted, retagged or grew) is
ignored, never served stale. Reading never raises; :func:`generate` is the
one function that calls a model, and nothing calls it on its own
(``scripts/recap_sessions.py`` does).
"""

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

from src import sessions
from src.llm.campaign import RECAP_PREFIX, Note

logger = logging.getLogger(__name__)

SCHEMA = 1
SUFFIX = ".recap.json"
# The topics ride along with every question (the index), so they stay short.
TOPICS_MAX_CHARS = 300
TOPICS_LABEL = "TOPICS:"

QUESTION = (
    "Write a recap of this whole session for players who will ask about it "
    "weeks from now. First line: 'TOPICS: ' followed by the people, places, "
    "factions, items and events that came up, comma-separated, most "
    "important first, at most 25 entries. Then a blank line and the recap: "
    "what happened in order, with the time of each major event, the "
    "decisions the party made, who they dealt with, what they gained or "
    "paid, and what was left unresolved. Plain facts only, about 300 to 500 "
    "words, no introduction."
)


class RecapError(Exception):
    """A recap could not be made (nothing transcribed, or an empty answer)."""


@dataclass
class Recap:
    text: str
    topics: str = ""
    model: str = ""
    generated_at: str = ""


def recap_path(files) -> Optional[Path]:
    """Where the recap of the session made of ``files`` is kept."""
    files = sorted(Path(f) for f in files)
    if not files:
        return None
    return files[0].with_name(files[0].stem + SUFFIX)


def fingerprint(files) -> list:
    out = []
    for f in sorted(Path(f) for f in files):
        try:
            out.append([f.name, f.stat().st_size])
        except OSError:
            out.append([f.name, -1])
    return out


def load(files) -> Optional[Recap]:
    """The session's cached recap, or None when there is none, it cannot be
    read, or the recordings changed since it was written."""
    path = recap_path(files)
    if path is None:
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict) or data.get("files") != fingerprint(files):
        return None
    text = str(data.get("recap") or "").strip()
    if not text:
        return None
    return Recap(
        text=text,
        topics=str(data.get("topics") or "").strip(),
        model=str(data.get("model") or ""),
        generated_at=str(data.get("generated_at") or ""),
    )


def save(files, recap: Recap) -> Path:
    path = recap_path(files)
    payload = {
        "schema": SCHEMA,
        "files": fingerprint(files),
        "generated_at": recap.generated_at,
        "model": recap.model,
        "topics": recap.topics,
        "recap": recap.text,
    }
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    return path


def parse(answer: str):
    """``(topics, recap text)`` from the model's answer. The TOPICS line is
    optional: a model that leaves it out still yields a usable recap."""
    lines = (answer or "").strip().splitlines()
    topics = ""
    for i, line in enumerate(lines[:3]):
        bare = line.strip().lstrip("*# ").replace("**", "")
        if bare.upper().startswith(TOPICS_LABEL):
            topics = " ".join(bare[len(TOPICS_LABEL) :].split())
            del lines[i]
            break
    if len(topics) > TOPICS_MAX_CHARS:
        cut = topics.rfind(",", 0, TOPICS_MAX_CHARS)
        topics = topics[: cut if cut > 0 else TOPICS_MAX_CHARS].rstrip(", ")
    return topics, "\n".join(lines).strip()


def generate(files, *, header: str = "", ask=None) -> Recap:
    """Write and cache the recap of the session made of ``files``. One
    blocking model call through ``provider.ask`` (so the GM check and the
    backend's limits apply); raises its ``AskFailed``/``AskBlocked``, or
    :class:`RecapError`. ``header`` is the session's ``SESSION:`` line; its
    ``STYLE:`` line is dropped, because a recap is reference material."""
    if ask is None:
        from src.llm import provider

        ask = provider.ask
    body = sessions.model_text(sessions.join(files)).strip()
    if not body:
        raise RecapError("nothing was transcribed in this session")
    plain = "\n".join(
        line for line in header.splitlines() if not line.startswith("STYLE:")
    )
    result = ask(body, QUESTION, header=plain)
    topics, text = parse(result.answer)
    if not text:
        raise RecapError("the model returned an empty recap")
    recap = Recap(
        text=text,
        topics=topics,
        model=result.model,
        generated_at=datetime.now().isoformat(timespec="seconds"),
    )
    save(files, recap)
    return recap


def note(day: str, nice: str, recap: Recap) -> Note:
    """The recap as an on-demand note, listed ahead of the transcript."""
    return Note(
        id=RECAP_PREFIX + day,
        title=f"Recap of the session on {nice}",
        summary=(
            "short recap of that session"
            + (f"; topics: {recap.topics}" if recap.topics else "")
        ),
        body=(
            recap.text
            + f"\n\n(A recap, not the full record. For exact words or a detail "
            f"it leaves out, read {sessions.EARLIER_PREFIX}{day}.)"
        ),
    )
