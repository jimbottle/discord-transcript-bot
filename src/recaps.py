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

SUMMARY_LABEL = "SUMMARY:"
RECAP_LABEL = "RECAP:"
# The summary is posted to Discord as ONE message (2000 characters with its
# title). Models overshoot a stated count by up to a third, so the target
# is well under the ceiling; session_summary trims anything still over.
SUMMARY_TARGET_CHARS = 1200
SUMMARY_MAX_CHARS = 1700

QUESTION = (
    "Write up this whole session in three parts, in exactly this layout.\n\n"
    "TOPICS: the people, places, factions, items and events that came up, "
    "comma-separated, most important first, at most 25 entries.\n\n"
    "SUMMARY:\n"
    "For the players, posted to the table's Discord as a single message: aim "
    f"for about {SUMMARY_TARGET_CHARS} characters and never more than "
    f"{SUMMARY_MAX_CHARS}. Only what happened in the game world: what the "
    "characters did, the choices they made, who they dealt with, fights, "
    "deals and discoveries, what they gained or lost, and what happened as "
    "a result. Use character names. Leave out table talk, rules "
    "and dice discussion, clock times, and anything said out of character. "
    "Two or three short paragraphs of prose, no headings, no bullet lists. "
    "Follow the SUMMARY RULES given with the session, and write it in the "
    "SUMMARY STYLE given there, if there is one.\n\n"
    "RECAP:\n"
    "A reference record for answering questions weeks from now, in plain "
    "language with no style: what happened in order, with the time of each "
    "major event, the decisions the party made, who they dealt with, what "
    "they gained or paid, and what was left unresolved. About 300 to 500 "
    "words, no introduction."
)

# What the SUMMARY may contain, whatever the game. Sent in the system
# prompt with the session, ahead of the style. The players read this at the
# table: it retells the night and must not steer the next one (owner's
# direction, 2026-11-11).
SUMMARY_RULES = (
    "SUMMARY RULES: The SUMMARY is a retelling of what happened in this "
    "session and nothing else. Tell what the characters did and what "
    "happened in response. Do not suggest, hint at or list what the party "
    "should or could do next, and do not point out open threads, unfinished "
    "business or opportunities. Do not analyse, explain or evaluate: no "
    "judging whether a choice was wise, no lessons, no commentary on how "
    "well things went, no speculation about why anyone acted or what an "
    "NPC intends, wants or is planning. Report only actions, events and "
    "words as they occurred, including plans and intentions only where a "
    "character stated them aloud. When something went especially badly, "
    "say that it happened, with relish if the style allows, but without "
    "assessing it."
)

# How the SUMMARY should sound, per game: sent in the system prompt, under
# the SESSION line. Stronger than the light ``voice`` /ask answers use,
# because a summary is a piece of the game's fiction, not a lookup. A game
# whose ``voice`` is switched off in library.yml (``voice: ""``) gets none.
SUMMARY_STYLES = {
    "cyberpunk-red": (
        "Write the SUMMARY the way a Night City fixer or a street-level "
        "screamsheet would tell it: Cyberpunk RED slang used naturally "
        "(chooms, eddies, corpos, flatlined, preem, nova, delta), clipped, "
        "gritty and neon-noir. Every name, deal, payment and outcome stays "
        "exactly as it happened."
    ),
    "dnd-5e": (
        "Write the SUMMARY the way a tavern chronicler or travelling bard "
        "would tell the tale: high-fantasy diction, a little flourish, the "
        "party as heroes of a saga. Every name, deed, treasure and outcome "
        "stays exactly as it happened."
    ),
}


def summary_style(lib, meta) -> str:
    """The ``SUMMARY STYLE:`` line for the session's game, or ''."""
    game_id = getattr(meta, "game", None)
    game = lib.games.get(game_id) if (lib is not None and game_id) else None
    if game is None or not game.voice:
        return ""
    style = SUMMARY_STYLES.get(game_id) or (
        f"Write the SUMMARY in the voice, vocabulary and mood of {game.name}, "
        "as someone inside that world would tell it. Every name, event and "
        "outcome stays exactly as it happened."
    )
    return "SUMMARY STYLE: " + style


class RecapError(Exception):
    """A recap could not be made (nothing transcribed, or an empty answer)."""


@dataclass
class Recap:
    text: str
    topics: str = ""
    # The short, in-world, themed version posted in Discord ('' on recaps
    # written before it existed).
    summary: str = ""
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
        summary=str(data.get("summary") or "").strip(),
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
        "summary": recap.summary,
        "recap": recap.text,
    }
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    return path


def _label(line: str, label: str):
    """The text after ``label`` when ``line`` starts with it (markdown
    dressing such as ``**SUMMARY:**`` or ``## Recap`` allowed), else None."""
    bare = line.strip().lstrip("*# ").replace("**", "")
    name = label.rstrip(":")
    if not bare.upper().startswith(name):
        return None
    rest = bare[len(name) :]
    # "SUMMARY:" or a bare heading line, never a sentence that happens to
    # begin with the word ("Summary of the heist ...").
    if rest.strip() and not rest.lstrip().startswith(":"):
        return None
    return rest.lstrip(": \t")


def parse(answer: str):
    """``(topics, summary, recap text)`` from the model's answer. Every
    label is optional: an answer without them is all recap, so a model that
    ignores the layout still yields a usable reference."""
    topics = ""
    parts = {"recap": [], "summary": []}
    where = "recap"
    seen_section = False
    for i, line in enumerate((answer or "").strip().splitlines()):
        if not seen_section and not topics and i < 3:
            found = _label(line, TOPICS_LABEL)
            if found is not None:
                topics = " ".join(found.split())
                continue
        for name, label in (("summary", SUMMARY_LABEL), ("recap", RECAP_LABEL)):
            found = _label(line, label)
            if found is not None and not (name == where and seen_section):
                where, seen_section = name, True
                line = found
                break
        parts[where].append(line)
    if len(topics) > TOPICS_MAX_CHARS:
        cut = topics.rfind(",", 0, TOPICS_MAX_CHARS)
        topics = topics[: cut if cut > 0 else TOPICS_MAX_CHARS].rstrip(", ")
    return (
        topics,
        "\n".join(parts["summary"]).strip(),
        "\n".join(parts["recap"]).strip(),
    )


def generate(files, *, header: str = "", style: str = "", ask=None) -> Recap:
    """Write and cache the recap of the session made of ``files``. One
    blocking model call through ``provider.ask`` (so the GM check and the
    backend's limits apply); raises its ``AskFailed``/``AskBlocked``, or
    :class:`RecapError`. ``header`` is the session's ``SESSION:`` line. Its
    ``STYLE:`` line (the light voice of /ask answers) is replaced by
    ``SUMMARY_RULES`` and ``style`` (:func:`summary_style`), which go into
    the system prompt and govern the SUMMARY part only; the RECAP part
    stays plain."""
    if ask is None:
        from src.llm import provider

        ask = provider.ask
    body = sessions.model_text(sessions.join(files)).strip()
    if not body:
        raise RecapError("nothing was transcribed in this session")
    plain = "\n".join(
        line for line in header.splitlines() if not line.startswith("STYLE:")
    )
    plain = (plain + "\n" if plain else "") + SUMMARY_RULES
    if style:
        plain += "\n" + style
    result = ask(body, QUESTION, header=plain)
    topics, summary, text = parse(result.answer)
    if not text:
        raise RecapError("the model returned an empty recap")
    recap = Recap(
        text=text,
        topics=topics,
        summary=summary,
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
