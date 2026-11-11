"""Session tagging: which group, game, campaign and game night a recording
belongs to (discord-transcript-bot-157).

Each transcript ``transcripts/<ts>.txt`` gets a sidecar
``transcripts/<ts>.meta.json`` written when /scribe starts. Recordings that
share a campaign and a *session day* are one session, so restarting the
bot mid-game splits the transcript into files but not the session: /ask
and the dashboard read them back as one.

The session day is the local date with a cutoff (``SESSION_DAY_CUTOFF_HOUR``,
default 6): a restart at 00:40 still belongs to the night it started.
"""

import json
import logging
import os
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

SCHEMA = 1
_STAMP = re.compile(r"^(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})$")
RESTART_MARK = "--- recording restarted at {time} ---"


def cutoff_hour() -> int:
    try:
        return max(0, min(23, int(os.getenv("SESSION_DAY_CUTOFF_HOUR", "6"))))
    except ValueError:
        return 6


def session_day(when: datetime) -> str:
    """Local date of the game night ``when`` falls in (cutoff-aware)."""
    if when.hour < cutoff_hour():
        when = when - timedelta(days=1)
    return when.strftime("%Y-%m-%d")


def started_at_from_name(path) -> Optional[datetime]:
    m = _STAMP.match(Path(path).stem)
    if not m:
        return None
    return datetime.strptime(m.group(1), "%Y-%m-%d_%H-%M-%S")


@dataclass
class SessionMeta:
    session_day: str
    group: Optional[str] = None
    game: Optional[str] = None
    campaign: Optional[str] = None
    guild_id: Optional[int] = None
    guild: Optional[str] = None
    channel_id: Optional[int] = None
    channel: Optional[str] = None
    started_at: Optional[str] = None
    # "auto" = resolved from library.yml at /scribe; "manual" = set by
    # scripts/tag_session.py. Informational only.
    source: str = "auto"
    schema: int = SCHEMA

    @property
    def key(self) -> str:
        """Recordings with equal keys are one session."""
        owner = self.campaign or (
            f"guild-{self.guild_id}" if self.guild_id else "untagged"
        )
        return f"{owner}/{self.session_day}"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "SessionMeta":
        known = {k: data.get(k) for k in cls.__dataclass_fields__ if k in data}
        if not known.get("session_day"):
            raise ValueError("session_day missing")
        return cls(**known)


def meta_path(transcript_path) -> Path:
    p = Path(transcript_path)
    return p.with_name(p.stem + ".meta.json")


def build_meta(
    lib,
    *,
    guild_id=None,
    guild=None,
    channel_id=None,
    channel=None,
    started_at: Optional[datetime] = None,
    group=None,
    campaign=None,
    source="auto",
) -> SessionMeta:
    """Tag a recording. ``group``/``campaign`` override the library's
    resolution (used by the retag script)."""
    started_at = started_at or datetime.now()
    if campaign and not group:
        c = lib.campaigns.get(campaign)
        group = c.group if c else None
    if not (group or campaign):
        group, campaign = lib.resolve(guild_id, channel_id)
    return SessionMeta(
        session_day=session_day(started_at),
        group=group,
        game=lib.game_for(campaign),
        campaign=campaign,
        guild_id=int(guild_id) if guild_id is not None else None,
        guild=guild,
        channel_id=int(channel_id) if channel_id is not None else None,
        channel=channel,
        started_at=started_at.isoformat(timespec="seconds"),
        source=source,
    )


def default_tag(lib, guild_id=None, channel_id=None, transcripts_dir=None):
    """``(group_id, campaign_id)`` a new recording should start with
    (discord-transcript-bot-5bd): the campaign most recently used in this
    Discord server, so picking a campaign once (dashboard, retag) carries
    forward, else the library's binding (the group's ``active_campaign``).

    "Most recently used" is the newest recording here with a campaign the
    library still has. It must be this table's: when the library binds the
    server or channel to a group, another group's campaign is skipped (two
    tables can share a server by channel); when it binds nothing, only a
    recording made in the same voice channel counts. Never raises."""
    group, campaign = lib.resolve(guild_id, channel_id)
    if guild_id is None or transcripts_dir is None:
        return group, campaign
    try:
        recordings = sorted(Path(transcripts_dir).glob("*.txt"), reverse=True)
    except OSError:
        return group, campaign
    for txt in recordings:
        m = read_meta(txt)
        if m is None or m.guild_id != int(guild_id) or not m.campaign:
            continue
        c = lib.campaigns.get(m.campaign)
        if c is None:
            continue
        owner = c.group or m.group
        if group:
            if owner and owner != group:
                continue
            # No group on the campaign or the recording: go by where it
            # was recorded, so a table bound to the server does not pick
            # up one played in another table's channel (roborev #7125).
            if not owner and lib.resolve(guild_id, m.channel_id)[0] != group:
                continue
        elif channel_id is None or m.channel_id != int(channel_id):
            continue
        return owner or group, m.campaign
    return group, campaign


def write_meta(transcript_path, meta: SessionMeta) -> Path:
    path = meta_path(transcript_path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(meta.to_dict(), indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    return path


def read_meta(transcript_path) -> Optional[SessionMeta]:
    path = meta_path(transcript_path)
    try:
        return SessionMeta.from_dict(json.loads(path.read_text(encoding="utf-8")))
    except FileNotFoundError:
        return None
    except (OSError, ValueError, TypeError) as e:
        logger.warning("unreadable session metadata %s: %s", path, e)
        return None


def session_files(transcript_path) -> list:
    """Every transcript of the same session, oldest first. An untagged
    transcript is a session of one."""
    target = Path(transcript_path)
    meta = read_meta(target)
    if meta is None:
        return [target]
    files = []
    for txt in sorted(target.parent.glob("*.txt")):
        other = read_meta(txt)
        if other is not None and other.key == meta.key:
            files.append(txt)
    return files or [target]


def latest_for_guild(guild_id, transcripts_dir) -> Optional[Path]:
    """The newest recording tagged with ``guild_id`` in ``transcripts_dir``,
    or None. Recording names are timestamps, so name order is time order.
    Lets /generate_pdf find the last session after /stop, /disconnect or a
    bot restart, when no sink is left to ask."""
    d = Path(transcripts_dir)
    if guild_id is None or not d.is_dir():
        return None
    for txt in sorted(d.glob("*.txt"), reverse=True):
        meta = read_meta(txt)
        if meta is not None and meta.guild_id == int(guild_id):
            return txt
    return None


def join(files) -> str:
    """Recordings joined oldest first with a restart marker between them."""
    parts = []
    for i, f in enumerate(files):
        try:
            body = f.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if not body.strip():
            continue
        if i and parts:
            start = started_at_from_name(f)
            parts.append(
                RESTART_MARK.format(
                    time=start.strftime("%H:%M:%S") if start else f.stem
                )
            )
        parts.append(body.rstrip("\n"))
    return "\n".join(parts) + ("\n" if parts else "")


def session_transcript(transcript_path):
    """``(text, meta, files)``: all of a session's recordings joined in
    order with a restart marker between them. ``meta`` is the target
    file's tag (None when untagged)."""
    files = session_files(transcript_path)
    return join(files), read_meta(transcript_path), files


def describe(lib, meta: Optional[SessionMeta], recordings: int = 1) -> dict:
    """Display fields for one session (names resolved from the library)."""
    if meta is None:
        return {"tagged": False, "recordings": recordings}
    group = lib.groups.get(meta.group) if meta.group else None
    return {
        "tagged": True,
        "key": meta.key,
        "session_day": meta.session_day,
        "game": lib.name("game", meta.game),
        "game_id": meta.game,
        "campaign": lib.name("campaign", meta.campaign),
        "campaign_id": meta.campaign,
        "group": lib.name("group", meta.group),
        "group_id": meta.group,
        "gm": group.gm if group else None,
        "channel": meta.channel,
        "recordings": recordings,
    }


def scribe_line(lib, meta: Optional[SessionMeta]) -> str:
    """What /scribe adds to its reply about the tag (5bd). The campaign
    decides which notes and earlier sessions /ask can read, so a wrong or
    missing one should be seen when recording starts, not at the first
    question."""
    d = describe(lib, meta)
    if not d["tagged"] or not d.get("campaign"):
        return (
            "\nNo campaign is set for this recording, so questions get no "
            "campaign notes or earlier sessions. Pick one on the dashboard's "
            "Live page."
        )
    game = f" ({d['game']})" if d.get("game") else ""
    return (
        f"\nCampaign: **{d['campaign']}**{game}. If that's wrong, change it on "
        "the dashboard's Live page."
    )


def header_text(lib, meta: Optional[SessionMeta], recordings: int = 1) -> str:
    """One line of session context for the /ask prompt; '' when untagged."""
    d = describe(lib, meta, recordings)
    if not d["tagged"]:
        return ""
    bits = []
    if d["game"]:
        bits.append(f"Game: {d['game']}")
    if d["campaign"]:
        bits.append(f"Campaign: {d['campaign']}")
    if d["group"]:
        bits.append(f"Group: {d['group']}" + (f" (GM: {d['gm']})" if d["gm"] else ""))
    bits.append(f"Session: {d['session_day']}")
    if recordings > 1:
        bits.append(
            f"{recordings} recordings joined (the bot was restarted mid-session)"
        )
    line = "SESSION: " + " · ".join(bits)
    game = lib.games.get(meta.game) if meta.game else None
    if game and game.voice:
        line += "\nSTYLE: " + game.voice
    return line


# "[8:26 PM] Player (Character) [254468368969629697]: text": the Discord id
# is noise to a model and, measured on the 2026-10-20 session, 44% of a
# line's tokens (12,000 characters: 5,805 tokens with ids, 3,225 without).
_SPEAKER_ID = re.compile(r"^(\[[^\]\n]+\] [^\n]*?\([^)\n]*\)) \[\d+\]:", re.M)


def model_text(transcript: str) -> str:
    """The transcript as a model reads it: 12-hour times and no Discord ids.
    The files keep both; only what is sent changes."""
    from src.utils.timefmt import for_model

    return _SPEAKER_ID.sub(r"\1:", for_model(transcript))


def ask_context(transcript_path, lib=None):
    """Everything /ask needs for the session ``transcript_path`` belongs to:
    ``(text, header, note_dirs, files)``."""
    if lib is None:
        from src import library

        lib = library.load()
    text, meta, files = session_transcript(transcript_path)
    # The model sees (and quotes) 12-hour times and no Discord ids; the files
    # keep 24-hour stamps and ids.
    text = model_text(text)
    if meta is None:
        return text, "", [], files
    header = header_text(lib, meta, len(files))
    dirs = lib.note_dirs(game=meta.game, group=meta.group, campaign=meta.campaign)
    return text, header, dirs, files


# ── earlier sessions (discord-transcript-bot-ij0) ─────────────────────

EARLIER_PREFIX = "session-"
# One earlier session is read whole when the model asks for it. The cap
# (characters, most recent kept) only guards against a runaway file.
EARLIER_MAX_CHARS_ENV = "EARLIER_SESSION_MAX_CHARS"
EARLIER_MAX_CHARS = 400_000
EARLIER_TRIM_MARK = "[... the start of this session is not included ...]"
_SPEAKER = re.compile(r"^\[[^\]\n]+\] ([^\[\n:]+?)(?: \[\d+\])?:", re.M)


def earlier_max_chars() -> int:
    try:
        return max(0, int(os.getenv(EARLIER_MAX_CHARS_ENV, "") or EARLIER_MAX_CHARS))
    except ValueError:
        return EARLIER_MAX_CHARS


def _same_story(a: SessionMeta, b: SessionMeta) -> bool:
    """Whether two recordings belong to the same ongoing game. The same
    campaign when both name one; otherwise the same game *and* the same
    group, so a recording tagged without a campaign still connects but
    never to another table that happens to play the same game: one bot
    serves several groups, and an earlier transcript is that group's."""
    if a.campaign and b.campaign:
        return a.campaign == b.campaign
    return bool(a.game and a.group) and a.game == b.game and a.group == b.group


def earlier_sessions(transcript_path) -> list:
    """``[(session_day, [files])]`` for every session before the one
    ``transcript_path`` belongs to, in the same campaign (or game), oldest
    first. Empty for an untagged recording."""
    target = Path(transcript_path)
    meta = read_meta(target)
    if meta is None:
        return []
    by_day = {}
    for txt in sorted(target.parent.glob("*.txt")):
        other = read_meta(txt)
        if (
            other is not None
            and other.session_day < meta.session_day
            and _same_story(meta, other)
        ):
            by_day.setdefault(other.session_day, []).append(txt)
    return sorted(by_day.items())


def earlier_session_notes(transcript_path) -> list:
    """Each earlier session as a note the model can read on demand through
    ``read_campaign_notes`` (id ``session-YYYY-MM-DD``, body = that night's
    transcript as the model reads it). Only the one-line index entry rides
    along with a question. Never raises: /ask must work without them."""
    from src.llm.campaign import Note

    notes = []
    try:
        sessions_ = earlier_sessions(transcript_path)
    except OSError as e:
        logger.warning("could not list earlier sessions: %s", e)
        return []
    cap = earlier_max_chars()
    for day, files in sessions_:
        body = model_text(join(files)).strip()
        if not body:
            continue
        lines = body.count("\n") + 1
        # Recordings from before the roster carry bare Discord ids: not names.
        speakers = dedupe_keep_order(
            s for s in _SPEAKER.findall(body) if not s.strip().isdigit()
        )[:8]
        if cap and len(body) > cap:
            cut = body.find("\n", len(body) - cap)
            body = EARLIER_TRIM_MARK + "\n" + body[cut + 1 :]
        try:
            nice = datetime.strptime(day, "%Y-%m-%d").strftime("%A, %B %-d, %Y")
        except ValueError:
            nice = day
        notes.append(
            Note(
                id=EARLIER_PREFIX + day,
                title=f"Earlier session, {nice}",
                summary=(
                    f"full transcript of the session played on {nice} "
                    f"(about {lines} lines"
                    + (f"; speakers: {', '.join(speakers)}" if speakers else "")
                    + ")"
                ),
                body=body,
                keywords=["earlier session", "last time", "previously"],
            )
        )
    return notes


def dedupe_keep_order(items) -> list:
    seen = set()
    return [i for i in items if not (i in seen or seen.add(i))]


def campaign_header(lib, campaign_id: str) -> str:
    """CAMPAIGN line (+ STYLE) for a notes-only question; '' if unknown."""
    c = lib.campaigns.get(campaign_id)
    if c is None:
        return ""
    group = lib.groups.get(c.group) if c.group else None
    bits = []
    if c.game:
        bits.append(f"Game: {lib.name('game', c.game)}")
    bits.append(f"Campaign: {c.name}")
    if group:
        bits.append(f"Group: {group.name}" + (f" (GM: {group.gm})" if group.gm else ""))
    line = "CAMPAIGN: " + " · ".join(bits)
    game = lib.games.get(c.game) if c.game else None
    if game and game.voice:
        line += "\nSTYLE: " + game.voice
    return line


class TagError(ValueError):
    """A retag request names an id the library does not have."""


def retag(
    transcript_path, lib, *, campaign=None, group=None, game=None, whole_session=True
):
    """Set a recording's (or its whole session's) campaign/group/game.

    Ids are validated against the library; ``None``/"" clears a field.
    Choosing a campaign fills in its group and game unless those are given
    explicitly; a campaign that belongs to no group leaves each recording's
    group as it was (the Live picker sends only the campaign, and losing
    the group would drop its notes and GM from /ask, roborev #7125).
    Discord fields and the session day are kept; ``source``
    becomes "manual". Returns the rewritten paths (oldest first)."""
    for kind, value, table in (
        ("campaign", campaign, lib.campaigns),
        ("group", group, lib.groups),
        ("game", game, lib.games),
    ):
        if value and value not in table:
            raise TagError(f"unknown {kind} '{value}'")
    keep_group = False
    if campaign:
        c = lib.campaigns[campaign]
        keep_group = group is None and not c.group
        group = group if group is not None else c.group
        game = game if game is not None else c.game
    targets = (
        session_files(transcript_path) if whole_session else [Path(transcript_path)]
    )
    written = []
    for txt in targets:
        old = read_meta(txt)
        started = started_at_from_name(txt)
        if old is not None:
            meta = old
        else:
            if started is None:
                raise TagError(f"{Path(txt).name} has no timestamp to date it by")
            meta = SessionMeta(
                session_day=session_day(started),
                started_at=started.isoformat(timespec="seconds"),
            )
        meta.campaign = campaign or None
        if not keep_group:
            meta.group = group or None
        meta.game = game or None
        meta.source = "manual"
        write_meta(txt, meta)
        written.append(Path(txt))
    return written
