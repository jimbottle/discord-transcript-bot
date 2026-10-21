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


def session_transcript(transcript_path):
    """``(text, meta, files)``: all of a session's recordings joined in
    order with a restart marker between them. ``meta`` is the target
    file's tag (None when untagged)."""
    files = session_files(transcript_path)
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
    return "\n".join(parts) + ("\n" if parts else ""), read_meta(transcript_path), files


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
    return "SESSION: " + " · ".join(bits)


def ask_context(transcript_path, lib=None):
    """Everything /ask needs for the session ``transcript_path`` belongs to:
    ``(text, header, note_dirs, files)``."""
    if lib is None:
        from src import library

        lib = library.load()
    text, meta, files = session_transcript(transcript_path)
    if meta is None:
        return text, "", [], files
    header = header_text(lib, meta, len(files))
    dirs = lib.note_dirs(game=meta.game, group=meta.group, campaign=meta.campaign)
    return text, header, dirs, files
