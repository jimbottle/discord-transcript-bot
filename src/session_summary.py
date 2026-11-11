"""Session summaries posted in Discord (discord-transcript-bot-qef).

Two ways in, one path through:

* ``/summary session_id:<id>``: the id is a random code on the session's
  tag that only the dashboard shows (``sessions.ensure_session_id``), so
  the command is usable only by someone with dashboard access.
* ``/disconnect summary:True``: summarises the session the bot just
  recorded in that server.

Either way the summary *is* the session's cached recap (src/recaps.py): it
is generated once with one model call, saved beside the transcripts, posted
to the channel, and from then on read by ``/ask`` before the transcript. A
session that already has a recap is posted from the cache, with no model
call, unless a refresh is asked for.

Everything here is plain, blocking Python with no Discord objects, so it is
unit-tested; main.py only defers, calls :func:`summarise` in a thread and
posts :func:`messages`.
"""

import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from src import recaps, sessions
from src.utils.answer import split_message

NOT_FOUND = (
    "No session with that id on this server. The id is on the dashboard's "
    "Transcripts page."
)
STILL_RECORDING = (
    "That session is still being recorded. Use `/stop` (or `/disconnect`) first."
)
NOTHING_RECORDED = "Nothing was recorded this time, so there is no summary to write."
NOTHING_SAID = "Nothing was transcribed in that session, so there is no summary."
ALREADY_RUNNING = "A summary of that session is already being written."


class SummaryError(Exception):
    """The summary cannot be made; ``user_message`` is ready for Discord."""

    def __init__(self, user_message: str):
        super().__init__(user_message)
        self.user_message = user_message


@dataclass
class Summary:
    recap: recaps.Recap
    files: list
    meta: object  # sessions.SessionMeta
    # False when an existing recap was reused (no model call).
    generated: bool = True


# One summary per session at a time in this process: a second request
# while the first is still reading the transcript would pay for it twice.
_running = set()
_running_lock = threading.Lock()


def owner_guilds(meta, lib) -> set:
    """The Discord servers a session belongs to: the one it was recorded
    in, or, for a recording tagged by hand with no server on it, the
    servers its group is bound to in library.yml. Empty when neither is
    known."""
    if meta.guild_id is not None:
        return {int(meta.guild_id)}
    group = lib.groups.get(meta.group) if (lib and meta.group) else None
    out = set()
    for g in getattr(group, "discord_guilds", None) or []:
        try:
            out.add(int(g))
        except (TypeError, ValueError):
            continue
    return out


def find(transcripts_dir, session_id, guild_id, lib=None) -> Path:
    """The recording ``session_id`` names, for a command run in
    ``guild_id``. Fails closed: the session must be known to belong to
    that server (:func:`owner_guilds`), and one that belongs to another
    server, or to none we can tell, is reported exactly like an unknown
    id. The bot serves several tables, and one must not post another's
    summary."""
    path = sessions.find_by_session_id(transcripts_dir, session_id)
    meta = sessions.read_meta(path) if path else None
    if meta is None or guild_id is None:
        raise SummaryError(NOT_FOUND)
    if lib is None:
        from src import library

        lib = library.load()
    if int(guild_id) not in owner_guilds(meta, lib):
        raise SummaryError(NOT_FOUND)
    return path


def summarise(path, lib, *, recording=(), refresh=False, ask=None) -> Summary:
    """The summary of the session ``path`` belongs to. Blocking: one model
    call unless a recap is cached. ``recording`` is the files being written
    right now; a session that includes one is refused. Raises
    :class:`SummaryError`, or the provider's ``AskFailed``/``AskBlocked``."""
    meta = sessions.read_meta(path)
    if meta is None:
        raise SummaryError(NOTHING_RECORDED)
    # The night as /ask groups it, which is what the recap is checked against.
    files = sessions.story_night_files(path)
    live = {Path(p).name for p in recording if p}
    if live & {f.name for f in files}:
        raise SummaryError(STILL_RECORDING)
    if not refresh:
        cached = recaps.load(files)
        if cached is not None:
            return Summary(cached, files, meta, generated=False)
    key = str(recaps.recap_path(files))
    with _running_lock:
        if key in _running:
            raise SummaryError(ALREADY_RUNNING)
        _running.add(key)
    try:
        recap = recaps.generate(
            files, header=sessions.header_text(lib, meta, len(files)), ask=ask
        )
    except recaps.RecapError as e:
        raise SummaryError(NOTHING_SAID) from e
    except FileNotFoundError as e:  # tagged, but nobody spoke: no .txt
        raise SummaryError(NOTHING_SAID) from e
    finally:
        with _running_lock:
            _running.discard(key)
    return Summary(recap, files, meta, generated=True)


def title(lib, meta) -> str:
    try:
        day = datetime.strptime(meta.session_day, "%Y-%m-%d").strftime("%A, %B %-d, %Y")
    except (TypeError, ValueError):
        day = str(meta.session_day)
    campaign = lib.name("campaign", meta.campaign) if meta.campaign else None
    return (
        f"**Session summary: {campaign}, {day}**"
        if campaign
        else (f"**Session summary: {day}**")
    )


def messages(lib, summary: Summary, limit: int = 2000) -> list:
    """The Discord messages to post, in order. The session id is never in
    them."""
    return split_message(f"{title(lib, summary.meta)}\n\n{summary.recap.text}", limit)
