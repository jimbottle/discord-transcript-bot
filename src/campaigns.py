"""The campaign data model (discord-transcript-bot-4n1).

A campaign is the thing everything else hangs off: it belongs to a group,
is played in a game, owns note folders, and is played over sessions (game
nights), each made of one or more recordings. That decides what `/ask` may
read, so the bot, the dashboard and the question log all resolve it here
instead of each reaching into the files their own way.

    Model ── campaigns() ──> Campaign ── sessions ──> Session ── files
      │                         │ game, group, note_dirs
      ├── session_of(path) ─────┘
      ├── ask_context(path) ──> AskContext (what one question is sent with)
      └── default_for(guild, channel) -> the tag a new recording starts with

Storage is unchanged and stays where it was: ``library/library.yml`` and
``library/dashboard.yml`` (src/library.py) say which campaigns exist, and
each recording's ``.meta.json`` (src/sessions.py) says which one it belongs
to. This module only reads them and never raises on bad files, like the two
it is built on. A ``Model`` is a snapshot: make a new one per request.
"""

from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Optional

from src import library, sessions


@dataclass(frozen=True)
class Session:
    """One game night: every recording with the same campaign (or, with no
    campaign, the same server) and session day. An untagged recording is a
    session of its own."""

    key: Optional[str]
    day: Optional[str]
    files: tuple
    campaign_id: Optional[str] = None
    game_id: Optional[str] = None
    group_id: Optional[str] = None
    # The tag of the recording the session was looked up by (None: untagged).
    meta: Optional[sessions.SessionMeta] = None

    @property
    def tagged(self) -> bool:
        return self.meta is not None

    @property
    def recordings(self) -> int:
        return len(self.files)

    @property
    def file_names(self) -> list:
        return [Path(f).name for f in self.files]

    def transcript(self) -> str:
        """The recordings joined oldest first, as stored (24-hour stamps,
        Discord ids), with a marker where the bot was restarted."""
        return sessions.join(self.files)

    def model_text(self) -> str:
        """The transcript as a model reads it."""
        return sessions.model_text(self.transcript())


@dataclass(frozen=True)
class Campaign:
    id: str
    name: str
    game: Optional[library.Game] = None
    group: Optional[library.Group] = None
    # Ids as written in the library, kept even when the game or group they
    # name is no longer defined there.
    game_id: Optional[str] = None
    group_id: Optional[str] = None
    # Existing note folders, general to specific (game, group, campaign).
    note_dirs: tuple = ()
    _model: Optional["Model"] = field(default=None, repr=False, compare=False)

    @cached_property
    def sessions(self) -> tuple:
        """Its sessions, oldest first. Looked up on first use (it reads
        every recording's tag), so a dropdown or a Lore question, which
        only need the name and the note folders, never scan the
        transcripts (roborev #7126)."""
        if self._model is None:
            return ()
        return tuple(self._model.sessions(self.id))

    @property
    def last_played(self) -> Optional[str]:
        """Session day of the newest session, or None if never recorded."""
        return self.sessions[-1].day if self.sessions else None

    def session_notes(self) -> list:
        """Every session of the campaign as on-demand notes: its cached
        recap when there is one, then its transcript
        (``sessions.campaign_session_notes``). For questions about the
        campaign as a whole (the Lore page). Reads the transcripts, so call
        it off the event loop."""
        if self._model is None or self._model.transcripts_dir is None:
            return []
        return sessions.campaign_session_notes(self._model.transcripts_dir, self.id)

    def sessions_before(self, day: str) -> list:
        return [s for s in self.sessions if s.day and s.day < day]

    def choice(self) -> dict:
        """Row for a dashboard dropdown."""
        return {
            "id": self.id,
            "name": self.name,
            "group": self.group_id or "",
            "game": self.game_id or "",
        }


@dataclass
class AskContext:
    """What a question about one session is sent with."""

    transcript: str = ""
    header: str = ""
    note_dirs: list = field(default_factory=list)
    files: list = field(default_factory=list)
    session: Optional[Session] = None
    _path: Optional[Path] = None
    _model: Optional["Model"] = None

    @property
    def campaign(self) -> Optional[Campaign]:
        """The session's campaign, with its sessions. Looked up when asked
        for (it reads every recording's tag), not when the context is made:
        Discord's /ask builds the context before it has acknowledged the
        command."""
        if self._model is None or self.session is None:
            return None
        return self._model.campaign(self.session.campaign_id)

    def earlier_notes(self) -> list:
        """The campaign's earlier sessions as on-demand notes
        (``sessions.earlier_session_notes``). Reads every earlier transcript,
        so call it off the event loop."""
        return sessions.earlier_session_notes(self._path) if self._path else []


class Model:
    """Campaigns and sessions resolved from one library and one transcripts
    folder, as they are when the model is made."""

    def __init__(self, transcripts_dir=None, lib: Optional[library.Library] = None):
        self.lib = lib if lib is not None else library.load()
        self.transcripts_dir = Path(transcripts_dir) if transcripts_dir else None
        self._metas = None

    # ── recordings ─────────────────────────────────────────────────────

    def _recordings(self) -> list:
        """``[(path, meta)]`` for every tagged recording, oldest first."""
        if self._metas is None:
            found = []
            if self.transcripts_dir is not None:
                try:
                    paths = sorted(self.transcripts_dir.glob("*.txt"))
                except OSError:
                    paths = []
                for txt in paths:
                    meta = sessions.read_meta(txt)
                    if meta is not None:
                        found.append((txt, meta))
            self._metas = found
        return self._metas

    def _session(self, key, files, meta) -> Session:
        return Session(
            key=key,
            day=meta.session_day if meta else None,
            files=tuple(files),
            campaign_id=meta.campaign if meta else None,
            game_id=meta.game if meta else None,
            group_id=meta.group if meta else None,
            meta=meta,
        )

    # ── sessions ───────────────────────────────────────────────────────

    def sessions(self, campaign_id: Optional[str] = None) -> list:
        """Every tagged session, oldest first; only ``campaign_id``'s when
        given."""
        by_key = {}
        for txt, meta in self._recordings():
            if campaign_id is not None and meta.campaign != campaign_id:
                continue
            by_key.setdefault(meta.key, (meta, []))[1].append(txt)
        out = [self._session(key, files, meta) for key, (meta, files) in by_key.items()]
        return sorted(out, key=lambda s: (s.day or "", s.key or ""))

    def session_of(self, transcript_path) -> Session:
        """The session a recording belongs to."""
        path = Path(transcript_path)
        meta = sessions.read_meta(path)
        if meta is None:
            return self._session(None, [path], None)
        return self._session(meta.key, sessions.session_files(path), meta)

    def describe(self, transcript_path) -> dict:
        """Display fields for a recording's session (names looked up), plus
        ``files``; ``{"tagged": False, ...}`` when untagged."""
        s = self.session_of(transcript_path)
        info = sessions.describe(self.lib, s.meta, s.recordings)
        info["files"] = s.file_names
        return info

    # ── campaigns ──────────────────────────────────────────────────────

    def campaign(self, campaign_id: Optional[str]) -> Optional[Campaign]:
        c = self.lib.campaigns.get(campaign_id) if campaign_id else None
        if c is None:
            return None
        return Campaign(
            id=c.id,
            name=c.name,
            game=self.lib.games.get(c.game) if c.game else None,
            group=self.lib.groups.get(c.group) if c.group else None,
            game_id=c.game,
            group_id=c.group,
            note_dirs=tuple(
                self.lib.note_dirs(game=c.game, group=c.group, campaign=c.id)
            ),
            _model=self,
        )

    def campaigns(self) -> list:
        """Every campaign the library has, by name."""
        found = [self.campaign(cid) for cid in self.lib.campaigns]
        return sorted(found, key=lambda c: c.name.lower())

    def campaign_of(self, transcript_path) -> Optional[Campaign]:
        return self.campaign(self.session_of(transcript_path).campaign_id)

    def campaign_header(self, campaign_id: str) -> str:
        """The CAMPAIGN (+ STYLE) lines for a notes-only question."""
        return sessions.campaign_header(self.lib, campaign_id)

    def default_for(self, guild_id=None, channel_id=None):
        """``(group_id, campaign_id)`` a new recording starts with: the
        campaign used last in this Discord server, else the library's
        binding (``sessions.default_tag``)."""
        return sessions.default_tag(
            self.lib, guild_id, channel_id, self.transcripts_dir
        )

    # ── /ask ───────────────────────────────────────────────────────────

    def ask_context(self, transcript_path) -> AskContext:
        """Everything a question about ``transcript_path``'s session needs:
        the whole session's transcript as the model reads it, the SESSION
        header, the note folders, and (on demand) the earlier sessions."""
        path = Path(transcript_path)
        # One pass over the tags: Discord's /ask gets here before it has
        # acknowledged the command (roborev #7126). Same result as
        # ``sessions.ask_context``, which tests/test_campaigns.py holds it to.
        session = self.session_of(path)
        meta = session.meta
        header, dirs = "", []
        if meta is not None:
            header = sessions.header_text(self.lib, meta, session.recordings)
            dirs = self.lib.note_dirs(
                game=meta.game, group=meta.group, campaign=meta.campaign
            )
        return AskContext(
            transcript=session.model_text(),
            header=header,
            note_dirs=list(dirs),
            files=list(session.files),
            session=session,
            _path=path,
            _model=self,
        )


def for_recording(transcript_path, lib: Optional[library.Library] = None) -> Model:
    """The model whose transcripts folder is the one ``transcript_path`` is in."""
    return Model(Path(transcript_path).parent, lib)
