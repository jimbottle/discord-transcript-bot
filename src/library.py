"""Games, groups and campaigns (discord-transcript-bot-157).

``library/library.yml`` says which games exist, which play group owns
which Discord server or voice channel, and which campaign each group is
currently running. Recordings are tagged from it when /scribe starts
(src/sessions.py), and /ask draws notes from the matching folders:

    library/games/<game>/notes/          rules or setting, shared by every campaign
    library/groups/<group>/notes/        the table: players, house rules, GM tables
    library/campaigns/<campaign>/notes/  the story: NPCs, places, arcs, secrets

Read fresh on every call (a few KB) so edits apply without a restart.
Never raises on bad config: a broken library means untagged recordings and
no notes, not a broken bot.
"""

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LIBRARY_DIR = PROJECT_ROOT / "library"


def library_dir() -> Path:
    override = os.getenv("LIBRARY_DIR", "").strip()
    return Path(override) if override else DEFAULT_LIBRARY_DIR


@dataclass
class Game:
    id: str
    name: str


@dataclass
class Group:
    id: str
    name: str
    gm: str = ""
    discord_guilds: list = field(default_factory=list)
    discord_channels: list = field(default_factory=list)
    active_campaign: Optional[str] = None


@dataclass
class Campaign:
    id: str
    name: str
    group: Optional[str] = None
    game: Optional[str] = None


def _ids(value) -> list:
    """Discord ids as ints; YAML may give ints, strings or a single value."""
    if value is None:
        return []
    if not isinstance(value, (list, tuple)):
        value = [value]
    out = []
    for v in value:
        try:
            out.append(int(str(v).strip()))
        except ValueError:
            logger.warning("library: ignoring non-numeric Discord id %r", v)
    return out


@dataclass
class Library:
    root: Path
    games: dict = field(default_factory=dict)
    groups: dict = field(default_factory=dict)
    campaigns: dict = field(default_factory=dict)

    # ── lookups ────────────────────────────────────────────────────────

    def resolve(self, guild_id=None, channel_id=None):
        """``(group_id, campaign_id)`` for a recording, or ``(None, None)``.

        A group listing the voice channel wins over one listing the whole
        server, so two groups can share a Discord server.
        """
        group = None
        if channel_id is not None:
            group = next(
                (
                    g
                    for g in self.groups.values()
                    if int(channel_id) in g.discord_channels
                ),
                None,
            )
        if group is None and guild_id is not None:
            group = next(
                (g for g in self.groups.values() if int(guild_id) in g.discord_guilds),
                None,
            )
        if group is None:
            return None, None
        return group.id, group.active_campaign

    def game_for(self, campaign_id: Optional[str]) -> Optional[str]:
        c = self.campaigns.get(campaign_id) if campaign_id else None
        return c.game if c else None

    def name(self, kind: str, item_id: Optional[str]) -> Optional[str]:
        """Display name for a game/group/campaign id; the id itself when the
        library no longer knows it (renamed or removed), None for None."""
        if not item_id:
            return None
        table = {"game": self.games, "group": self.groups, "campaign": self.campaigns}[
            kind
        ]
        item = table.get(item_id)
        return item.name if item else item_id

    def note_dirs(self, game=None, group=None, campaign=None) -> list:
        """Existing note folders for a session, general to specific. A later
        folder's note overrides an earlier one with the same id."""
        candidates = []
        if game:
            candidates.append(self.root / "games" / game / "notes")
        if group:
            candidates.append(self.root / "groups" / group / "notes")
        if campaign:
            candidates.append(self.root / "campaigns" / campaign / "notes")
        return [d for d in candidates if d.is_dir()]


def _load_section(raw, key, build):
    items = {}
    section = raw.get(key) or {}
    if not isinstance(section, dict):
        logger.warning("library: '%s' must be a mapping; ignoring it", key)
        return items
    for item_id, body in section.items():
        body = body if isinstance(body, dict) else {}
        try:
            items[str(item_id)] = build(str(item_id), body)
        except (TypeError, ValueError) as e:
            logger.warning("library: skipping %s '%s': %s", key, item_id, e)
    return items


def load(root: Optional[Path] = None) -> Library:
    root = Path(root) if root else library_dir()
    lib = Library(root=root)
    path = root / "library.yml"
    if not path.is_file():
        return lib
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, ValueError, yaml.YAMLError) as e:
        # ValueError: UnicodeDecodeError from a file saved as cp1252/Latin-1.
        logger.warning("library: cannot read %s: %s", path, e)
        return lib
    if not isinstance(raw, dict):
        logger.warning("library: %s is not a mapping", path)
        return lib
    lib.games = _load_section(
        raw, "games", lambda i, b: Game(id=i, name=str(b.get("name") or i))
    )
    lib.groups = _load_section(
        raw,
        "groups",
        lambda i, b: Group(
            id=i,
            name=str(b.get("name") or i),
            gm=str(b.get("gm") or ""),
            discord_guilds=_ids(b.get("discord_guilds")),
            discord_channels=_ids(b.get("discord_channels")),
            active_campaign=(
                str(b["active_campaign"]) if b.get("active_campaign") else None
            ),
        ),
    )
    lib.campaigns = _load_section(
        raw,
        "campaigns",
        lambda i, b: Campaign(
            id=i,
            name=str(b.get("name") or i),
            group=(str(b["group"]) if b.get("group") else None),
            game=(str(b["game"]) if b.get("game") else None),
        ),
    )
    for c in lib.campaigns.values():
        if c.group and c.group not in lib.groups:
            logger.warning(
                "library: campaign '%s' names unknown group '%s'", c.id, c.group
            )
        if c.game and c.game not in lib.games:
            logger.warning(
                "library: campaign '%s' names unknown game '%s'", c.id, c.game
            )
    for g in lib.groups.values():
        if g.active_campaign and g.active_campaign not in lib.campaigns:
            logger.warning(
                "library: group '%s' active_campaign '%s' is not a campaign",
                g.id,
                g.active_campaign,
            )
    return lib
