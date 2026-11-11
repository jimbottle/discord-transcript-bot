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


# A light, game-flavoured voice for /ask answers. Kept to a word or two of
# colour so answers stay easy to read; facts are never stylised. Override or
# disable per game with ``voice:`` in library.yml (``voice: ""`` turns it off).
DEFAULT_VOICES = {
    "cyberpunk-red": (
        "Give answers a light touch of Night City street flavour: now and then a "
        "single word of Cyberpunk RED slang (choom, eddies, preem, nova, delta) "
        "or a fixer's-briefing tone, varied, and not in every answer. Keep names, times, numbers and facts plain and "
        "exact, keep the answer as clear and short as it would be without the "
        "flavour, and leave the flavour out when it would feel forced."
    ),
    "dnd-5e": (
        "Give answers a light touch of tavern-chronicler flavour: now and then a "
        "phrase of fantasy colour, varied, and not in every answer. Keep names, times, numbers and facts plain "
        "and exact, keep the answer as clear and short as it would be without "
        "the flavour, and leave the flavour out when it would feel forced."
    ),
}


@dataclass
class Game:
    id: str
    name: str
    voice: str = ""


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


# Campaigns made from the dashboard live in their own machine-written file,
# so library.yml stays hand-edited with its comments (5bd). library.yml wins
# when both define an id.
DASHBOARD_FILE = "dashboard.yml"
MAX_CAMPAIGN_NAME = 80


class LibraryError(ValueError):
    """A dashboard change to the library was refused; the message is
    safe to show."""


def _campaign_from(i, b) -> "Campaign":
    return Campaign(
        id=i,
        name=str(b.get("name") or i),
        group=(str(b["group"]) if b.get("group") else None),
        game=(str(b["game"]) if b.get("game") else None),
    )


def _read_dashboard(root: Path) -> dict:
    """The dashboard file's ``campaigns`` mapping; {} when absent or bad."""
    path = root / DASHBOARD_FILE
    if not path.is_file():
        return {}
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, ValueError, yaml.YAMLError) as e:
        logger.warning("library: cannot read %s: %s", path, e)
        return {}
    section = raw.get("campaigns") if isinstance(raw, dict) else None
    return section if isinstance(section, dict) else {}


def slug(name: str) -> str:
    out = "".join(ch if ch.isalnum() else "-" for ch in name.lower())
    return "-".join(part for part in out.split("-") if part)[:60]


def add_campaign(name, game=None, group=None, root: Optional[Path] = None) -> Campaign:
    """Create a campaign from the dashboard: an entry in ``dashboard.yml``
    and an empty ``campaigns/<id>/notes`` folder for its documents. The id
    is a slug of the name. Raises :class:`LibraryError` on a bad name, an
    unknown game or group, or a name already taken."""
    root = Path(root) if root else library_dir()
    lib = load(root)
    name = " ".join(str(name or "").split())
    if not name:
        raise LibraryError("campaign name is required")
    if len(name) > MAX_CAMPAIGN_NAME:
        raise LibraryError(f"campaign name is too long (max {MAX_CAMPAIGN_NAME})")
    if game and game not in lib.games:
        raise LibraryError(f"unknown game '{game}'")
    if group and group not in lib.groups:
        raise LibraryError(f"unknown group '{group}'")
    cid = slug(name)
    if not cid:
        raise LibraryError("campaign name needs at least one letter or digit")
    taken = {c.name.lower() for c in lib.campaigns.values()}
    if cid in lib.campaigns or name.lower() in taken:
        raise LibraryError(f"a campaign called '{name}' already exists")
    entries = _read_dashboard(root)
    entries[cid] = {"name": name}
    if game:
        entries[cid]["game"] = game
    if group:
        entries[cid]["group"] = group
    try:
        root.mkdir(parents=True, exist_ok=True)
        path = root / DASHBOARD_FILE
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(
            "# Campaigns created from the dashboard. Written by the app; edit\n"
            "# library.yml for anything else.\n"
            + yaml.safe_dump(
                {"campaigns": entries}, sort_keys=True, allow_unicode=True
            ),
            encoding="utf-8",
        )
        os.replace(tmp, path)
        (root / "campaigns" / cid / "notes").mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise LibraryError(f"could not save the campaign: {e}") from e
    return _campaign_from(cid, entries[cid])


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
    lib = _load_main(root)
    for cid, body in _read_dashboard(root).items():
        if str(cid) not in lib.campaigns and isinstance(body, dict):
            lib.campaigns[str(cid)] = _campaign_from(str(cid), body)
    return lib


def _load_main(root: Path) -> Library:
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
        raw,
        "games",
        lambda i, b: Game(
            id=i,
            name=str(b.get("name") or i),
            voice=(
                # absent -> built-in default; present but empty or null
                # (`voice:` / `voice: ~` / `voice: ""`) -> off (roborev #6459)
                (str(b["voice"]).strip() if b["voice"] is not None else "")
                if "voice" in b
                else DEFAULT_VOICES.get(i, "")
            ),
        ),
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
    lib.campaigns = _load_section(raw, "campaigns", _campaign_from)
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
