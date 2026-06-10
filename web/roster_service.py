"""Roster (Discord id -> player/character) assembly for the dashboard.

Kept separate from transcript_service so that module stays focused on
reading transcripts/logs. All persistence goes through the shared
``src.player_map_store`` so the web editor writes player_map.yml exactly
the way the bot's ``/add_player`` does (atomic, non-dict guarded). The web
process never signals the running bot — edits land on disk and take effect
on the bot's next start.
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
# The dashboard runs from web/, so the project root (where ``src`` lives)
# isn't importable by default. Add it once so ``player_map_store`` — the
# bot's own roster writer — is shared rather than duplicated.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import player_map_store  # noqa: E402  (after sys.path setup)

from transcript_service import get_bot_state, get_live_session  # noqa: E402

# Names the bot writes for an unmapped speaker — never offer these as a
# pre-filled suggestion (the user is naming the person precisely to
# replace them).
_PLACEHOLDER_NAMES = {None, "", "Unknown", "None"}


def roster_path():
    """Path to player_map.yml.

    Honours ``PLAYER_MAP_FILE_PATH`` when set, else the project-root file
    the bot uses by default. A *relative* env value (the bundled .env uses
    ``./player_map.yml``) is resolved against the project root, NOT the
    dashboard's cwd (``web/``) — otherwise the web editor would read/write
    a different file than the bot.
    """
    configured = os.getenv("PLAYER_MAP_FILE_PATH")
    if not configured:
        return str(PROJECT_ROOT / "player_map.yml")
    path = Path(configured)
    return str(path if path.is_absolute() else PROJECT_ROOT / path)


def get_roster():
    """Current roster as a render-ready, name-sorted list.

    Returns ``{"entries": [...], "invalid": bool}``. ``invalid`` is True
    when the file is valid YAML but not a mapping (hand-edited to a
    list/scalar) — the page warns instead of 500-ing, and writes are
    refused by the store anyway.
    """
    try:
        data = player_map_store.load(roster_path())
        invalid = False
    except ValueError:
        data, invalid = {}, True
    entries = [
        {
            "user_id": uid,
            "player": (val or {}).get("player"),
            "character": (val or {}).get("character"),
        }
        for uid, val in data.items()
    ]
    entries.sort(key=lambda e: (str(e["player"] or "").lower(), str(e["user_id"])))
    return {"entries": entries, "invalid": invalid}


def _clean(name, user_id):
    """A usable suggested name, or None — drop placeholders and the raw id."""
    if name in _PLACEHOLDER_NAMES or name == str(user_id):
        return None
    return name


def get_unmapped_speakers():
    """People to offer naming for: present on the call or recently spoke,
    minus anyone already in the roster.

    Two on-disk signals, neither of which touches the running bot:
      - ``members`` of each connected guild in bot_state.json (present now,
        even if silent) — source ``on_call``.
      - speakers in the current/most-recent session transcript — source
        ``spoke``, carrying their last utterance for context.
    """
    try:
        mapped = set(player_map_store.load(roster_path()).keys())
    except ValueError:
        mapped = set()

    found = {}

    def add(uid, name, source, last_text=None):
        try:
            uid = int(uid)
        except (TypeError, ValueError):
            return
        if uid in mapped:
            return
        row = found.get(uid)
        name = _clean(name, uid)
        if row is None:
            found[uid] = {
                "user_id": uid,
                "suggested_name": name,
                "source": source,
                "last_text": last_text,
            }
            return
        # Already seen from the other source — fill gaps, prefer "on_call"
        # (they're here right now) and keep an utterance if we have one.
        if not row.get("suggested_name") and name:
            row["suggested_name"] = name
        if not row.get("last_text") and last_text:
            row["last_text"] = last_text
        if source == "on_call":
            row["source"] = "on_call"

    state = get_bot_state() or {}
    for guild in state.get("guilds", []) or []:
        for member in guild.get("members", []) or []:
            add(
                member.get("id"),
                member.get("display_name") or member.get("name"),
                "on_call",
            )

    live = get_live_session()
    for entry in live.get("entries", []):
        add(
            entry.get("user_id"),
            entry.get("player") or entry.get("character"),
            "spoke",
            entry.get("text"),
        )

    # On the call first (actionable now), then by id for stable ordering.
    return sorted(
        found.values(), key=lambda r: (r["source"] != "on_call", r["user_id"])
    )
