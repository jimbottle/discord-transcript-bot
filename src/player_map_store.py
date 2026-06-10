"""Atomic, dependency-free reader/writer for the player_map.yml roster.

Shared by the Discord bot (``VoloBot.upsert_player_entry``) and the web
dashboard's roster editor so the load -> non-dict guard -> merge -> atomic
write logic lives in exactly one place. No Discord or Flask imports — pure
file I/O — so the web process can import it without pulling in Pycord.

The roster maps an integer Discord user id to ``{"player", "character"}``.
"""

import os

import yaml


def _atomic_dump(path, data):
    """Write ``data`` as YAML to ``path`` via tmp + ``os.replace``.

    A kill mid-write can't truncate/corrupt the whole roster, and a failed
    dump/replace never leaves an orphaned ``.tmp`` behind (it's removed and
    the error re-raised). The roster is written by ``/add_player`` mid-call
    and by the dashboard, so this can race a live session — never leave a
    partial file.
    """
    tmp = path + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
        os.replace(tmp, path)
    except Exception:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise


def load(path):
    """Return the roster mapping at ``path`` (dict keyed by int user id).

    Missing file (or falsy path) -> ``{}``. Raises ``ValueError`` if the
    file is valid YAML but not a mapping (hand-edited to a list/scalar), so
    callers never silently treat a non-mapping as empty and then clobber
    whatever was there.
    """
    if not path:
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except FileNotFoundError:
        return {}
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(
            f"{path} is not a YAML mapping; refusing to overwrite it. "
            "Fix the roster file by hand to persist changes."
        )
    return data


def upsert(path, user_id, player, character):
    """Merge one ``user_id -> {player, character}`` entry into the on-disk
    roster, preserving every other entry, and write it atomically.

    Coerces ``user_id`` to int (Discord snowflake). Raises ``ValueError``
    on a non-mapping file (see ``load``); the file is left untouched then.
    """
    file_map = load(path)
    file_map[int(user_id)] = {"player": player, "character": character}
    _atomic_dump(path, file_map)


def delete(path, user_id):
    """Remove ``user_id`` from the on-disk roster, preserving other entries.

    Returns ``True`` if an entry was removed, ``False`` if it wasn't
    present. Raises ``ValueError`` on a non-mapping file (see ``load``).
    """
    file_map = load(path)
    if int(user_id) not in file_map:
        return False
    del file_map[int(user_id)]
    _atomic_dump(path, file_map)
    return True
