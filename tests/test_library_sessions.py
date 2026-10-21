"""Library (games/groups/campaigns) and session tagging
(src/library.py, src/sessions.py; discord-transcript-bot-157)."""

import json
from datetime import datetime

import pytest

from src import library, sessions

LIB = """
games:
  cyberpunk-red: {name: Cyberpunk RED}
  dnd-5e: {name: D&D 5e}
groups:
  noah-gm:
    name: Noah's group
    gm: Noah
    discord_guilds: [111]
    active_campaign: return-to-night-city
  side-table:
    name: Side table
    discord_guilds: ["111"]
    discord_channels: [999]
    active_campaign: lost-mine
campaigns:
  return-to-night-city: {name: Return to Night City, group: noah-gm, game: cyberpunk-red}
  lost-mine: {name: Lost Mine, group: side-table, game: dnd-5e}
"""


@pytest.fixture
def lib(tmp_path):
    (tmp_path / "library.yml").write_text(LIB, encoding="utf-8")
    for d in (
        "games/cyberpunk-red/notes",
        "groups/noah-gm/notes",
        "campaigns/return-to-night-city/notes",
    ):
        (tmp_path / d).mkdir(parents=True)
    return library.load(tmp_path)


# ── library ───────────────────────────────────────────────────────────


def test_load_reads_games_groups_campaigns(lib):
    assert set(lib.games) == {"cyberpunk-red", "dnd-5e"}
    assert lib.groups["noah-gm"].discord_guilds == [111]
    assert lib.campaigns["return-to-night-city"].game == "cyberpunk-red"


def test_resolve_by_guild(lib):
    assert lib.resolve(guild_id=111) == ("noah-gm", "return-to-night-city")


def test_channel_binding_beats_guild_binding(lib):
    # side-table also lists guild 111 but claims channel 999 specifically
    assert lib.resolve(guild_id=111, channel_id=999) == ("side-table", "lost-mine")
    assert lib.resolve(guild_id=111, channel_id=5) == (
        "noah-gm",
        "return-to-night-city",
    )


def test_unknown_guild_resolves_to_nothing(lib):
    assert lib.resolve(guild_id=222) == (None, None)


def test_note_dirs_general_to_specific_and_only_existing(lib, tmp_path):
    dirs = lib.note_dirs("cyberpunk-red", "noah-gm", "return-to-night-city")
    assert [d.relative_to(tmp_path).parts[0] for d in dirs] == [
        "games",
        "groups",
        "campaigns",
    ]
    assert lib.note_dirs("dnd-5e", "side-table", "lost-mine") == []


def test_missing_or_broken_library_is_empty_not_an_error(tmp_path):
    assert library.load(tmp_path).campaigns == {}
    (tmp_path / "library.yml").write_text(
        "games: [not, a, mapping]\n: :", encoding="utf-8"
    )
    assert library.load(tmp_path).campaigns == {}


def test_non_utf8_library_is_empty_not_an_error(tmp_path):
    # roborev #6450: a cp1252-saved library.yml raised UnicodeDecodeError.
    (tmp_path / "library.yml").write_bytes(b"games:\n  x: {name: Caf\xe9}\n")
    assert library.load(tmp_path).games == {}


def test_health_check_survives_a_library_that_raises(monkeypatch):
    from src.bot.health import HealthCheck

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")

    def boom(*a, **k):
        raise RuntimeError("library exploded")

    monkeypatch.setattr(library, "load", boom)
    hc = HealthCheck()
    hc._check_ask_provider()
    check = hc.checks["ask_provider"]
    assert check["ok"] is True and check["critical"] is False
    assert "0 campaigns" in check["message"]


def test_game_voice_defaults_override_and_off(tmp_path):
    (tmp_path / "library.yml").write_text(
        "games:\n"
        "  cyberpunk-red: {name: Cyberpunk RED}\n"
        "  dnd-5e: {name: D&D 5e, voice: ''}\n"
        "  nulled: {name: Nulled, voice: }\n"
        "  custom: {name: Custom, voice: Speak like a ship's log.}\n"
        "  unknown: {name: Unknown}\n"
    )
    games = library.load(tmp_path).games
    assert games["cyberpunk-red"].voice == library.DEFAULT_VOICES["cyberpunk-red"]
    assert games["dnd-5e"].voice == ""
    assert games["nulled"].voice == ""  # roborev #6459: was the string "None"
    assert games["custom"].voice == "Speak like a ship's log."
    assert games["unknown"].voice == ""


def test_name_falls_back_to_the_id_for_a_removed_item(lib):
    assert lib.name("campaign", "return-to-night-city") == "Return to Night City"
    assert lib.name("campaign", "gone") == "gone"
    assert lib.name("game", None) is None


# ── session day ───────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "when, day",
    [
        (datetime(2026, 10, 20, 19, 38), "2026-10-20"),
        (datetime(2026, 10, 21, 0, 40), "2026-10-20"),  # after midnight, same night
        (datetime(2026, 10, 21, 6, 0), "2026-10-21"),
    ],
)
def test_session_day_has_a_morning_cutoff(when, day, monkeypatch):
    monkeypatch.delenv("SESSION_DAY_CUTOFF_HOUR", raising=False)
    assert sessions.session_day(when) == day


# ── tagging + grouping recordings into sessions ──────────────────────


def _rec(dirpath, stamp, text, lib, **kw):
    txt = dirpath / f"{stamp}.txt"
    txt.write_text(text, encoding="utf-8")
    meta = sessions.build_meta(lib, started_at=sessions.started_at_from_name(txt), **kw)
    sessions.write_meta(txt, meta)
    return txt


def test_build_meta_resolves_from_guild(lib):
    m = sessions.build_meta(
        lib,
        guild_id=111,
        guild="personal",
        channel_id=5,
        channel="General",
        started_at=datetime(2026, 10, 20, 19, 38, 38),
    )
    assert (m.group, m.game, m.campaign, m.session_day) == (
        "noah-gm",
        "cyberpunk-red",
        "return-to-night-city",
        "2026-10-20",
    )
    assert m.key == "return-to-night-city/2026-10-20"
    assert m.started_at == "2026-10-20T19:38:38"


def test_explicit_campaign_fills_in_its_group(lib):
    m = sessions.build_meta(
        lib, campaign="lost-mine", started_at=datetime(2026, 1, 1, 20)
    )
    assert (m.group, m.game) == ("side-table", "dnd-5e")


def test_meta_round_trips_and_ignores_unknown_fields(lib, tmp_path):
    txt = tmp_path / "2026-10-20_19-38-38.txt"
    m = sessions.build_meta(
        lib, guild_id=111, started_at=datetime(2026, 10, 20, 19, 38)
    )
    p = sessions.write_meta(txt, m)
    assert p.name == "2026-10-20_19-38-38.meta.json"
    data = json.loads(p.read_text())
    data["future_field"] = 1
    p.write_text(json.dumps(data))
    assert sessions.read_meta(txt) == m


def test_restart_joins_recordings_of_the_same_session(lib, tmp_path):
    a = _rec(
        tmp_path,
        "2026-10-20_19-38-38",
        "[19:40:00] A (B) [1]: first\n",
        lib,
        guild_id=111,
    )
    b = _rec(
        tmp_path,
        "2026-10-20_20-15-26",
        "[20:16:00] A (B) [1]: second\n",
        lib,
        guild_id=111,
    )
    # different night, same campaign: not part of it
    _rec(
        tmp_path,
        "2026-10-13_19-00-00",
        "[19:01:00] A (B) [1]: last week\n",
        lib,
        guild_id=111,
    )
    # same night, other group: not part of it
    _rec(
        tmp_path,
        "2026-10-20_19-50-00",
        "[19:51:00] C (D) [2]: other table\n",
        lib,
        guild_id=111,
        channel_id=999,
    )
    assert sessions.session_files(b) == [a, b]
    text, meta, files = sessions.session_transcript(b)
    assert text == (
        "[19:40:00] A (B) [1]: first\n"
        "--- recording restarted at 20:15:26 ---\n"
        "[20:16:00] A (B) [1]: second\n"
    )
    assert meta.key == "return-to-night-city/2026-10-20"


def test_untagged_recording_is_a_session_of_one(tmp_path):
    txt = tmp_path / "2026-05-18_17-39-56.txt"
    txt.write_text("[17:40:00] A (B) [1]: hi\n", encoding="utf-8")
    text, meta, files = sessions.session_transcript(txt)
    assert meta is None and files == [txt]
    assert text == "[17:40:00] A (B) [1]: hi\n"


def test_header_text_names_game_campaign_group_and_restart(lib):
    m = sessions.build_meta(
        lib, guild_id=111, started_at=datetime(2026, 10, 20, 19, 38)
    )
    h = sessions.header_text(lib, m, recordings=2)
    h, style = h.split("\n")
    assert style.startswith("STYLE: ") and "Night City" in style
    assert h == (
        "SESSION: Game: Cyberpunk RED · Campaign: Return to Night City · "
        "Group: Noah's group (GM: Noah) · Session: 2026-10-20 · "
        "2 recordings joined (the bot was restarted mid-session)"
    )
    assert sessions.header_text(lib, None) == ""


def test_ask_context_returns_session_text_header_and_note_dirs(lib, tmp_path):
    t = tmp_path / "t"
    t.mkdir()
    a = _rec(t, "2026-10-20_19-38-38", "x\n", lib, guild_id=111)
    text, header, dirs, files = sessions.ask_context(a, lib)
    assert text == "x\n" and header.startswith("SESSION: Game: Cyberpunk RED")
    assert [d.parent.name for d in dirs] == [
        "cyberpunk-red",
        "noah-gm",
        "return-to-night-city",
    ]


# ── the retag script ─────────────────────────────────────────────────


def test_tag_session_script_tags_files(lib, tmp_path, monkeypatch, capsys):
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "tag_session", Path(library.PROJECT_ROOT) / "scripts" / "tag_session.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setenv("LIBRARY_DIR", str(tmp_path))
    txt = tmp_path / "2026-10-21_00-30-00.txt"
    txt.write_text("x", encoding="utf-8")
    assert mod.main([str(txt), "--campaign", "return-to-night-city"]) == 0
    m = sessions.read_meta(txt)
    assert m.key == "return-to-night-city/2026-10-20"  # past midnight, same night
    assert m.source == "manual" and m.game == "cyberpunk-red"
    with pytest.raises(SystemExit):
        mod.main([str(txt), "--campaign", "nope"])
