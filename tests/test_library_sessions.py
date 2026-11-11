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


def test_ask_context_gives_the_model_12_hour_times(lib, tmp_path):
    t = tmp_path / "t12"
    t.mkdir()
    a = _rec(
        t, "2026-10-20_19-38-38", "[19:40:00] A (B) [1]: first\n", lib, guild_id=111
    )
    _rec(t, "2026-10-20_20-15-26", "[20:16:00] A (B) [1]: second\n", lib, guild_id=111)
    text, *_ = sessions.ask_context(a, lib)
    assert "[7:40 PM] A (B): first" in text  # 12-hour, no Discord id
    assert "--- recording restarted at 8:15 PM ---" in text
    assert "19:40" not in text and "20:15" not in text
    # the files themselves keep 24-hour stamps and ids
    assert a.read_text().startswith("[19:40:00] A (B) [1]:")


# ── retag (dashboard tag editor) ─────────────────────────────────────


def test_retag_whole_session_moves_every_recording(lib, tmp_path):
    t = tmp_path / "rt"
    t.mkdir()
    a = _rec(t, "2026-10-20_19-38-38", "x\n", lib, guild_id=111)
    b = _rec(t, "2026-10-20_20-15-26", "y\n", lib, guild_id=111)
    written = sessions.retag(b, lib, campaign="lost-mine")
    assert written == [a, b]
    for f in (a, b):
        m = sessions.read_meta(f)
        assert (m.campaign, m.group, m.game) == ("lost-mine", "side-table", "dnd-5e")
        assert m.source == "manual" and m.guild_id == 111  # Discord fields kept
        assert m.session_day == "2026-10-20"


def test_retag_one_recording_only(lib, tmp_path):
    t = tmp_path / "rt1"
    t.mkdir()
    a = _rec(t, "2026-10-20_19-38-38", "x\n", lib, guild_id=111)
    b = _rec(t, "2026-10-20_20-15-26", "y\n", lib, guild_id=111)
    sessions.retag(b, lib, campaign="lost-mine", whole_session=False)
    assert sessions.read_meta(a).campaign == "return-to-night-city"
    assert sessions.read_meta(b).campaign == "lost-mine"


def test_retag_explicit_game_and_group_override_the_campaigns(lib, tmp_path):
    t = tmp_path / "rt2"
    t.mkdir()
    a = _rec(t, "2026-10-20_19-38-38", "x\n", lib, guild_id=111)
    sessions.retag(
        a, lib, campaign="return-to-night-city", game="dnd-5e", group="side-table"
    )
    m = sessions.read_meta(a)
    assert (m.campaign, m.game, m.group) == (
        "return-to-night-city",
        "dnd-5e",
        "side-table",
    )


def test_retag_clears_with_empty_and_tags_an_untagged_file(lib, tmp_path):
    txt = tmp_path / "2026-10-21_00-30-00.txt"
    txt.write_text("x")
    sessions.retag(txt, lib, campaign="return-to-night-city")
    m = sessions.read_meta(txt)
    assert m.key == "return-to-night-city/2026-10-20"  # 6am cutoff
    sessions.retag(txt, lib, campaign="", group="", game="")
    m = sessions.read_meta(txt)
    assert (m.campaign, m.group, m.game) == (None, None, None)


def test_retag_rejects_unknown_ids(lib, tmp_path):
    t = tmp_path / "rt3"
    t.mkdir()
    a = _rec(t, "2026-10-20_19-38-38", "x\n", lib, guild_id=111)
    for kw in ({"campaign": "nope"}, {"group": "nope"}, {"game": "nope"}):
        with pytest.raises(sessions.TagError):
            sessions.retag(a, lib, **kw)
    assert sessions.read_meta(a).campaign == "return-to-night-city"  # untouched


def test_latest_for_guild_picks_the_newest_tagged_recording(lib, tmp_path):
    """/generate_pdf finds the last session after /stop or a restart (fwv)."""
    _rec(
        tmp_path, "2026-10-13_19-00-00", "[19:01:00] A (B) [1]: x\n", lib, guild_id=111
    )
    newest = _rec(
        tmp_path, "2026-10-20_20-53-37", "[20:54:00] A (B) [1]: y\n", lib, guild_id=111
    )
    _rec(
        tmp_path, "2026-10-21_19-00-00", "[19:01:00] C (D) [2]: z\n", lib, guild_id=222
    )
    (tmp_path / "2026-10-22_19-00-00.txt").write_text("untagged\n")
    assert sessions.latest_for_guild(111, tmp_path) == newest
    assert sessions.latest_for_guild(333, tmp_path) is None
    assert sessions.latest_for_guild(None, tmp_path) is None
    assert sessions.latest_for_guild(111, tmp_path / "missing") is None


def test_model_text_drops_discord_ids_and_uses_12_hour_times():
    t = (
        "[21:12:05] Reiko Tanaka (Gus) [254468368969629697]: I'll call Fixie [later].\n"
        "--- recording restarted at 20:15:26 ---\n"
        "an unrecognised line [123]: kept as is\n"
    )
    assert sessions.model_text(t) == (
        "[9:12 PM] Reiko Tanaka (Gus): I'll call Fixie [later].\n"
        "--- recording restarted at 8:15 PM ---\n"
        "an unrecognised line [123]: kept as is\n"
    )


# ── earlier sessions (discord-transcript-bot-ij0) ─────────────────────


def _earlier_fixture(lib, t):
    _rec(
        t,
        "2026-10-06_19-00-00",
        "[19:01:00] Jim (Jill) [371]: we met the fixer\n",
        lib,
        guild_id=111,
    )
    _rec(
        t,
        "2026-10-13_19-00-00",
        "[19:01:00] Noah (GM) [420]: the deal went bad\n",
        lib,
        guild_id=111,
    )
    _rec(
        t,
        "2026-10-13_21-00-00",
        "[21:01:00] Jim (Jill) [371]: we ran\n",
        lib,
        guild_id=111,
    )
    # another campaign and game: never offered
    _rec(
        t,
        "2026-10-12_19-00-00",
        "[19:01:00] C (D) [2]: goblins\n",
        lib,
        guild_id=111,
        channel_id=999,
    )
    # untagged: never offered
    (t / "2026-10-01_19-00-00.txt").write_text("[19:00:00] X (Y) [9]: test\n")
    return _rec(
        t,
        "2026-10-20_19-38-38",
        "[19:40:00] Jim (Jill) [371]: tonight\n",
        lib,
        guild_id=111,
    )


def test_earlier_sessions_are_the_same_campaigns_previous_nights(lib, tmp_path):
    now = _earlier_fixture(lib, tmp_path)
    found = sessions.earlier_sessions(now)
    assert [(day, [f.name for f in files]) for day, files in found] == [
        ("2026-10-06", ["2026-10-06_19-00-00.txt"]),
        ("2026-10-13", ["2026-10-13_19-00-00.txt", "2026-10-13_21-00-00.txt"]),
    ]
    # the first session has nothing before it; an untagged file has none
    assert sessions.earlier_sessions(tmp_path / "2026-10-06_19-00-00.txt") == []
    assert sessions.earlier_sessions(tmp_path / "2026-10-01_19-00-00.txt") == []


def test_earlier_session_notes_hold_the_model_facing_transcript(lib, tmp_path):
    now = _earlier_fixture(lib, tmp_path)
    notes = sessions.earlier_session_notes(now)
    assert [n.id for n in notes] == ["session-2026-10-06", "session-2026-10-13"]
    last = notes[-1]
    assert last.title == "Earlier session, Tuesday, October 13, 2026"
    assert "speakers: Noah (GM), Jim (Jill)" in last.summary
    # joined across the restart, 12-hour times, no Discord ids
    assert last.body == (
        "[7:01 PM] Noah (GM): the deal went bad\n"
        "--- recording restarted at 9:00 PM ---\n"
        "[9:01 PM] Jim (Jill): we ran"
    )
    assert "tonight" not in "".join(n.body for n in notes)
    assert last.index_line().startswith("- session-2026-10-13 — Earlier session")


def test_a_game_and_group_tag_connects_without_a_campaign(lib, tmp_path):
    now = _earlier_fixture(lib, tmp_path)
    old = tmp_path / "2026-09-01_19-00-00.txt"
    old.write_text("[19:00:00] Jim (Jill) [371]: session zero\n")
    sessions.retag(old, lib, game="cyberpunk-red", group="noah-gm", whole_session=False)
    days = [day for day, _ in sessions.earlier_sessions(now)]
    assert days == ["2026-09-01", "2026-10-06", "2026-10-13"]


def test_another_groups_recordings_of_the_same_game_are_never_offered(lib, tmp_path):
    now = _earlier_fixture(lib, tmp_path)
    # same game, other table; and same game with no group at all
    other = tmp_path / "2026-09-02_19-00-00.txt"
    other.write_text("[19:00:00] C (D) [2]: their secret heist\n")
    sessions.retag(
        other, lib, game="cyberpunk-red", group="side-table", whole_session=False
    )
    loose = tmp_path / "2026-09-03_19-00-00.txt"
    loose.write_text("[19:00:00] E (F) [3]: nobody's table\n")
    sessions.retag(loose, lib, game="cyberpunk-red", whole_session=False)
    days = [day for day, _ in sessions.earlier_sessions(now)]
    assert days == ["2026-10-06", "2026-10-13"]
    assert "secret heist" not in "".join(
        n.body for n in sessions.earlier_session_notes(now)
    )


def test_an_oversized_earlier_session_keeps_its_most_recent_part(
    lib, tmp_path, monkeypatch
):
    now = _earlier_fixture(lib, tmp_path)
    monkeypatch.setenv("EARLIER_SESSION_MAX_CHARS", "30")
    body = sessions.earlier_session_notes(now)[-1].body
    assert body == sessions.EARLIER_TRIM_MARK + "\n[9:01 PM] Jim (Jill): we ran"


# ── campaign defaults and dashboard-made campaigns (discord-transcript-bot-5bd) ──


def _tagged(t, stamp, lib, **kw):
    return _rec(t, stamp, "[19:00:00] A (B) [1]: x\n", lib, **kw)


def test_default_tag_is_the_library_binding_until_something_is_recorded(lib, tmp_path):
    assert sessions.default_tag(lib, 111, 5, tmp_path) == (
        "noah-gm",
        "return-to-night-city",
    )
    assert sessions.default_tag(lib, 222, 5, tmp_path) == (None, None)


def test_default_tag_follows_the_campaign_used_last_in_the_server(lib, tmp_path):
    new = library.add_campaign(
        "Second Arc", game="cyberpunk-red", group="noah-gm", root=lib.root
    )
    lib = library.load(lib.root)
    _tagged(tmp_path, "2026-10-13_19-00-00", lib, guild_id=111, channel_id=5)
    last = _tagged(tmp_path, "2026-10-20_19-00-00", lib, guild_id=111, channel_id=5)
    sessions.retag(last, lib, campaign=new.id)
    assert sessions.default_tag(lib, 111, 5, tmp_path) == ("noah-gm", "second-arc")
    # an older recording being retagged does not change the default
    sessions.retag(tmp_path / "2026-10-13_19-00-00.txt", lib, campaign="lost-mine")
    assert sessions.default_tag(lib, 111, 5, tmp_path) == ("noah-gm", "second-arc")


def test_default_tag_never_borrows_another_tables_campaign(lib, tmp_path):
    # the side table shares the server through its own voice channel
    _tagged(tmp_path, "2026-10-20_19-00-00", lib, guild_id=111, channel_id=999)
    assert sessions.default_tag(lib, 111, 5, tmp_path) == (
        "noah-gm",
        "return-to-night-city",
    )
    assert sessions.default_tag(lib, 111, 999, tmp_path) == ("side-table", "lost-mine")
    # another server's recordings never count
    _tagged(tmp_path, "2026-10-21_19-00-00", lib, guild_id=333, campaign="lost-mine")
    assert sessions.default_tag(lib, 111, 5, tmp_path)[1] == "return-to-night-city"


def test_default_tag_for_an_unbound_server_needs_the_same_channel(lib, tmp_path):
    f = tmp_path / "2026-10-20_19-00-00.txt"
    f.write_text("x\n")
    sessions.write_meta(
        f,
        sessions.build_meta(
            lib,
            guild_id=222,
            channel_id=7,
            started_at=datetime(2026, 10, 20, 19),
            campaign="return-to-night-city",
        ),
    )
    assert sessions.default_tag(lib, 222, 7, tmp_path) == (
        "noah-gm",
        "return-to-night-city",
    )
    assert sessions.default_tag(lib, 222, 8, tmp_path) == (None, None)
    # a campaign the library no longer has is not a default
    sessions.write_meta(
        f,
        sessions.SessionMeta(
            session_day="2026-10-20", campaign="gone", guild_id=222, channel_id=7
        ),
    )
    assert sessions.default_tag(lib, 222, 7, tmp_path) == (None, None)


def test_add_campaign_writes_the_dashboard_file_and_a_notes_folder(lib):
    before = (lib.root / "library.yml").read_text()
    c = library.add_campaign(
        "  The  Night Market! ", game="cyberpunk-red", group="noah-gm", root=lib.root
    )
    assert (c.id, c.name, c.game, c.group) == (
        "the-night-market",
        "The Night Market!",
        "cyberpunk-red",
        "noah-gm",
    )
    assert (lib.root / "campaigns" / "the-night-market" / "notes").is_dir()
    assert (
        lib.root / "library.yml"
    ).read_text() == before  # hand-written file untouched
    again = library.load(lib.root)
    assert again.campaigns["the-night-market"].name == "The Night Market!"
    assert again.game_for("the-night-market") == "cyberpunk-red"
    assert "return-to-night-city" in again.campaigns
    library.add_campaign("Another", root=lib.root)
    assert {"the-night-market", "another"} <= set(library.load(lib.root).campaigns)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"name": "   "}, "name is required"),
        ({"name": "x" * 81}, "too long"),
        ({"name": "!!!"}, "at least one letter"),
        ({"name": "Ok", "game": "nope"}, "unknown game"),
        ({"name": "Ok", "group": "nope"}, "unknown group"),
        ({"name": "return to night city"}, "already exists"),
        ({"name": "Lost Mine"}, "already exists"),
    ],
)
def test_add_campaign_refuses_bad_input(lib, kwargs, message):
    with pytest.raises(library.LibraryError, match=message):
        library.add_campaign(root=lib.root, **kwargs)
    assert not (lib.root / library.DASHBOARD_FILE).exists()


def test_library_yml_wins_over_a_dashboard_entry_and_bad_files_are_ignored(lib):
    (lib.root / library.DASHBOARD_FILE).write_text(
        "campaigns:\n  lost-mine: {name: Hijacked}\n  extra: {name: Extra}\n"
    )
    loaded = library.load(lib.root)
    assert loaded.campaigns["lost-mine"].name == "Lost Mine"
    assert loaded.campaigns["extra"].name == "Extra"
    (lib.root / library.DASHBOARD_FILE).write_text("campaigns: [not, a, mapping")
    assert "lost-mine" in library.load(lib.root).campaigns


def test_scribe_line_names_the_campaign_or_says_there_is_none(lib):
    m = sessions.build_meta(lib, guild_id=111, started_at=datetime(2026, 10, 20, 19))
    assert sessions.scribe_line(lib, m).startswith(
        "\nCampaign: **Return to Night City** (Cyberpunk RED)."
    )
    assert "No campaign is set" in sessions.scribe_line(lib, None)
    untagged = sessions.build_meta(lib, guild_id=222, started_at=datetime(2026, 10, 20))
    assert "No campaign is set" in sessions.scribe_line(lib, untagged)


# ── roborev #7125 ─────────────────────────────────────────────────────


def test_retag_to_a_campaign_with_no_group_keeps_the_sessions_group(lib, tmp_path):
    loose = library.add_campaign("One Shot", game="cyberpunk-red", root=lib.root)
    lib = library.load(lib.root)
    f = _tagged(tmp_path, "2026-10-20_19-00-00", lib, guild_id=111, channel_id=5)
    # what the Live picker sends: the campaign alone
    sessions.retag(f, lib, campaign=loose.id)
    meta = sessions.read_meta(f)
    assert (meta.campaign, meta.group, meta.game) == (
        "one-shot",
        "noah-gm",
        "cyberpunk-red",
    )
    # the group's notes and GM are still part of the question
    assert "Noah" in sessions.header_text(lib, meta)
    # an explicit empty group still clears it (the tag editor)
    sessions.retag(f, lib, campaign=loose.id, group="")
    assert sessions.read_meta(f).group is None


def test_retag_to_a_campaign_with_no_game_keeps_the_sessions_game(lib, tmp_path):
    # roborev #7135: "New campaign…" with the game left empty
    bare = library.add_campaign("Side Job", group="noah-gm", root=lib.root)
    lib = library.load(lib.root)
    f = _tagged(tmp_path, "2026-10-20_19-00-00", lib, guild_id=111, channel_id=5)
    sessions.retag(f, lib, campaign=bare.id)
    meta = sessions.read_meta(f)
    assert (meta.campaign, meta.group, meta.game) == (
        "side-job",
        "noah-gm",
        "cyberpunk-red",
    )
    assert "Cyberpunk RED" in sessions.header_text(lib, meta)
    # an explicit empty game still clears it (the tag editor)
    sessions.retag(f, lib, campaign=bare.id, game="")
    assert sessions.read_meta(f).game is None


def test_default_tag_skips_a_groupless_campaign_from_another_tables_channel(
    lib, tmp_path
):
    loose = library.add_campaign("One Shot", root=lib.root)
    lib = library.load(lib.root)
    f = tmp_path / "2026-10-20_19-00-00.txt"
    f.write_text("x\n")
    # recorded in the side table's channel, with no group on the tag
    sessions.write_meta(
        f,
        sessions.SessionMeta(
            session_day="2026-10-20", campaign=loose.id, guild_id=111, channel_id=999
        ),
    )
    assert sessions.default_tag(lib, 111, 5, tmp_path) == (
        "noah-gm",
        "return-to-night-city",
    )
    assert sessions.default_tag(lib, 111, 999, tmp_path) == ("side-table", "one-shot")
    # the same groupless campaign played in this table's own channel counts
    g = tmp_path / "2026-10-21_19-00-00.txt"
    g.write_text("x\n")
    sessions.write_meta(
        g,
        sessions.SessionMeta(
            session_day="2026-10-21", campaign=loose.id, guild_id=111, channel_id=5
        ),
    )
    assert sessions.default_tag(lib, 111, 5, tmp_path) == ("noah-gm", "one-shot")


def test_add_campaign_at_the_same_moment_keeps_every_campaign(lib):
    import threading

    names = [f"Arc {i}" for i in range(12)]
    errors = []
    go = threading.Barrier(len(names))

    def create(name):
        go.wait()
        try:
            library.add_campaign(name, root=lib.root)
        except Exception as e:  # noqa: BLE001 - any failure fails the test
            errors.append(e)

    threads = [threading.Thread(target=create, args=(n,)) for n in names]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    assert {f"arc-{i}" for i in range(12)} <= set(library.load(lib.root).campaigns)
    assert not list(lib.root.glob("*.tmp"))
