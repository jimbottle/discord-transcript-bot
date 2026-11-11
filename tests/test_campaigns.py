"""The campaign data model (src/campaigns.py; discord-transcript-bot-4n1)."""

from datetime import datetime

import pytest

from src import campaigns, library, sessions

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
campaigns:
  return-to-night-city: {name: Return to Night City, group: noah-gm, game: cyberpunk-red}
  lost-mine: {name: Lost Mine, game: dnd-5e}
  orphan: {name: An Orphan, group: gone-group, game: gone-game}
"""


@pytest.fixture
def lib(tmp_path):
    root = tmp_path / "lib"
    root.mkdir()
    (root / "library.yml").write_text(LIB, encoding="utf-8")
    for d in ("games/cyberpunk-red/notes", "campaigns/return-to-night-city/notes"):
        (root / d).mkdir(parents=True)
    return library.load(root)


@pytest.fixture
def tdir(tmp_path):
    d = tmp_path / "transcripts"
    d.mkdir()
    return d


def _rec(tdir, stamp, text, lib, **kw):
    txt = tdir / f"{stamp}.txt"
    txt.write_text(text, encoding="utf-8")
    sessions.write_meta(
        txt,
        sessions.build_meta(lib, started_at=sessions.started_at_from_name(txt), **kw),
    )
    return txt


@pytest.fixture
def played(lib, tdir):
    """Two nights of the campaign (the second split by a restart), one night
    of another campaign, and an untagged test recording."""
    a = _rec(
        tdir,
        "2026-10-13_19-00-00",
        "[19:01:00] Noah (GM) [1]: last week\n",
        lib,
        guild_id=111,
    )
    b = _rec(
        tdir,
        "2026-10-20_19-38-38",
        "[19:40:00] Jim (Jill) [2]: before\n",
        lib,
        guild_id=111,
    )
    c = _rec(
        tdir,
        "2026-10-20_20-15-26",
        "[20:16:00] Jim (Jill) [2]: after\n",
        lib,
        guild_id=111,
    )
    d = _rec(
        tdir,
        "2026-10-21_19-00-00",
        "[19:00:00] X (Y) [3]: goblins\n",
        lib,
        campaign="lost-mine",
    )
    e = tdir / "2026-05-18_17-39-56.txt"
    e.write_text("[17:40:00] X (Y) [3]: mic test\n")
    return a, b, c, d, e


def test_a_campaign_knows_its_game_group_notes_and_sessions(lib, tdir, played):
    a, b, c, _d, _e = played
    model = campaigns.Model(tdir, lib)
    rtnc = model.campaign("return-to-night-city")
    assert (rtnc.name, rtnc.game.name, rtnc.group.name, rtnc.group.gm) == (
        "Return to Night City",
        "Cyberpunk RED",
        "Noah's group",
        "Noah",
    )
    assert [p.parent.name for p in rtnc.note_dirs] == [
        "cyberpunk-red",
        "return-to-night-city",
    ]
    assert [(s.day, s.file_names) for s in rtnc.sessions] == [
        ("2026-10-13", [a.name]),
        ("2026-10-20", [b.name, c.name]),
    ]
    assert rtnc.last_played == "2026-10-20"
    assert [s.day for s in rtnc.sessions_before("2026-10-20")] == ["2026-10-13"]
    assert rtnc.choice() == {
        "id": "return-to-night-city",
        "name": "Return to Night City",
        "group": "noah-gm",
        "game": "cyberpunk-red",
    }


def test_campaigns_are_listed_by_name_and_survive_dangling_ids(lib, tdir, played):
    model = campaigns.Model(tdir, lib)
    assert [c.id for c in model.campaigns()] == [
        "orphan",
        "lost-mine",
        "return-to-night-city",
    ]
    orphan = model.campaign("orphan")
    # the ids are kept; the objects they no longer name are None
    assert (orphan.game, orphan.group) == (None, None)
    assert (orphan.game_id, orphan.group_id) == ("gone-game", "gone-group")
    assert orphan.sessions == () and orphan.last_played is None
    assert model.campaign("nope") is None and model.campaign(None) is None
    assert model.campaign("lost-mine").last_played == "2026-10-21"


def test_a_session_is_every_recording_of_the_night(lib, tdir, played):
    _a, b, c, _d, e = played
    model = campaigns.Model(tdir, lib)
    s = model.session_of(c)
    assert (s.key, s.day, s.recordings, s.tagged) == (
        "return-to-night-city/2026-10-20",
        "2026-10-20",
        2,
        True,
    )
    assert (s.campaign_id, s.game_id, s.group_id) == (
        "return-to-night-city",
        "cyberpunk-red",
        "noah-gm",
    )
    assert s.files == (b, c)
    assert s.transcript() == (
        "[19:40:00] Jim (Jill) [2]: before\n"
        "--- recording restarted at 20:15:26 ---\n"
        "[20:16:00] Jim (Jill) [2]: after\n"
    )
    assert "[7:40 PM] Jim (Jill): before" in s.model_text()
    assert model.campaign_of(c).id == "return-to-night-city"
    # an untagged recording is a session of its own, in no campaign
    lone = model.session_of(e)
    assert (lone.key, lone.day, lone.files, lone.tagged) == (None, None, (e,), False)
    assert model.campaign_of(e) is None


def test_sessions_lists_every_tagged_night_or_one_campaigns(lib, tdir, played):
    model = campaigns.Model(tdir, lib)
    assert [s.key for s in model.sessions()] == [
        "return-to-night-city/2026-10-13",
        "return-to-night-city/2026-10-20",
        "lost-mine/2026-10-21",
    ]
    assert [s.key for s in model.sessions("lost-mine")] == ["lost-mine/2026-10-21"]
    assert campaigns.Model(None, lib).sessions() == []
    assert campaigns.Model(tdir / "missing", lib).sessions() == []


def test_describe_matches_the_storage_layer_and_adds_files(lib, tdir, played):
    _a, b, c, _d, e = played
    model = campaigns.Model(tdir, lib)
    d = model.describe(c)
    assert d == {
        **sessions.describe(lib, sessions.read_meta(c), 2),
        "files": [b.name, c.name],
    }
    assert d["campaign"] == "Return to Night City" and d["recordings"] == 2
    assert model.describe(e) == {"tagged": False, "recordings": 1, "files": [e.name]}


def test_ask_context_carries_the_session_its_campaign_and_earlier_notes(
    lib, tdir, played
):
    _a, b, c, _d, e = played
    asked = campaigns.for_recording(c, lib).ask_context(c)
    # the same four things the storage layer gives a question
    assert (asked.transcript, asked.header, asked.note_dirs, asked.files) == tuple(
        sessions.ask_context(c, lib)
    )
    assert asked.header.startswith("SESSION: Game: Cyberpunk RED")
    assert asked.session.key == "return-to-night-city/2026-10-20"
    assert asked.campaign.id == "return-to-night-city"
    assert [n.id for n in asked.earlier_notes()] == ["session-2026-10-13"]
    # untagged: transcript only
    lone = campaigns.for_recording(e, lib).ask_context(e)
    assert lone.transcript and (lone.header, lone.note_dirs) == ("", [])
    assert lone.campaign is None and lone.earlier_notes() == []
    # nothing asked about
    empty = campaigns.AskContext()
    assert (empty.transcript, empty.header, empty.note_dirs, empty.files) == (
        "",
        "",
        [],
        [],
    )
    assert empty.earlier_notes() == []


def test_default_for_is_the_campaign_used_last_in_the_server(lib, tdir, played):
    model = campaigns.Model(tdir, lib)
    assert model.default_for(111, 5) == ("noah-gm", "return-to-night-city")
    assert model.default_for(222, 5) == (None, None)
    assert model.default_for(111, 5) == sessions.default_tag(lib, 111, 5, tdir)


def test_campaign_header_is_the_lore_header(lib, tdir):
    model = campaigns.Model(tdir, lib)
    assert model.campaign_header("return-to-night-city") == sessions.campaign_header(
        lib, "return-to-night-city"
    )
    assert model.campaign_header("nope") == ""


def test_the_model_is_a_snapshot_of_the_tags(lib, tdir, played):
    _a, _b, c, _d, _e = played
    model = campaigns.Model(tdir, lib)
    assert len(model.campaign("return-to-night-city").sessions) == 2
    sessions.retag(c, lib, campaign="lost-mine")
    # this model already read the tags; a new one sees the change
    assert len(model.sessions("return-to-night-city")) == 2
    fresh = campaigns.Model(tdir, lib)
    assert [s.day for s in fresh.sessions("return-to-night-city")] == ["2026-10-13"]
    assert [s.day for s in fresh.sessions("lost-mine")] == ["2026-10-20", "2026-10-21"]


def test_a_broken_tag_file_is_skipped_not_fatal(lib, tdir, played):
    _a, _b, c, _d, _e = played
    sessions.meta_path(c).write_text("{not json")
    model = campaigns.Model(tdir, lib)
    assert [s.recordings for s in model.sessions("return-to-night-city")] == [1, 1]
    assert model.session_of(c).tagged is False


def test_build_meta_dates_from_the_recording(lib):
    # guards the fixture helper: the session day comes from the file name
    m = sessions.build_meta(lib, guild_id=111, started_at=datetime(2026, 10, 20, 19))
    assert m.session_day == "2026-10-20"
