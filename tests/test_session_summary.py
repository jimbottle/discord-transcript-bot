"""Session ids and Discord session summaries (discord-transcript-bot-qef)."""

import re

import pytest

from src import recaps, session_summary, sessions
from src.llm.common import AskFailed, AskResult
from tests import test_library_sessions as _base
from tests.test_library_sessions import _earlier_fixture, _rec

lib = _base.lib  # the same library fixture, under its usual name

ANSWER = "TOPICS: Fixie, Militech\n\nAt 7:01 PM the deal went bad.\nThen they ran."


def _fake_ask(seen=None, answer=ANSWER):
    def ask(transcript, question, *, header=""):
        if seen is not None:
            seen.append(transcript)
        return AskResult(answer=answer, model="test-model")

    return ask


def _night_13(t):
    return [t / "2026-10-13_19-00-00.txt", t / "2026-10-13_21-00-00.txt"]


# ── session ids ───────────────────────────────────────────────────────


def test_a_session_id_is_random_typeable_and_shared_by_the_session(lib, tmp_path):
    _earlier_fixture(lib, tmp_path)
    first, late = _night_13(tmp_path)
    assert sessions.read_meta(first).session_id is None  # tagged before ids
    sid = sessions.ensure_session_id(late)
    assert re.fullmatch(r"[A-HJKMNP-Z2-9]{4}-[A-HJKMNP-Z2-9]{4}", sid)
    assert sessions.read_meta(first).session_id == sid
    assert sessions.read_meta(late).session_id == sid
    assert sessions.ensure_session_id(first) == sid  # stable
    # another night gets its own
    other = sessions.ensure_session_id(tmp_path / "2026-10-06_19-00-00.txt")
    assert other != sid
    assert len({sessions.new_session_id() for _ in range(200)}) == 200


def test_ensure_session_id_is_none_for_an_untagged_recording(lib, tmp_path):
    _earlier_fixture(lib, tmp_path)
    assert sessions.ensure_session_id(tmp_path / "2026-10-01_19-00-00.txt") is None
    assert sessions.ensure_session_id(tmp_path / "missing.txt") is None


def test_a_restarted_recording_joins_the_sessions_id_before_anyone_speaks(
    lib, tmp_path
):
    """/scribe tags the recording before its .txt exists."""
    _earlier_fixture(lib, tmp_path)
    sid = sessions.ensure_session_id(_night_13(tmp_path)[0])
    restart = tmp_path / "2026-10-13_22-30-00.txt"  # not created yet
    meta = sessions.build_meta(
        lib, guild_id=111, started_at=sessions.started_at_from_name(restart)
    )
    sessions.write_meta(restart, meta)
    assert sessions.ensure_session_id(restart) == sid


def test_find_by_session_id_ignores_case_dashes_and_spaces(lib, tmp_path):
    _earlier_fixture(lib, tmp_path)
    first, late = _night_13(tmp_path)
    sid = sessions.ensure_session_id(first)
    for typed in (sid, sid.lower(), sid.replace("-", ""), f"  {sid[:4]} {sid[5:]} "):
        assert sessions.find_by_session_id(tmp_path, typed) == late  # newest
    for wrong in ("", None, "AAAA-AAAA", sid[:-1], sid + "X", "2026-10-13"):
        assert sessions.find_by_session_id(tmp_path, wrong) is None


def test_retagging_one_recording_away_does_not_carry_the_id_along(lib, tmp_path):
    _earlier_fixture(lib, tmp_path)
    first, late = _night_13(tmp_path)
    sid = sessions.ensure_session_id(first)
    sessions.retag(late, lib, campaign="", whole_session=False)
    moved = sessions.read_meta(late)
    assert moved.key != sessions.read_meta(first).key
    assert moved.session_id and moved.session_id != sid
    assert sessions.read_meta(first).session_id == sid
    # retagging the whole session keeps its id
    sessions.retag(first, lib, campaign="return-to-night-city")
    assert sessions.read_meta(first).session_id == sid


def test_the_id_survives_the_tag_round_trip_and_reaches_describe(lib, tmp_path):
    _earlier_fixture(lib, tmp_path)
    first = _night_13(tmp_path)[0]
    sid = sessions.ensure_session_id(first)
    assert sessions.describe(lib, sessions.read_meta(first))["session_id"] == sid


# ── summaries ─────────────────────────────────────────────────────────


def test_find_needs_the_right_id_and_the_right_server(lib, tmp_path):
    _earlier_fixture(lib, tmp_path)
    first, late = _night_13(tmp_path)
    sid = sessions.ensure_session_id(first)
    assert session_summary.find(tmp_path, sid, guild_id=111) == late
    for args in ((tmp_path, "AAAA-AAAA", 111), (tmp_path, sid, 222)):
        with pytest.raises(session_summary.SummaryError) as e:
            session_summary.find(*args)
        # another server's session reads exactly like an unknown id
        assert e.value.user_message == session_summary.NOT_FOUND
    assert sid not in session_summary.NOT_FOUND


def test_summarise_writes_the_recap_once_then_reuses_it(lib, tmp_path):
    now = _earlier_fixture(lib, tmp_path)
    first, late = _night_13(tmp_path)
    seen = []
    s = session_summary.summarise(late, lib, ask=_fake_ask(seen))
    assert s.generated and s.files == [first, late]
    assert "the deal went bad" in seen[0] and "we ran" in seen[0]
    # saved where /ask looks for it
    assert recaps.load([first, late]) == s.recap
    assert "recap-2026-10-13" in [n.id for n in sessions.earlier_session_notes(now)]
    again = session_summary.summarise(first, lib, ask=_fake_ask(seen))
    assert not again.generated and again.recap == s.recap and len(seen) == 1
    fresh = session_summary.summarise(first, lib, refresh=True, ask=_fake_ask(seen))
    assert fresh.generated and len(seen) == 2


def test_a_session_still_being_recorded_is_refused(lib, tmp_path):
    _earlier_fixture(lib, tmp_path)
    first, late = _night_13(tmp_path)
    with pytest.raises(session_summary.SummaryError) as e:
        session_summary.summarise(
            first, lib, recording=[None, f"transcripts/{late.name}"], ask=_fake_ask()
        )
    assert e.value.user_message == session_summary.STILL_RECORDING
    # some other recording being live does not block it
    session_summary.summarise(
        first, lib, recording=["transcripts/2026-11-10_20-04-24.txt"], ask=_fake_ask()
    )


def test_silent_and_untagged_sessions_have_no_summary(lib, tmp_path):
    _earlier_fixture(lib, tmp_path)
    silent = tmp_path / "2026-10-27_19-00-00.txt"  # tagged, nobody spoke
    sessions.write_meta(
        silent,
        sessions.build_meta(
            lib, guild_id=111, started_at=sessions.started_at_from_name(silent)
        ),
    )
    with pytest.raises(session_summary.SummaryError) as e:
        session_summary.summarise(silent, lib, ask=_fake_ask())
    assert e.value.user_message == session_summary.NOTHING_SAID
    with pytest.raises(session_summary.SummaryError) as e:
        session_summary.summarise(tmp_path / "2026-10-01_19-00-00.txt", lib)
    assert e.value.user_message == session_summary.NOTHING_RECORDED


def test_a_failed_model_call_leaves_no_lock_behind(lib, tmp_path):
    _earlier_fixture(lib, tmp_path)
    first = _night_13(tmp_path)[0]

    def down(transcript, question, *, header=""):
        raise AskFailed("The model server is down.")

    with pytest.raises(AskFailed):
        session_summary.summarise(first, lib, ask=down)
    assert session_summary.summarise(first, lib, ask=_fake_ask()).generated


def test_a_second_request_for_the_same_session_does_not_pay_twice(lib, tmp_path):
    _earlier_fixture(lib, tmp_path)
    first = _night_13(tmp_path)[0]
    inner = {}

    def slow(transcript, question, *, header=""):
        with pytest.raises(session_summary.SummaryError) as e:
            session_summary.summarise(first, lib, ask=_fake_ask())
        inner["msg"] = e.value.user_message
        return AskResult(answer=ANSWER, model="m")

    session_summary.summarise(first, lib, ask=slow)
    assert inner["msg"] == session_summary.ALREADY_RUNNING


def test_messages_fit_discord_and_never_contain_the_id(lib, tmp_path):
    _earlier_fixture(lib, tmp_path)
    first = _night_13(tmp_path)[0]
    sid = sessions.ensure_session_id(first)
    long = "TOPICS: a\n\n" + "\n\n".join(
        f"Paragraph {i}. " + "word " * 80 for i in range(12)
    )
    s = session_summary.summarise(first, lib, ask=_fake_ask(answer=long))
    parts = session_summary.messages(lib, s)
    assert len(parts) > 1 and all(len(p) <= 2000 for p in parts)
    assert parts[0].startswith(
        "**Session summary: Return to Night City, Tuesday, October 13, 2026**"
    )
    joined = "\n".join(parts)
    assert "Paragraph 11." in joined and "TOPICS" not in joined
    assert sid not in joined and sid.replace("-", "") not in joined


# ── the Discord commands (no slash-command harness: source checks) ────


def _main_source():
    import main

    with open(main.__file__, encoding="utf-8") as f:
        return f.read()


def test_summary_command_hides_the_id_and_works_off_the_event_loop():
    src = _main_source()
    body = src[src.index('name="summary"') : src.index('name="disconnect"')]
    assert "ctx.defer(ephemeral=True)" in body
    assert "asyncio.to_thread" in body and "session_summary.find" in body
    assert "InteractionContextType.guild}" in body  # no DMs: needs a server
    deliver = src[src.index("async def _deliver_summary") : src.index('name="summary"')]
    assert "asyncio.to_thread" in deliver and "session_summary.summarise" in deliver


def test_disconnect_summarises_only_what_this_process_recorded():
    src = _main_source()
    body = src[src.index('name="disconnect"') :]
    body = body[: body.index("@bot.slash_command", 10)]
    assert "summary: discord.Option(" in body and "default=False" in body
    assert "bot.guild_last_session_file.get(guild_id)" in body
    assert "session_file_for_export" not in body  # never an older night from disk


def test_discord_never_prints_the_session_id():
    """Knowing the id is what allows /summary, so only the dashboard shows
    it: main.py and src/bot/ may pass it around but never read it out."""
    import pathlib

    import main

    root = pathlib.Path(main.__file__).parent
    for path in [root / "main.py", *sorted((root / "src" / "bot").glob("*.py"))]:
        text = path.read_text(encoding="utf-8")
        assert '["session_id"]' not in text and "['session_id']" not in text, path
        for line in text.splitlines():
            if ".session_id" in line:
                assert "ensure_session_id" in line, (path, line)


def test_summary_failures_are_reported_even_after_the_token_expires():
    """roborev #7151: every error reply goes through the helper that falls
    back to the channel, not a bare follow-up that can raise."""
    src = _main_source()
    deliver = src[src.index("async def _deliver_summary") : src.index('name="summary"')]
    errors = deliver[deliver.index("except session_summary.SummaryError") :]
    errors = errors[: errors.index("parts = session_summary.messages")]
    assert errors.count("await _tell(") == 3 and "ctx.followup.send" not in errors
    tell = deliver[deliver.index("async def _tell") : deliver.index("# Blocking")]
    assert "ctx.followup.send(text, ephemeral=True)" in tell
    assert "ctx.channel.send(text)" in tell


def test_the_tag_script_keeps_a_sessions_id(lib, tmp_path, monkeypatch):
    """roborev #7151: retagging by script used to drop the id, so the one
    the GM had copied for /summary stopped working."""
    import importlib.util
    from pathlib import Path

    from src import library

    spec = importlib.util.spec_from_file_location(
        "tag_session", Path(library.PROJECT_ROOT) / "scripts" / "tag_session.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _earlier_fixture(lib, tmp_path)
    monkeypatch.setenv("LIBRARY_DIR", str(tmp_path))
    first, late = _night_13(tmp_path)
    sid = sessions.ensure_session_id(first)
    args = [str(first), str(late), "--campaign", "return-to-night-city"]
    assert mod.main(args) == 0
    assert sessions.read_meta(first).session_id == sid
    assert sessions.read_meta(late).session_id == sid
    assert sessions.find_by_session_id(tmp_path, sid) == late
    # moved to another session: it does not take the old id with it
    assert mod.main([str(late), "--game", "dnd-5e", "--group", "side-table"]) == 0
    moved = sessions.read_meta(late)
    assert moved.campaign is None and moved.session_id and moved.session_id != sid


def test_find_fails_closed_when_the_sessions_server_is_unknown(lib, tmp_path):
    """Security review: a session with no server on its tag used to be
    summarisable from any server. Now its group's servers decide, and
    with neither it is not found."""
    _earlier_fixture(lib, tmp_path)
    # tagged by hand: a campaign (group noah-gm, bound to server 111), no guild
    manual = _rec(
        tmp_path,
        "2026-09-08_19-00-00",
        "[19:01:00] Jim (Jill) [371]: by hand\n",
        lib,
        campaign="return-to-night-city",
    )
    assert sessions.read_meta(manual).guild_id is None
    sid = sessions.ensure_session_id(manual)
    assert session_summary.owner_guilds(sessions.read_meta(manual), lib) == {111}
    assert session_summary.find(tmp_path, sid, 111, lib) == manual
    for guild in (222, None):
        with pytest.raises(session_summary.SummaryError) as e:
            session_summary.find(tmp_path, sid, guild, lib)
        assert e.value.user_message == session_summary.NOT_FOUND
    # no server and no group that names one: nobody may summarise it
    loose = tmp_path / "2026-09-09_19-00-00.txt"
    loose.write_text("[19:01:00] E (F) [3]: nobody's table\n")
    sessions.retag(loose, lib, game="cyberpunk-red", whole_session=False)
    loose_id = sessions.ensure_session_id(loose)
    assert (
        loose_id
        and session_summary.owner_guilds(sessions.read_meta(loose), lib) == set()
    )
    with pytest.raises(session_summary.SummaryError):
        session_summary.find(tmp_path, loose_id, 111, lib)


def test_the_tag_script_keeps_the_id_when_a_whole_night_changes_campaign(
    lib, tmp_path, monkeypatch
):
    """roborev #7152: the documented usage, on a night that had no
    campaign, changes every file's key; the id must survive that."""
    import importlib.util
    from pathlib import Path

    from src import library

    spec = importlib.util.spec_from_file_location(
        "tag_session", Path(library.PROJECT_ROOT) / "scripts" / "tag_session.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _earlier_fixture(lib, tmp_path)
    monkeypatch.setenv("LIBRARY_DIR", str(tmp_path))
    first, late = _night_13(tmp_path)
    sessions.retag(first, lib, campaign="")  # the whole night: no campaign
    before = sessions.read_meta(first).key
    sid = sessions.ensure_session_id(first)
    assert mod.main([str(first), str(late), "--campaign", "return-to-night-city"]) == 0
    assert sessions.read_meta(first).key != before
    assert sessions.read_meta(first).session_id == sid
    assert sessions.read_meta(late).session_id == sid
    # one recording moved into a night that has its own id takes that id
    other = tmp_path / "2026-10-06_19-00-00.txt"
    other_id = sessions.ensure_session_id(other)
    stray = tmp_path / "2026-10-06_23-00-00.txt"
    stray.write_text("x")
    sessions.retag(stray, lib, game="cyberpunk-red", group="noah-gm")
    stray_id = sessions.ensure_session_id(stray)
    assert stray_id not in (None, other_id)
    assert mod.main([str(stray), "--campaign", "return-to-night-city"]) == 0
    assert sessions.read_meta(stray).session_id == other_id
    assert sessions.read_meta(other).session_id == other_id
