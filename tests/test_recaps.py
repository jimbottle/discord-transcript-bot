"""Cached session recaps (discord-transcript-bot-fkj): storage, parsing,
how they reach /ask's notes index, and the script that writes them."""

import importlib.util
import json
from pathlib import Path

import pytest

from src import library, recaps, sessions
from src.llm.common import AskResult
from tests import test_library_sessions as _base
from tests.test_library_sessions import _earlier_fixture, _rec

lib = _base.lib  # the same library fixture, under its usual name

ANSWER = (
    "TOPICS: Fixie, the Afterlife, Militech deal, stolen datashard\n"
    "\n"
    "At 7:01 PM the deal with Militech went bad.\n"
    "At 9:01 PM the crew ran."
)


def _fake_ask(answer=ANSWER, seen=None):
    def ask(transcript, question, *, header="", **context):
        if seen is not None:
            seen.append(
                {"transcript": transcript, "q": question, "header": header, **context}
            )
        return AskResult(answer=answer, model="test-model")

    return ask


def _night_13(t):
    return [t / "2026-10-13_19-00-00.txt", t / "2026-10-13_21-00-00.txt"]


# ── parsing ───────────────────────────────────────────────────────────


def test_parse_splits_topics_from_the_recap():
    topics, summary, text = recaps.parse(ANSWER)
    assert topics == "Fixie, the Afterlife, Militech deal, stolen datashard"
    assert summary == ""  # no SUMMARY section: everything is recap
    assert text.startswith("At 7:01 PM") and "TOPICS" not in text


FULL = (
    "TOPICS: Fixie, Militech\n"
    "\n"
    "SUMMARY:\n"
    "The crew's deal with Militech went sideways, choom.\n"
    "\n"
    "They delta'd before the corpos closed in.\n"
    "\n"
    "RECAP:\n"
    "At 7:01 PM the deal with Militech went bad.\n"
    "At 9:01 PM the crew ran."
)


def test_parse_separates_the_summary_from_the_reference_recap():
    topics, summary, text = recaps.parse(FULL)
    assert topics == "Fixie, Militech"
    assert summary == (
        "The crew's deal with Militech went sideways, choom.\n\n"
        "They delta'd before the corpos closed in."
    )
    assert (
        text == "At 7:01 PM the deal with Militech went bad.\nAt 9:01 PM the crew ran."
    )
    # markdown dressing, text on the label's own line, and either order
    dressed = (
        "**TOPICS:** a, b\n\n## Recap\nplain facts\n\n**SUMMARY:** short and themed"
    )
    assert recaps.parse(dressed) == ("a, b", "short and themed", "plain facts")
    # a line that merely starts with the word is not a label
    assert recaps.parse("SUMMARY:\nx\nRECAP:\nRecapping the night, they left.")[2] == (
        "Recapping the night, they left."
    )


def test_parse_accepts_markdown_dressing_and_a_missing_topics_line():
    assert recaps.parse("**TOPICS:** a, b\n\nthe recap") == ("a, b", "", "the recap")
    # a sentence that begins with a label's word is not a label
    assert recaps.parse("Summary of the heist: it failed.")[2].startswith("Summary of")
    assert recaps.parse("Just the recap.") == ("", "", "Just the recap.")
    assert recaps.parse("") == ("", "", "")


def test_parse_cuts_long_topics_at_a_comma():
    many = ", ".join(f"topic number {i}" for i in range(60))
    topics, _, _ = recaps.parse(f"TOPICS: {many}\n\nrecap")
    assert len(topics) <= recaps.TOPICS_MAX_CHARS
    assert topics.endswith(tuple("0123456789")) and not topics.endswith(",")


def test_the_summary_is_stored_and_the_style_goes_in_the_system_header(lib, tmp_path):
    _earlier_fixture(lib, tmp_path)
    files = _night_13(tmp_path)
    meta = sessions.read_meta(files[0])
    style = recaps.summary_style(lib, meta)
    assert style.startswith("SUMMARY STYLE: ") and "Night City" in style
    seen = []
    recap = recaps.generate(
        files,
        header="SESSION: Game: X\nSTYLE: a light touch",
        style=style,
        ask=_fake_ask(FULL, seen=seen),
    )
    # the light /ask voice is replaced by the summary's own directions
    assert seen[0]["header"] == (
        "SESSION: Game: X\n" + recaps.SUMMARY_RULES + "\n" + style
    )
    assert "SUMMARY STYLE" in seen[0]["q"] and "game world" in seen[0]["q"]
    assert recap.summary.startswith("The crew's deal") and "choom" in recap.summary
    assert "choom" not in recap.text  # the reference stays plain
    assert recaps.load(files).summary == recap.summary
    # /ask is offered the plain recap, never the themed summary
    note = recaps.note("2026-10-13", "Tuesday, October 13, 2026", recap)
    assert "choom" not in note.body and "7:01 PM" in note.body


def test_summary_style_per_game(lib):
    from types import SimpleNamespace as NS

    assert "bard" in recaps.summary_style(lib, NS(game="dnd-5e"))
    assert recaps.summary_style(lib, NS(game=None)) == ""
    assert recaps.summary_style(lib, NS(game="unknown-game")) == ""
    # any other game with a voice gets directions built from its name
    lib.games["mothership"] = NS(name="Mothership", voice="grim")
    other = recaps.summary_style(lib, NS(game="mothership"))
    assert other.startswith("SUMMARY STYLE: ") and "Mothership" in other
    # voice switched off in library.yml: no theming
    lib.games["cyberpunk-red"].voice = ""
    assert recaps.summary_style(lib, NS(game="cyberpunk-red")) == ""


# ── storage ───────────────────────────────────────────────────────────


def test_generate_writes_a_recap_beside_the_first_recording(lib, tmp_path):
    _earlier_fixture(lib, tmp_path)
    files = _night_13(tmp_path)
    seen = []
    recap = recaps.generate(
        files, header="SESSION: Game: X\nSTYLE: neon slang", ask=_fake_ask(seen=seen)
    )
    assert recap.topics.startswith("Fixie") and recap.model == "test-model"
    path = tmp_path / "2026-10-13_19-00-00.recap.json"
    assert json.loads(path.read_text())["recap"] == recap.text
    # the model read the whole night, as /ask would, without the voice
    assert "the deal went bad" in seen[0]["transcript"]
    assert "we ran" in seen[0]["transcript"]
    assert "[371]" not in seen[0]["transcript"]
    assert seen[0]["header"] == "SESSION: Game: X\n" + recaps.SUMMARY_RULES
    assert recaps.load(files) == recap
    assert recaps.load(list(reversed(files))) == recap


def test_a_recap_is_dropped_when_the_session_changes(lib, tmp_path):
    _earlier_fixture(lib, tmp_path)
    files = _night_13(tmp_path)
    recaps.generate(files, ask=_fake_ask())
    # a recording grew
    with files[1].open("a") as f:
        f.write("[21:05:00] Jim (Jill) [371]: one more thing\n")
    assert recaps.load(files) is None
    recaps.generate(files, ask=_fake_ask())
    # a recording left the session
    assert recaps.load(files[:1]) is None
    assert recaps.load(files) is not None


def test_load_never_raises_on_a_bad_file(lib, tmp_path):
    _earlier_fixture(lib, tmp_path)
    files = _night_13(tmp_path)
    path = recaps.recap_path(files)
    assert recaps.load(files) is None
    path.write_text("{not json")
    assert recaps.load(files) is None
    path.write_text(json.dumps(["a list"]))
    assert recaps.load(files) is None
    path.write_text(json.dumps({"files": recaps.fingerprint(files), "recap": " "}))
    assert recaps.load(files) is None
    assert recaps.load([]) is None


def test_generate_refuses_an_empty_session_or_an_empty_answer(lib, tmp_path):
    _earlier_fixture(lib, tmp_path)
    empty = tmp_path / "2026-10-14_19-00-00.txt"
    empty.write_text("")
    with pytest.raises(recaps.RecapError):
        recaps.generate([empty], ask=_fake_ask())
    with pytest.raises(recaps.RecapError):
        recaps.generate(_night_13(tmp_path), ask=_fake_ask("TOPICS: a, b\n"))
    assert not recaps.recap_path(_night_13(tmp_path)).exists()


# ── what /ask is offered ──────────────────────────────────────────────


def test_a_cached_recap_is_listed_ahead_of_its_transcript(lib, tmp_path):
    now = _earlier_fixture(lib, tmp_path)
    recaps.generate(_night_13(tmp_path), ask=_fake_ask())
    notes = sessions.earlier_session_notes(now)
    # the 6th has no recap: its transcript is offered exactly as before
    assert [n.id for n in notes] == [
        "session-2026-10-06",
        "recap-2026-10-13",
        "session-2026-10-13",
    ]
    old, recap, full = notes
    assert "recap" not in old.summary
    assert "topics: Fixie, the Afterlife" in recap.index_line()
    assert recap.body.startswith("At 7:01 PM the deal with Militech went bad.")
    assert "session-2026-10-13" in recap.body  # where the detail is
    assert "read recap-2026-10-13 first" in full.summary
    assert "the deal went bad" in full.body
    # the recap is a fraction of what rides in the index and is read first
    assert len(recap.body) < 400


def test_the_recap_follows_the_session_not_the_file(lib, tmp_path):
    now = _earlier_fixture(lib, tmp_path)
    recaps.generate(_night_13(tmp_path), ask=_fake_ask())
    # a third recording of that night appears: the recap no longer covers it
    _rec(
        tmp_path,
        "2026-10-13_22-00-00",
        "[22:01:00] Jim (Jill) [371]: epilogue\n",
        lib,
        guild_id=111,
    )
    assert [n.id for n in sessions.earlier_session_notes(now)] == [
        "session-2026-10-06",
        "session-2026-10-13",
    ]


# ── scripts/recap_sessions.py ─────────────────────────────────────────


def _script(name):
    spec = importlib.util.spec_from_file_location(
        name, Path(library.PROJECT_ROOT) / "scripts" / f"{name}.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_script_lists_without_calling_a_model(lib, tmp_path, monkeypatch, capsys):
    _earlier_fixture(lib, tmp_path)
    monkeypatch.setenv("LIBRARY_DIR", str(tmp_path))

    def boom(*a, **k):
        raise AssertionError("listing must not call a model")

    assert _script("recap_sessions").main(["--dir", str(tmp_path)], ask=boom) == 0
    out = capsys.readouterr().out
    assert out.count("no recap") == 4
    assert not list(tmp_path.glob("*.recap.json"))


def test_script_generates_missing_recaps_once(lib, tmp_path, monkeypatch, capsys):
    _earlier_fixture(lib, tmp_path)
    monkeypatch.setenv("LIBRARY_DIR", str(tmp_path))
    mod = _script("recap_sessions")
    seen = []
    args = ["--dir", str(tmp_path), "--generate", "--campaign", "return-to-night-city"]
    assert mod.main(args, ask=_fake_ask(seen=seen)) == 0
    # three nights of that campaign (the fixture's "tonight" is long past),
    # one call each; the two recordings of the 13th are one session
    assert len(seen) == 3
    assert sorted(p.name for p in tmp_path.glob("*.recap.json")) == [
        "2026-10-06_19-00-00.recap.json",
        "2026-10-13_19-00-00.recap.json",
        "2026-10-20_19-38-38.recap.json",
    ]
    assert seen[0]["header"].startswith("SESSION: ")
    # a second run keeps them; --force with --day rewrites one
    assert mod.main(args, ask=_fake_ask(seen=seen)) == 0
    assert len(seen) == 3
    assert (
        mod.main(args + ["--force", "--day", "2026-10-06"], ask=_fake_ask(seen=seen))
        == 0
    )
    assert len(seen) == 4
    assert "kept" in capsys.readouterr().out


def test_script_reports_a_failure_and_carries_on(lib, tmp_path, monkeypatch, capsys):
    from src.llm.common import AskFailed

    _earlier_fixture(lib, tmp_path)
    monkeypatch.setenv("LIBRARY_DIR", str(tmp_path))
    calls = []

    def flaky(transcript, question, *, header="", **context):
        calls.append(1)
        if len(calls) == 1:
            raise AskFailed("The model server is down.")
        return AskResult(answer=ANSWER, model="m")

    args = ["--dir", str(tmp_path), "--generate", "--campaign", "return-to-night-city"]
    assert _script("recap_sessions").main(args, ask=flaky) == 1
    assert "FAILED: The model server is down." in capsys.readouterr().err
    assert len(list(tmp_path.glob("*.recap.json"))) == 2


def test_script_skips_tonights_session(lib, tmp_path, monkeypatch):
    from datetime import datetime

    monkeypatch.setenv("LIBRARY_DIR", str(tmp_path))
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    _rec(tmp_path, stamp, "[19:01:00] Jim (Jill) [371]: live\n", lib, guild_id=111)
    mod = _script("recap_sessions")
    seen = []
    assert (
        mod.main(["--dir", str(tmp_path), "--generate"], ask=_fake_ask(seen=seen)) == 0
    )
    assert seen == []
    assert (
        mod.main(
            ["--dir", str(tmp_path), "--generate", "--today"], ask=_fake_ask(seen=seen)
        )
        == 0
    )
    assert len(seen) == 1


def test_tag_script_sets_a_game_without_a_campaign(lib, tmp_path, monkeypatch):
    monkeypatch.setenv("LIBRARY_DIR", str(tmp_path))
    mod = _script("tag_session")
    txt = tmp_path / "2026-09-01_19-00-00.txt"
    txt.write_text("x", encoding="utf-8")
    assert mod.main([str(txt), "--game", "cyberpunk-red", "--group", "noah-gm"]) == 0
    m = sessions.read_meta(txt)
    assert (m.game, m.group, m.campaign) == ("cyberpunk-red", "noah-gm", None)
    with pytest.raises(SystemExit):
        mod.main([str(txt), "--game", "nope"])


def test_tag_script_refuses_a_game_that_contradicts_the_campaign(
    lib, tmp_path, monkeypatch
):
    """roborev #7146: --game used to overwrite the campaign's game."""
    monkeypatch.setenv("LIBRARY_DIR", str(tmp_path))
    mod = _script("tag_session")
    txt = tmp_path / "2026-09-01_19-00-00.txt"
    txt.write_text("x", encoding="utf-8")
    with pytest.raises(SystemExit):
        mod.main([str(txt), "--campaign", "return-to-night-city", "--game", "dnd-5e"])
    assert sessions.read_meta(txt) is None  # nothing written
    args = [str(txt), "--campaign", "return-to-night-city", "--game", "cyberpunk-red"]
    assert mod.main(args) == 0
    m = sessions.read_meta(txt)
    assert (m.campaign, m.game) == ("return-to-night-city", "cyberpunk-red")


def test_script_recaps_an_unevenly_tagged_night_as_ask_groups_it(
    lib, tmp_path, monkeypatch
):
    """roborev #7146: one recording of the 13th names only the game and
    group. /ask still treats the night as one earlier session, so the
    script must write one recap for all of it, and /ask must find it."""
    now = _earlier_fixture(lib, tmp_path)
    monkeypatch.setenv("LIBRARY_DIR", str(tmp_path))
    first, late = _night_13(tmp_path)
    sessions.retag(
        late, lib, game="cyberpunk-red", group="noah-gm", whole_session=False
    )
    assert sessions.read_meta(late).campaign is None
    assert sessions.story_night_files(first) == [first, late]
    assert sessions.story_night_files(late) == [first, late]

    seen = []
    mod = _script("recap_sessions")
    args = ["--dir", str(tmp_path), "--generate", "--day", "2026-10-13"]
    assert mod.main(args, ask=_fake_ask(seen=seen)) == 0
    assert len(seen) == 1  # once, not once per differently-tagged half
    assert "the deal went bad" in seen[0]["transcript"]
    assert "we ran" in seen[0]["transcript"]
    assert "recap-2026-10-13" in [n.id for n in sessions.earlier_session_notes(now)]
    assert mod.main(args, ask=_fake_ask(seen=seen)) == 0 and len(seen) == 1


def test_the_summary_rules_forbid_nudges_and_analysis():
    """Owner's direction: the summary retells the session; it never steers
    the next one or interprets this one. The rules ride in the system
    prompt on every recap call, with or without a style."""
    rules = recaps.SUMMARY_RULES.lower()
    for phrase in (
        "do not suggest",
        "should or could do next",
        "open threads",
        "do not analyse",
        "judging whether a choice was wise",
        "what an npc intends",
        "what happened in response",
        "especially badly",
    ):
        assert phrase in rules, phrase
    # and the question no longer asks for loose ends
    assert "left hanging" not in recaps.QUESTION
    assert "SUMMARY RULES" in recaps.QUESTION
