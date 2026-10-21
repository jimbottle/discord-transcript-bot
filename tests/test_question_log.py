"""Question history (src/question_log.py; discord-transcript-bot-jmj)."""

import json
from datetime import datetime
from types import SimpleNamespace

from src import library, question_log, sessions


def _result():
    return SimpleNamespace(
        model="claude-opus-5-5",
        input_tokens=120,
        cached_tokens=100,
        output_tokens=9,
        rounds=2,
        notes_used=["npcs"],
    )


def test_record_appends_one_json_line_with_result_fields():
    e = question_log.record(
        source="discord",
        question="who is kane?",
        status=question_log.ANSWERED,
        answer="A fixer.",
        result=_result(),
        asker="Jobby Jill",
        asker_id=371442040141119490,
        guild_id=111,
        latency_s=3.456,
    )
    lines = question_log.history_file().read_text().splitlines()
    assert len(lines) == 1
    stored = json.loads(lines[0])
    assert stored == e
    assert stored["asker_id"] == "371442040141119490"  # snowflake as a string
    assert stored["notes_used"] == ["npcs"] and stored["rounds"] == 2
    assert stored["latency_s"] == 3.46
    assert stored["session_key"] is None


def test_record_tags_the_session(tmp_path, monkeypatch):
    libdir = tmp_path / "lib"
    libdir.mkdir()
    (libdir / "library.yml").write_text(
        "games: {cyberpunk-red: {name: Cyberpunk RED}}\n"
        "groups: {noah-gm: {name: N, discord_guilds: [111], active_campaign: rtnc}}\n"
        "campaigns: {rtnc: {name: Return to Night City, group: noah-gm, game: cyberpunk-red}}\n"
    )
    monkeypatch.setenv("LIBRARY_DIR", str(libdir))
    lib = library.load()
    tdir = tmp_path / "t"
    tdir.mkdir()
    paths = []
    for stamp in ("2026-10-20_19-38-38", "2026-10-20_20-15-26"):
        p = tdir / f"{stamp}.txt"
        p.write_text("x\n")
        sessions.write_meta(
            p,
            sessions.build_meta(
                lib,
                guild_id=111,
                started_at=datetime.strptime(stamp, "%Y-%m-%d_%H-%M-%S"),
            ),
        )
        paths.append(p)
    e = question_log.record(
        source="dashboard", question="q", status="answered", transcript_path=paths[1]
    )
    assert e["session_key"] == "rtnc/2026-10-20"
    assert (e["group"], e["game"], e["campaign"]) == (
        "noah-gm",
        "cyberpunk-red",
        "rtnc",
    )
    assert e["files"] == ["2026-10-20_19-38-38.txt", "2026-10-20_20-15-26.txt"]


def test_read_is_newest_first_filters_and_skips_bad_lines():
    for i, (src, status) in enumerate(
        [
            ("discord", "answered"),
            ("dashboard", "failed"),
            ("discord-public", "answered"),
        ]
    ):
        question_log.record(source=src, question=f"q{i}", status=status)
    with open(question_log.history_file(), "a") as fh:
        fh.write("not json\n[1, 2]\n")
    assert [e["question"] for e in question_log.read()] == ["q2", "q1", "q0"]
    assert [e["question"] for e in question_log.read(source="discord")] == ["q0"]
    assert [e["question"] for e in question_log.read(status="failed")] == ["q1"]
    assert [e["question"] for e in question_log.read(limit=1)] == ["q2"]


def test_read_without_a_file_is_empty():
    assert question_log.read() == []


def test_record_never_raises(monkeypatch, tmp_path):
    # history path is a directory: open() fails; record logs and returns None
    monkeypatch.setenv("QUESTION_LOG_FILE", str(tmp_path))
    assert (
        question_log.record(source="discord", question="q", status="answered") is None
    )


def test_include_gm_false_hides_gm_note_answers():
    # roborev #6462: Lore history must not resurface spoiler answers
    question_log.record(
        source="dashboard-lore",
        question="spoiler",
        status="answered",
        campaign="rtnc",
        include_gm=True,
    )
    question_log.record(
        source="dashboard-lore",
        question="safe",
        status="answered",
        campaign="rtnc",
        include_gm=False,
    )
    question_log.record(source="discord", question="no flag", status="answered")
    assert [e["question"] for e in question_log.read(include_gm=False)] == [
        "no flag",
        "safe",
    ]
    assert len(question_log.read()) == 3
