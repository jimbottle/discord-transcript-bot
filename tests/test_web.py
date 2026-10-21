"""HTTP surface tests for the Flask dashboard.

Covers the CSRF gate, the status endpoint shape, and path-traversal defense
in transcript_service. Run via `make test` or `./venv/bin/python -m pytest`.

Every test gets a fake BotManager via the `fake_bot` fixture — no real
`python main.py` subprocess is ever spawned, so the suite leaves no
artifacts in .logs/ or transcripts/ and tests stay independent.
"""

import json
import os
import time as _time
from unittest.mock import MagicMock

import pytest
import yaml

import app as web_app
import roster_service
import transcript_service


class FakeBotManager:
    """In-memory stand-in for BotManager that records calls and returns a
    deterministic status. Lets us tighten the CSRF / status assertions
    without spawning the real bot process."""

    def __init__(self):
        self._status = {"status": "stopped", "pid": None}
        self.start_calls = 0
        self.stop_calls = 0

    def start(self):
        self.start_calls += 1
        self._status = {"status": "ready", "pid": 12345, "checks": {}}
        return True

    def stop(self):
        self.stop_calls += 1
        self._status = {"status": "stopped", "pid": None}
        return True

    def status(self):
        return dict(self._status)


@pytest.fixture
def fake_bot(monkeypatch):
    fb = FakeBotManager()
    monkeypatch.setattr(web_app, "bot", fb)
    return fb


@pytest.fixture
def client(fake_bot):
    web_app.app.config["TESTING"] = True
    with web_app.app.test_client() as c:
        yield c


# ── CSRF gate ─────────────────────────────────────────────────────────


def test_post_start_no_header_is_403(client):
    assert client.post("/bot/start").status_code == 403


def test_post_stop_no_header_is_403(client):
    assert client.post("/bot/stop").status_code == 403


def test_post_start_with_csrf_header_invokes_bot_start(client, fake_bot):
    r = client.post("/bot/start", headers={"X-Requested-With": "XMLHttpRequest"})
    assert r.status_code == 200
    assert fake_bot.start_calls == 1
    assert r.get_json()["status"] == "ready"


def test_post_stop_with_csrf_header_invokes_bot_stop(client, fake_bot):
    r = client.post("/bot/stop", headers={"X-Requested-With": "XMLHttpRequest"})
    assert r.status_code == 200
    assert fake_bot.stop_calls == 1
    assert r.get_json()["status"] == "stopped"


def test_post_with_mismatched_origin_is_403(client, fake_bot):
    r = client.post(
        "/bot/stop",
        headers={
            "X-Requested-With": "XMLHttpRequest",
            "Origin": "http://evil.example",
        },
    )
    assert r.status_code == 403
    # Rejected before reaching the BotManager
    assert fake_bot.stop_calls == 0


def test_post_with_matching_origin_passes(client, fake_bot):
    # Flask test client uses 'http://localhost/' as host_url
    r = client.post(
        "/bot/stop",
        headers={
            "X-Requested-With": "XMLHttpRequest",
            "Origin": "http://localhost",
        },
    )
    assert r.status_code == 200
    assert fake_bot.stop_calls == 1


# ── Status endpoint shape ─────────────────────────────────────────────


def test_status_returns_stopped_initially(client):
    r = client.get("/bot/status")
    assert r.status_code == 200
    data = r.get_json()
    assert data == {"status": "stopped", "pid": None}


def test_status_reflects_started_state(client, fake_bot):
    client.post("/bot/start", headers={"X-Requested-With": "XMLHttpRequest"})
    r = client.get("/bot/status")
    assert r.status_code == 200
    data = r.get_json()
    assert data["status"] == "ready"
    assert data["pid"] == 12345


# ── Page renders (smoke) ──────────────────────────────────────────────


def test_index_renders(client):
    r = client.get("/")
    assert r.status_code == 200
    assert b"Dashboard" in r.data


def test_transcripts_renders(client):
    r = client.get("/transcripts")
    assert r.status_code == 200


def test_search_renders(client):
    r = client.get("/search?q=test")
    assert r.status_code == 200


def test_live_page_renders(client):
    r = client.get("/live")
    assert r.status_code == 200
    assert b"Live Session" in r.data


def test_live_data_returns_expected_shape(client):
    r = client.get("/live/data")
    assert r.status_code == 200
    data = r.get_json()
    for key in ("filename", "entries", "modified", "age_seconds", "live"):
        assert key in data
    assert isinstance(data["entries"], list)
    assert isinstance(data["live"], bool)


# ── live transcript line parser ───────────────────────────────────────


def test_parse_transcript_line_structured():
    line = "[18:08:09] Jobby Jill (Jim) [371442040141119490]: Drain test one two."
    e = transcript_service.parse_transcript_line(line)
    assert e == {
        "time": "18:08:09",
        "player": "Jobby Jill",
        "character": "Jim",
        "user_id": "371442040141119490",
        "text": "Drain test one two.",
    }


def test_parse_transcript_line_blank_is_none():
    assert transcript_service.parse_transcript_line("") is None
    assert transcript_service.parse_transcript_line("   \n") is None


def test_parse_transcript_line_unmatched_kept_as_raw():
    # Older format / manual edit must not be silently dropped.
    e = transcript_service.parse_transcript_line("just some freeform text")
    assert e["text"] == "just some freeform text"
    assert e["player"] is None and e["time"] is None


def test_parse_transcript_line_text_with_brackets_and_colons():
    line = "[09:00:01] A B (C) [42]: see [note]: it's 3:00 sharp"
    e = transcript_service.parse_transcript_line(line)
    assert e["player"] == "A B"
    assert e["character"] == "C"
    assert e["user_id"] == "42"
    assert e["text"] == "see [note]: it's 3:00 sharp"


def test_get_live_session_shape():
    data = transcript_service.get_live_session()
    assert set(data) >= {"filename", "entries", "modified", "age_seconds", "live"}
    assert isinstance(data["entries"], list)


def test_get_live_session_picks_newest_and_freshness(tmp_path, monkeypatch):
    """roborev #795: the newest-file selection and the mtime-based
    `live` flag are the core logic — exercise them directly."""
    monkeypatch.setattr(transcript_service, "TRANSCRIPTS_DIR", tmp_path)
    (tmp_path / "2026-05-18_10-00-00.txt").write_text(
        "[10:00:00] A (A) [1]: old line\n", encoding="utf-8"
    )
    new = tmp_path / "2026-05-18_12-00-00.txt"
    new.write_text("[12:00:00] B (B) [2]: new line\n", encoding="utf-8")

    s = transcript_service.get_live_session()
    assert s["filename"] == "2026-05-18_12-00-00.txt"  # lexicographically newest
    assert s["live"] is True  # just written
    assert s["entries"][0]["text"] == "new line"

    stale = _time.time() - (transcript_service.LIVE_FRESH_SECONDS + 60)
    os.utime(new, (stale, stale))
    assert transcript_service.get_live_session()["live"] is False


def test_get_log_entries_parses_and_skips(tmp_path, monkeypatch):
    """roborev #796: positive path — valid rows parsed (event_source
    -> source), silence (empty data) and malformed lines skipped."""
    monkeypatch.setattr(transcript_service, "LOGS_DIR", tmp_path)
    log = tmp_path / "2026-05-18-transcription.log"
    log.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "date": "2026-05-18",
                        "begin": "18:00:01",
                        "end": "18:00:03",
                        "user_id": 7,
                        "player": "P",
                        "character": "C",
                        "event_source": "Discord",
                        "data": "hello there",
                    }
                ),
                json.dumps({"player": "P", "data": "   "}),  # silence
                "{ not valid json",  # malformed
                "",  # blank
                json.dumps(
                    {"player": "Q", "event_source": "Discord", "data": "second"}
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    entries = transcript_service.get_log_entries("2026-05-18-transcription.log")
    assert [e["text"] for e in entries] == ["hello there", "second"]
    assert entries[0]["source"] == "Discord"  # event_source -> source
    assert entries[0]["player"] == "P"
    assert entries[0]["begin"] == "18:00:01"


def test_get_bot_state_reads_valid_missing_and_corrupt(tmp_path, monkeypatch):
    f = tmp_path / "bot_state.json"
    monkeypatch.setattr(transcript_service, "BOT_STATE_FILE", f)
    # missing
    assert transcript_service.get_bot_state() is None
    # valid
    f.write_text(json.dumps({"guilds": [], "started_at": 1.0}), encoding="utf-8")
    assert transcript_service.get_bot_state() == {"guilds": [], "started_at": 1.0}
    # corrupt
    f.write_text("{ not json", encoding="utf-8")
    assert transcript_service.get_bot_state() is None


def test_get_bot_state_non_dict_json_is_none(tmp_path, monkeypatch):
    """roborev #812: valid JSON that isn't an object must read as None
    so the index route doesn't 500 on state.get(...)."""
    f = tmp_path / "bot_state.json"
    monkeypatch.setattr(transcript_service, "BOT_STATE_FILE", f)
    f.write_text("[1, 2, 3]", encoding="utf-8")
    assert transcript_service.get_bot_state() is None
    f.write_text('"just a string"', encoding="utf-8")
    assert transcript_service.get_bot_state() is None


def test_index_shows_authoritative_card_when_alive(client, fake_bot, monkeypatch):
    fake_bot._status = {"status": "ready", "pid": 1, "checks": {}}
    monkeypatch.setattr(
        web_app,
        "get_bot_state",
        lambda: {
            "started_at": _time.time() - 90,
            "guilds": [{"guild": "personal", "channel": "voice-1", "recording": True}],
        },
    )
    body = client.get("/").data.decode()
    assert "Connected:" in body and "personal" in body and "voice-1" in body
    assert "Recording:" in body
    assert "Uptime:" in body


def test_index_no_authoritative_card_without_state(client, fake_bot, monkeypatch):
    fake_bot._status = {"status": "ready", "pid": 1, "checks": {}}
    monkeypatch.setattr(web_app, "get_bot_state", lambda: None)
    assert b"Connected:" not in client.get("/").data


def test_index_no_authoritative_card_when_bot_not_alive(client, fake_bot, monkeypatch):
    fake_bot._status = {"status": "stopped", "pid": None}
    monkeypatch.setattr(
        web_app,
        "get_bot_state",
        lambda: {"started_at": _time.time(), "guilds": [{"guild": "x"}]},
    )
    # gated out when the bot isn't alive even if a (stale) state file exists
    assert b"Connected:" not in client.get("/").data


def test_format_uptime():
    fu = transcript_service.format_uptime
    assert fu(None) is None
    assert fu(0) is None  # falsy start -> no uptime
    now = _time.time()
    assert fu(now - 5).endswith("s")
    assert fu(now - 125) == "2m 5s"
    assert fu(now - 3 * 3600 - 4 * 60).startswith("3h 4m")
    # clock skew: bot's start is "in the future" -> clamp, never negative
    assert fu(now + 999) == "0s"


def test_format_duration():
    fd = transcript_service.format_duration
    assert fd(None) is None
    assert fd(-5) == "0s"  # clamp negatives
    assert fd(5) == "5s"
    assert fd(125) == "2m 5s"
    assert fd(3 * 3600 + 4 * 60) == "3h 4m"


def test_heartbeat_age():
    ha = transcript_service.heartbeat_age
    assert ha(None) is None
    assert ha({}) is None
    assert ha({"updated_at": 0}) is None  # falsy -> None
    now = _time.time()
    assert 49 <= ha({"updated_at": now - 50}) <= 61
    assert ha({"updated_at": now + 999}) == 0.0  # clock skew clamped


# ── Liveness: heartbeat-stale + stalled-recording warnings on the index ──


def test_index_warns_when_heartbeat_stale(client, fake_bot, monkeypatch):
    fake_bot._status = {"status": "ready", "pid": 1, "checks": {}}
    monkeypatch.setattr(
        web_app,
        "get_bot_state",
        lambda: {
            "started_at": _time.time() - 600,
            "updated_at": _time.time() - 300,  # well past STATE_STALE_SECONDS
            "guilds": [{"guild": "x", "channel": "c", "recording": True}],
        },
    )
    body = client.get("/").data.decode()
    assert "not responding" in body.lower()


def test_index_shows_stalled_recording_badge(client, fake_bot, monkeypatch):
    fake_bot._status = {"status": "ready", "pid": 1, "checks": {}}
    monkeypatch.setattr(
        web_app,
        "get_bot_state",
        lambda: {
            "started_at": _time.time() - 600,
            "updated_at": _time.time(),  # fresh heartbeat -> not "not responding"
            "guilds": [
                {
                    "guild": "x",
                    "channel": "c",
                    "recording": True,
                    "stalled": True,
                    "output_age": 300,
                }
            ],
        },
    )
    body = client.get("/").data.decode()
    assert "STALLED" in body
    assert "5m 0s" in body  # output_age formatted via the duration filter
    assert "not responding" not in body.lower()  # heartbeat is fresh


def test_view_log_renders_known_file(tmp_path, monkeypatch, client):
    monkeypatch.setattr(transcript_service, "LOGS_DIR", tmp_path)
    (tmp_path / "ok.log").write_text(
        json.dumps({"player": "Zed", "event_source": "Discord", "data": "ping"}) + "\n",
        encoding="utf-8",
    )
    r = client.get("/logs/ok.log")
    assert r.status_code == 200
    assert b"Zed" in r.data and b"ping" in r.data


# ── Path traversal defense ────────────────────────────────────────────


@pytest.mark.parametrize(
    "evil",
    [
        "../../../etc/passwd",
        "..\\..\\Windows\\System32",
        "a/b",
        "x/../y",
    ],
)
def test_get_transcript_rejects_traversal(evil):
    assert transcript_service.get_transcript(evil) is None


def test_get_transcript_unknown_file_returns_none():
    assert transcript_service.get_transcript("definitely-not-here-9999.txt") is None


@pytest.mark.parametrize(
    "evil",
    ["../../../etc/passwd", "a/b", "x/../y", "..\\w", ""],
)
def test_entry_helpers_reject_traversal(evil):
    assert transcript_service.get_transcript_entries(evil) is None
    assert transcript_service.get_log_entries(evil) is None


def test_entry_helpers_unknown_file_returns_none():
    assert transcript_service.get_transcript_entries("nope-9999.txt") is None
    assert transcript_service.get_log_entries("nope-9999.log") is None


def test_view_transcript_unknown_is_404(client):
    assert client.get("/transcripts/nope-9999.txt").status_code == 404


def test_view_log_unknown_is_404(client):
    assert client.get("/logs/nope-9999.log").status_code == 404


def test_view_transcript_traversal_is_404(client):
    # Flask treats the slash as a path separator -> no route match (404).
    assert client.get("/transcripts/..%2f..%2fetc%2fpasswd").status_code == 404


# ── transcript_service basics ─────────────────────────────────────────


def test_list_transcripts_returns_list():
    result = transcript_service.list_transcripts()
    assert isinstance(result, list)


def test_search_logs_empty_query_returns_empty():
    assert transcript_service.search_logs("") == []


# ── Roster editor (CRUD over player_map.yml) ──────────────────────────


@pytest.fixture
def roster_file(tmp_path, monkeypatch):
    """Point the roster at an isolated tmp file so tests never touch the
    real player_map.yml. conftest loads .env, which sets
    PLAYER_MAP_FILE_PATH to the real ./player_map.yml — override it with an
    absolute tmp path (roster_path resolves absolute paths as-is)."""
    pm = tmp_path / "player_map.yml"
    monkeypatch.setenv("PLAYER_MAP_FILE_PATH", str(pm))
    return pm


_XHR = {"X-Requested-With": "XMLHttpRequest"}


def test_roster_page_renders(client, roster_file):
    roster_file.write_text(
        yaml.dump({371442040141119490: {"player": "Jobby Jill", "character": "Jim"}}),
        encoding="utf-8",
    )
    r = client.get("/roster")
    assert r.status_code == 200
    assert b"371442040141119490" in r.data
    assert b"Jobby Jill" in r.data


def test_roster_upsert_no_csrf_is_403(client, roster_file):
    r = client.post(
        "/roster/entry", json={"user_id": 7, "player": "A", "character": "B"}
    )
    assert r.status_code == 403
    assert not roster_file.exists()  # rejected before any write


def test_roster_upsert_persists(client, roster_file):
    r = client.post(
        "/roster/entry",
        json={"user_id": 7, "player": "Ed", "character": "Volo"},
        headers=_XHR,
    )
    assert r.status_code == 200
    # user_id returned as a string (snowflakes exceed JS safe-int range)
    assert r.get_json() == {"user_id": "7", "player": "Ed", "character": "Volo"}
    assert yaml.safe_load(roster_file.read_text()) == {
        7: {"player": "Ed", "character": "Volo"}
    }


def test_roster_upsert_returns_snowflake_id_without_precision_loss(client, roster_file):
    # A real 19-digit Discord id must survive the round trip exactly — it
    # would lose precision as a JSON number, hence the string contract.
    sid = "421900550829768715"
    r = client.post(
        "/roster/entry",
        json={"user_id": sid, "player": "Ed", "character": "Cody"},
        headers=_XHR,
    )
    assert r.status_code == 200
    assert r.get_json()["user_id"] == sid
    assert int(sid) in yaml.safe_load(roster_file.read_text())


def test_roster_upsert_invalid_user_id_400(client, roster_file):
    r = client.post(
        "/roster/entry",
        json={"user_id": "abc", "player": "A", "character": "B"},
        headers=_XHR,
    )
    assert r.status_code == 400
    assert "error" in r.get_json()
    assert not roster_file.exists()


def test_roster_upsert_missing_names_400(client, roster_file):
    r = client.post(
        "/roster/entry",
        json={"user_id": 7, "player": "", "character": "B"},
        headers=_XHR,
    )
    assert r.status_code == 400


def test_roster_upsert_non_dict_file_400_and_untouched(client, roster_file):
    roster_file.write_text(yaml.dump(["not", "a", "map"]), encoding="utf-8")
    r = client.post(
        "/roster/entry",
        json={"user_id": 7, "player": "A", "character": "B"},
        headers=_XHR,
    )
    assert r.status_code == 400
    assert "mapping" in r.get_json()["error"]
    assert yaml.safe_load(roster_file.read_text()) == ["not", "a", "map"]


def test_roster_delete_no_csrf_is_403(client, roster_file):
    assert client.post("/roster/entry/delete", json={"user_id": 7}).status_code == 403


def test_roster_delete_removes_entry(client, roster_file):
    roster_file.write_text(
        yaml.dump(
            {
                7: {"player": "Go", "character": "Go"},
                9: {"player": "Stay", "character": "Stay"},
            }
        ),
        encoding="utf-8",
    )
    r = client.post("/roster/entry/delete", json={"user_id": 7}, headers=_XHR)
    assert r.status_code == 200
    assert r.get_json() == {"user_id": "7", "deleted": True}
    assert yaml.safe_load(roster_file.read_text()) == {
        9: {"player": "Stay", "character": "Stay"}
    }


def test_roster_delete_invalid_user_id_400(client, roster_file):
    r = client.post("/roster/entry/delete", json={"user_id": "-3"}, headers=_XHR)
    assert r.status_code == 400


def test_get_unmapped_speakers_excludes_mapped_includes_members(
    roster_file, monkeypatch
):
    roster_file.write_text(
        yaml.dump({7: {"player": "Named", "character": "Char"}}), encoding="utf-8"
    )
    # 7 already mapped (excluded); 8 on the call; 9 spoke.
    monkeypatch.setattr(
        roster_service,
        "get_bot_state",
        lambda: {
            "guilds": [
                {
                    "members": [
                        {"id": 7, "name": "named", "display_name": "Named"},
                        {"id": 8, "name": "alice", "display_name": "Alice"},
                    ]
                }
            ]
        },
    )
    monkeypatch.setattr(
        roster_service,
        "get_live_session",
        lambda: {
            "entries": [
                {"user_id": "9", "player": "Bob", "character": None, "text": "hello"},
                {"user_id": "7", "player": "Named", "character": "Char", "text": "x"},
            ]
        },
    )
    result = roster_service.get_unmapped_speakers()
    by_id = {r["user_id"]: r for r in result}
    assert set(by_id) == {8, 9}  # 7 excluded (mapped)
    assert by_id[8]["source"] == "on_call" and by_id[8]["suggested_name"] == "Alice"
    assert by_id[9]["source"] == "spoke" and by_id[9]["suggested_name"] == "Bob"
    assert by_id[9]["last_text"] == "hello"
    assert result[0]["user_id"] == 8  # on-call sorted first


# ── /ask: question the live session from the dashboard ──────────────────
# Same module as Discord's /ask (src.llm.claude); these tests stub it so no
# API call is ever made and check the HTTP contract the Live page relies on.


@pytest.fixture
def live_transcript(tmp_path, monkeypatch):
    """A transcripts dir with one session file, the 'live' one."""
    monkeypatch.setattr(transcript_service, "TRANSCRIPTS_DIR", tmp_path)
    f = tmp_path / "2026-10-20_19-38-38.txt"
    f.write_text(
        "[19:40:01] Jobby Jill (Jim) [371442040141119490]: We head for the keep.\n",
        encoding="utf-8",
    )
    return f


@pytest.fixture
def ask_configured(monkeypatch):
    monkeypatch.setattr(web_app.claude, "configured", lambda: True)


def _fake_ask(calls, answer="They decided to head for the keep."):
    def ask(transcript, question, header="", note_dirs=()):
        calls.append((transcript, question))
        return web_app.claude.AskResult(
            answer=answer,
            model="claude-opus-5-5",
            input_tokens=120,
            cached_tokens=100,
            output_tokens=12,
        )

    return ask


def test_ask_requires_csrf_header(client, live_transcript, ask_configured):
    r = client.post("/ask", json={"question": "What happened?"})
    assert r.status_code == 403


def test_ask_rejects_empty_question(client, live_transcript, ask_configured):
    r = client.post("/ask", json={"question": "   "}, headers=_XHR)
    assert r.status_code == 400
    assert "question" in r.get_json()["error"]


def test_ask_rejects_oversized_question(client, live_transcript, ask_configured):
    r = client.post(
        "/ask",
        json={"question": "x" * (web_app.MAX_QUESTION_CHARS + 1)},
        headers=_XHR,
    )
    assert r.status_code == 400


def test_ask_without_key_is_503(client, live_transcript, monkeypatch):
    monkeypatch.setattr(web_app.claude, "configured", lambda: False)
    called = []
    monkeypatch.setattr(web_app.claude, "ask", _fake_ask(called))
    r = client.post("/ask", json={"question": "Why?"}, headers=_XHR)
    assert r.status_code == 503
    assert r.get_json()["error"] == web_app.claude.NO_KEY_MSG
    assert called == []  # never reaches the API without a key


def test_ask_with_no_session_is_409(client, tmp_path, monkeypatch, ask_configured):
    monkeypatch.setattr(transcript_service, "TRANSCRIPTS_DIR", tmp_path)
    r = client.post("/ask", json={"question": "Why?"}, headers=_XHR)
    assert r.status_code == 409


def test_ask_with_empty_session_is_409(client, live_transcript, ask_configured):
    live_transcript.write_text("", encoding="utf-8")
    r = client.post("/ask", json={"question": "Why?"}, headers=_XHR)
    assert r.status_code == 409


def test_ask_unknown_filename_is_404(client, live_transcript, ask_configured):
    r = client.post(
        "/ask", json={"question": "Why?", "filename": "nope.txt"}, headers=_XHR
    )
    assert r.status_code == 404


def test_ask_path_traversal_is_404(client, live_transcript, ask_configured):
    r = client.post(
        "/ask", json={"question": "Why?", "filename": "../.env"}, headers=_XHR
    )
    assert r.status_code == 404


def test_ask_answers_from_the_live_session(
    client, live_transcript, ask_configured, monkeypatch
):
    calls = []
    monkeypatch.setattr(web_app.claude, "ask", _fake_ask(calls))
    r = client.post("/ask", json={"question": "Where next?"}, headers=_XHR)
    assert r.status_code == 200
    body = r.get_json()
    assert body == {
        "filename": "2026-10-20_19-38-38.txt",
        "question": "Where next?",
        "answer": "They decided to head for the keep.",
        "model": "claude-opus-5-5",
        "input_tokens": 120,
        "cached_tokens": 100,
        "output_tokens": 12,
        "notes_used": [],
        "recordings": 1,
    }
    # The model saw the session file with 12-hour line stamps.
    from src.utils.timefmt import for_model

    assert calls == [
        (for_model(live_transcript.read_text(encoding="utf-8")), "Where next?")
    ]


def test_ask_picks_the_newest_session_by_default(
    client, live_transcript, ask_configured, monkeypatch
):
    newer = live_transcript.parent / "2026-10-21_09-00-00.txt"
    newer.write_text("[09:00:01] A (B) [1]: newer session\n", encoding="utf-8")
    calls = []
    monkeypatch.setattr(web_app.claude, "ask", _fake_ask(calls))
    r = client.post("/ask", json={"question": "q"}, headers=_XHR)
    assert r.status_code == 200
    assert r.get_json()["filename"] == newer.name
    assert "newer session" in calls[0][0]


def test_ask_named_transcript(client, live_transcript, ask_configured, monkeypatch):
    calls = []
    monkeypatch.setattr(web_app.claude, "ask", _fake_ask(calls))
    r = client.post(
        "/ask",
        json={"question": "q", "filename": live_transcript.name},
        headers=_XHR,
    )
    assert r.status_code == 200
    assert r.get_json()["filename"] == live_transcript.name


def test_ask_failed_maps_to_502_with_user_message(
    client, live_transcript, ask_configured, monkeypatch
):
    def boom(transcript, question, **kw):
        raise web_app.claude.AskFailed(web_app.claude.RATE_LIMIT_MSG)

    monkeypatch.setattr(web_app.claude, "ask", boom)
    r = client.post("/ask", json={"question": "q"}, headers=_XHR)
    assert r.status_code == 502
    assert r.get_json()["error"] == web_app.claude.RATE_LIMIT_MSG


def test_ask_empty_answer_is_502(client, live_transcript, ask_configured, monkeypatch):
    monkeypatch.setattr(web_app.claude, "ask", _fake_ask([], answer="  "))
    r = client.post("/ask", json={"question": "q"}, headers=_XHR)
    assert r.status_code == 502
    assert r.get_json()["error"] == web_app.claude.EMPTY_MSG


def test_live_data_reports_ask_configured(client, monkeypatch):
    monkeypatch.setattr(web_app.claude, "configured", lambda: False)
    assert client.get("/live/data").get_json()["ask_configured"] is False
    monkeypatch.setattr(web_app.claude, "configured", lambda: True)
    assert client.get("/live/data").get_json()["ask_configured"] is True


def test_live_page_has_the_ask_box(client):
    r = client.get("/live")
    assert b'id="ask-form"' in r.data
    assert b'id="ask-question"' in r.data
    assert b'data-filename=""' in r.data  # follows the newest session
    assert b'data-static="false"' in r.data


def test_transcript_page_has_an_ask_box_for_that_file(client, live_transcript):
    r = client.get("/transcripts/" + live_transcript.name)
    assert r.status_code == 200
    assert b'id="ask-form"' in r.data
    assert ('data-filename="%s"' % live_transcript.name).encode() in r.data
    assert b'data-static="true"' in r.data


def test_ask_non_string_filename_is_404(client, live_transcript, ask_configured):
    # roborev #6442: an int used to reach _safe_path's substring tests → 500.
    r = client.post("/ask", json={"question": "q", "filename": 123}, headers=_XHR)
    assert r.status_code == 404
    r = client.post("/ask", json={"question": "q", "filename": ["a.txt"]}, headers=_XHR)
    assert r.status_code == 404


# ── Host / Origin allowlist (DNS rebinding) ──────────────────────────────
# A rebinding page reaches the loopback server with the attacker's hostname
# in Host and Origin. Both checks compare against fixed loopback values.


def test_foreign_host_is_rejected_everywhere(client):
    assert client.get("/", headers={"Host": "attacker.example"}).status_code == 403
    assert (
        client.get("/live/data", headers={"Host": "attacker.example:5001"}).status_code
        == 403
    )


def test_loopback_hosts_are_accepted(client):
    for host in ("localhost", "127.0.0.1", "localhost:5001", "127.0.0.1:5001"):
        assert client.get("/bot/status", headers={"Host": host}).status_code == 200


def test_origin_must_be_loopback_even_when_host_matches(client, fake_bot):
    # Host spoofed to loopback but Origin says attacker: still refused.
    r = client.post(
        "/bot/start",
        headers={**_XHR, "Host": "127.0.0.1:5001", "Origin": "http://attacker.example"},
    )
    assert r.status_code == 403
    assert fake_bot.start_calls == 0
    ok = client.post(
        "/bot/start",
        headers={**_XHR, "Host": "127.0.0.1:5001", "Origin": "http://127.0.0.1:5001"},
    )
    assert ok.status_code == 200


# ── sessions: the dashboard asks over every recording of the session ─────


@pytest.fixture
def tagged_session(tmp_path, monkeypatch):
    """Two recordings of one campaign night (a restart) plus an unrelated
    recording, under a temp library and transcripts dir."""
    from datetime import datetime

    from src import library, sessions

    libdir = tmp_path / "lib"
    libdir.mkdir()
    (libdir / "library.yml").write_text(
        "games: {cyberpunk-red: {name: Cyberpunk RED}}\n"
        "groups: {noah-gm: {name: Noah's group, gm: Noah, discord_guilds: [111],"
        " active_campaign: rtnc}}\n"
        "campaigns: {rtnc: {name: Return to Night City, group: noah-gm,"
        " game: cyberpunk-red}}\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("LIBRARY_DIR", str(libdir))
    tdir = tmp_path / "transcripts"
    tdir.mkdir()
    monkeypatch.setattr(transcript_service, "TRANSCRIPTS_DIR", tdir)
    lib = library.load()
    files = []
    for stamp, line in (
        ("2026-10-20_19-38-38", "[19:40:00] A (B) [1]: before the restart\n"),
        ("2026-10-20_20-15-26", "[20:16:00] A (B) [1]: after the restart\n"),
    ):
        f = tdir / f"{stamp}.txt"
        f.write_text(line, encoding="utf-8")
        sessions.write_meta(
            f,
            sessions.build_meta(
                lib,
                guild_id=111,
                started_at=datetime.strptime(stamp, "%Y-%m-%d_%H-%M-%S"),
            ),
        )
        files.append(f)
    (tdir / "2026-05-18_17-39-56.txt").write_text("[17:40:00] X (Y) [2]: old test\n")
    return files


def test_ask_covers_every_recording_of_the_session(
    client, tagged_session, ask_configured, monkeypatch
):
    seen = {}

    def ask(transcript, question, header="", note_dirs=()):
        seen.update(transcript=transcript, header=header)
        return web_app.claude.AskResult(answer="ok", model="m")

    monkeypatch.setattr(web_app.claude, "ask", ask)
    r = client.post("/ask", json={"question": "q"}, headers=_XHR)
    assert r.status_code == 200
    assert r.get_json()["recordings"] == 2
    assert "before the restart" in seen["transcript"]
    assert "--- recording restarted at 8:15 PM ---" in seen["transcript"]
    assert seen["header"].startswith(
        "SESSION: Game: Cyberpunk RED · Campaign: Return to Night City"
    )
    assert "\nSTYLE: " in seen["header"]


def test_transcripts_page_shows_session_columns(client, tagged_session):
    r = client.get("/transcripts")
    assert r.status_code == 200
    body = r.get_data(as_text=True)
    assert "Return to Night City" in body and "Cyberpunk RED" in body
    assert "untagged" in body  # the old test recording


def test_live_data_carries_the_session(client, tagged_session):
    s = client.get("/live/data").get_json()["session"]
    assert s["campaign"] == "Return to Night City"
    assert s["session_day"] == "2026-10-20"
    assert s["recordings"] == 2
    assert s["files"] == ["2026-10-20_19-38-38.txt", "2026-10-20_20-15-26.txt"]


def test_transcript_page_shows_the_session_tag(client, tagged_session):
    r = client.get("/transcripts/2026-10-20_19-38-38.txt")
    body = r.get_data(as_text=True)
    assert "Return to Night City" in body
    assert "2 recordings" in body


# ── question history ─────────────────────────────────────────────────────


def test_ask_records_answered_questions(
    client, live_transcript, ask_configured, monkeypatch
):
    from src import question_log

    monkeypatch.setattr(web_app.claude, "ask", _fake_ask([]))
    client.post("/ask", json={"question": "Where next?"}, headers=_XHR)
    (e,) = question_log.read()
    assert e["source"] == "dashboard" and e["status"] == "answered"
    assert e["question"] == "Where next?"
    assert e["answer"] == "They decided to head for the keep."
    assert e["model"] == "claude-opus-5-5" and e["input_tokens"] == 120
    assert e["files"] == [live_transcript.name]
    assert e["latency_s"] is not None


def test_ask_records_failures_and_rejections(client, live_transcript, monkeypatch):
    from src import question_log

    monkeypatch.setattr(web_app.claude, "configured", lambda: False)
    client.post("/ask", json={"question": "no key"}, headers=_XHR)
    monkeypatch.setattr(web_app.claude, "configured", lambda: True)

    def boom(transcript, question, **kw):
        raise web_app.claude.AskFailed(web_app.claude.RATE_LIMIT_MSG)

    monkeypatch.setattr(web_app.claude, "ask", boom)
    client.post("/ask", json={"question": "limited"}, headers=_XHR)
    failed, rejected = question_log.read()
    assert (failed["status"], failed["error"]) == (
        "failed",
        web_app.claude.RATE_LIMIT_MSG,
    )
    assert (rejected["status"], rejected["error"]) == (
        "rejected",
        web_app.claude.NO_KEY_MSG,
    )


def test_bad_requests_are_not_recorded(client, live_transcript, ask_configured):
    from src import question_log

    client.post("/ask", json={"question": "  "}, headers=_XHR)
    client.post("/ask", json={"question": "q"})  # no CSRF header
    assert question_log.read() == []


def test_questions_page_lists_and_filters_history(client):
    from src import question_log

    question_log.record(
        source="discord",
        question="Who is Kane?",
        status="answered",
        answer="A fixer.",
        asker="Jobby Jill",
    )
    question_log.record(
        source="dashboard",
        question="Where next?",
        status="failed",
        error="rate limited",
    )
    body = client.get("/questions").get_data(as_text=True)
    assert "Who is Kane?" in body and "A fixer." in body and "Jobby Jill" in body
    assert "Where next?" in body and "rate limited" in body
    only = client.get("/questions?source=discord").get_data(as_text=True)
    assert "Who is Kane?" in only and "Where next?" not in only


def test_questions_data_filters_by_session(client):
    from src import question_log

    question_log.record(source="discord", question="a", status="answered")
    data = client.get("/questions/data?session=rtnc/2026-10-20").get_json()
    assert data == []
    assert len(client.get("/questions/data").get_json()) == 1


def test_questions_link_in_nav(client):
    assert b'href="/questions"' in client.get("/live").data


# ── Lore page (dashboard-only campaign Q&A) ──────────────────────────────


@pytest.fixture
def lore_library(tmp_path, monkeypatch):
    libdir = tmp_path / "lib"
    notes = libdir / "campaigns" / "rtnc" / "notes"
    notes.mkdir(parents=True)
    (libdir / "library.yml").write_text(
        "games: {cyberpunk-red: {name: Cyberpunk RED}}\n"
        "groups: {noah-gm: {name: Noah's group, gm: Noah, discord_guilds: [111],"
        " active_campaign: rtnc}}\n"
        "campaigns: {rtnc: {name: Return to Night City, group: noah-gm,"
        " game: cyberpunk-red}, empty: {name: Empty, group: noah-gm}}\n",
        encoding="utf-8",
    )
    (notes / "npcs.md").write_text(
        "---\nid: npcs\ntitle: Named NPCs\nsummary: Who is who.\n---\nKane runs Little Europe.\n"
    )
    (notes / "gm-secrets.md").write_text(
        "---\nid: gm-secrets\ntitle: Secrets\nsummary: Spoilers.\naudience: gm\n---\nTaka.\n"
    )
    monkeypatch.setenv("LIBRARY_DIR", str(libdir))
    return libdir


def test_lore_page_defaults_to_the_active_campaign_and_lists_notes(
    client, lore_library
):
    body = client.get("/lore").get_data(as_text=True)
    assert "Ask about Return to Night City" in body
    assert 'data-campaign="rtnc"' in body
    assert "Named NPCs" in body and "GM only" in body


def test_lore_ask_answers_from_notes_and_records_history(
    client, lore_library, ask_configured, monkeypatch
):
    from src import question_log

    seen = {}

    def ask_lore(question, header="", note_dirs=(), include_gm=False):
        seen.update(header=header, dirs=list(note_dirs), include_gm=include_gm)
        return web_app.claude.AskResult(
            answer="A fixer, choom.", model="m", notes_used=["npcs"]
        )

    monkeypatch.setattr(web_app.claude, "ask_lore", ask_lore)
    r = client.post(
        "/lore/ask", json={"question": "who is kane?", "campaign": "rtnc"}, headers=_XHR
    )
    assert r.status_code == 200 and r.get_json()["answer"] == "A fixer, choom."
    assert seen["header"].startswith(
        "CAMPAIGN: Game: Cyberpunk RED · Campaign: Return to Night City"
    )
    assert "\nSTYLE: " in seen["header"]
    assert seen["include_gm"] is False
    (e,) = question_log.read()
    assert (e["source"], e["campaign"], e["game"], e["include_gm"]) == (
        "dashboard-lore",
        "rtnc",
        "cyberpunk-red",
        False,
    )
    assert e["notes_used"] == ["npcs"]


def test_lore_ask_passes_the_gm_toggle(
    client, lore_library, ask_configured, monkeypatch
):
    seen = {}

    def ask_lore(question, **kw):
        seen.update(kw)
        return web_app.claude.AskResult(answer="x", model="m")

    monkeypatch.setattr(web_app.claude, "ask_lore", ask_lore)
    client.post(
        "/lore/ask",
        json={"question": "q", "campaign": "rtnc", "include_gm": True},
        headers=_XHR,
    )
    assert seen["include_gm"] is True
    client.post(
        "/lore/ask",
        json={"question": "q", "campaign": "rtnc", "include_gm": "yes"},
        headers=_XHR,
    )
    assert seen["include_gm"] is False  # only a real true opts in


def test_lore_ask_rejects_bad_input(client, lore_library, ask_configured):
    assert (
        client.post("/lore/ask", json={"question": "q", "campaign": "rtnc"}).status_code
        == 403
    )
    assert (
        client.post(
            "/lore/ask", json={"question": " ", "campaign": "rtnc"}, headers=_XHR
        ).status_code
        == 400
    )
    assert (
        client.post(
            "/lore/ask", json={"question": "q", "campaign": "nope"}, headers=_XHR
        ).status_code
        == 404
    )
    assert (
        client.post(
            "/lore/ask", json={"question": "q", "campaign": ["rtnc"]}, headers=_XHR
        ).status_code
        == 404
    )


def test_lore_ask_without_notes_is_409_and_rejected(
    client, lore_library, ask_configured
):
    from src import question_log

    r = client.post(
        "/lore/ask", json={"question": "q", "campaign": "empty"}, headers=_XHR
    )
    assert r.status_code == 409
    assert question_log.read()[0]["status"] == "rejected"


def test_lore_in_nav(client):
    assert b'href="/lore"' in client.get("/live").data


def test_questions_page_renders_answer_markdown(client):
    from src import question_log

    question_log.record(
        source="discord",
        question="q",
        status="answered",
        answer="**Kane** runs it.\n- a\n- <b>b</b>",
    )
    body = client.get("/questions").get_data(as_text=True)
    assert "<strong>Kane</strong> runs it." in body
    assert "<li>a</li>" in body and "&lt;b&gt;b&lt;/b&gt;" in body


# ── GM check outcomes on the dashboard ───────────────────────────────────


def _blocked(*a, **k):
    from src.llm import guard

    v = guard.Verdict(False, "strategy", "That asks for tactics.", model="g")
    raise web_app.claude.AskBlocked(guard.block_message(v), v)


def test_blocked_dashboard_question_is_a_200_refusal_and_recorded(
    client, live_transcript, ask_configured, monkeypatch
):
    from src import question_log

    monkeypatch.setattr(web_app.claude, "ask", _blocked)
    r = client.post(
        "/ask", json={"question": "best approach to beat this?"}, headers=_XHR
    )
    assert r.status_code == 200
    body = r.get_json()
    assert body["blocked"] is True and body["category"] == "strategy"
    assert "coaching the play" in body["answer"]
    (e,) = question_log.read()
    assert e["status"] == "blocked"
    assert e["guard"] == {
        "allowed": False,
        "category": "strategy",
        "reason": "That asks for tactics.",
        "model": "g",
    }


def test_blocked_lore_question(client, lore_library, ask_configured, monkeypatch):
    from src import question_log

    monkeypatch.setattr(web_app.claude, "ask_lore", _blocked)
    r = client.post(
        "/lore/ask", json={"question": "best build?", "campaign": "rtnc"}, headers=_XHR
    )
    assert r.status_code == 200 and r.get_json()["blocked"] is True
    assert question_log.read()[0]["status"] == "blocked"


def test_questions_page_shows_blocked_entries(client):
    from src import question_log
    from src.llm import guard

    question_log.record(
        source="discord",
        question="how do we win the fight?",
        status="blocked",
        error="The GM wouldn't want me coaching the play on that one.",
        verdict=guard.Verdict(False, "strategy", "Tactics.", model="g"),
    )
    body = client.get("/questions?status=blocked").get_data(as_text=True)
    assert "how do we win the fight?" in body and "GM check: strategy" in body


def test_lore_history_query_hides_gm_answers_by_default(client):
    from src import question_log

    question_log.record(
        source="dashboard-lore", question="spoiler", status="answered", include_gm=True
    )
    question_log.record(
        source="dashboard-lore", question="safe", status="answered", include_gm=False
    )
    default = client.get(
        "/questions/data?source=dashboard-lore&include_gm=false"
    ).get_json()
    assert [e["question"] for e in default] == ["safe"]
    both = client.get("/questions/data?source=dashboard-lore").get_json()
    assert [e["question"] for e in both] == ["safe", "spoiler"]


def test_lore_gm_badge_follows_the_note_not_its_id(client, lore_library):
    # roborev #6462: a campaign gm note overriding a group table note with
    # the same id must still be badged GM only
    group = lore_library / "groups" / "noah-gm" / "notes"
    group.mkdir(parents=True)
    (group / "gm-secrets.md").write_text(
        "---\nid: gm-secrets\ntitle: Table version\nsummary: s\n---\nx\n"
    )
    body = client.get("/lore").get_data(as_text=True)
    i = body.index("<strong>Secrets</strong>")
    assert "GM only" in body[i : i + 120]
    assert "Table version" not in body


def test_lore_page_reloads_history_with_the_gm_filter(client, lore_library):
    body = client.get("/lore").get_data(as_text=True)
    assert 'data-history-query="source=dashboard-lore&campaign=rtnc"' in body
    js = client.get("/static/ask.js").get_data(as_text=True)
    assert '"&include_gm=false"' in js


def test_lore_shows_the_gm_only_hint_before_asking(client, lore_library):
    # roborev #6470: table_count was computed but unused
    assert b'id="gm-only-hint"' not in client.get("/lore").data
    (lore_library / "campaigns" / "rtnc" / "notes" / "npcs.md").unlink()
    assert b'id="gm-only-hint"' in client.get("/lore").data


def test_ask_js_history_races_behaviourally():
    """Runs tests/js/ask_history.test.mjs: the real ask.js against a DOM
    stand-in with hand-released fetches (roborev #6472 replaced a
    source-grep test). Verified to fail on the pre-fix script."""
    import pathlib
    import shutil
    import subprocess

    node = shutil.which("node")
    if node is None:
        pytest.skip("node not installed")
    script = pathlib.Path(__file__).parent / "js" / "ask_history.test.mjs"
    r = subprocess.run([node, str(script)], capture_output=True, text=True, timeout=30)
    assert r.returncode == 0, r.stderr or r.stdout


def test_dashboard_shows_12_hour_times(client, live_transcript):
    from src import question_log

    body = client.get("/transcripts/" + live_transcript.name).get_data(as_text=True)
    styled = body[body.index('id="parsed-view"') : body.index('id="raw-view"')]
    assert "7:40 PM" in styled and "19:40" not in styled  # Raw view stays the file
    question_log.record(source="discord", question="q", status="answered", answer="a")
    e = question_log.read()[0]
    assert web_app.timefmt.when(e["ts"]) in client.get("/questions").get_data(
        as_text=True
    )
    assert "live-feed.js" in client.get("/live").get_data(as_text=True)


# ── session-tag editor ───────────────────────────────────────────────────


def test_retag_endpoint_updates_the_whole_session(client, tagged_session):
    from src import sessions

    r = client.post(
        "/transcripts/2026-10-20_20-15-26.txt/tag",
        json={"campaign": "", "group": "noah-gm", "game": "cyberpunk-red"},
        headers=_XHR,
    )
    assert r.status_code == 200
    body = r.get_json()
    assert body["updated"] == ["2026-10-20_19-38-38.txt", "2026-10-20_20-15-26.txt"]
    for f in tagged_session:
        m = sessions.read_meta(f)
        assert (m.campaign, m.group, m.game, m.source) == (
            None,
            "noah-gm",
            "cyberpunk-red",
            "manual",
        )


def test_retag_endpoint_single_recording(client, tagged_session):
    from src import sessions

    r = client.post(
        "/transcripts/2026-10-20_20-15-26.txt/tag",
        json={"campaign": "", "whole_session": False},
        headers=_XHR,
    )
    assert r.get_json()["updated"] == ["2026-10-20_20-15-26.txt"]
    assert sessions.read_meta(tagged_session[0]).campaign == "rtnc"


def test_retag_endpoint_tags_an_untagged_recording(client, tagged_session):
    from src import sessions

    r = client.post(
        "/transcripts/2026-05-18_17-39-56.txt/tag",
        json={"campaign": "rtnc"},
        headers=_XHR,
    )
    assert r.status_code == 200
    m = sessions.read_meta(tagged_session[0].parent / "2026-05-18_17-39-56.txt")
    assert (m.campaign, m.group, m.game) == ("rtnc", "noah-gm", "cyberpunk-red")


def test_retag_endpoint_rejects_bad_input(client, tagged_session):
    url = "/transcripts/2026-10-20_20-15-26.txt/tag"
    assert client.post(url, json={"campaign": "rtnc"}).status_code == 403  # CSRF
    assert client.post(url, json={"campaign": "nope"}, headers=_XHR).status_code == 400
    assert client.post(url, json={"campaign": 5}, headers=_XHR).status_code == 400
    assert (
        client.post("/transcripts/missing.txt/tag", json={}, headers=_XHR).status_code
        == 404
    )
    assert (
        client.post("/transcripts/..%2F.env/tag", json={}, headers=_XHR).status_code
        == 404
    )


def test_tag_editor_is_on_the_list_and_transcript_pages(client, tagged_session):
    listing = client.get("/transcripts").get_data(as_text=True)
    assert listing.count('class="tag-form"') == 3
    assert 'data-group="noah-gm"' in listing and "tag-editor.js" in listing
    page = client.get("/transcripts/2026-10-20_19-38-38.txt").get_data(as_text=True)
    assert 'class="tag-form"' in page
    import re

    assert re.search(
        r'<option value="rtnc"[^>]*\bselected>Return to Night City</option>', page
    )
    assert "Apply to all 2 recordings" in page
    assert 'id="edit-tag-link"' in client.get("/live").get_data(as_text=True)


# ── deleting a transcript ────────────────────────────────────────────────


@pytest.fixture
def deletable(tmp_path, monkeypatch):
    """Two old recordings (one tagged, one with captured audio) in temp
    transcripts/ and captures/ dirs, plus a bot state file."""
    import os as _os
    from datetime import datetime

    from src import library, sessions

    tdir, cdir = tmp_path / "transcripts", tmp_path / "captures"
    tdir.mkdir()
    cdir.mkdir()
    monkeypatch.setattr(transcript_service, "TRANSCRIPTS_DIR", tdir)
    monkeypatch.setattr(transcript_service, "CAPTURES_DIR", cdir)
    monkeypatch.setattr(
        transcript_service, "BOT_STATE_FILE", tmp_path / "bot_state.json"
    )
    old = _time.time() - 3600
    files = []
    for stamp in ("2026-10-13_19-00-00", "2026-10-20_19-38-38"):
        f = tdir / f"{stamp}.txt"
        f.write_text("[19:40:00] A (B) [1]: hi\n")
        sessions.write_meta(
            f,
            sessions.build_meta(
                library.load(tmp_path / "nolib"),
                guild_id=1,
                started_at=datetime.strptime(stamp, "%Y-%m-%d_%H-%M-%S"),
            ),
        )
        _os.utime(f, (old, old))
        files.append(f)
    (cdir / "2026-10-20_19-38-38").mkdir()
    (cdir / "2026-10-20_19-38-38" / "0001.wav").write_bytes(b"RIFF")
    return tdir, cdir, files


def _delete(client, name, **body):
    return client.post(f"/transcripts/{name}/delete", json=body, headers=_XHR)


def test_delete_requires_csrf_and_explicit_confirmation(client, deletable):
    tdir, _, files = deletable
    url = f"/transcripts/{files[0].name}/delete"
    assert client.post(url, json={"confirm": True}).status_code == 403
    assert _delete(client, files[0].name).status_code == 400
    assert _delete(client, files[0].name, confirm="yes").status_code == 400
    assert files[0].exists()


def test_delete_removes_transcript_and_tag_but_keeps_audio_by_default(
    client, deletable
):
    tdir, cdir, files = deletable
    r = _delete(client, files[1].name, confirm=True)
    assert r.status_code == 200
    assert not files[1].exists()
    assert not (tdir / "2026-10-20_19-38-38.meta.json").exists()
    assert (cdir / "2026-10-20_19-38-38").is_dir()  # audio kept unless asked
    assert files[0].exists()  # other recordings untouched


def test_delete_with_audio(client, deletable):
    _, cdir, files = deletable
    r = _delete(client, files[1].name, confirm=True, include_audio=True)
    assert r.status_code == 200
    assert not (cdir / "2026-10-20_19-38-38").exists()
    assert any("captures" in p for p in r.get_json()["deleted"])


def test_delete_refuses_the_live_recording(client, deletable, tmp_path):
    _, _, files = deletable
    (tmp_path / "bot_state.json").write_text(
        json.dumps(
            {
                "updated_at": _time.time(),
                "guilds": [{"recording": True, "session_file": files[1].name}],
            }
        )
    )
    r = _delete(client, files[1].name, confirm=True)
    assert r.status_code == 409 and files[1].exists()
    assert "Stop recording first" in r.get_json()["error"]


def test_delete_refuses_a_file_written_seconds_ago(client, deletable):
    _, _, files = deletable
    files[0].write_text("still growing\n")  # fresh mtime, no bot state
    assert _delete(client, files[0].name, confirm=True).status_code == 409


def test_delete_rejects_unknown_and_traversal(client, deletable):
    assert _delete(client, "nope.txt", confirm=True).status_code == 404
    assert (
        client.post(
            "/transcripts/..%2F.env/delete", json={"confirm": True}, headers=_XHR
        ).status_code
        == 404
    )


def test_delete_controls_render_with_an_inline_confirmation(client, deletable):
    _, _, files = deletable
    listing = client.get("/transcripts").get_data(as_text=True)
    assert listing.count('class="delete-transcript"') == 2
    assert "Yes, delete" in listing and "delete-transcript.js" in listing
    assert "Also delete its captured audio" in listing  # only the one with audio
    page = client.get(f"/transcripts/{files[0].name}").get_data(as_text=True)
    assert 'data-after="/transcripts"' in page


# ── newest / oldest order toggle ─────────────────────────────────────────


def _run_node_test(name):
    import pathlib
    import shutil
    import subprocess

    node = shutil.which("node")
    if node is None:
        pytest.skip("node not installed")
    script = pathlib.Path(__file__).parent / "js" / name
    r = subprocess.run([node, str(script)], capture_output=True, text=True, timeout=30)
    assert r.returncode == 0, r.stderr or r.stdout


def test_live_feed_order_behaviour():
    """tests/js/live_feed.test.mjs: both orders, follow + place-keeping,
    switching, empty state, and the remembered toggle."""
    _run_node_test("live_feed.test.mjs")


def test_order_toggle_is_on_live_and_transcript_pages(client, live_transcript):
    live = client.get("/live").get_data(as_text=True)
    assert 'id="order-toggle"' in live and "feed-order.js" in live
    assert 'FeedOrder.get(ORDER_KEY, "newest")' in live
    page = client.get("/transcripts/" + live_transcript.name).get_data(as_text=True)
    assert 'id="order-toggle"' in page and 'data-idx="0"' in page
    assert 'FeedOrder.get(KEY, "oldest")' in page


def test_list_page_offers_whole_session_choice(client, tagged_session):
    # roborev #6491: each list row knows its real session size
    listing = client.get("/transcripts").get_data(as_text=True)
    assert listing.count("Apply to all 2 recordings") == 2


def test_stale_bot_state_does_not_block_delete(client, deletable, tmp_path):
    # roborev #6491: a dead bot's leftover "recording: true" is ignored
    _, _, files = deletable
    (tmp_path / "bot_state.json").write_text(
        json.dumps(
            {
                "updated_at": _time.time() - 3600,
                "guilds": [{"recording": True, "session_file": files[1].name}],
            }
        )
    )
    assert _delete(client, files[1].name, confirm=True).status_code == 200


def test_live_page_has_no_server_rendered_placeholder(client):
    assert b'id="empty-state"' not in client.get("/live").data


def test_dashboard_binds_loopback_unless_told_otherwise(monkeypatch):
    """The dashboard has no login. Only the container (Dockerfile) sets
    DASHBOARD_HOST=0.0.0.0, and compose publishes it on host loopback."""
    import importlib

    monkeypatch.delenv("DASHBOARD_HOST", raising=False)
    assert importlib.reload(web_app).DASHBOARD_HOST == "127.0.0.1"
    monkeypatch.setenv("DASHBOARD_HOST", "0.0.0.0")
    assert importlib.reload(web_app).DASHBOARD_HOST == "0.0.0.0"
    monkeypatch.delenv("DASHBOARD_HOST")
    importlib.reload(web_app)


def test_docker_context_excludes_private_material():
    """The repo is public and the image may be pushed: secrets, GM notes,
    recordings, audio and history must never be copied into it."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(root, ".dockerignore")) as f:
        rules = {ln.strip() for ln in f if ln.strip() and not ln.startswith("#")}
    for must in (
        ".env",
        "op.env",
        "transcripts",
        "captures",
        "history",
        "library/*",
        "*.meta.json",
        "player_map.yml",
        ".beads",
        ".git",
    ):
        assert must in rules, must
