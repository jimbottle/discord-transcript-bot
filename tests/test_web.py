"""HTTP surface tests for the Flask dashboard.

Covers the CSRF gate, the status endpoint shape, and path-traversal defense
in transcript_service. Run via `make test` or `./venv/bin/python -m pytest`.

Every test gets a fake BotManager via the `fake_bot` fixture — no real
`python main.py` subprocess is ever spawned, so the suite leaves no
artifacts in .logs/ or transcripts/ and tests stay independent.
"""

from unittest.mock import MagicMock

import pytest

import app as web_app
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
