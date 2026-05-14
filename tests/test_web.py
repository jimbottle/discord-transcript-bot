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


# ── Path traversal defense ────────────────────────────────────────────

@pytest.mark.parametrize("evil", [
    "../../../etc/passwd",
    "..\\..\\Windows\\System32",
    "a/b",
    "x/../y",
])
def test_get_transcript_rejects_traversal(evil):
    assert transcript_service.get_transcript(evil) is None


def test_get_transcript_unknown_file_returns_none():
    assert transcript_service.get_transcript("definitely-not-here-9999.txt") is None


# ── transcript_service basics ─────────────────────────────────────────

def test_list_transcripts_returns_list():
    result = transcript_service.list_transcripts()
    assert isinstance(result, list)


def test_search_logs_empty_query_returns_empty():
    assert transcript_service.search_logs("") == []
