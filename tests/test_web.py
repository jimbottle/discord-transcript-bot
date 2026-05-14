"""HTTP surface tests for the Flask dashboard.

Covers the CSRF gate, the status endpoint shape, and path-traversal defense
in transcript_service. Run via `make test` or `./venv/bin/python -m pytest`.
"""

import json
from pathlib import Path

import pytest

import app as web_app
import transcript_service


@pytest.fixture
def client():
    web_app.app.config["TESTING"] = True
    with web_app.app.test_client() as c:
        yield c


# ── CSRF gate ─────────────────────────────────────────────────────────

def test_post_start_no_header_is_403(client):
    assert client.post("/bot/start").status_code == 403


def test_post_stop_no_header_is_403(client):
    assert client.post("/bot/stop").status_code == 403


def test_post_with_csrf_header_passes(client):
    r = client.post("/bot/start", headers={"X-Requested-With": "XMLHttpRequest"})
    assert r.status_code == 200
    # Always clean up the subprocess we just spawned
    client.post("/bot/stop", headers={"X-Requested-With": "XMLHttpRequest"})


def test_post_with_mismatched_origin_is_403(client):
    r = client.post(
        "/bot/stop",
        headers={
            "X-Requested-With": "XMLHttpRequest",
            "Origin": "http://evil.example",
        },
    )
    assert r.status_code == 403


def test_post_with_matching_origin_passes(client):
    # Flask test client uses 'http://localhost/' as host_url
    r = client.post(
        "/bot/stop",
        headers={
            "X-Requested-With": "XMLHttpRequest",
            "Origin": "http://localhost",
        },
    )
    assert r.status_code == 200


# ── Status endpoint shape ─────────────────────────────────────────────

def test_status_returns_json_with_expected_keys(client):
    r = client.get("/bot/status")
    assert r.status_code == 200
    data = r.get_json()
    assert "status" in data
    assert data["status"] in {"stopped", "starting", "initializing", "ready", "unhealthy", "crashed"}


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
