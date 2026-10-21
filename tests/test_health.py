"""Smoke tests for the HealthCheck system.

These run without a bot instance and without autofix. They verify the check
runner is wired up correctly. The whisper_model end-to-end decode test is
marked `integration` so the default `make test` doesn't load
faster-whisper large-v3 — conftest's `_fast_whisper_model` fixture stubs
`audio_model` for the fast suite.
"""

import os
from unittest.mock import MagicMock

import pytest

from src.bot.health import HealthCheck
from src.bot import health as health_module


@pytest.fixture(autouse=True)
def _isolate_status_file(monkeypatch, tmp_path):
    """No test may write .logs/health_status.json.

    The web dashboard reads that exact path (web/bot_manager.py), and the
    bot only rewrites it at startup. A test that calls run_all(autofix=True)
    with most checks stubbed would leave the UI reporting a handful of fake
    checks — as "ready" — for a live bot whose real results had been erased,
    and it would stay wrong for the rest of the session.
    """
    monkeypatch.setattr(
        health_module, "STATUS_FILE", str(tmp_path / "health_status.json")
    )


def test_run_all_returns_dict_with_expected_keys():
    hc = HealthCheck()
    results = hc.run_all(autofix=False, bot=None)
    assert isinstance(results, dict)
    expected = {
        "env_vars",
        "ffmpeg",
        "opus",
        "whisper_model",
        "openai_api",
        "transcripts_dir",
        "log_dirs",
        "player_map",
        "ask_provider",
        "discord_gateway",
    }
    assert expected.issubset(results.keys())


def test_each_check_has_required_fields():
    hc = HealthCheck()
    results = hc.run_all(autofix=False, bot=None)
    for name, info in results.items():
        assert "ok" in info, f"{name} missing 'ok'"
        assert "message" in info, f"{name} missing 'message'"
        assert "critical" in info, f"{name} missing 'critical'"
        assert isinstance(info["ok"], bool)


@pytest.mark.integration
def test_whisper_check_actually_decodes():
    """Regression test for roborev #512: the transcribe() generator was
    never consumed, so the check passed silently even if decoding broke.
    Marked integration because it loads faster-whisper large-v3 (~10s on
    first run); the conftest fixture stubs the model for the fast suite."""
    hc = HealthCheck()
    hc._check_whisper_model()
    result = hc.checks["whisper_model"]
    assert result["ok"] is True, f"Real model failed to decode: {result['message']}"
    assert result["message"] != ""


def test_summary_renders_one_line_per_check():
    hc = HealthCheck()
    hc.run_all(autofix=False, bot=None)
    summary = hc.summary()
    assert summary.count("\n") + 1 >= len(hc.checks)


# ── ask_provider ──────────────────────────────────────────────────────
# /ask is the only thing that needs the Anthropic key. Voice transcription
# does not, and on_ready aborts if all_ok() is False, so this check must
# never be critical.


def test_ask_provider_passes_with_a_key(monkeypatch):
    monkeypatch.delenv("ASK_PROVIDER", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setenv("ASK_MODEL", "claude-sonnet-5-5")
    hc = HealthCheck()
    hc._check_ask_provider()
    check = hc.checks["ask_provider"]
    assert check["ok"] is True
    assert "claude-sonnet-5-5" in check["message"]
    assert check["critical"] is False


def test_ask_provider_warns_without_a_key(monkeypatch):
    monkeypatch.setenv("ASK_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "  ")
    hc = HealthCheck()
    hc._check_ask_provider()
    check = hc.checks["ask_provider"]
    assert check["ok"] is False
    assert "ANTHROPIC_API_KEY" in check["message"]
    assert check["critical"] is False


def test_all_ok_when_only_ask_provider_fails():
    """Every critical check passes and only ask_provider is failing —
    all_ok() must return True so a missing key never blocks startup."""
    hc = HealthCheck()
    hc.checks = {
        "env_vars": {"ok": True, "message": "ok", "critical": True},
        "ffmpeg": {"ok": True, "message": "ok", "critical": True},
        "opus": {"ok": True, "message": "ok", "critical": True},
        "whisper_model": {"ok": True, "message": "ok", "critical": True},
        "ask_provider": {"ok": False, "message": "no key", "critical": False},
    }
    assert hc.all_ok() is True


def test_run_all_never_calls_anthropic(monkeypatch):
    """The health check must not spend tokens: no client is ever built."""
    import anthropic

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")

    def boom(*a, **k):
        raise AssertionError("health check constructed an Anthropic client")

    monkeypatch.setattr(anthropic, "Anthropic", boom)
    hc = HealthCheck()
    hc._check_ask_provider()
    assert hc.checks["ask_provider"]["ok"] is True


def test_health_checks_never_touch_the_live_status_file(monkeypatch, tmp_path):
    """Regression: a stubbed run_all(autofix=True) once overwrote
    .logs/health_status.json, leaving the dashboard reporting three fake
    checks as 'ready' for a bot whose real results were gone."""
    live = os.path.join(os.getcwd(), ".logs", "health_status.json")
    before = os.path.exists(live) and os.path.getmtime(live)

    hc = HealthCheck()
    for name in (
        "_check_env_vars",
        "_check_ffmpeg",
        "_check_opus",
        "_check_whisper_model",
        "_check_openai_api",
        "_check_player_map",
        "_check_discord_gateway",
        "_check_ask_provider",
    ):
        monkeypatch.setattr(hc, name, lambda *a, **kw: None)
    monkeypatch.setattr(hc, "_check_transcripts_dir", lambda **kw: None)
    monkeypatch.setattr(hc, "_check_log_dirs", lambda **kw: None)

    hc.run_all(autofix=True)

    after = os.path.exists(live) and os.path.getmtime(live)
    assert after == before, "run_all wrote to the live health status file"


# ── whisper_model: cold cache is announced, not silently downloaded ────
# discord-transcript-bot-56t. get_backend() downloads ~3 GB on a cold
# cache from inside this check; the operator must be told BEFORE the stall.


def _cache_state(cached):
    from src.asr.model_cache import CacheState

    return CacheState(
        backend="mlx-whisper",
        model_id="large-v3",
        repo="mlx-community/whisper-large-v3-mlx",
        cached=cached,
        cache_dir="/nowhere/hub",
        approx_gb=3.0,
    )


def test_whisper_check_warns_before_a_cold_cache_download(monkeypatch, caplog):
    import logging

    from src.asr import model_cache

    monkeypatch.setattr(model_cache, "probe", lambda: _cache_state(False))
    hc = HealthCheck()
    written = []
    provisional = []

    def capture_status(phase, check=None):
        written.append((phase, check))
        provisional.append(dict(hc.checks.get("whisper_model", {})))

    monkeypatch.setattr(hc, "_write_status", capture_status)
    with caplog.at_level(logging.WARNING, logger="src.bot.health"):
        hc._check_whisper_model(autofix=True)

    assert any("downloading ~3 GB" in r.message for r in caplog.records)
    # The provisional record persisted to the dashboard is an in-progress
    # notice (warn styling), not a critical failure (red ✗) — roborev #4307.
    assert provisional and provisional[0]["ok"] is False
    assert provisional[0]["critical"] is False
    # The provisional "downloading" state reached the dashboard status file
    # before the (stubbed) load ran.
    assert ("initializing", "whisper_model") in written
    # ...and the final verdict still reflects the real load.
    result = hc.checks["whisper_model"]
    assert result["ok"] is True
    assert "make prewarm" in result["message"]


def test_whisper_check_is_quiet_on_a_warm_cache(monkeypatch, caplog):
    import logging

    from src.asr import model_cache

    monkeypatch.setattr(model_cache, "probe", lambda: _cache_state(True))
    hc = HealthCheck()
    written = []
    monkeypatch.setattr(
        hc, "_write_status", lambda phase, check=None: written.append((phase, check))
    )
    with caplog.at_level(logging.WARNING, logger="src.bot.health"):
        hc._check_whisper_model(autofix=True)

    assert not any("downloading" in r.message for r in caplog.records)
    assert written == []
    assert hc.checks["whisper_model"]["ok"] is True
    assert "downloaded" not in hc.checks["whisper_model"]["message"]


def test_whisper_check_survives_a_broken_probe(monkeypatch):
    from src.asr import model_cache

    def boom():
        raise RuntimeError("probe exploded")

    monkeypatch.setattr(model_cache, "probe", boom)
    hc = HealthCheck()
    hc._check_whisper_model()
    assert hc.checks["whisper_model"]["ok"] is True


# ── ask_provider: local model ─────────────────────────────────────────


def _local(monkeypatch, **status):
    from src.llm import local

    base = {"url": "http://localhost:11434/v1", "model": "gemma4:26b"}
    base.update(status)
    monkeypatch.setenv("ASK_PROVIDER", "local")
    monkeypatch.setattr(local, "status", lambda: base)


def test_local_provider_ready(monkeypatch):
    _local(monkeypatch, reachable=True, has_model=True)
    hc = HealthCheck()
    hc._check_ask_provider()
    check = hc.checks["ask_provider"]
    assert check["ok"] is True and "gemma4:26b" in check["message"]
    assert check["critical"] is False


def test_local_provider_server_down_names_the_fix(monkeypatch):
    _local(monkeypatch, reachable=False, has_model=False)
    hc = HealthCheck()
    hc._check_ask_provider()
    check = hc.checks["ask_provider"]
    assert check["ok"] is False and check["critical"] is False
    assert "Ollama" in check["message"] and "ASK_PROVIDER=anthropic" in check["message"]


def test_local_provider_missing_model_says_pull(monkeypatch):
    _local(monkeypatch, reachable=True, has_model=False)
    hc = HealthCheck()
    hc._check_ask_provider()
    assert "ollama pull gemma4:26b" in hc.checks["ask_provider"]["message"]


def test_unconfigured_checkout_probes_the_local_server_not_anthropic(monkeypatch):
    """A fresh checkout (no key, no ASK_PROVIDER) is local-first; the probe
    hits the conftest's closed port and reports it, quickly."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    hc = HealthCheck()
    hc._check_ask_provider()
    check = hc.checks["ask_provider"]
    assert check["ok"] is False and "127.0.0.1:9" in check["message"]
