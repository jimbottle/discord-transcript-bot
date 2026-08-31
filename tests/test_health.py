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


def _stub_ollama_list(monkeypatch, installed_names):
    """Stub ollama.list() so _check_ollama_model runs without a daemon."""
    import ollama

    fake = MagicMock()
    fake.models = [MagicMock(model=n) for n in installed_names]
    monkeypatch.setattr(ollama, "list", lambda: fake)


def test_ollama_model_uses_env_override(monkeypatch):
    """Regression for roborev #768 (MEDIUM): _check_ollama_model must
    honor ASK_OLLAMA_MODEL, not a hardcoded literal."""
    monkeypatch.setenv("ASK_OLLAMA_MODEL", "custom/model:99")
    _stub_ollama_list(monkeypatch, [])  # nothing installed
    hc = HealthCheck()
    hc._check_ollama_model()
    assert "custom/model:99" in hc.checks["ollama_model"]["message"]


def test_ollama_model_default_when_env_unset(monkeypatch):
    monkeypatch.delenv("ASK_OLLAMA_MODEL", raising=False)
    _stub_ollama_list(monkeypatch, [])
    hc = HealthCheck()
    hc._check_ollama_model()
    assert "gemma4:26b" in hc.checks["ollama_model"]["message"]


def test_ollama_model_empty_env_falls_back_to_default(monkeypatch):
    monkeypatch.setenv("ASK_OLLAMA_MODEL", "  ")
    _stub_ollama_list(monkeypatch, [])
    hc = HealthCheck()
    hc._check_ollama_model()
    assert "gemma4:26b" in hc.checks["ollama_model"]["message"]


def test_ollama_model_ok_when_resolved_model_installed(monkeypatch):
    monkeypatch.setenv("ASK_OLLAMA_MODEL", "gemma4:26b")
    _stub_ollama_list(monkeypatch, ["gemma4:26b"])
    hc = HealthCheck()
    hc._check_ollama_model()
    assert hc.checks["ollama_model"]["ok"] is True


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
        "ollama_server",
        "ollama_model",
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


# ── ollama_model is non-critical ──────────────────────────────────────
# Regression guards for "bot won't start because /ask's model isn't pulled."
# /ask is the only thing that uses the Ollama model. Voice transcription
# does not. on_ready aborts if all_ok() is False, so ollama_model being
# critical was effectively blocking the entire bot on a 4GB model pull.


def test_ollama_model_is_non_critical():
    hc = HealthCheck()
    hc.run_all(autofix=False, bot=None)
    assert (
        hc.checks["ollama_model"]["critical"] is False
    ), "ollama_model must stay non-critical so a missing /ask model doesn't block startup"


def test_all_ok_when_only_ollama_model_fails():
    """Synthesize a check set where every critical check passes and only
    ollama_model is failing — all_ok() must return True."""
    hc = HealthCheck()
    hc.checks = {
        "env_vars": {"ok": True, "message": "ok", "critical": True},
        "ffmpeg": {"ok": True, "message": "ok", "critical": True},
        "opus": {"ok": True, "message": "ok", "critical": True},
        "whisper_model": {"ok": True, "message": "ok", "critical": True},
        "ollama_server": {"ok": True, "message": "ok", "critical": True},
        "ollama_model": {"ok": False, "message": "not installed", "critical": False},
    }
    assert (
        hc.all_ok() is True
    ), "all_ok() must return True if only non-critical ollama_model is failing"


def test_missing_model_suggests_a_command_that_can_work(monkeypatch):
    """A pullable-but-absent model gets the pull command."""
    monkeypatch.setenv("ASK_OLLAMA_MODEL", "gemma4:26b")
    _stub_ollama_list(monkeypatch, [])
    hc = HealthCheck()
    hc._check_ollama_model()
    assert "ollama pull gemma4:26b" in hc.checks["ollama_model"]["message"]


def test_docker_named_model_does_not_suggest_a_failing_pull(monkeypatch):
    """The regression: health used to tell users to run `ollama pull
    ai/mistral:latest` — the exact command that had just 404'd for them,
    sending them in a circle instead of naming the real problem."""
    monkeypatch.setenv("ASK_OLLAMA_MODEL", "ai/mistral:latest")
    _stub_ollama_list(monkeypatch, [])
    hc = HealthCheck()
    hc._check_ollama_model()
    message = hc.checks["ollama_model"]["message"]

    assert "ollama pull ai/mistral:latest" not in message
    assert "Docker Hub" in message
    assert "ASK_OLLAMA_MODEL" in message


# ── ask_providers ────────────────────────────────────────────────────


def _clear_provider_env(monkeypatch):
    for var in (
        "OPENROUTER_API_KEY",
        "CEREBRAS_PAID_API_KEY",
        "CEREBRAS_API_KEY",
        "ASK_DISABLE_OLLAMA",
    ):
        monkeypatch.delenv(var, raising=False)


def test_ask_providers_reports_configured_cloud_tiers(monkeypatch):
    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
    monkeypatch.setenv("CEREBRAS_PAID_API_KEY", "cb-key")

    hc = HealthCheck()
    hc._check_ask_providers()
    check = hc.checks["ask_providers"]

    assert check["ok"] is True
    assert "openrouter" in check["message"]
    assert "cerebras" in check["message"]


def test_ask_providers_passes_with_no_keys_only_when_ollama_is_reachable(monkeypatch):
    """Local-only /ask is supported, but the check must verify the local
    tier can actually answer — not merely that the env var permits it."""
    _clear_provider_env(monkeypatch)

    hc = HealthCheck()
    hc.checks["ollama_server"] = {"ok": True, "message": "", "critical": False}
    hc.checks["ollama_model"] = {"ok": True, "message": "", "critical": False}
    hc._check_ask_providers()

    assert hc.checks["ask_providers"]["ok"] is True
    assert "local Ollama" in hc.checks["ask_providers"]["message"]


def test_ask_providers_fails_when_no_key_and_ollama_unreachable(monkeypatch):
    """The case the check exists to catch: nothing at all can answer.
    Reading ollama_enabled() alone would wrongly report PASS here."""
    _clear_provider_env(monkeypatch)

    hc = HealthCheck()
    hc.checks["ollama_server"] = {"ok": False, "message": "", "critical": False}
    hc.checks["ollama_model"] = {"ok": False, "message": "", "critical": False}
    hc._check_ask_providers()

    assert hc.checks["ask_providers"]["ok"] is False
    assert "not reachable" in hc.checks["ask_providers"]["message"]


def test_ask_providers_passes_on_cloud_key_alone_without_ollama(monkeypatch):
    """A cloud-only host is healthy even with no ollama at all."""
    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
    monkeypatch.setenv("ASK_DISABLE_OLLAMA", "1")

    hc = HealthCheck()
    hc.checks["ollama_server"] = {"ok": False, "message": "", "critical": False}
    hc._check_ask_providers()

    assert hc.checks["ask_providers"]["ok"] is True


def test_cloud_only_host_does_not_probe_or_spawn_ollama(monkeypatch):
    """ASK_DISABLE_OLLAMA=1 must not start a daemon the operator disabled."""
    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
    monkeypatch.setenv("ASK_DISABLE_OLLAMA", "1")

    hc = HealthCheck()
    probed = []
    monkeypatch.setattr(
        hc, "_check_ollama_server", lambda **kw: probed.append("server")
    )
    monkeypatch.setattr(hc, "_check_ollama_model", lambda: probed.append("model"))
    # Everything except the ollama pair is stubbed out; we only care that
    # the disabled tier is never touched.
    for name in (
        "_check_env_vars",
        "_check_ffmpeg",
        "_check_opus",
        "_check_whisper_model",
        "_check_openai_api",
        "_check_player_map",
        "_check_discord_gateway",
    ):
        monkeypatch.setattr(hc, name, lambda *a, **kw: None)
    monkeypatch.setattr(hc, "_check_transcripts_dir", lambda **kw: None)
    monkeypatch.setattr(hc, "_check_log_dirs", lambda **kw: None)

    hc.run_all(autofix=True)

    assert probed == []
    assert hc.checks["ollama_server"]["ok"] is True
    assert "cloud-only" in hc.checks["ollama_server"]["message"]
    assert hc.checks["ollama_server"]["critical"] is False


def test_ask_providers_is_never_critical(monkeypatch):
    """Transcription does not depend on /ask, so no provider state — not
    even 'nothing can answer' — may stop the bot from starting."""
    _clear_provider_env(monkeypatch)

    for ollama_ok in (True, False):
        hc = HealthCheck()
        hc.checks["ollama_server"] = {"ok": ollama_ok, "message": "", "critical": False}
        hc.checks["ollama_model"] = {"ok": ollama_ok, "message": "", "critical": False}
        hc._check_ask_providers()
        assert hc.checks["ask_providers"]["critical"] is False


def test_ask_providers_fails_when_model_is_not_installed(monkeypatch):
    """A reachable daemon is not enough: without the model pulled,
    _ollama_call gets 'manifest unknown' and the chain latches the tier.
    Reporting PASS here would repeat, one level down, the defect that
    keying this check on real results was meant to fix."""
    _clear_provider_env(monkeypatch)

    hc = HealthCheck()
    hc.checks["ollama_server"] = {"ok": True, "message": "", "critical": False}
    hc.checks["ollama_model"] = {"ok": False, "message": "", "critical": False}
    hc._check_ask_providers()

    assert hc.checks["ask_providers"]["ok"] is False
    assert "model is not installed" in hc.checks["ask_providers"]["message"]


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
        "_check_ollama_server",
        "_check_ollama_model",
        "_check_ask_providers",
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
