"""Smoke tests for the HealthCheck system.

These run without a bot instance and without autofix. They verify the check
runner is wired up correctly. The whisper_model end-to-end decode test is
marked `integration` so the default `make test` doesn't load
faster-whisper large-v3 — conftest's `_fast_whisper_model` fixture stubs
`audio_model` for the fast suite.
"""

import pytest

from src.bot.health import HealthCheck


def test_run_all_returns_dict_with_expected_keys():
    hc = HealthCheck()
    results = hc.run_all(autofix=False, bot=None)
    assert isinstance(results, dict)
    expected = {
        "env_vars", "ffmpeg", "opus", "whisper_model", "openai_api",
        "transcripts_dir", "log_dirs", "player_map",
        "ollama_server", "ollama_model", "discord_gateway",
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
# /ask is the only thing that uses ai/mistral:latest. Voice transcription
# does not. on_ready aborts if all_ok() is False, so ollama_model being
# critical was effectively blocking the entire bot on a 4GB model pull.

def test_ollama_model_is_non_critical():
    hc = HealthCheck()
    hc.run_all(autofix=False, bot=None)
    assert hc.checks["ollama_model"]["critical"] is False, \
        "ollama_model must stay non-critical so a missing /ask model doesn't block startup"


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
    assert hc.all_ok() is True, \
        "all_ok() must return True if only non-critical ollama_model is failing"
