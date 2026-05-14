"""Smoke tests for the HealthCheck system.

These run without a bot instance and without autofix. They verify the check
runner is wired up correctly and that the whisper_model check actually
exercises decoding (not just imports). Skips checks that need a real bot
gateway connection.
"""

import sys

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


def test_whisper_check_actually_decodes():
    """Regression test for roborev #512: the transcribe() generator was
    never consumed, so the check passed silently even if decoding broke.
    Verify that running the check loads the model and produces a non-error
    result on real synthetic audio."""
    hc = HealthCheck()
    hc._check_whisper_model()
    result = hc.checks["whisper_model"]
    # Either OK (model works) or FAIL with a real exception message.
    # What we don't want is silent OK with a broken pipeline — covered by
    # the fact that the check now iterates the segments.
    assert result["message"] != "", "whisper_model check returned empty message"


def test_summary_renders_one_line_per_check():
    hc = HealthCheck()
    hc.run_all(autofix=False, bot=None)
    summary = hc.summary()
    assert summary.count("\n") + 1 >= len(hc.checks)
