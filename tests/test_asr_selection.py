"""Selection + graceful-fallback logic for the ASR backend
(discord-transcript-bot-d8g). Both real backends are mocked so no model loads;
these assert the auto-select / env-override / fallback decision table."""

from types import SimpleNamespace

import pytest

from src.asr import selection
from src.asr.base import BackendUnavailable


@pytest.fixture
def mocked_builders(monkeypatch):
    """Replace the two backend constructors with sentinels that record the
    model id they were built with, and start from an empty backend cache."""
    selection.reset_backend_cache()
    built = {}

    def fake_fw(model_id):
        built["faster-whisper"] = model_id
        return SimpleNamespace(name="faster-whisper", model_id=model_id)

    def fake_mlx(model_id):
        built["mlx-whisper"] = model_id
        return SimpleNamespace(name="mlx-whisper", model_id=model_id)

    monkeypatch.setattr(selection, "_build_faster_whisper", fake_fw)
    monkeypatch.setattr(selection, "_build_mlx", fake_mlx)
    yield built
    selection.reset_backend_cache()


def _force_apple(monkeypatch, is_apple):
    monkeypatch.setattr(selection, "_is_apple_silicon", lambda: is_apple)


def test_auto_on_apple_silicon_picks_mlx(monkeypatch, mocked_builders):
    monkeypatch.delenv("ASR_BACKEND", raising=False)
    _force_apple(monkeypatch, True)
    assert selection.get_backend().name == "mlx-whisper"


def test_auto_off_apple_picks_faster_whisper(monkeypatch, mocked_builders):
    monkeypatch.delenv("ASR_BACKEND", raising=False)
    _force_apple(monkeypatch, False)
    assert selection.get_backend().name == "faster-whisper"


def test_explicit_mlx_picks_mlx_even_off_apple(monkeypatch, mocked_builders):
    monkeypatch.setenv("ASR_BACKEND", "mlx")
    _force_apple(monkeypatch, False)
    assert selection.get_backend().name == "mlx-whisper"


def test_explicit_faster_whisper_picks_it_on_apple(monkeypatch, mocked_builders):
    monkeypatch.setenv("ASR_BACKEND", "faster-whisper")
    _force_apple(monkeypatch, True)
    assert selection.get_backend().name == "faster-whisper"


def test_mlx_failure_falls_back_to_faster_whisper(monkeypatch, mocked_builders, caplog):
    """Graceful degradation: an MLX build failure must NOT raise — it falls
    back to faster-whisper (the hard requirement)."""
    import logging

    monkeypatch.setenv("ASR_BACKEND", "mlx")
    _force_apple(monkeypatch, True)

    def boom(_model_id):
        raise BackendUnavailable("no mlx here")

    monkeypatch.setattr(selection, "_build_mlx", boom)
    with caplog.at_level(logging.ERROR):
        backend = selection.get_backend()
    assert backend.name == "faster-whisper"
    assert any("MLX-Whisper unavailable" in r.message for r in caplog.records)


def test_auto_apple_mlx_failure_falls_back(monkeypatch, mocked_builders):
    monkeypatch.delenv("ASR_BACKEND", raising=False)
    _force_apple(monkeypatch, True)
    monkeypatch.setattr(
        selection, "_build_mlx", lambda _m: (_ for _ in ()).throw(RuntimeError("x"))
    )
    assert selection.get_backend().name == "faster-whisper"


def test_whisper_model_env_flows_into_backend(monkeypatch, mocked_builders):
    monkeypatch.setenv("ASR_BACKEND", "faster-whisper")
    monkeypatch.setenv("WHISPER_MODEL", "large-v3-turbo")
    assert selection.get_backend().model_id == "large-v3-turbo"
    assert mocked_builders["faster-whisper"] == "large-v3-turbo"


def test_default_model_is_large_v3(monkeypatch, mocked_builders):
    """Accuracy-first default (d29): unset WHISPER_MODEL -> large-v3, not turbo."""
    monkeypatch.setenv("ASR_BACKEND", "faster-whisper")
    monkeypatch.delenv("WHISPER_MODEL", raising=False)
    assert selection.get_backend().model_id == "large-v3"


def test_get_backend_is_memoized(monkeypatch, mocked_builders):
    monkeypatch.setenv("ASR_BACKEND", "faster-whisper")
    first = selection.get_backend()
    second = selection.get_backend()
    assert first is second


def test_mlx_repo_mapping_and_override(monkeypatch):
    monkeypatch.delenv("MLX_WHISPER_MODEL", raising=False)
    assert selection._mlx_repo_for("large-v3") == "mlx-community/whisper-large-v3-mlx"
    # Unknown id passes through verbatim (lets a full HF repo be given directly).
    assert selection._mlx_repo_for("my-org/custom") == "my-org/custom"
    # MLX_WHISPER_MODEL overrides everything.
    monkeypatch.setenv("MLX_WHISPER_MODEL", "mlx-community/whatever")
    assert selection._mlx_repo_for("large-v3") == "mlx-community/whatever"
