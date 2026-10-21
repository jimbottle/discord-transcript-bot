"""Cold-cache detection for the local ASR model (discord-transcript-bot-56t).

The probe must answer "would get_backend() download?" without loading a
model, mirror selection's backend choice, and never raise."""

import os

import pytest

from src.asr import model_cache, selection


@pytest.fixture
def hub(tmp_path, monkeypatch):
    """An empty, isolated HF hub cache."""
    monkeypatch.setattr(model_cache, "hub_cache_dir", lambda: tmp_path)
    monkeypatch.delenv("MLX_WHISPER_MODEL", raising=False)
    return tmp_path


def _warm(hub, repo, weight_name):
    snap = hub / f"models--{repo.replace('/', '--')}" / "snapshots" / "abc123"
    snap.mkdir(parents=True)
    (snap / "config.json").write_text("{}")
    (snap / weight_name).write_bytes(b"\x00" * 16)
    return snap


def test_faster_whisper_cold_by_default(hub, monkeypatch):
    monkeypatch.setenv("ASR_BACKEND", "faster-whisper")
    monkeypatch.setenv("WHISPER_MODEL", "large-v3")
    state = model_cache.probe()
    assert state.backend == "faster-whisper"
    assert state.repo == "Systran/faster-whisper-large-v3"
    assert state.cached is False
    assert state.approx_gb == 3.0
    assert "make prewarm" in state.cold_message()
    assert "COLD" in state.describe()


def test_faster_whisper_warm_once_model_bin_is_in_a_snapshot(hub, monkeypatch):
    monkeypatch.setenv("ASR_BACKEND", "faster-whisper")
    monkeypatch.setenv("WHISPER_MODEL", "large-v3")
    _warm(hub, "Systran/faster-whisper-large-v3", "model.bin")
    state = model_cache.probe()
    assert state.cached is True
    assert "WARM" in state.describe()


def test_partial_download_is_still_cold(hub, monkeypatch):
    """huggingface_hub only links a file into snapshots/ when complete, so a
    snapshot dir with config.json but no weights means the download was
    interrupted — the next load WILL download."""
    monkeypatch.setenv("ASR_BACKEND", "faster-whisper")
    monkeypatch.setenv("WHISPER_MODEL", "large-v3")
    snap = hub / "models--Systran--faster-whisper-large-v3" / "snapshots" / "x"
    snap.mkdir(parents=True)
    (snap / "config.json").write_text("{}")
    assert model_cache.probe().cached is False


def test_explicit_mlx_maps_to_mlx_community_repo(hub, monkeypatch):
    monkeypatch.setenv("ASR_BACKEND", "mlx")
    monkeypatch.setenv("WHISPER_MODEL", "large-v3")
    state = model_cache.probe()
    assert state.backend == "mlx-whisper"
    assert state.repo == "mlx-community/whisper-large-v3-mlx"
    assert state.cached is False
    _warm(hub, "mlx-community/whisper-large-v3-mlx", "weights.npz")
    assert model_cache.probe().cached is True


def test_auto_on_apple_silicon_without_mlx_installed_reports_faster_whisper(
    hub, monkeypatch
):
    """selection's auto path falls back to faster-whisper when mlx_whisper
    can't be imported; the probe must predict the same repo, or it would
    warn about (or vouch for) weights the bot never loads."""
    monkeypatch.setenv("ASR_BACKEND", "auto")
    monkeypatch.setenv("WHISPER_MODEL", "large-v3-turbo")
    monkeypatch.setattr(selection, "_is_apple_silicon", lambda: True)
    monkeypatch.setattr(model_cache, "_mlx_importable", lambda: False)
    state = model_cache.probe()
    assert state.backend == "faster-whisper"
    assert state.repo == "mobiuslabsgmbh/faster-whisper-large-v3-turbo"
    assert state.approx_gb == 1.6

    monkeypatch.setattr(model_cache, "_mlx_importable", lambda: True)
    assert model_cache.probe().backend == "mlx-whisper"


def test_local_model_directory_is_always_warm(hub, tmp_path, monkeypatch):
    local = tmp_path / "my-ct2-model"
    local.mkdir()
    monkeypatch.setenv("ASR_BACKEND", "faster-whisper")
    monkeypatch.setenv("WHISPER_MODEL", str(local))
    state = model_cache.probe()
    assert state.repo == str(local)
    assert state.cached is True
    assert state.approx_gb is None
    assert "several GB" in state.size_hint


def test_probe_never_raises(monkeypatch):
    def boom():
        raise RuntimeError("no cache for you")

    monkeypatch.setattr(model_cache, "hub_cache_dir", boom)
    assert model_cache.probe() is None


def test_hub_cache_dir_honors_hf_hub_cache_when_hub_is_unavailable(
    tmp_path, monkeypatch
):
    import builtins

    real_import = builtins.__import__

    def no_hub(name, *a, **kw):
        if name.startswith("huggingface_hub"):
            raise ImportError(name)
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", no_hub)
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hub"))
    assert model_cache.hub_cache_dir() == tmp_path / "hub"
    monkeypatch.delenv("HF_HUB_CACHE")
    monkeypatch.setenv("HF_HOME", str(tmp_path / "home"))
    assert model_cache.hub_cache_dir() == tmp_path / "home" / "hub"
