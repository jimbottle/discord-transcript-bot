"""Cold-cache detection for the local transcription model
(discord-transcript-bot-56t).

Whisper weights are not vendored: ``selection.get_backend()`` downloads them
lazily on first use — ~3 GB for large-v3 — from inside the startup health
check, so on a fresh machine the bot simply looks hung for minutes. This
module answers "would that load hit the network?" WITHOUT loading anything,
so ``health.py`` can say so before it happens and ``prewarm_models.py
--check`` can report it.

It mirrors ``selection._build_backend``'s decision (ASR_BACKEND /
WHISPER_MODEL / MLX_WHISPER_MODEL, Apple-Silicon auto-pick) rather than
calling it, because calling it IS the download. Keep the two in step.

Everything here is best-effort: ``probe()`` never raises — a wrong guess
costs one misleading log line, whereas an exception here would fail a
critical health check for a bot that could transcribe fine.
"""

import importlib.util
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

from . import selection

logger = logging.getLogger(__name__)

# A snapshot is usable once one of these exists in it. faster-whisper
# (CTranslate2) ships model.bin; mlx-community repos ship weights.npz or
# weights.safetensors. huggingface_hub only links a file into snapshots/
# after the download completes (partial blobs stay as *.incomplete), so
# presence == complete.
_WEIGHT_FILES = ("model.bin", "weights.npz", "weights.safetensors", "model.safetensors")

# Rough download sizes, by WHISPER_MODEL id, for the heads-up message. Both
# the MLX fp16 and the CTranslate2 float16 packagings of large-v3 are ~3 GB.
_APPROX_GB = {
    "large-v3": 3.0,
    "large-v2": 3.0,
    "large-v1": 3.0,
    "large-v3-turbo": 1.6,
    "medium": 1.5,
    "small": 0.5,
}


@dataclass
class CacheState:
    backend: str  # "mlx-whisper" | "faster-whisper"
    model_id: str  # WHISPER_MODEL as configured
    repo: str  # HF repo id (or a local model directory)
    cached: bool
    cache_dir: str
    approx_gb: Optional[float] = None

    @property
    def size_hint(self) -> str:
        return f"~{self.approx_gb:g} GB" if self.approx_gb else "several GB"

    def cold_message(self) -> str:
        return (
            f"{self.backend} model {self.repo} is not in the model cache "
            f"({self.cache_dir}) — downloading {self.size_hint} now. The bot "
            "will look hung for several minutes while this runs. Run "
            "`make prewarm` before a session to avoid the wait."
        )

    def describe(self) -> str:
        state = (
            "WARM (cached)"
            if self.cached
            else f"COLD (would download {self.size_hint})"
        )
        return f"{self.backend} -> {self.repo}: {state}"


def hub_cache_dir() -> Path:
    """The huggingface_hub cache root, honoring HF_HUB_CACHE / HF_HOME the
    same way the hub does (asked of the hub itself when importable)."""
    try:
        from huggingface_hub import constants

        return Path(constants.HF_HUB_CACHE)
    except Exception:  # noqa: BLE001 - fall back to the hub's documented defaults
        explicit = os.getenv("HF_HUB_CACHE")
        if explicit:
            return Path(explicit)
        home = os.getenv("HF_HOME") or os.path.join(
            Path.home(), ".cache", "huggingface"
        )
        return Path(home) / "hub"


def _mlx_importable() -> bool:
    try:
        return importlib.util.find_spec("mlx_whisper") is not None
    except Exception:  # noqa: BLE001 - a broken finder counts as "not installed"
        return False


def _faster_whisper_repo(model_id: str) -> str:
    if os.path.isdir(model_id):
        return model_id
    try:
        from faster_whisper.utils import _MODELS

        return _MODELS.get(model_id, model_id)
    except Exception:  # noqa: BLE001 - faster_whisper not importable here
        return model_id if "/" in model_id else f"Systran/faster-whisper-{model_id}"


def resolved_repo() -> Tuple[str, str, str]:
    """(backend name, WHISPER_MODEL id, repo) that ``selection.get_backend()``
    would load right now — computed without loading it.

    Mirrors ``selection._build_backend``: an explicit ``mlx`` request is
    reported as MLX even if the package is missing (that load will fail and
    health reports THAT); ``auto`` on Apple Silicon only picks MLX when
    ``mlx_whisper`` is importable, matching the auto path's fallback.
    """
    requested = (os.getenv("ASR_BACKEND") or "auto").strip().lower()
    model_id = os.getenv("WHISPER_MODEL", selection.DEFAULT_WHISPER_MODEL)
    want_mlx = requested == "mlx" or (
        requested == "auto" and selection._is_apple_silicon() and _mlx_importable()
    )
    if want_mlx:
        return "mlx-whisper", model_id, selection._mlx_repo_for(model_id)
    return "faster-whisper", model_id, _faster_whisper_repo(model_id)


def _snapshot_has_weights(repo: str, cache_dir: Path) -> bool:
    snapshots = cache_dir / f"models--{repo.replace('/', '--')}" / "snapshots"
    if not snapshots.is_dir():
        return False
    for snap in snapshots.iterdir():
        for name in _WEIGHT_FILES:
            f = snap / name
            try:
                # Snapshot entries are symlinks into blobs/; stat() follows
                # them, so a dangling link (blob removed) reads as missing.
                if f.is_file() and f.stat().st_size > 0:
                    return True
            except OSError:
                continue
    return False


def is_cached(repo: str, cache_dir: Optional[Path] = None) -> bool:
    if os.path.isdir(repo):
        return True  # a local model directory needs no download
    return _snapshot_has_weights(repo, cache_dir or hub_cache_dir())


def probe() -> Optional[CacheState]:
    """Cache state for the model the bot is about to load. Never raises;
    returns None when it cannot tell (callers then say nothing)."""
    try:
        backend, model_id, repo = resolved_repo()
        cache_dir = hub_cache_dir()
        return CacheState(
            backend=backend,
            model_id=model_id,
            repo=repo,
            cached=is_cached(repo, cache_dir),
            cache_dir=str(cache_dir),
            approx_gb=_APPROX_GB.get(model_id),
        )
    except Exception as e:  # noqa: BLE001 - diagnostics must never fail startup
        logger.debug("model cache probe failed (ignored): %s", e)
        return None
