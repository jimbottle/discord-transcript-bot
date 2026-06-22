"""Runtime ASR backend selection (discord-transcript-bot-d8g + -d29).

Picks the local transcription backend once, memoized, and shared by both the
sink and the health check so the (large) model loads exactly once:

  * Apple Silicon (Darwin/arm64) -> MLX-Whisper (Metal GPU), falling back to
    faster-whisper-CPU if MLX can't load.
  * everything else -> faster-whisper (CUDA float16 if available, else CPU int8).

Overridable with ``ASR_BACKEND`` (auto|mlx|faster-whisper). The model id comes
from ``WHISPER_MODEL`` (default ``large-v3`` — accuracy-first; the GPU backend
makes the most accurate model affordable in near-realtime, so we do NOT default
to turbo). Selection NEVER raises: the guaranteed terminal fallback is
faster-whisper-CPU, the bot's original behavior.
"""

import logging
import os
import platform

logger = logging.getLogger(__name__)

# Accuracy-first default (d29). large-v3-turbo stays selectable via WHISPER_MODEL.
DEFAULT_WHISPER_MODEL = "large-v3"

# Map a generic Whisper id to the MLX-community HF repo. Unknown ids pass
# through as-is, so a full repo can be given directly via WHISPER_MODEL or
# overridden wholesale with MLX_WHISPER_MODEL.
_MLX_REPO_BY_MODEL = {
    "large-v3": "mlx-community/whisper-large-v3-mlx",
    "large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
    "large-v2": "mlx-community/whisper-large-v2-mlx",
    "medium": "mlx-community/whisper-medium-mlx",
    "small": "mlx-community/whisper-small-mlx",
}

_backend = None


def get_backend():
    """Return the memoized live backend, building it on first call."""
    global _backend
    if _backend is None:
        _backend = _build_backend()
    return _backend


def reset_backend_cache():
    """Drop the memoized backend (tests / explicit re-selection)."""
    global _backend
    _backend = None


def _is_apple_silicon():
    return platform.system() == "Darwin" and platform.machine() == "arm64"


def _mlx_repo_for(model_id):
    return os.getenv("MLX_WHISPER_MODEL") or _MLX_REPO_BY_MODEL.get(model_id, model_id)


def _build_faster_whisper(model_id):
    """Build the faster-whisper backend with the same device/precision logic
    the bot used inline: CUDA float16 when a capable GPU is present, else CPU
    int8 (~3-4x faster and far lighter on RAM than float32, negligible accuracy
    loss for speech — see discord-transcript-bot-hin)."""
    from .faster_whisper_backend import FasterWhisperBackend

    device = "cpu"
    try:
        import torch

        if torch.cuda.is_available():
            gpu_ram = torch.cuda.get_device_properties(0).total_memory / 1024**3
            if gpu_ram >= 5.0:
                device = "cuda"
            else:
                logger.warning("GPU has less than 5GB of RAM. Using CPU.")
    except Exception as e:  # torch missing or CUDA probe failed -> CPU
        logger.debug("CUDA probe failed (%s); using CPU.", e)

    compute_type = "float16" if device == "cuda" else "int8"
    batch_size = int(os.getenv("WHISPER_BATCH_SIZE", "8"))
    return FasterWhisperBackend(model_id, device, compute_type, batch_size)


def _build_mlx(model_id):
    from .mlx_backend import MlxWhisperBackend

    return MlxWhisperBackend(_mlx_repo_for(model_id))


def _build_backend():
    requested = (os.getenv("ASR_BACKEND") or "auto").strip().lower()
    model_id = os.getenv("WHISPER_MODEL", DEFAULT_WHISPER_MODEL)

    if requested in ("faster-whisper", "faster_whisper"):
        backend = _build_faster_whisper(model_id)
        logger.info("ASR backend: %s (model=%s)", backend.name, backend.model_id)
        return backend

    want_mlx = requested == "mlx" or (requested == "auto" and _is_apple_silicon())

    if want_mlx:
        try:
            backend = _build_mlx(model_id)
            logger.info("ASR backend: %s (repo=%s)", backend.name, backend.model_id)
            return backend
        except Exception as e:
            # Graceful degradation is a hard requirement. An EXPLICIT mlx
            # request that fails is logged loudly; auto just notes the fallback.
            level = logging.ERROR if requested == "mlx" else logging.WARNING
            logger.log(
                level,
                "MLX-Whisper unavailable (%s); falling back to faster-whisper.",
                e,
            )

    backend = _build_faster_whisper(model_id)
    logger.info("ASR backend: %s (model=%s)", backend.name, backend.model_id)
    return backend
