"""Pluggable ASR (automatic speech recognition) backends.

See ``base.TranscriptionBackend`` for the interface and ``selection.get_backend``
for runtime selection (Apple Silicon -> MLX-Whisper, else faster-whisper).
"""

from .base import (
    BackendError,
    BackendUnavailable,
    NormalizedSegment,
    TranscribeResult,
    TranscriptionBackend,
)
from .selection import get_backend, reset_backend_cache

__all__ = [
    "BackendError",
    "BackendUnavailable",
    "NormalizedSegment",
    "TranscribeResult",
    "TranscriptionBackend",
    "get_backend",
    "reset_backend_cache",
]
