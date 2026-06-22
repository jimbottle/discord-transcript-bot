"""Transcription-backend abstraction (discord-transcript-bot-d8g).

The bot's accuracy logic lives in ``WhisperSink`` — the per-segment
hallucination filter (``_accept_segment``) and the roster ``initial_prompt``
biasing (``_build_initial_prompt`` / ``_is_prompt_echo``). Those layers MUST
keep working whichever engine actually runs the inference, so every backend
yields the SAME normalized segment shape: objects carrying ``text``,
``no_speech_prob``, ``avg_logprob`` and ``compression_ratio`` (the exact
fields ``_accept_segment`` inspects). A backend only owns "audio + decode
params + initial_prompt -> normalized segments"; everything downstream is
backend-agnostic.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional


class BackendError(RuntimeError):
    """Base class for transcription-backend failures."""


class BackendUnavailable(BackendError):
    """A backend cannot be constructed in this environment (missing
    dependency, unsupported hardware, model failed to load). Selection
    catches this and falls back to faster-whisper-CPU."""


@dataclass
class NormalizedSegment:
    """Engine-agnostic stand-in for a faster-whisper ``Segment``.

    Field names match exactly what ``WhisperSink._accept_segment`` reads via
    ``getattr``, so a ``NormalizedSegment`` is a drop-in. The defaults are the
    *keep* values (below every drop threshold): a backend that cannot supply a
    metric fails OPEN — it must never cause real speech to be dropped. See
    ``mlx_backend._normalize_mlx``.
    """

    text: str
    no_speech_prob: float = 0.0
    avg_logprob: float = 0.0
    compression_ratio: float = 1.0


@dataclass
class TranscribeResult:
    """A backend's transcription output: the normalized segments plus the
    engine's opaque ``info`` object (language/duration etc.) when it has one."""

    segments: List[NormalizedSegment] = field(default_factory=list)
    info: Optional[object] = None


class TranscriptionBackend(ABC):
    """Interface every ASR engine implements. Implementations:
    ``FasterWhisperBackend`` (CPU/CUDA) and ``MlxWhisperBackend`` (Apple
    Silicon Metal GPU). The OpenAI Whisper API path stays inline in
    ``WhisperSink.transcribe_audio`` (it is a hosted service, not a local
    engine).
    """

    #: Short engine name, e.g. "faster-whisper" / "mlx-whisper".
    name: str = "base"
    #: Model identifier this backend was built with, e.g. "large-v3".
    model_id: str = ""

    @abstractmethod
    def transcribe(
        self,
        audio,
        *,
        language,
        beam_size,
        best_of,
        initial_prompt,
        vad_filter,
        vad_parameters,
        no_speech_threshold,
    ) -> TranscribeResult:
        """Transcribe one in-memory WAV (a ``BytesIO`` as built by
        ``WhisperSink.transcribe``) into a ``TranscribeResult``.

        ``batch_size`` is deliberately NOT in this signature — it is a
        faster-whisper-only throughput knob with no MLX analogue, so that
        backend reads it from config itself. ``vad_filter`` / ``vad_parameters``
        are passed for backends that support VAD (faster-whisper); a backend
        without it documents that it ignores them.
        """

    @abstractmethod
    def healthcheck(self) -> None:
        """Decode a tiny silent clip to prove the model can run. Raise on
        failure. Used by ``HealthCheck._check_whisper_model``."""
