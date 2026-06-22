"""MLX-Whisper backend — Apple Silicon Metal GPU (discord-transcript-bot-d8g).

faster-whisper (CTranslate2) has no Metal backend, so on Apple Silicon it is
stuck on CPU int8. MLX-Whisper runs Whisper on the GPU via Apple's MLX,
~6-7x faster, at float16 (no int8 quantization loss). That speed headroom is
what lets us keep the most accurate model (large-v3) and raise beam_size for
accuracy instead of trading down — the accuracy-first goal (epic dkj).

``mlx_whisper`` is imported lazily inside the class so this module imports
(and the pure helpers below are testable) without it installed; an import or
load failure raises ``BackendUnavailable`` and selection falls back to
faster-whisper.

MLX-Whisper supports ``initial_prompt`` (so roster biasing is preserved) but
has no Silero VAD — it relies on Whisper's own ``no_speech`` gating plus the
sink's ``_accept_segment`` filter, so ``vad_filter`` / ``vad_parameters`` are
accepted and ignored.
"""

import io
import logging
import math
import wave

import numpy as np

from .base import (
    BackendUnavailable,
    NormalizedSegment,
    TranscribeResult,
    TranscriptionBackend,
)

logger = logging.getLogger(__name__)

# Whisper's mel filterbank assumes 16 kHz mono; an ndarray handed to
# mlx_whisper is NOT resampled internally, so we must deliver exactly this.
TARGET_SAMPLE_RATE = 16000

# Defaults chosen so a missing MLX metric makes _accept_segment KEEP the
# segment (fail-open) — a missing metric must never drop real speech. They
# match _accept_segment's own getattr defaults and NormalizedSegment's.
_KEEP_NO_SPEECH_PROB = 0.0
_KEEP_AVG_LOGPROB = 0.0
_KEEP_COMPRESSION_RATIO = 1.0


def _wav_bytesio_to_float32_mono16k(audio):
    """Convert an in-memory WAV (the sink's 48 kHz/16-bit/stereo BytesIO) to a
    float32 mono ndarray at 16 kHz — the format mlx_whisper expects for an
    ndarray input. Reads the actual WAV params rather than assuming them."""
    audio.seek(0)
    with wave.open(audio, "rb") as w:
        n_channels = w.getnchannels()
        sampwidth = w.getsampwidth()
        sample_rate = w.getframerate()
        frames = w.readframes(w.getnframes())

    if sampwidth != 2:
        raise ValueError(f"expected 16-bit PCM, got sampwidth={sampwidth}")

    samples = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    if n_channels > 1:
        samples = samples.reshape(-1, n_channels).mean(axis=1)

    if sample_rate != TARGET_SAMPLE_RATE and samples.size:
        samples = _resample(samples, sample_rate, TARGET_SAMPLE_RATE)

    return np.ascontiguousarray(samples, dtype=np.float32)


def _resample(samples, src_rate, dst_rate):
    """Resample mono float32 audio. Prefers scipy's polyphase resampler
    (anti-aliased, the quality that protects WER); falls back to numpy linear
    interpolation if scipy is unavailable."""
    g = math.gcd(src_rate, dst_rate)
    up, down = dst_rate // g, src_rate // g
    try:
        from scipy.signal import resample_poly

        return resample_poly(samples, up, down).astype(np.float32)
    except ImportError:
        logger.warning(
            "scipy unavailable; using linear resample (lower quality, may "
            "affect WER). Install scipy for anti-aliased resampling."
        )
        n_out = int(round(samples.size * dst_rate / src_rate))
        if n_out <= 0:
            return samples[:0]
        xp = np.linspace(0.0, 1.0, num=samples.size, endpoint=False)
        x = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
        return np.interp(x, xp, samples).astype(np.float32)


def _normalize_mlx(segments):
    """Map mlx_whisper result segments (Whisper-format dicts) to
    NormalizedSegment, failing OPEN per missing metric so real speech is never
    dropped just because a backend omitted a field. Logs once if any of the
    three quality metrics is missing, so the (degraded) metric-filtering is
    visible rather than silent — the prompt-echo / known-artifact backstops in
    the sink still run regardless."""
    norm = []
    missing = False
    for seg in segments:
        text = seg.get("text", "")
        for key in ("no_speech_prob", "avg_logprob", "compression_ratio"):
            if key not in seg:
                missing = True
        norm.append(
            NormalizedSegment(
                text=text,
                no_speech_prob=seg.get("no_speech_prob", _KEEP_NO_SPEECH_PROB),
                avg_logprob=seg.get("avg_logprob", _KEEP_AVG_LOGPROB),
                compression_ratio=seg.get("compression_ratio", _KEEP_COMPRESSION_RATIO),
            )
        )
    if missing:
        logger.warning(
            "MLX segments missing quality metrics; per-metric hallucination "
            "filtering degraded for those fields (echo/artifact backstops "
            "still active)."
        )
    return norm


class MlxWhisperBackend(TranscriptionBackend):
    name = "mlx-whisper"

    def __init__(self, model_repo):
        self.model_id = model_repo
        try:
            import mlx_whisper  # noqa: F401
        except ImportError as e:
            raise BackendUnavailable(f"mlx_whisper not installed: {e}") from e
        self._mlx = mlx_whisper
        logger.info("mlx-whisper backend ready: repo=%s", model_repo)

    def transcribe(
        self,
        audio,
        *,
        language,
        beam_size,  # mlx_whisper has no beam search; accepted and NOT forwarded.
        best_of,  # only used by Whisper's sampling path (temperature>0); skipped.
        initial_prompt,
        vad_filter,  # MLX has no Silero VAD; accepted and ignored.
        vad_parameters,
        no_speech_threshold,
    ) -> TranscribeResult:
        # NOTE: mlx_whisper decodes greedily with temperature fallback — its
        # beam-search decoder raises NotImplementedError, so beam_size/best_of
        # must NOT be passed. The GPU "raise beam for accuracy" lever therefore
        # applies only to the faster-whisper backend; MLX's accuracy levers are
        # the model id and temperature fallback.
        samples = _wav_bytesio_to_float32_mono16k(audio)
        result = self._mlx.transcribe(
            samples,
            path_or_hf_repo=self.model_id,
            language=language,
            initial_prompt=initial_prompt,
            no_speech_threshold=no_speech_threshold,
            condition_on_previous_text=False,
        )
        return TranscribeResult(
            segments=_normalize_mlx(result.get("segments", [])), info=result
        )

    def healthcheck(self) -> None:
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(TARGET_SAMPLE_RATE)
            w.writeframes(b"\x00" * 3200)  # 0.1s of silence
        buf.seek(0)
        self.transcribe(
            buf,
            language="en",
            beam_size=1,
            best_of=1,
            initial_prompt=None,
            vad_filter=False,
            vad_parameters=None,
            no_speech_threshold=0.6,
        )
