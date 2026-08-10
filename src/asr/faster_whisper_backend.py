"""faster-whisper (CTranslate2) backend — the universal fallback.

This is the engine the bot has always used: ``WhisperModel`` wrapped in a
``BatchedInferencePipeline`` for CPU throughput. CTranslate2 has no Apple
Metal backend, so on Apple Silicon this runs CPU ``int8``; it is also the
graceful-degradation target when MLX is unavailable (discord-transcript-bot-d8g).
"""

import io
import logging
import wave

from faster_whisper import BatchedInferencePipeline, WhisperModel

from .base import NormalizedSegment, TranscribeResult, TranscriptionBackend

logger = logging.getLogger(__name__)


class FasterWhisperBackend(TranscriptionBackend):
    name = "faster-whisper"

    def __init__(self, model_id, device, compute_type, batch_size):
        self.model_id = model_id
        self.device = device
        self.compute_type = compute_type
        self._batch_size = batch_size
        # Loaded eagerly here (not at import) so model construction happens
        # under the memoized get_backend(), once, when first needed.
        self._model = WhisperModel(model_id, device=device, compute_type=compute_type)
        self._batched = BatchedInferencePipeline(self._model)
        logger.info(
            "faster-whisper backend ready: model=%s device=%s compute=%s batch=%s",
            model_id,
            device,
            compute_type,
            batch_size,
        )

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
        audio.seek(0)
        segments, info = self._batched.transcribe(
            audio,
            language=language,
            beam_size=beam_size,
            best_of=best_of,
            batch_size=self._batch_size,
            vad_filter=vad_filter,
            vad_parameters=vad_parameters,
            no_speech_threshold=no_speech_threshold,
            initial_prompt=initial_prompt,
            # Don't let one hallucinated segment seed the next; the
            # roster-biased initial_prompt makes prompt-echo loops more likely
            # without this. (Preserved from the original inline call.)
            condition_on_previous_text=False,
        )
        # faster-whisper Segments already expose the four metric fields;
        # normalize to the uniform return type so downstream stays
        # backend-agnostic.
        norm = [
            NormalizedSegment(
                text=s.text,
                no_speech_prob=getattr(s, "no_speech_prob", 0.0) or 0.0,
                avg_logprob=getattr(s, "avg_logprob", 0.0) or 0.0,
                compression_ratio=getattr(s, "compression_ratio", 1.0) or 1.0,
            )
            for s in segments
        ]
        return TranscribeResult(segments=norm, info=info)

    def healthcheck(self) -> None:
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(16000)
            w.writeframes(b"\x00" * 3200)  # 0.1s of silence
        buf.seek(0)
        # Iterate the generator to actually exercise decoding — otherwise the
        # check passes without verifying the model can decode (roborev #512).
        segments, _info = self._model.transcribe(buf)
        list(segments)
