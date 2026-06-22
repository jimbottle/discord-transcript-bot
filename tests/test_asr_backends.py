"""Backend-level tests (discord-transcript-bot-d8g): MLX segment normalization
+ fail-open, the WAV->16k-mono conversion, that the per-segment hallucination
filter still fires on NormalizedSegments regardless of backend, and that the
faster-whisper backend forwards decode params (incl. batch_size) to the model.

These avoid loading any real model: the MLX helpers are pure (mlx_whisper is
imported lazily inside the class, not at module import), and the faster-whisper
backend is exercised via a fake batched pipeline.
"""

import io
import wave
from types import SimpleNamespace

import numpy as np

from src.asr import mlx_backend
from src.asr.base import NormalizedSegment, TranscribeResult
from src.asr.faster_whisper_backend import FasterWhisperBackend
from src.asr.mlx_backend import (
    _normalize_mlx,
    _wav_bytesio_to_float32_mono16k,
)


def _wav_bytesio(seconds=0.5, rate=48000, channels=2):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(rate)
        frame = b"\x00\x00" * channels  # one silent frame (2 bytes per channel)
        w.writeframes(frame * int(rate * seconds))
    buf.seek(0)
    return buf


# ── MLX segment normalization ─────────────────────────────────────────────


def test_normalize_mlx_maps_all_fields():
    segs = [
        {
            "text": "hello",
            "no_speech_prob": 0.2,
            "avg_logprob": -0.4,
            "compression_ratio": 1.3,
        }
    ]
    out = _normalize_mlx(segs)
    assert len(out) == 1
    assert isinstance(out[0], NormalizedSegment)
    assert out[0].text == "hello"
    assert out[0].no_speech_prob == 0.2
    assert out[0].avg_logprob == -0.4
    assert out[0].compression_ratio == 1.3


def test_normalize_mlx_fails_open_on_missing_metrics(caplog):
    """A segment missing the quality metrics must keep KEEP-defaults so real
    speech is never dropped just because the engine omitted a field — and the
    degradation is logged, not silent."""
    import logging

    with caplog.at_level(logging.WARNING):
        out = _normalize_mlx([{"text": "hi"}])
    seg = out[0]
    assert seg.no_speech_prob == 0.0  # below the 0.8 drop threshold -> kept
    assert seg.avg_logprob == 0.0  # above the -1.0 threshold -> kept
    assert seg.compression_ratio == 1.0  # below the 2.4 threshold -> kept
    assert any("missing quality metrics" in r.message for r in caplog.records)


def test_accept_segment_keeps_failopen_segment(tmp_path, monkeypatch):
    """The sink's layer-1 filter must KEEP a fail-open normalized segment..."""
    from tests.test_whisper_sink import _make_sink

    sink = _make_sink(tmp_path, monkeypatch, player_map={})
    seg = _normalize_mlx([{"text": "real words"}])[0]
    assert sink._accept_segment(seg) is True
    sink.close()


def test_accept_segment_still_drops_bad_normalized_segments(tmp_path, monkeypatch):
    """...but still DROPS normalized segments that trip the metric thresholds
    or are known artifacts — proving layer 1 works on any backend's output."""
    from tests.test_whisper_sink import _make_sink

    sink = _make_sink(tmp_path, monkeypatch, player_map={})
    silence = NormalizedSegment(text="x", no_speech_prob=0.95)
    low_conf = NormalizedSegment(text="x", avg_logprob=-1.5)
    repetition = NormalizedSegment(text="la la", compression_ratio=3.0)
    artifact = NormalizedSegment(text="Subtitles by the Amara.org community")
    assert sink._accept_segment(silence) is False
    assert sink._accept_segment(low_conf) is False
    assert sink._accept_segment(repetition) is False
    assert sink._accept_segment(artifact) is False
    sink.close()


# ── WAV -> 16k mono float32 conversion ─────────────────────────────────────


def test_wav_conversion_downmixes_and_resamples():
    out = _wav_bytesio_to_float32_mono16k(_wav_bytesio(seconds=0.5))
    assert out.dtype == np.float32
    assert out.ndim == 1  # mono
    # 0.5s at 16 kHz ~= 8000 samples (resampler edge effects allow slack).
    assert abs(out.size - 8000) <= 50


def test_wav_conversion_passthrough_when_already_16k_mono():
    out = _wav_bytesio_to_float32_mono16k(
        _wav_bytesio(seconds=0.25, rate=16000, channels=1)
    )
    assert out.dtype == np.float32
    assert abs(out.size - 4000) <= 5


# ── MLX backend unavailable -> BackendUnavailable ──────────────────────────


def test_mlx_backend_raises_when_mlx_not_installed(monkeypatch):
    """If mlx_whisper can't import, constructing the backend raises
    BackendUnavailable (which selection turns into a faster-whisper fallback)."""
    import builtins

    from src.asr.base import BackendUnavailable

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "mlx_whisper":
            raise ImportError("no mlx_whisper")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    try:
        mlx_backend.MlxWhisperBackend("mlx-community/whisper-large-v3-mlx")
        assert False, "expected BackendUnavailable"
    except BackendUnavailable:
        pass


def test_mlx_backend_forwards_params_and_normalizes():
    """MLX transcribe forwards the decode params (incl.
    condition_on_previous_text=False) + the repo + converted float32 samples to
    mlx_whisper.transcribe, and wraps the result via _normalize_mlx. Built via
    __new__ with a fake _mlx so no model loads — guards the MLX call signature,
    the likeliest place an upstream API change would silently break."""
    captured = {}

    def fake_mlx_transcribe(samples, **kwargs):
        captured["samples"] = samples
        captured.update(kwargs)
        return {
            "segments": [
                {
                    "text": "hi",
                    "no_speech_prob": 0.1,
                    "avg_logprob": -0.2,
                    "compression_ratio": 1.1,
                }
            ]
        }

    be = object.__new__(mlx_backend.MlxWhisperBackend)
    be.model_id = "mlx-community/whisper-large-v3-mlx"
    be._mlx = SimpleNamespace(transcribe=fake_mlx_transcribe)

    result = be.transcribe(
        _wav_bytesio(0.2),
        language="en",
        beam_size=7,
        best_of=9,
        initial_prompt="prompt here",
        vad_filter=False,  # MLX ignores VAD; must not be forwarded / must not error
        vad_parameters=None,
        no_speech_threshold=0.6,
    )

    assert captured["path_or_hf_repo"] == "mlx-community/whisper-large-v3-mlx"
    assert captured["language"] == "en"
    assert captured["beam_size"] == 7
    assert captured["best_of"] == 9
    assert captured["initial_prompt"] == "prompt here"
    assert captured["no_speech_threshold"] == 0.6
    assert captured["condition_on_previous_text"] is False
    assert isinstance(captured["samples"], np.ndarray)
    assert captured["samples"].dtype == np.float32
    assert isinstance(result, TranscribeResult)
    assert result.segments[0].text == "hi"
    assert result.segments[0].no_speech_prob == 0.1


# ── faster-whisper backend forwards decode params (incl. batch_size) ───────


def test_faster_whisper_backend_forwards_params():
    """beam_size/best_of/batch_size/initial_prompt/vad all reach the batched
    pipeline, and faster-whisper Segments are normalized to NormalizedSegment.
    Built via __new__ to avoid loading a real model."""
    captured = {}

    def fake_transcribe(audio, **kwargs):
        captured.update(kwargs)
        seg = SimpleNamespace(
            text="hi", no_speech_prob=0.1, avg_logprob=-0.2, compression_ratio=1.1
        )
        return ([seg], SimpleNamespace(language="en"))

    be = object.__new__(FasterWhisperBackend)
    be._batch_size = 4
    be._batched = SimpleNamespace(transcribe=fake_transcribe)

    result = be.transcribe(
        _wav_bytesio(0.2),
        language="en",
        beam_size=7,
        best_of=9,
        initial_prompt="prompt here",
        vad_filter=True,
        vad_parameters={"threshold": 0.8},
        no_speech_threshold=0.6,
    )

    assert captured["beam_size"] == 7
    assert captured["best_of"] == 9
    assert captured["batch_size"] == 4
    assert captured["initial_prompt"] == "prompt here"
    assert captured["vad_filter"] is True
    assert captured["condition_on_previous_text"] is False
    assert isinstance(result, TranscribeResult)
    assert result.segments[0].text == "hi"
    assert result.segments[0].no_speech_prob == 0.1
