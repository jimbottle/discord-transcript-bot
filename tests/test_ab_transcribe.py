"""Tests for the A/B harness orchestration (scripts/ab_transcribe.py).

The harness imports faster_whisper lazily, so loading the module and
exercising everything except the actual transcription is model-free. The one
test that needs a "transcription" stubs _transcribe_clip. discord-transcript-bot-61z.
"""

import importlib.util
import json
import wave
from pathlib import Path
from types import SimpleNamespace


def _load_harness():
    path = Path(__file__).resolve().parents[1] / "scripts" / "ab_transcribe.py"
    spec = importlib.util.spec_from_file_location("ab_transcribe", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ab = _load_harness()


def _write_wav(path, seconds=1.0):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(48000)
        w.writeframes(b"\x00\x00\x00\x00" * int(48000 * seconds))


def test_default_configs_cover_turbo_and_beam_questions():
    names = [c.name for c in ab.DEFAULT_CONFIGS]
    models = {c.model for c in ab.DEFAULT_CONFIGS}
    assert "large-v3" in models and "large-v3-turbo" in models
    # at least one unbatched and one beam=10 config to compare prior behavior
    assert any(c.batch_size == 0 for c in ab.DEFAULT_CONFIGS)
    assert any(c.beam_size == 10 for c in ab.DEFAULT_CONFIGS)
    assert len(names) == len(set(names)), "config names must be unique"


def test_load_manifest_reads_jsonl_inline_and_file_refs(tmp_path):
    ref_file = tmp_path / "noah.txt"
    ref_file.write_text("from the file")
    manifest = tmp_path / "clips.jsonl"
    manifest.write_text(
        json.dumps({"audio": "a.wav", "reference": "inline ref"})
        + "\n"
        + json.dumps({"audio": "b.wav", "reference_file": str(ref_file)})
        + "\n"
    )

    args = SimpleNamespace(manifest=str(manifest), audio=None, reference=None)
    clips = ab.load_manifest(args)
    base = manifest.resolve().parent
    # audio paths resolve against the manifest dir; an absolute reference_file
    # is honored as-is.
    assert clips == [
        (str(base / "a.wav"), "inline ref"),
        (str(base / "b.wav"), "from the file"),
    ]


def test_load_manifest_resolves_relative_paths_against_manifest_dir(
    tmp_path, monkeypatch
):
    """roborev #2011: relative audio/reference_file must resolve against the
    manifest's directory, not the cwd, so the harness runs from anywhere."""
    clipdir = tmp_path / "session"
    clipdir.mkdir()
    (clipdir / "gus.txt").write_text("hello there")
    manifest = clipdir / "clips.jsonl"
    manifest.write_text(
        json.dumps({"audio": "gus.wav", "reference_file": "gus.txt"}) + "\n"
    )

    other = tmp_path / "elsewhere"
    other.mkdir()
    monkeypatch.chdir(other)  # run from a different cwd

    clips = ab.load_manifest(
        SimpleNamespace(manifest=str(manifest), audio=None, reference=None)
    )
    audio, ref = clips[0]
    assert ref == "hello there"  # reference_file found despite the cwd
    assert Path(audio) == clipdir.resolve() / "gus.wav"


def test_no_speech_threshold_is_config_field_and_passed(monkeypatch):
    """roborev #2011: no_speech_threshold is a swept Config field, not a
    hardcoded transcribe kwarg."""
    assert ab.Config(name="x").no_speech_threshold == 0.6

    captured = {}

    class _FakeModel:
        def transcribe(self, audio, **kwargs):
            captured.update(kwargs)
            return ([], None)

    monkeypatch.setattr(ab, "_get_faster_whisper", lambda *a, **k: _FakeModel())
    cfg = ab.Config(name="x", no_speech_threshold=0.42, batch_size=0)
    ab._transcribe_clip(cfg, "clip.wav")
    assert captured["no_speech_threshold"] == 0.42


def test_run_config_computes_corpus_wer_and_rtf(tmp_path, monkeypatch):
    wav = tmp_path / "clip.wav"
    _write_wav(wav, seconds=2.0)

    # Stub the actual transcription: 1 substitution, 0.5s of "compute".
    monkeypatch.setattr(
        ab, "_transcribe_clip", lambda cfg, audio: ("the quick brown dog", 0.5)
    )

    cfg = ab.Config(name="test")
    result = ab.run_config(cfg, [(str(wav), "the quick brown fox")])

    assert result["wer"] == 0.25  # 1 edit / 4 ref words
    assert result["edits"] == 1
    assert result["ref_words"] == 4
    assert result["audio_s"] == 2.0
    assert result["proc_s"] == 0.5
    assert result["rtf"] == 0.25  # 0.5s proc / 2.0s audio


def test_format_table_sorts_by_wer():
    results = [
        {
            "config": "worse",
            "wer": 0.5,
            "edits": 5,
            "ref_words": 10,
            "rtf": 1.0,
            "proc_s": 2.0,
        },
        {
            "config": "better",
            "wer": 0.1,
            "edits": 1,
            "ref_words": 10,
            "rtf": 2.0,
            "proc_s": 4.0,
        },
    ]
    table = ab.format_table(results)
    assert table.index("better") < table.index("worse")


def test_unknown_backend_raises(tmp_path):
    cfg = ab.Config(name="parakeet", backend="parakeet")
    try:
        ab._transcribe_clip(cfg, "x.wav")
        assert False, "should have raised for unimplemented backend"
    except NotImplementedError as e:
        assert "parakeet" in str(e)


def test_mlx_backend_dispatches(tmp_path, monkeypatch):
    """A mlx-whisper config routes to _transcribe_mlx, reusing the production
    MlxWhisperBackend (mocked here so no model loads). discord-transcript-bot-d6j."""
    from src.asr.base import NormalizedSegment, TranscribeResult

    clip = tmp_path / "c.wav"
    clip.write_bytes(b"RIFFfake")  # _transcribe_mlx only reads bytes into BytesIO

    captured = {}

    class _FakeBackend:
        def transcribe(self, audio, **kwargs):
            captured.update(kwargs)
            captured["read_bytes"] = audio.read()
            return TranscribeResult(
                segments=[NormalizedSegment(text="hello world")], info={}
            )

    monkeypatch.setattr(ab, "_get_mlx", lambda model: _FakeBackend())
    cfg = ab.Config(name="mlx", backend="mlx-whisper", model="large-v3")
    text, secs = ab._transcribe_clip(cfg, str(clip))
    assert text == "hello world"
    assert isinstance(secs, float)
    assert captured["read_bytes"] == b"RIFFfake"
    assert captured["initial_prompt"] == cfg.initial_prompt
