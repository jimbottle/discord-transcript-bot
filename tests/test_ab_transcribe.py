"""Tests for the A/B harness orchestration (scripts/ab_transcribe.py).

The harness imports faster_whisper lazily, so loading the module and
exercising everything except the actual transcription is model-free. The one
test that needs a "transcription" stubs _transcribe_clip. discord-transcript-bot-61z.
"""

import importlib.util
import json
import subprocess
import sys
import wave
from pathlib import Path
from types import SimpleNamespace

import pytest


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


def test_main_skips_unavailable_backend_and_still_prints(tmp_path, monkeypatch, capsys):
    """A config whose backend is unavailable (e.g. mlx-whisper on a non-Apple
    host) must be SKIPPED with a notice, not abort the run — the completed
    configs' results must still print. Regression: roborev #2279."""
    from src.asr.base import BackendUnavailable

    wav = tmp_path / "clip.wav"
    _write_wav(wav, seconds=1.0)
    manifest = tmp_path / "clips.jsonl"
    manifest.write_text(json.dumps({"audio": str(wav), "reference": "hi"}) + "\n")
    configs = tmp_path / "cfg.json"
    configs.write_text(
        json.dumps(
            [
                {"name": "fw", "backend": "faster-whisper"},
                {"name": "mlx", "backend": "mlx-whisper"},
            ]
        )
    )

    def fake_run_config(cfg, clips, **kw):
        if cfg.backend == "mlx-whisper":
            raise BackendUnavailable("mlx_whisper not installed")
        return {
            "config": cfg.name,
            "wer": 0.0,
            "edits": 0,
            "ref_words": 1,
            "audio_s": 1.0,
            "proc_s": 0.1,
            "rtf": 0.1,
        }

    monkeypatch.setattr(ab, "run_config", fake_run_config)
    ab.main(["--manifest", str(manifest), "--configs", str(configs)])

    out = capsys.readouterr().out
    assert "SKIPPED mlx" in out, "unavailable backend must be skipped with a notice"
    assert "fw" in out, "the completed config's results must still print"


def test_main_raises_when_all_backends_unavailable(tmp_path, monkeypatch):
    wav = tmp_path / "clip.wav"
    _write_wav(wav, seconds=1.0)
    manifest = tmp_path / "clips.jsonl"
    manifest.write_text(json.dumps({"audio": str(wav), "reference": "hi"}) + "\n")
    configs = tmp_path / "cfg.json"
    configs.write_text(json.dumps([{"name": "mlx", "backend": "mlx-whisper"}]))

    from src.asr.base import BackendUnavailable

    def always_unavailable(cfg, clips, **kw):
        raise BackendUnavailable("nope")

    monkeypatch.setattr(ab, "run_config", always_unavailable)
    try:
        ab.main(["--manifest", str(manifest), "--configs", str(configs)])
        assert False, "should SystemExit when no config could run"
    except SystemExit:
        pass


# ── order effects / marginal clips (discord-transcript-bot-adg) ────────


def _three_clips(tmp_path):
    clips = []
    for name in ("a", "b", "c"):
        wav = tmp_path / f"{name}.wav"
        _write_wav(wav, seconds=1.0)
        clips.append((str(wav), "hello there"))
    return clips


def _order_dependent_stub(ab, monkeypatch):
    """b.wav decodes differently depending on whether a.wav ran right
    before it — the measured MLX behaviour on marginal audio. a and c are
    stable."""
    seen = []

    def fake(cfg, audio):
        prev = seen[-1] if seen else None
        seen.append(audio)
        if audio.endswith("b.wav"):
            return (
                "hello there" if prev and prev.endswith("a.wav") else "hello here",
                0.1,
            )
        return ("hello there", 0.1)

    monkeypatch.setattr(ab, "_transcribe_clip", fake)
    return seen


def test_trial_order_is_manifest_order_then_seeded_shuffles():
    assert ab._trial_order(4, 0) == [0, 1, 2, 3]
    t1 = ab._trial_order(4, 1)
    assert sorted(t1) == [0, 1, 2, 3]
    assert t1 == ab._trial_order(4, 1), "trials must be reproducible"


def test_single_trial_run_config_reports_per_clip_and_no_sensitivity(
    tmp_path, monkeypatch
):
    clips = _three_clips(tmp_path)
    _order_dependent_stub(ab, monkeypatch)
    r = ab.run_config(ab.Config(name="x"), clips)
    assert r["order_trials"] == 1
    assert r["order_sensitive"] == []
    assert [c["audio"] for c in r["clips"]] == ["a.wav", "b.wav", "c.wav"]
    assert r["clips"][1]["hyp"] == "hello there"  # ran right after a.wav
    assert r["wer"] == r["wer_min"] == r["wer_max"] == 0.0


def test_order_trials_flag_the_order_sensitive_clip(tmp_path, monkeypatch):
    clips = _three_clips(tmp_path)
    seen = _order_dependent_stub(ab, monkeypatch)
    r = ab.run_config(ab.Config(name="x"), clips, order_trials=3)

    assert r["order_trials"] == 3
    assert len(seen) == 9, "every clip transcribed once per trial"
    assert r["order_sensitive"] == [clips[1][0]]
    b = r["clips"][1]
    assert b["order_sensitive"] is True
    assert sorted(b["hyps"]) == ["hello here", "hello there"]
    assert all(not c["order_sensitive"] for c in r["clips"] if c["audio"] != "b.wav")
    # Headline numbers are trial 0 (manifest order) — identical to a plain run.
    assert r["wer"] == 0.0
    assert r["wer_min"] == 0.0 and r["wer_max"] > 0.0
    # Only trial 0's compute counts toward RTF.
    assert r["proc_s"] == pytest.approx(0.3)


def test_hypotheses_that_differ_only_in_case_or_punctuation_are_stable(
    tmp_path, monkeypatch
):
    clips = _three_clips(tmp_path)
    n = {"i": 0}

    def fake(cfg, audio):
        n["i"] += 1
        return ("Hello, there!" if n["i"] % 2 else "hello there", 0.1)

    monkeypatch.setattr(ab, "_transcribe_clip", fake)
    r = ab.run_config(ab.Config(name="x"), clips, order_trials=2)
    assert r["order_sensitive"] == []


def test_format_table_adds_range_and_unstable_columns_for_multi_trial():
    base = {"edits": 1, "ref_words": 10, "rtf": 1.0, "proc_s": 2.0}
    single = [{"config": "s", "wer": 0.1, **base}]
    assert "unstable" not in ab.format_table(single)
    multi = [
        {
            "config": "m",
            "wer": 0.1,
            "order_trials": 3,
            "wer_min": 0.1,
            "wer_max": 0.3,
            "order_sensitive": ["/x/b.wav"],
            **base,
        }
    ]
    table = ab.format_table(multi)
    assert "unstable" in table and "10.00-30.00%" in table


def test_format_order_sensitivity_lists_clips_and_hypotheses():
    results = [
        {
            "config": "m",
            "order_trials": 3,
            "clips": [
                {"audio": "a.wav", "order_sensitive": False, "hyps": ["x"]},
                {"audio": "b.wav", "order_sensitive": True, "hyps": ["x", "y"]},
            ],
        }
    ]
    text = ab.format_order_sensitivity(results)
    assert "b.wav" in text and "'x'" in text and "'y'" in text
    assert "a.wav" not in text
    assert ab.format_order_sensitivity([{"config": "c", "clips": []}]) == ""


def test_write_stable_manifest_drops_unstable_lines_verbatim(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    lines = [
        json.dumps({"audio": "a.wav", "reference": "one", "player": "Gus"}),
        json.dumps({"audio": "b.wav", "reference": "two", "note": "mumbled"}),
        json.dumps({"audio": "c.wav", "reference": "three"}),
    ]
    manifest.write_text("\n".join(lines) + "\n\n")
    out = tmp_path / "stable.jsonl"
    kept, dropped = ab.write_stable_manifest(manifest, {str(tmp_path / "b.wav")}, out)
    assert (kept, dropped) == (2, 1)
    assert out.read_text().splitlines() == [lines[0], lines[2]]


def test_main_order_trials_writes_stable_manifest(tmp_path, monkeypatch, capsys):
    clips = _three_clips(tmp_path)
    _order_dependent_stub(ab, monkeypatch)
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        "".join(
            json.dumps({"audio": Path(p).name, "reference": ref}) + "\n"
            for p, ref in clips
        )
    )
    cfgs = tmp_path / "cfg.json"
    cfgs.write_text(json.dumps([{"name": "only", "backend": "faster-whisper"}]))
    stable = tmp_path / "stable.jsonl"
    ab.main(
        [
            "--manifest",
            str(manifest),
            "--configs",
            str(cfgs),
            "--order-trials",
            "3",
            "--stable-manifest",
            str(stable),
        ]
    )
    out = capsys.readouterr().out
    assert "order-sensitive" in out and "b.wav" in out
    assert "kept 2 clip(s), dropped 1" in out
    assert [json.loads(l)["audio"] for l in stable.read_text().splitlines()] == [
        "a.wav",
        "c.wav",
    ]


def test_stable_manifest_requires_manifest_and_multiple_trials(tmp_path):
    wav = tmp_path / "c.wav"
    _write_wav(wav)
    ref = tmp_path / "r.txt"
    ref.write_text("hi")
    with pytest.raises(SystemExit):
        ab.main(
            ["--audio", str(wav), "--reference", str(ref), "--stable-manifest", "x"]
        )


def test_isolate_runs_each_config_in_its_own_subprocess(tmp_path, monkeypatch, capsys):
    wav = tmp_path / "clip.wav"
    _write_wav(wav)
    manifest = tmp_path / "m.jsonl"
    manifest.write_text(json.dumps({"audio": "clip.wav", "reference": "hi"}) + "\n")
    cfgs = tmp_path / "cfg.json"
    cfgs.write_text(
        json.dumps(
            [
                {"name": "fw", "backend": "faster-whisper"},
                {"name": "mlx", "backend": "mlx-whisper"},
            ]
        )
    )
    calls = []

    def fake_run(cmd, **kw):
        calls.append(cmd)
        cfg = json.loads(Path(cmd[cmd.index("--configs") + 1]).read_text())[0]
        out = Path(cmd[cmd.index("--json-out") + 1])
        if cfg["backend"] == "mlx-whisper":
            return subprocess.CompletedProcess(cmd, 1, "", "No configs ran")
        out.write_text(
            json.dumps(
                {
                    "results": [
                        {
                            "config": cfg["name"],
                            "wer": 0.0,
                            "edits": 0,
                            "ref_words": 1,
                            "audio_s": 1.0,
                            "proc_s": 0.1,
                            "rtf": 0.1,
                        }
                    ]
                }
            )
        )
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(ab.subprocess, "run", fake_run)
    monkeypatch.setattr(
        ab, "run_config", lambda *a, **k: pytest.fail("must not run in-process")
    )
    ab.main(
        [
            "--manifest",
            str(manifest),
            "--configs",
            str(cfgs),
            "--isolate",
            "--order-trials",
            "2",
        ]
    )

    assert len(calls) == 2, "one subprocess per config"
    for cmd in calls:
        assert cmd[0] == sys.executable and cmd[1].endswith("ab_transcribe.py")
        assert "--isolate" not in cmd, "child must not recurse"
        assert cmd[cmd.index("--order-trials") + 1] == "2"
        assert cmd[cmd.index("--manifest") + 1] == str(manifest)
    out = capsys.readouterr().out
    assert "SKIPPED mlx" in out and "fw" in out
