#!/usr/bin/env python3
"""End-to-end smoke test of reference-audio capture, using real speech.

Unit tests cover capture with a mocked engine; this proves the whole loop a
live session depends on, with the REAL selected backend:

    speech -> WhisperSink.transcribe -> clip on disk + draft manifest
           -> ab_transcribe reads that manifest -> WER/RTF

Run it before a session you intend to capture, so a broken link in that chain
shows up here rather than after four hours of play. Uses macOS ``say`` in
several voices to stand in for several speakers (macOS only; on other
platforms use a real recording instead).

USAGE
    python scripts/smoke_capture.py            # capture + score the loop
    python scripts/smoke_capture.py --keep     # leave the capture dir behind
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Capture must be on BEFORE whisper_sink is imported — the module reads the
# env flag at import time, as it does in the real bot process.
os.environ.setdefault("CAPTURE_SESSION_AUDIO", "1")

from unittest.mock import MagicMock  # noqa: E402

from src.sinks.whisper_sink import Speaker, WhisperSink  # noqa: E402

# Stand-ins for players on the call. The phrases are deliberately D&D-shaped
# and name-heavy: proper nouns are exactly what the accuracy work targets, so
# a decode that mangles them is the interesting failure.
SPEAKERS = [
    ("Alex", "Fenwick", "Alex", "I cast fireball at the goblin"),
    ("Samantha", "Thorin", "Sam", "Thorin swings his axe and misses"),
    ("Daniel", "Mirelle", "Dan", "Mirelle casts healing word on Thorin"),
]


def _say_to_pcm(voice, phrase):
    """Render a phrase with `say` and return raw 48kHz stereo PCM frames —
    the same shape Discord hands the sink. A None voice uses the system
    default (the named voices aren't installed on every machine)."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        path = tmp.name
    try:
        cmd = ["say"]
        if voice:
            cmd += ["-v", voice]
        cmd += ["-o", path, "--data-format=LEI16@48000", phrase]
        subprocess.run(cmd, check=True, capture_output=True, timeout=60)
        with wave.open(path, "rb") as w:
            frames = w.readframes(w.getnframes())
            channels = w.getnchannels()
        if channels == 1:
            # `say` gives mono; the sink expects Discord's stereo, so
            # duplicate each 16-bit sample into both channels.
            stereo = bytearray()
            for i in range(0, len(frames) - 1, 2):
                stereo += frames[i : i + 2] * 2
            frames = bytes(stereo)
        return frames
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def _available_voice(preferred):
    """`say -v` voices vary by machine; fall back to the default voice."""
    try:
        out = subprocess.run(
            ["say", "-v", "?"], capture_output=True, text=True, timeout=15
        ).stdout
        return preferred if preferred in out else None
    except (OSError, subprocess.SubprocessError):
        return None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--keep", action="store_true", help="keep the capture directory"
    )
    args = parser.parse_args(argv)

    if sys.platform != "darwin":
        print("SKIP: needs macOS `say` to synthesize speakers.")
        return 0

    workdir = Path(tempfile.mkdtemp(prefix="smoke_capture_"))
    cwd = os.getcwd()
    os.chdir(workdir)
    try:
        sink = WhisperSink(
            transcript_queue=MagicMock(),
            loop=MagicMock(),
            transcriber_type="local",
            player_map={},
        )
        if sink.capture is None:
            print("FAIL: capture did not initialize (CAPTURE_SESSION_AUDIO unset?)")
            return 1

        print(f"Capture dir: {sink.capture.directory}\n")
        expected = []
        for seq, (voice, character, player, phrase) in enumerate(SPEAKERS):
            v = _available_voice(voice)
            pcm = _say_to_pcm(v, phrase) if v else _say_to_pcm(None, phrase)
            speaker = Speaker(
                user=1000 + seq, player=player, character=character, data=pcm
            )
            speaker.seq = seq
            print(f"  [{player}] said: {phrase!r}")
            text = sink.transcribe(speaker)
            print(f"          heard: {text.strip()!r}")
            sink.write_transcription_log(speaker, text)
            expected.append(phrase)

        capture_dir = Path(sink.capture.directory)
        summary = sink.capture.summary()
        sink.close()
        print(f"\n{summary}")

        # --- verify the artifacts are what the harness expects -------------
        manifest = capture_dir / "manifest.draft.jsonl"
        problems = []
        if not manifest.exists():
            problems.append("no manifest.draft.jsonl written")
        else:
            entries = [
                json.loads(line)
                for line in manifest.read_text().splitlines()
                if line.strip()
            ]
            if len(entries) != len(SPEAKERS):
                problems.append(
                    f"manifest has {len(entries)} lines, expected {len(SPEAKERS)}"
                )
            for e in entries:
                clip = capture_dir / e["audio"]
                if not clip.exists():
                    problems.append(f"manifest points at missing clip {e['audio']}")
                elif clip.stat().st_size == 0:
                    problems.append(f"clip {e['audio']} is empty")
        if not (capture_dir / "README.md").exists():
            problems.append("no README.md written")

        # --- close the loop: score the capture with the real harness ------
        # Correct the draft into ground truth (here we KNOW what was said —
        # in a real session a human does this) and confirm the harness reads
        # it and produces a number.
        if not problems:
            corrected = capture_dir / "manifest.jsonl"
            lines = []
            for e, phrase in zip(
                [
                    json.loads(line)
                    for line in manifest.read_text().splitlines()
                    if line.strip()
                ],
                expected,
            ):
                e["reference"] = phrase
                e.pop("needs_reference", None)
                lines.append(json.dumps(e))
            corrected.write_text("\n".join(lines) + "\n")

            from scripts.ab_transcribe import Config, run_config
            from scripts.ab_transcribe import load_manifest
            from types import SimpleNamespace

            clips = load_manifest(SimpleNamespace(manifest=str(corrected)))
            # Score with the engine that actually produced the clips, not a
            # hardcoded one — otherwise on a faster-whisper host (ASR_BACKEND
            # set, or no mlx_whisper installed) the scoring stage raises
            # BackendUnavailable and this reports the capture loop as broken
            # when only the config was wrong. Backend .name values match
            # Config.backend exactly ("mlx-whisper" / "faster-whisper").
            # cfg.model stays the SHORT id: the harness maps it to an MLX repo
            # itself, so passing backend.model_id would double-map it.
            resolved = getattr(sink._backend, "name", "faster-whisper")
            model = os.getenv("WHISPER_MODEL", "large-v3")
            cfg = Config(
                name=f"{resolved} {model} (as configured)",
                model=model,
                backend=resolved,
            )
            print("\nScoring the captured clips with the real harness...")
            try:
                result = run_config(cfg, clips)
                print(
                    f"  {result['config']}: WER {result['wer'] * 100:.1f}% "
                    f"({result['edits']}/{result['ref_words']} words), "
                    f"RTF {result['rtf']:.2f}"
                )
                print(
                    "\n  (WER here is against synthetic TTS speech — it says the "
                    "loop works,\n   not how the bot does on real players.)"
                )
            except Exception as e:
                problems.append(f"harness failed on the captured manifest: {e}")

        if problems:
            print("\nFAIL:")
            for p in problems:
                print(f"  - {p}")
            return 1

        print("\nOK: capture -> manifest -> harness loop works end to end.")
        return 0
    finally:
        os.chdir(cwd)
        if args.keep:
            print(f"\nCapture kept at: {workdir}")
        else:
            shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
