#!/usr/bin/env python3
"""Download + warm the transcription model BEFORE a live session.

Model weights are fetched lazily on first transcription. For the MLX path
that is a ~3 GB download, so a cold cache means the first person to speak
triggers a multi-minute stall — during a real game, on the exact code path
whose backlog-snowball failure mode we just fixed
(discord-transcript-bot-hin). Run this once before a session; it is a no-op
when the weights are already cached.

Resolves the SAME backend the bot will use (src/asr/selection honoring
WHISPER_MODEL / ASR_BACKEND / MLX_WHISPER_MODEL), so what gets warmed is
what runs. Optionally transcribes a real spoken clip end-to-end to prove the
whole path works — not just that the files downloaded.

USAGE
    python scripts/prewarm_models.py            # download + warm
    python scripts/prewarm_models.py --check    # report cache state, download nothing
"""

import argparse
import io
import os
import subprocess
import sys
import tempfile
import time
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.asr import selection  # noqa: E402
from src.asr.base import BackendUnavailable  # noqa: E402

# Spoken by macOS `say` for the end-to-end check. Deliberately D&D-flavored
# with a proper noun, so a totally broken decode is obvious in the output.
SMOKE_PHRASE = "I cast fireball at the goblin"


def _silent_wav(seconds=0.6):
    """Discord-format silence, as a fallback when `say` isn't available."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(48000)
        w.writeframes(b"\x00\x00\x00\x00" * int(48000 * seconds))
    return buf.getvalue()


def _spoken_wav():
    """Real speech via macOS `say`, in the bot's 48kHz stereo format. Returns
    (wav_bytes, is_real_speech)."""
    if sys.platform != "darwin":
        return _silent_wav(), False
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            path = tmp.name
        subprocess.run(
            ["say", "-o", path, "--data-format=LEI16@48000", SMOKE_PHRASE],
            check=True,
            capture_output=True,
            timeout=30,
        )
        data = Path(path).read_bytes()
        os.unlink(path)
        return data, True
    except (OSError, subprocess.SubprocessError):
        return _silent_wav(), False


def describe_cache():
    """Human summary of what's already on disk."""
    hub = Path.home() / ".cache" / "huggingface" / "hub"
    if not hub.exists():
        return "empty (no ~/.cache/huggingface/hub)"
    repos = [p.name for p in hub.glob("models--*")]
    if not repos:
        return "empty (no models in ~/.cache/huggingface/hub)"
    total = sum(f.stat().st_size for f in hub.rglob("*") if f.is_file())
    return f"{len(repos)} model(s), {total / 1024**3:.1f} GB: {', '.join(repos)}"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--check",
        action="store_true",
        help="report cache state and exit without downloading",
    )
    args = parser.parse_args(argv)

    # This script's whole job is narrating a multi-minute download, so its
    # progress must not sit in a pipe buffer when the output is redirected.
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except AttributeError:
        pass

    model = os.getenv("WHISPER_MODEL", "large-v3")
    backend_pref = os.getenv("ASR_BACKEND", "auto")
    print(f"WHISPER_MODEL={model}  ASR_BACKEND={backend_pref}")
    print(f"Model cache: {describe_cache()}")

    if args.check:
        return 0

    print("\nResolving backend (downloads weights on a cold cache)...")
    start = time.monotonic()
    try:
        backend = selection.get_backend()
    except BackendUnavailable as e:
        print(f"FAIL: no transcription backend available: {e}")
        return 1
    load_s = time.monotonic() - start
    print(f"  backend: {type(backend).__name__}  (ready in {load_s:.1f}s)")

    wav, is_speech = _spoken_wav()
    label = f"real speech ({SMOKE_PHRASE!r})" if is_speech else "silence"
    print(f"\nTranscribing a smoke clip — {label}...")
    start = time.monotonic()
    result = backend.transcribe(
        io.BytesIO(wav),
        language="en",
        beam_size=5,
        best_of=5,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=150, threshold=0.8),
        no_speech_threshold=0.6,
        initial_prompt="You are writing the transcriptions for a D&D game.",
    )
    infer_s = time.monotonic() - start
    text = "".join(seg.text for seg in result.segments).strip()
    print(f"  -> {text!r}  ({infer_s:.2f}s)")

    print(f"\nModel cache now: {describe_cache()}")
    if is_speech and not text:
        print(
            "\nWARNING: real speech transcribed to nothing. The engine loaded "
            "but is not decoding — investigate before relying on a session."
        )
        return 1
    print("\nOK: model warm, engine decoding. Safe to start a session.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
