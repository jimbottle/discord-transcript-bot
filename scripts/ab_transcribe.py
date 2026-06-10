#!/usr/bin/env python3
"""A/B accuracy + speed harness for transcription configs.

Runs several transcription configurations over the SAME recorded session
audio and reports WER (vs a reference transcript) and real-time factor, so
engine/model/param choices are decided on real noisy, overlapping,
proper-noun-heavy D&D audio instead of clean leaderboard benchmarks.
discord-transcript-bot-61z.

This is validation tooling — it is NOT imported by the bot and loads models
only when you run it. faster_whisper is imported lazily inside the worker so
importing this module (e.g. for tests) costs nothing.

USAGE
-----
Single clip:
    python scripts/ab_transcribe.py --audio clip.wav --reference ref.txt

Many per-speaker clips (recommended — mirrors the bot's per-user streams),
via a JSONL manifest, one object per line:
    {"audio": "clips/gus_01.wav", "reference": "I cast fireball"}
    {"audio": "clips/noah_01.wav", "reference_file": "clips/noah_01.txt"}

    python scripts/ab_transcribe.py --manifest clips.jsonl

Custom configs (JSON list of objects with the Config fields below); omit to
use DEFAULT_CONFIGS, which target the turbo-vs-large-v3 and
beam/batched questions:
    python scripts/ab_transcribe.py --manifest clips.jsonl --configs cfg.json

PROPER-NOUN BIASING CAVEAT
--------------------------
The bot biases ``initial_prompt`` per session by appending the roster's
character/player names (``whisper_sink._build_initial_prompt``,
discord-transcript-bot-cul) — a deliberate accuracy lever for names. This
harness applies each Config's static ``initial_prompt`` (default: the base
hint only), so default-config WER UNDER-represents the bot's real accuracy
on proper nouns. To measure like-for-like, set a roster-augmented
``initial_prompt`` on the config(s) you score (it's a per-Config field,
overridable via --configs JSON).

EXTENDING (discord-transcript-bot-1s7 / -mni)
---------------------------------------------
Each Config has a ``backend`` field (default "faster-whisper"). Add an MLX or
Parakeet backend by handling it in ``_transcribe_clip`` and adding configs;
the WER/RTF reporting is backend-agnostic. Note: A/B-ing the Silero VAD
*version* (v4/v5/v6) needs a faster-whisper version swap, not a transcribe
param, so it isn't expressible as a Config here.
"""

import argparse
import json
import sys
import time
import wave
from dataclasses import dataclass, field, asdict
from pathlib import Path

# Keep the bot's package importable when run from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.wer import aggregate_wer, word_error_rate  # noqa: E402


@dataclass
class Config:
    name: str
    model: str = "large-v3"
    compute_type: str = "int8"
    beam_size: int = 5
    best_of: int = 5
    batch_size: int = 8  # 0 = unbatched (plain WhisperModel.transcribe)
    vad_filter: bool = True
    vad_parameters: dict = field(
        default_factory=lambda: dict(min_silence_duration_ms=150, threshold=0.8)
    )
    no_speech_threshold: float = 0.6
    language: str = "en"
    initial_prompt: str = "You are writing the transcriptions for a D&D game."
    backend: str = "faster-whisper"


# Targets the deep-research open questions: current bot baseline vs turbo,
# and the beam_size=10/unbatched prior behavior.
DEFAULT_CONFIGS = [
    Config(name="large-v3 int8 beam5 batched", model="large-v3"),
    Config(name="turbo int8 beam5 batched", model="large-v3-turbo"),
    Config(name="large-v3 int8 beam10 unbatched", beam_size=10, batch_size=0),
    Config(name="turbo int8 beam5 unbatched", model="large-v3-turbo", batch_size=0),
]


def wav_duration_seconds(path):
    with wave.open(str(path), "rb") as w:
        return w.getnframes() / float(w.getframerate())


def load_manifest(args):
    """Return a list of (audio_path, reference_text) pairs from either the
    --manifest JSONL or the single --audio/--reference pair.

    Relative ``audio`` / ``reference_file`` paths in a manifest resolve
    against the MANIFEST's directory, not the cwd, so the harness runs from
    anywhere (the docstring's "clips/gus_01.wav" works as written).
    Absolute paths are used as-is.
    """
    clips = []
    if args.manifest:
        base = Path(args.manifest).resolve().parent
        for lineno, line in enumerate(Path(args.manifest).read_text().splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            audio = base / obj["audio"]
            if "reference_file" in obj:
                ref = (base / obj["reference_file"]).read_text()
            else:
                ref = obj["reference"]
            clips.append((str(audio), ref))
        if not clips:
            raise SystemExit(f"Manifest {args.manifest} had no clips")
    else:
        ref = Path(args.reference).read_text()
        clips.append((args.audio, ref))
    return clips


# Cache loaded models across configs so we don't reload the same weights.
_MODEL_CACHE = {}


def _get_faster_whisper(model, compute_type, batched):
    """Lazily load (and cache) a faster-whisper model, optionally wrapped in
    a BatchedInferencePipeline."""
    from faster_whisper import BatchedInferencePipeline, WhisperModel

    try:
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        device = "cpu"

    key = (model, compute_type, device)
    if key not in _MODEL_CACHE:
        _MODEL_CACHE[key] = WhisperModel(
            model, device=device, compute_type=compute_type
        )
    base = _MODEL_CACHE[key]
    return BatchedInferencePipeline(base) if batched else base


def _transcribe_clip(cfg, audio_path):
    """Transcribe one clip with one config; return (text, proc_seconds)."""
    if cfg.backend != "faster-whisper":
        raise NotImplementedError(
            f"backend '{cfg.backend}' not implemented yet — add it here "
            "(discord-transcript-bot-1s7 / -mni)"
        )
    batched = cfg.batch_size > 0
    model = _get_faster_whisper(cfg.model, cfg.compute_type, batched)
    kwargs = dict(
        language=cfg.language,
        beam_size=cfg.beam_size,
        best_of=cfg.best_of,
        vad_filter=cfg.vad_filter,
        vad_parameters=cfg.vad_parameters,
        no_speech_threshold=cfg.no_speech_threshold,
        initial_prompt=cfg.initial_prompt,
    )
    if batched:
        kwargs["batch_size"] = cfg.batch_size

    start = time.monotonic()
    segments, _ = model.transcribe(str(audio_path), **kwargs)
    text = "".join(seg.text for seg in segments)
    return text, time.monotonic() - start


def run_config(cfg, clips):
    """Run one config over all clips; return a result dict with corpus WER
    and RTF."""
    scores = []
    proc_total = 0.0
    audio_total = 0.0
    for audio_path, reference in clips:
        text, proc = _transcribe_clip(cfg, audio_path)
        scores.append(word_error_rate(reference, text))
        proc_total += proc
        audio_total += wav_duration_seconds(audio_path)
    agg = aggregate_wer(scores)
    return {
        "config": cfg.name,
        "wer": agg["wer"],
        "edits": agg["edits"],
        "ref_words": agg["ref_words"],
        "audio_s": audio_total,
        "proc_s": proc_total,
        "rtf": (proc_total / audio_total) if audio_total else 0.0,
    }


def load_configs(path):
    raw = json.loads(Path(path).read_text())
    return [Config(**obj) for obj in raw]


def format_table(results):
    header = f"{'config':<34} {'WER':>8} {'edits/ref':>12} {'RTF':>7} {'proc_s':>8}"
    lines = [header, "-" * len(header)]
    for r in sorted(results, key=lambda r: r["wer"]):
        wer_pct = f"{r['wer'] * 100:.2f}%"
        ratio = f"{r['edits']}/{r['ref_words']}"
        lines.append(
            f"{r['config']:<34} {wer_pct:>8} {ratio:>12} "
            f"{r['rtf']:>7.2f} {r['proc_s']:>8.1f}"
        )
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--audio", help="single WAV file")
    parser.add_argument("--reference", help="reference transcript .txt for --audio")
    parser.add_argument("--manifest", help="JSONL of {audio, reference|reference_file}")
    parser.add_argument(
        "--configs", help="JSON list of config objects (default set if omitted)"
    )
    parser.add_argument("--json-out", help="write full results JSON here")
    args = parser.parse_args(argv)

    if not args.manifest and not (args.audio and args.reference):
        parser.error("provide --manifest, or both --audio and --reference")

    clips = load_manifest(args)
    configs = load_configs(args.configs) if args.configs else DEFAULT_CONFIGS

    print(f"Scoring {len(configs)} config(s) over {len(clips)} clip(s)...\n")
    results = []
    for cfg in configs:
        print(f"  running: {cfg.name} ...", flush=True)
        results.append(run_config(cfg, clips))

    print("\n" + format_table(results))

    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps(
                {"configs": [asdict(c) for c in configs], "results": results}, indent=2
            )
        )
        print(f"\nWrote {args.json_out}")
    return results


if __name__ == "__main__":
    main()
