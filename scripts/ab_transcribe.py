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

ORDER EFFECTS AND MARGINAL CLIPS (discord-transcript-bot-adg)
-------------------------------------------------------------
Measured on real captured clips with the production MLX backend: for
near-unintelligible audio the decode depends on what was transcribed earlier
in the same process (GPU numerical state flipping a marginal argmax), while
repeats within a process are stable and clear speech is unaffected. Because
run_config loops every clip through one process, such clips contribute WER
that depends on clip order and on which config ran first — differences
between configs on them are noise, not signal.

Two flags make that visible instead of folding it into the corpus number:

    --order-trials 3     run each config 3 times with the clip order shuffled
                         (trial 0 = manifest order); clips whose hypothesis
                         changes between trials are reported as
                         ORDER-SENSITIVE and counted in the table
    --stable-manifest P  write the manifest minus every order-sensitive clip
                         to P, ready to score for real
    --isolate            run each config in a fresh subprocess so config B's
                         numbers cannot depend on config A having run first

Recommended methodology: `--order-trials 3 --stable-manifest stable.jsonl`
once, review the flagged clips (delete near-unintelligible ones from the
reference rather than guessing their text), then score the stable manifest
with `--isolate`. Per-clip WER + hypotheses are always included in
--json-out for inspection.

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
import os
import random
import subprocess
import sys
import tempfile
import time
import wave
from dataclasses import dataclass, field, asdict
from pathlib import Path

# Keep the bot's package importable when run from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.asr.base import BackendUnavailable  # noqa: E402
from src.wer import aggregate_wer, normalize_text, word_error_rate  # noqa: E402


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


# Targets the deep-research open questions: current CPU baseline vs turbo and
# the beam10 prior behavior on faster-whisper, PLUS the GPU path the MLX backend
# unlocks — MLX large-v3 vs turbo. NOTE: mlx_whisper has no beam-search decoder
# (greedy + temperature fallback only), so beam_size is a no-op for MLX configs;
# the beam5-vs-beam10 question lives on the faster-whisper rows.
# discord-transcript-bot-d6j. On a host without mlx_whisper the MLX configs
# raise BackendUnavailable; main() catches that and skips them with a notice
# (printing a warning), so the default set still runs on non-Apple-Silicon hosts.
DEFAULT_CONFIGS = [
    Config(name="large-v3 int8 beam5 batched", model="large-v3"),
    Config(name="turbo int8 beam5 batched", model="large-v3-turbo"),
    Config(name="large-v3 int8 beam10 unbatched", beam_size=10, batch_size=0),
    Config(name="turbo int8 beam5 unbatched", model="large-v3-turbo", batch_size=0),
    Config(name="mlx large-v3 fp16 greedy", model="large-v3", backend="mlx-whisper"),
    Config(
        name="mlx large-v3-turbo fp16 greedy",
        model="large-v3-turbo",
        backend="mlx-whisper",
    ),
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
        if ".draft" in Path(args.manifest).name:
            # A capture's manifest.draft.jsonl holds MACHINE transcriptions as
            # its references (src/session_capture.py). Scoring against those
            # compares Whisper to itself and reports a meaninglessly low WER,
            # which looks like a great result — so say so loudly rather than
            # silently producing a bogus bake-off.
            print(
                f"WARNING: {Path(args.manifest).name} looks like an "
                "uncorrected capture draft. Its 'reference' fields are "
                "machine output, not ground truth, so the WER below is "
                "meaningless. Correct the text and save as manifest.jsonl "
                "first — see the README in the capture directory.\n",
                file=sys.stderr,
            )
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


def _get_mlx(model):
    """Lazily load (and cache) the production MLX-Whisper backend for a model
    id, so the bake-off measures the SAME engine + 16k-mono conversion the bot
    actually runs (src/asr/mlx_backend). discord-transcript-bot-d6j."""
    from src.asr.mlx_backend import MlxWhisperBackend
    from src.asr.selection import _mlx_repo_for

    key = ("mlx", model)
    if key not in _MODEL_CACHE:
        _MODEL_CACHE[key] = MlxWhisperBackend(_mlx_repo_for(model))
    return _MODEL_CACHE[key]


def _transcribe_faster_whisper(cfg, audio_path):
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


def _transcribe_mlx(cfg, audio_path):
    import io

    backend = _get_mlx(cfg.model)
    with open(audio_path, "rb") as f:
        wav = io.BytesIO(f.read())

    start = time.monotonic()
    result = backend.transcribe(
        wav,
        language=cfg.language,
        beam_size=cfg.beam_size,
        best_of=cfg.best_of,
        initial_prompt=cfg.initial_prompt,
        vad_filter=cfg.vad_filter,
        vad_parameters=cfg.vad_parameters,
        no_speech_threshold=cfg.no_speech_threshold,
    )
    # Raw engine output (no _accept_segment filter) so WER reflects the model
    # itself, matching the faster-whisper branch.
    text = "".join(seg.text for seg in result.segments)
    return text, time.monotonic() - start


def _transcribe_clip(cfg, audio_path):
    """Transcribe one clip with one config; return (text, proc_seconds)."""
    if cfg.backend == "faster-whisper":
        return _transcribe_faster_whisper(cfg, audio_path)
    if cfg.backend == "mlx-whisper":
        return _transcribe_mlx(cfg, audio_path)
    raise NotImplementedError(
        f"backend '{cfg.backend}' not implemented yet — add it here "
        "(Parakeet: discord-transcript-bot-mni)"
    )


def _trial_order(n_clips, trial):
    """Clip order for one trial: 0 = manifest order; later trials are
    seeded shuffles so a run is reproducible and two configs see the SAME
    orders."""
    order = list(range(n_clips))
    if trial > 0:
        random.Random(trial).shuffle(order)
    return order


def run_config(cfg, clips, *, order_trials=1):
    """Run one config over all clips; return a result dict with corpus WER,
    RTF and per-clip scores.

    With ``order_trials > 1`` the clips are transcribed that many times in
    different orders (see _trial_order). A clip whose normalized hypothesis
    differs between trials is ORDER-SENSITIVE — its decode depends on what
    ran before it, so its contribution to WER is noise
    (discord-transcript-bot-adg). Headline numbers (wer/edits/proc_s/rtf)
    always come from trial 0, the manifest order, so a plain run and the
    first trial of a multi-trial run report identically.
    """
    n = len(clips)
    hyps = [[] for _ in range(n)]  # per clip, one hypothesis per trial
    trial_wers = []
    proc0 = audio_total = 0.0
    scores0 = []
    for trial in range(max(1, order_trials)):
        scores = [None] * n
        for i in _trial_order(n, trial):
            audio_path, reference = clips[i]
            text, proc = _transcribe_clip(cfg, audio_path)
            hyps[i].append(text)
            scores[i] = word_error_rate(reference, text)
            if trial == 0:
                proc0 += proc
                audio_total += wav_duration_seconds(audio_path)
        if trial == 0:
            scores0 = scores
        trial_wers.append(aggregate_wer(scores)["wer"])

    per_clip = []
    for i, (audio_path, _reference) in enumerate(clips):
        distinct = []
        for h in hyps[i]:
            key = normalize_text(h)
            if key not in (normalize_text(d) for d in distinct):
                distinct.append(h.strip())
        per_clip.append(
            {
                "audio": os.path.basename(audio_path),
                "path": audio_path,
                "wer": scores0[i]["wer"],
                "edits": scores0[i]["edits"],
                "ref_words": scores0[i]["ref_words"],
                "hyp": hyps[i][0].strip(),
                "order_sensitive": len(distinct) > 1,
                "hyps": distinct,
            }
        )

    agg = aggregate_wer(scores0)
    return {
        "config": cfg.name,
        "wer": agg["wer"],
        "edits": agg["edits"],
        "ref_words": agg["ref_words"],
        "audio_s": audio_total,
        "proc_s": proc0,
        "rtf": (proc0 / audio_total) if audio_total else 0.0,
        "order_trials": len(trial_wers),
        "wer_min": min(trial_wers),
        "wer_max": max(trial_wers),
        "order_sensitive": [c["path"] for c in per_clip if c["order_sensitive"]],
        "clips": per_clip,
    }


def write_stable_manifest(src_manifest, unstable_paths, out_path):
    """Copy ``src_manifest`` to ``out_path`` minus the clips whose resolved
    audio path is in ``unstable_paths``. Lines are copied verbatim so every
    field a human added during correction survives. Returns (kept, dropped)."""
    base = Path(src_manifest).resolve().parent
    kept = dropped = 0
    unstable = {str(Path(p)) for p in unstable_paths}
    out_lines = []
    for line in Path(src_manifest).read_text().splitlines():
        if not line.strip():
            continue
        audio = str(base / json.loads(line)["audio"])
        if audio in unstable:
            dropped += 1
            continue
        out_lines.append(line)
        kept += 1
    Path(out_path).write_text("\n".join(out_lines) + ("\n" if out_lines else ""))
    return kept, dropped


def load_configs(path):
    raw = json.loads(Path(path).read_text())
    return [Config(**obj) for obj in raw]


def format_table(results):
    multi = any(r.get("order_trials", 1) > 1 for r in results)
    header = f"{'config':<34} {'WER':>8} {'edits/ref':>12} {'RTF':>7} {'proc_s':>8}"
    if multi:
        header += f" {'WER range':>16} {'unstable':>8}"
    lines = [header, "-" * len(header)]
    for r in sorted(results, key=lambda r: r["wer"]):
        wer_pct = f"{r['wer'] * 100:.2f}%"
        ratio = f"{r['edits']}/{r['ref_words']}"
        line = (
            f"{r['config']:<34} {wer_pct:>8} {ratio:>12} "
            f"{r['rtf']:>7.2f} {r['proc_s']:>8.1f}"
        )
        if multi:
            rng = f"{r.get('wer_min', r['wer']) * 100:.2f}-{r.get('wer_max', r['wer']) * 100:.2f}%"
            line += f" {rng:>16} {len(r.get('order_sensitive', [])):>8}"
        lines.append(line)
    return "\n".join(lines)


def format_order_sensitivity(results):
    """Per-config list of order-sensitive clips with their competing
    hypotheses, so the human can decide which to cut from the reference."""
    out = []
    for r in results:
        flagged = [c for c in r.get("clips", []) if c.get("order_sensitive")]
        if not flagged:
            continue
        out.append(
            f"\n{r['config']}: {len(flagged)} order-sensitive clip(s) over "
            f"{r['order_trials']} trials — these decodes depend on what ran "
            "before them; treat their WER as noise:"
        )
        for c in flagged:
            out.append(f"  {c['audio']}")
            for h in c["hyps"]:
                out.append(f"      -> {h!r}")
    if out:
        out.append(
            "\nDelete near-unintelligible clips from the reference rather than "
            "guessing their text; --stable-manifest writes the manifest without them."
        )
    return "\n".join(out)


def _run_isolated(cfg, args):
    """Run ONE config in a fresh interpreter (same manifest/flags, minus
    --isolate) and return its result dict, or None if the child failed —
    e.g. its backend was unavailable. A fresh process means no model
    cache and no GPU numerical state carried over from an earlier config
    (discord-transcript-bot-adg)."""
    with tempfile.TemporaryDirectory(prefix="ab_transcribe_") as tmp:
        cfg_path = Path(tmp) / "config.json"
        out_path = Path(tmp) / "result.json"
        cfg_path.write_text(json.dumps([asdict(cfg)]))
        cmd = [sys.executable, str(Path(__file__).resolve())]
        if args.manifest:
            cmd += ["--manifest", args.manifest]
        else:
            cmd += ["--audio", args.audio, "--reference", args.reference]
        cmd += [
            "--configs",
            str(cfg_path),
            "--json-out",
            str(out_path),
            "--order-trials",
            str(args.order_trials),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if proc.returncode != 0 or not out_path.exists():
            tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-3:]
            for line in tail:
                print(f"    | {line}", flush=True)
            return None
        results = json.loads(out_path.read_text())["results"]
        return results[0] if results else None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--audio", help="single WAV file")
    parser.add_argument("--reference", help="reference transcript .txt for --audio")
    parser.add_argument("--manifest", help="JSONL of {audio, reference|reference_file}")
    parser.add_argument(
        "--configs", help="JSON list of config objects (default set if omitted)"
    )
    parser.add_argument("--json-out", help="write full results JSON here")
    parser.add_argument(
        "--order-trials",
        type=int,
        default=1,
        metavar="N",
        help="transcribe every clip N times in different orders and flag clips "
        "whose decode changes (order-sensitive = noise; default 1)",
    )
    parser.add_argument(
        "--stable-manifest",
        metavar="PATH",
        help="with --manifest and --order-trials>1: write the manifest minus "
        "every order-sensitive clip here",
    )
    parser.add_argument(
        "--isolate",
        action="store_true",
        help="run each config in a fresh subprocess so no config's numbers "
        "depend on which config ran first",
    )
    args = parser.parse_args(argv)

    if not args.manifest and not (args.audio and args.reference):
        parser.error("provide --manifest, or both --audio and --reference")
    if args.order_trials < 1:
        parser.error("--order-trials must be >= 1")
    if args.stable_manifest and not (args.manifest and args.order_trials > 1):
        parser.error("--stable-manifest needs --manifest and --order-trials > 1")

    clips = load_manifest(args)
    configs = load_configs(args.configs) if args.configs else DEFAULT_CONFIGS

    trials_note = f", {args.order_trials} order trials" if args.order_trials > 1 else ""
    iso_note = ", one subprocess per config" if args.isolate else ""
    print(
        f"Scoring {len(configs)} config(s) over {len(clips)} clip(s)"
        f"{trials_note}{iso_note}...\n"
    )
    results = []
    skipped = []
    for cfg in configs:
        print(f"  running: {cfg.name} ...", flush=True)
        if args.isolate:
            result = _run_isolated(cfg, args)
            if result is None:
                print(
                    f"  SKIPPED {cfg.name}: subprocess failed (see above)", flush=True
                )
                skipped.append(cfg.name)
            else:
                results.append(result)
            continue
        try:
            results.append(run_config(cfg, clips, order_trials=args.order_trials))
        except BackendUnavailable as e:
            # e.g. an mlx-whisper config on a host without mlx_whisper. Skip it
            # with a notice rather than aborting the whole run (which would
            # discard every already-completed config's results), so the default
            # set still works on non-Apple-Silicon hosts.
            print(f"  SKIPPED {cfg.name}: backend unavailable ({e})", flush=True)
            skipped.append(cfg.name)

    if not results:
        raise SystemExit(
            "No configs ran — all backends were unavailable. "
            f"Skipped: {', '.join(skipped)}"
        )

    print("\n" + format_table(results))
    if skipped:
        print(f"\nSkipped {len(skipped)} config(s): {', '.join(skipped)}")
    sensitivity = format_order_sensitivity(results)
    if sensitivity:
        print(sensitivity)

    if args.stable_manifest:
        unstable = set()
        for r in results:
            unstable.update(r.get("order_sensitive", []))
        kept, dropped = write_stable_manifest(
            args.manifest, unstable, args.stable_manifest
        )
        print(
            f"\nWrote {args.stable_manifest}: kept {kept} clip(s), dropped "
            f"{dropped} order-sensitive clip(s)"
        )

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
