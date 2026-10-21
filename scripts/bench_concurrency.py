#!/usr/bin/env python3
"""How many transcriptions should run at once? (discord-transcript-bot-4uk)

WhisperSink hands every finished utterance to a ThreadPoolExecutor. On
Apple Silicon all of those workers share one Metal GPU, and mlx_whisper
says nothing about thread safety; on faster-whisper (CPU) each call already
uses every core. This replays real captured clips through the production
path (``WhisperSink.transcribe_audio``: the selected backend plus the
hallucination filter) at several worker counts and reports:

- throughput (clips/s and audio-seconds per wall-second),
- per-clip latency from submit to result (p50/p95/max) for a burst,
  i.e. how far the ordered commit falls behind when everyone talks at once,
- how many texts differ from the 1-worker run, against a control: a second
  1-worker run, since marginal audio is order/state dependent anyway
  (discord-transcript-bot-adg). Differences well above the control mean
  concurrent decoding is changing results.
- errors raised by the backend.

USAGE
    python scripts/bench_concurrency.py --capture captures/2026-10-20_20-53-37
    python scripts/bench_concurrency.py --capture DIR --clips 60 --workers 1,2,4
"""

import argparse
import io
import json
import os
import statistics
import sys
import tempfile
import time
import wave
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["CAPTURE_SESSION_AUDIO"] = "0"  # never write a capture from a bench

from unittest.mock import MagicMock  # noqa: E402

from src.sinks.whisper_sink import WhisperSink  # noqa: E402


def pick_clips(capture_dir: Path, n: int):
    """``n`` clips with non-empty draft text, spread evenly over the session
    (deterministic, so runs are comparable)."""
    manifest = capture_dir / "manifest.draft.jsonl"
    rows = []
    with open(manifest, encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if (row.get("reference") or "").strip() and (
                capture_dir / row["audio"]
            ).exists():
                rows.append(row)
    if len(rows) <= n:
        return rows
    step = len(rows) / n
    return [rows[int(i * step)] for i in range(n)]


def wav_seconds(data: bytes) -> float:
    with wave.open(io.BytesIO(data), "rb") as w:
        return w.getnframes() / float(w.getframerate())


def run(sink, clips, workers):
    """Submit every clip at once; return per-clip (text, latency, error)."""
    results = [None] * len(clips)

    def job(i):
        t = time.perf_counter()
        try:
            text = sink.transcribe_audio(io.BytesIO(clips[i]["bytes"]))
            err = None
        except Exception as e:  # noqa: BLE001 - counted, not fatal
            text, err = None, f"{type(e).__name__}: {e}"
        return i, text, time.perf_counter() - t, err

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(job, i) for i in range(len(clips))]
        for fut in futures:
            i, text, _work, err = fut.result()
            # Latency as the sink sees it: from the burst's submit time.
            results[i] = {"text": text, "done": time.perf_counter() - t0, "err": err}
    wall = time.perf_counter() - t0
    return wall, results


def summarize(label, wall, results, audio_s, baseline):
    lat = sorted(r["done"] for r in results)
    q = statistics.quantiles(lat, n=20) if len(lat) > 1 else lat * 19
    diffs = (
        sum(1 for r, b in zip(results, baseline) if r["text"] != b["text"])
        if baseline
        else 0
    )
    return {
        "run": label,
        "wall_s": round(wall, 2),
        "clips_per_s": round(len(results) / wall, 2),
        "audio_x_realtime": round(audio_s / wall, 1),
        "latency_p50_s": round(statistics.median(lat), 2),
        "latency_p95_s": round(q[18], 2),
        "latency_max_s": round(lat[-1], 2),
        "text_diffs_vs_1_worker": diffs,
        "errors": sum(1 for r in results if r["err"]),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--capture", required=True, type=Path)
    ap.add_argument("--clips", type=int, default=96)
    ap.add_argument("--workers", default="1,2,4,8")
    ap.add_argument("--json-out", type=Path)
    args = ap.parse_args()

    clips = pick_clips(args.capture, args.clips)
    if not clips:
        print(f"No usable clips in {args.capture}")
        return 1
    for c in clips:
        c["bytes"] = (args.capture / c["audio"]).read_bytes()
    audio_s = sum(wav_seconds(c["bytes"]) for c in clips)

    cwd = os.getcwd()
    os.chdir(tempfile.mkdtemp(prefix="bench_concurrency_"))
    try:
        sink = WhisperSink(
            transcript_queue=MagicMock(), loop=MagicMock(), transcriber_type="local"
        )
        backend = sink._backend
        print(
            f"{len(clips)} clips, {audio_s:.0f}s of speech, backend "
            f"{backend.name}/{backend.model_id}"
        )
        sink.transcribe_audio(io.BytesIO(clips[0]["bytes"]))  # load + warm

        rows = []
        wall, baseline = run(sink, clips, 1)
        rows.append(summarize("1 worker", wall, baseline, audio_s, None))
        wall, control = run(sink, clips, 1)
        rows.append(summarize("1 worker (control)", wall, control, audio_s, baseline))
        for w in [int(x) for x in args.workers.split(",") if x.strip()]:
            if w == 1:
                continue
            wall, res = run(sink, clips, w)
            rows.append(summarize(f"{w} workers", wall, res, audio_s, baseline))
        sink.close()
    finally:
        os.chdir(cwd)

    cols = list(rows[0])
    print("\n" + "  ".join(f"{c:>22}" if i else f"{c:<20}" for i, c in enumerate(cols)))
    for r in rows:
        print(
            "  ".join(
                f"{r[c]!s:>22}" if i else f"{r[c]!s:<20}" for i, c in enumerate(cols)
            )
        )
    if args.json_out:
        args.json_out.write_text(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
