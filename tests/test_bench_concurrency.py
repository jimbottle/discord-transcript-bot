"""scripts/bench_concurrency.py counts backend errors that transcribe_audio
swallows (roborev #6520)."""

import importlib.util
import logging
from pathlib import Path


def _load():
    path = Path(__file__).resolve().parents[1] / "scripts" / "bench_concurrency.py"
    spec = importlib.util.spec_from_file_location("bench_concurrency", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _Sink:
    def transcribe_audio(self, f):
        logging.getLogger("src.sinks.whisper_sink").error(
            "Error transcribing audio: GPU fell over"
        )
        return ""


def test_swallowed_backend_errors_are_counted():
    bench = _load()
    clips = [{"bytes": b""}, {"bytes": b""}, {"bytes": b""}]
    wall, results, errors = bench.run(_Sink(), clips, 2)
    assert errors == 3
    assert [r["text"] for r in results] == ["", "", ""]
