"""Tests for WhisperSink file-handle lifecycle.

These cover the roborev #514 fixes: lazy file open, race-safe close, and
no empty file created when a sink is constructed but never used.
"""

import os
import threading
from unittest.mock import MagicMock

from src.sinks.whisper_sink import WhisperSink


def _make_sink(tmp_path, monkeypatch):
    """Construct a WhisperSink with cwd pointed at a temp dir so the
    transcripts/ side effect lands there."""
    monkeypatch.chdir(tmp_path)
    sink = WhisperSink(
        transcript_queue=MagicMock(),
        loop=MagicMock(),
        data_length=50000,
        max_speakers=10,
        transcriber_type="local",
    )
    return sink


def test_constructed_sink_does_not_create_file(tmp_path, monkeypatch):
    sink = _make_sink(tmp_path, monkeypatch)
    assert not os.path.exists(sink.session_file), \
        "session file should not exist until first write"
    sink.close()


def test_lazy_open_creates_file_on_first_write(tmp_path, monkeypatch):
    sink = _make_sink(tmp_path, monkeypatch)
    fh = sink._get_session_fh()
    assert fh is not None
    fh.write("hello\n")
    fh.flush()
    assert os.path.exists(sink.session_file)
    sink.close()


def test_get_session_fh_returns_none_after_close(tmp_path, monkeypatch):
    sink = _make_sink(tmp_path, monkeypatch)
    sink._get_session_fh()  # open it
    sink.close()
    assert sink._get_session_fh() is None, \
        "after close, lazy-open must refuse to re-open the file"


def test_close_is_idempotent(tmp_path, monkeypatch):
    sink = _make_sink(tmp_path, monkeypatch)
    sink._get_session_fh()
    sink.close()
    # Second close must not raise. running is already False, fh is None.
    sink.close()


def test_concurrent_lazy_open_only_opens_once(tmp_path, monkeypatch):
    """Multiple executor workers may race to first-write. The lock must
    ensure exactly one file handle is opened."""
    sink = _make_sink(tmp_path, monkeypatch)
    handles = []

    def grab():
        handles.append(sink._get_session_fh())

    threads = [threading.Thread(target=grab) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # All threads got the same handle (the one stored on the sink).
    assert all(h is sink._session_fh for h in handles)
    assert len({id(h) for h in handles}) == 1
    sink.close()
