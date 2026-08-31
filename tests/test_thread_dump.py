"""Sampled thread dump (discord-transcript-bot-309): must rank a spinning
thread above a parked one and never raise."""

import logging
import os
import signal
import threading
import time

from src.utils import thread_dump


def _spin(stop):
    x = 0
    while not stop.is_set():
        x += 1
        y = x * 2
        x = y - x  # several distinct lines so the sampler sees movement


def _park(stop):
    while not stop.is_set():
        time.sleep(0.02)


def test_sampler_ranks_spinning_thread_above_parked_thread():
    stop = threading.Event()
    spinner = threading.Thread(target=_spin, args=(stop,), name="spinner", daemon=True)
    parker = threading.Thread(target=_park, args=(stop,), name="parker", daemon=True)
    spinner.start()
    parker.start()
    try:
        samples = thread_dump.sample_threads(
            seconds=0.3, interval=0.002, exclude_ident=threading.get_ident()
        )
    finally:
        stop.set()
        spinner.join(1)
        parker.join(1)

    assert threading.get_ident() not in samples, "sampling thread excluded"
    spin = samples[spinner.ident]
    park = samples[parker.ident]
    assert spin.name == "spinner" and park.name == "parker"
    assert spin.samples > 10 and park.samples > 10
    assert spin.hot_score > park.hot_score
    assert park.modal_fraction > 0.9, "a sleeping thread sits on its sleep line"
    assert "_park" in park.locations.most_common(1)[0][0]

    text = thread_dump.render(samples, cpu_pct=142.0)
    assert "Process CPU over the sampling window: 142%" in text
    assert text.index("== spinner") < text.index("== parker"), "hottest first"
    assert "last stack:" in text


def test_dump_to_file_writes_ranked_report(tmp_path, caplog):
    # Sample from a worker thread, as the signal handler does, so the main
    # thread (this test) is one of the sampled threads.
    result = {}

    def worker():
        result["path"] = thread_dump.dump_to_file(str(tmp_path), seconds=0.05)

    with caplog.at_level(logging.WARNING, logger="src.utils.thread_dump"):
        t = threading.Thread(target=worker)
        t.start()
        t.join(5)
    path = result.get("path")
    assert path and os.path.exists(path)
    body = open(path, encoding="utf-8").read()
    assert body.startswith("Thread dump ")
    assert "MainThread" in body
    assert any("Thread dump written to" in r.message for r in caplog.records)


def test_dump_to_file_never_raises(tmp_path, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("no frames for you")

    monkeypatch.setattr(thread_dump, "sample_threads", boom)
    assert thread_dump.dump_to_file(str(tmp_path), seconds=0.01) is None


def test_signal_handler_writes_dump_off_the_main_thread(tmp_path):
    previous = signal.getsignal(signal.SIGUSR1)
    try:
        assert thread_dump.install_signal_handler(str(tmp_path)) is True
        os.kill(os.getpid(), signal.SIGUSR1)
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and not list(tmp_path.glob("thread_dump-*")):
            time.sleep(0.05)
        dumps = list(tmp_path.glob("thread_dump-*.txt"))
        assert dumps, "SIGUSR1 must produce a dump file"
        # Wait for the dump thread to finish before restoring the handler.
        for t in threading.enumerate():
            if t.name == "thread-dump":
                t.join(5)
    finally:
        signal.signal(signal.SIGUSR1, previous)


def test_install_signal_handler_off_main_thread_returns_false():
    result = {}

    def run():
        result["ok"] = thread_dump.install_signal_handler()

    t = threading.Thread(target=run)
    t.start()
    t.join()
    assert result["ok"] is False
