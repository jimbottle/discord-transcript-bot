"""Sampled all-thread stack dump — a poor man's py-spy that needs no root
(discord-transcript-bot-309).

The bot once pegged a full core while recording a silent room, and nothing
in-process could say which thread. py-spy needs root on macOS, so this
module does the next best thing from inside the process: sample every
thread's innermost frame a few hundred times over a second and rank
threads by how much they MOVE. A thread parked in ``select`` / ``wait`` /
``sleep`` shows the same file:line every sample; a thread spinning through
a loop lands on different lines. The dump lists each thread's top
locations with counts plus its full stack, and the process-wide CPU% over
the sampling window, so a human can tell "which thread" at a glance.

Wire-up: ``install_signal_handler()`` makes ``kill -USR1 <pid>`` write
``.logs/thread_dump-<timestamp>.txt`` and log its path. The handler only
starts a daemon thread, so it never blocks the event loop.
"""

import logging
import os
import signal
import sys
import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from traceback import format_stack
from typing import Dict, Optional

logger = logging.getLogger(__name__)

DEFAULT_SECONDS = 1.0
DEFAULT_INTERVAL = 0.005


@dataclass
class ThreadSample:
    ident: int
    name: str
    daemon: bool
    samples: int = 0
    locations: Counter = field(default_factory=Counter)  # "file:line func" -> n
    last_stack: str = ""

    @property
    def modal_fraction(self) -> float:
        """Share of samples at the single most common location — ~1.0 for a
        thread blocked on one call, lower for one moving through a loop."""
        if not self.samples:
            return 1.0
        return self.locations.most_common(1)[0][1] / self.samples

    @property
    def hot_score(self) -> float:
        """0 = parked on one line the whole window; higher = moving.
        Ranking key, not a CPU measurement (Python has no per-thread CPU
        time on macOS)."""
        return 1.0 - self.modal_fraction


def _location(frame) -> str:
    code = frame.f_code
    return f"{os.path.basename(code.co_filename)}:{frame.f_lineno} {code.co_name}"


def sample_threads(
    seconds: float = DEFAULT_SECONDS,
    interval: float = DEFAULT_INTERVAL,
    exclude_ident: Optional[int] = None,
) -> Dict[int, ThreadSample]:
    """Sample every live thread's innermost frame for ``seconds``.
    ``exclude_ident`` drops the sampling thread itself from the result."""
    samples: Dict[int, ThreadSample] = {}
    deadline = time.monotonic() + seconds
    while True:
        names = {t.ident: t for t in threading.enumerate()}
        for ident, frame in sys._current_frames().items():
            if ident == exclude_ident:
                continue
            entry = samples.get(ident)
            if entry is None:
                t = names.get(ident)
                entry = samples[ident] = ThreadSample(
                    ident=ident,
                    name=getattr(t, "name", f"thread-{ident}"),
                    daemon=bool(getattr(t, "daemon", False)),
                )
            entry.samples += 1
            entry.locations[_location(frame)] += 1
            entry.last_stack = "".join(format_stack(frame))
        if time.monotonic() >= deadline:
            return samples
        time.sleep(interval)


def render(samples: Dict[int, ThreadSample], cpu_pct: Optional[float] = None) -> str:
    """Human-readable dump, hottest thread first."""
    lines = [
        f"Thread dump {datetime.now().isoformat(timespec='seconds')}  pid={os.getpid()}",
    ]
    if cpu_pct is not None:
        lines.append(f"Process CPU over the sampling window: {cpu_pct:.0f}%")
    lines.append(
        "Threads ranked by movement (hot_score 0.00 = parked on one call all "
        "window; a spinning loop scores high and cycles through several lines):"
    )
    for s in sorted(samples.values(), key=lambda s: s.hot_score, reverse=True):
        lines.append("")
        lines.append(
            f"== {s.name}  ident={s.ident}  daemon={s.daemon}  "
            f"samples={s.samples}  hot_score={s.hot_score:.2f}"
        )
        for loc, n in s.locations.most_common(5):
            lines.append(f"   {n:>5}  {loc}")
        lines.append("   last stack:")
        for ln in s.last_stack.rstrip().splitlines():
            lines.append(f"   | {ln}")
    return "\n".join(lines) + "\n"


def dump_to_file(
    directory: str = ".logs", seconds: float = DEFAULT_SECONDS
) -> Optional[str]:
    """Sample, render and write the dump. Returns the file path, or None on
    failure (logged). Safe to call from any thread."""
    try:
        cpu0, wall0 = time.process_time(), time.monotonic()
        samples = sample_threads(seconds, exclude_ident=threading.get_ident())
        wall = time.monotonic() - wall0
        cpu_pct = (time.process_time() - cpu0) / wall * 100 if wall > 0 else None
        os.makedirs(directory, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        path = os.path.join(directory, f"thread_dump-{stamp}.txt")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(render(samples, cpu_pct))
        logger.warning(
            "Thread dump written to %s (process CPU %s%% during the %.1fs sample)",
            path,
            f"{cpu_pct:.0f}" if cpu_pct is not None else "?",
            seconds,
        )
        return path
    except Exception as e:  # noqa: BLE001 - a diagnostic must never take the bot down
        logger.error("Thread dump failed: %s", e)
        return None


def install_signal_handler(directory: str = ".logs", signum=None) -> bool:
    """Make ``kill -<signum> <pid>`` (default SIGUSR1) write a thread dump.
    Returns False where the signal isn't available (Windows) or when not
    on the main thread — never raises."""
    if signum is None:
        signum = getattr(signal, "SIGUSR1", None)
    if signum is None:
        return False

    def _handler(_signum, _frame):
        # Do the sampling off the main thread so the event loop isn't
        # blocked for the sampling window.
        threading.Thread(
            target=dump_to_file, args=(directory,), name="thread-dump", daemon=True
        ).start()

    try:
        signal.signal(signum, _handler)
    except (ValueError, OSError) as e:  # not main thread / unsupported
        logger.debug("thread-dump signal handler not installed: %s", e)
        return False
    logger.info(
        "Thread-dump handler installed: `kill -USR1 %s` writes %s/thread_dump-*.txt",
        os.getpid(),
        directory,
    )
    return True
