"""Startup integration test — verifies the bot subprocess can reach the
'ready' phase end-to-end.

This is the closest automated proxy for "slash commands will actually
respond." Catches the class of bugs where the bot connects to the gateway
but never finishes on_ready (health checks fail, autofix blocks
indefinitely, on_ready raises silently, etc.) — which presents to the
user as "The application did not respond" on every interaction.

Requires a real DISCORD_BOT_TOKEN in the environment. Auto-skips if only
the conftest placeholder is set so unit-only runs (CI without secrets,
new contributor checkout) still pass.
"""

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
HEALTH_STATUS_FILE = PROJECT_ROOT / ".logs" / "health_status.json"
VENV_PYTHON = PROJECT_ROOT / "venv" / "bin" / "python"

# How long to wait for the bot to reach phase=ready. Whisper model load
# can take 10-30s cold; gateway connect adds a few seconds; total budget
# is generous so a slow first load doesn't flake the test.
READY_TIMEOUT = 120
POLL_INTERVAL = 0.5


def _has_real_token():
    token = os.getenv("DISCORD_BOT_TOKEN", "")
    return bool(token) and token != "test-placeholder"


@pytest.mark.integration
def test_bot_subprocess_reaches_ready_phase():
    # Evaluate prerequisites at test run time, not module import — so a
    # post-collection change to env or venv state is observed correctly.
    if not _has_real_token():
        pytest.skip("DISCORD_BOT_TOKEN missing or placeholder — startup test needs a real token")
    if not VENV_PYTHON.exists():
        pytest.skip(f"venv/bin/python not found at {VENV_PYTHON}")

    # Clean any prior health status file so we don't observe stale state
    HEALTH_STATUS_FILE.unlink(missing_ok=True)

    stderr_log = PROJECT_ROOT / ".logs" / "bot_stderr_test.log"
    stderr_log.parent.mkdir(parents=True, exist_ok=True)
    stderr_log.unlink(missing_ok=True)

    def _read_stderr_tail(n=2000):
        try:
            return stderr_log.read_text(errors="replace")[-n:]
        except FileNotFoundError:
            return "(no stderr captured)"

    proc = None
    stderr_fh = open(stderr_log, "w")
    try:
        proc = subprocess.Popen(
            [str(VENV_PYTHON), "main.py"],
            cwd=str(PROJECT_ROOT),
            stdout=subprocess.DEVNULL,
            stderr=stderr_fh,
        )

        deadline = time.time() + READY_TIMEOUT
        final_phase = None
        final_status = None
        premature_exit_message = None

        while time.time() < deadline:
            if proc.poll() is not None:
                stderr_fh.flush()
                premature_exit_message = (
                    f"Bot subprocess exited prematurely (rc={proc.returncode}). "
                    f"Last stderr:\n{_read_stderr_tail()}"
                )
                break

            try:
                final_status = json.loads(HEALTH_STATUS_FILE.read_text())
                final_phase = final_status.get("phase")
            except (FileNotFoundError, json.JSONDecodeError):
                pass

            if final_phase in ("ready", "failed"):
                break
            time.sleep(POLL_INTERVAL)
    finally:
        # Always tear down the subprocess and close the stderr handle
        # before reading the file. SIGINT is what main.py expects.
        if proc is not None and proc.poll() is None:
            proc.send_signal(signal.SIGINT)
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)
        stderr_fh.close()

    if premature_exit_message:
        pytest.fail(premature_exit_message)

    assert final_phase == "ready", (
        f"Bot did not reach phase=ready within {READY_TIMEOUT}s "
        f"(observed phase={final_phase!r}). "
        f"Status: {json.dumps(final_status, indent=2) if final_status else 'never written'}. "
        f"Stderr tail: {_read_stderr_tail(1500)}"
    )

    assert "checks" in final_status
    critical_failures = [
        name for name, info in final_status["checks"].items()
        if info.get("critical") and not info.get("ok")
    ]
    assert not critical_failures, \
        f"Ready but critical checks failing: {critical_failures}"
