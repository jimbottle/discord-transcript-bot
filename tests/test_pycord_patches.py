"""Guarded Pycord runtime patches (discord-transcript-bot-309).

The keep-alive thread on the pinned fix/voice-rec-2 branch retries a
failed UDP send with no delay — thousands of iterations a second — and
logs it only at DEBUG. The patch must (1) bound the retry rate, (2) keep
upstream semantics otherwise, (3) refuse to install if upstream changes."""

import logging
import time
from unittest.mock import MagicMock

import pytest

from src.bot import pycord_patches

reader = pytest.importorskip("discord.voice.receive.reader")


@pytest.fixture
def pristine_keepalive():
    """Run each test against the UNPATCHED upstream class and restore it."""
    cls = reader.UDPKeepAlive
    original_run = getattr(cls, "_volo_original_run", None) or cls.run
    cls.run = original_run
    for attr in ("_volo_backoff_patched", "_volo_original_run"):
        if attr in cls.__dict__:
            delattr(cls, attr)
    yield cls
    cls.run = original_run
    for attr in ("_volo_backoff_patched", "_volo_original_run"):
        if attr in cls.__dict__:
            delattr(cls, attr)


def _failing_client():
    client = MagicMock()
    client.wait_until_connected.return_value = True
    client.is_connected.return_value = True
    client._connection.socket.sendto.side_effect = OSError(
        49, "Can't assign requested address"
    )
    return client


def _iterations(client):
    return client._connection.socket.sendto.call_count


def _wait_for(predicate, timeout=5.0):
    """Poll instead of asserting after a fixed sleep: under load (a busy CI
    box, preflight next to a build) a thread can go unscheduled for longer
    than the sleep, which made these tests flaky (discord-transcript-bot-pzo)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return predicate()


def test_patch_applies_and_is_idempotent(pristine_keepalive):
    cls = pristine_keepalive
    assert pycord_patches.patch_udp_keepalive_backoff() is True
    assert cls.run is pycord_patches._keepalive_run
    assert cls._volo_original_run is not cls.run
    assert pycord_patches.patch_udp_keepalive_backoff() is True
    assert pycord_patches.apply_all() == {"udp_keepalive_backoff": True}


def test_unpatched_loop_spins_on_persistent_send_failure(pristine_keepalive):
    """The defect being worked around: characterise it so a future Pycord
    bump that fixes it makes this test (and the patch) removable."""
    client = _failing_client()
    ka = pristine_keepalive(client)
    t0 = time.monotonic()
    ka.start()
    spun = _wait_for(lambda: _iterations(client) > 200)
    elapsed = time.monotonic() - t0
    ka.stop()
    ka.join(2)
    # 200 sends with delay=5000 between successes; only a no-delay retry
    # loop gets there, however slowly the machine schedules it.
    assert spun and elapsed < 5, "upstream retries with no delay"


def test_patched_loop_backs_off_and_logs_visibly(
    pristine_keepalive, monkeypatch, caplog
):
    monkeypatch.setattr(pycord_patches, "KEEPALIVE_FAILURE_BACKOFF_S", 0.05)
    # A starved run can reach the 60th failure and log again (roborev #6519).
    monkeypatch.setattr(pycord_patches, "_LOG_EVERY", 10**6)
    assert pycord_patches.patch_udp_keepalive_backoff()
    client = _failing_client()
    ka = pristine_keepalive(client)
    with caplog.at_level(logging.WARNING, logger="src.bot.pycord_patches"):
        t0 = time.monotonic()
        ka.start()
        assert _wait_for(lambda: _iterations(client) >= 2), "it retries"
        time.sleep(0.2)
        ka.stop()
        ka.join(2)
        elapsed = time.monotonic() - t0
    assert not ka.is_alive(), "stop() must interrupt the backoff"
    # At most one send per 0.05s backoff window (plus the first), however
    # long the run took; unpatched it would be thousands.
    assert _iterations(client) <= elapsed / 0.05 + 2, "≈ one retry per window"
    warnings = [r for r in caplog.records if "keep-alive send failed" in r.message]
    assert len(warnings) == 1, "first failure logged at WARNING, then throttled"
    assert "OSError" in warnings[0].message


def test_patched_loop_keeps_upstream_success_path_but_stop_is_prompt(
    pristine_keepalive, caplog
):
    assert pycord_patches.patch_udp_keepalive_backoff()
    client = MagicMock()
    client.wait_until_connected.return_value = True
    client.is_connected.return_value = True
    ka = pristine_keepalive(client)
    ka.start()
    assert _wait_for(lambda: ka.counter == 1), "one send"
    time.sleep(0.05)
    assert _iterations(client) == 1, "one send, then sleep for `delay`"
    assert ka.counter == 1
    t0 = time.monotonic()
    ka.stop()
    ka.join(2)
    assert (
        not ka.is_alive() and time.monotonic() - t0 < 1.0
    ), "upstream time.sleep(delay) would park the thread for delay seconds"


def test_patched_loop_exits_when_connection_is_gone(pristine_keepalive, monkeypatch):
    monkeypatch.setattr(pycord_patches, "KEEPALIVE_FAILURE_BACKOFF_S", 0.05)
    assert pycord_patches.patch_udp_keepalive_backoff()
    client = _failing_client()
    client.is_connected.return_value = False
    ka = pristine_keepalive(client)
    ka.start()
    ka.join(2)
    assert not ka.is_alive(), "upstream semantics: break once disconnected"
    assert _iterations(client) == 1


def test_patch_refuses_when_upstream_shape_changes(
    pristine_keepalive, monkeypatch, caplog
):
    cls = pristine_keepalive
    monkeypatch.setattr(
        pycord_patches.inspect, "getsource", lambda obj: "def run(self):\n    pass\n"
    )
    with caplog.at_level(logging.WARNING, logger="src.bot.pycord_patches"):
        assert pycord_patches.patch_udp_keepalive_backoff() is False
    assert cls.run is not pycord_patches._keepalive_run
    assert any("NOT applied" in r.message for r in caplog.records)
    assert pycord_patches.apply_all() == {"udp_keepalive_backoff": False}


def test_apply_all_swallows_patch_exceptions(monkeypatch):
    def boom():
        raise RuntimeError("bad patch")

    monkeypatch.setattr(pycord_patches, "patch_udp_keepalive_backoff", boom)
    assert pycord_patches.apply_all() == {"udp_keepalive_backoff": False}
