"""Unit tests for BotManager status / external-bot detection.

BotManager() has no side effects at construction. We exercise status()
with _find_bot_pid and _read_health monkeypatched so the tests never
shell out to pgrep or read the real health file.

Regression guard: the dashboard used to report "stopped" for any bot
it didn't spawn itself (`make start`, a detached run), so it was blind
to the bot it's supposed to monitor.
"""

from bot_manager import BotManager


def _bm(monkeypatch, *, pid, health):
    bm = BotManager()
    monkeypatch.setattr(bm, "_find_bot_pid", lambda: pid)
    monkeypatch.setattr(bm, "_read_health", lambda: health)
    return bm


def test_not_spawned_no_external_process_is_stopped(monkeypatch):
    bm = _bm(monkeypatch, pid=None, health={"phase": "ready"})
    assert bm.status() == {"status": "stopped", "pid": None, "external": False}


def test_stale_health_without_process_is_stopped(monkeypatch):
    # health file present but no live process -> still stopped (don't
    # let a crashed run's leftover file read as running).
    bm = _bm(monkeypatch, pid=None, health={"phase": "ready"})
    assert bm.status()["status"] == "stopped"


def test_process_without_health_is_stopped(monkeypatch):
    bm = _bm(monkeypatch, pid=4321, health=None)
    assert bm.status() == {"status": "stopped", "pid": None, "external": False}


def test_external_ready_detected(monkeypatch):
    bm = _bm(
        monkeypatch,
        pid=4321,
        health={"phase": "ready", "checks": {"x": {"ok": True}}},
    )
    s = bm.status()
    assert s["status"] == "ready"
    assert s["pid"] == 4321
    assert s["external"] is True
    assert s["checks"] == {"x": {"ok": True}}


def test_external_failed_maps_to_unhealthy(monkeypatch):
    bm = _bm(monkeypatch, pid=99, health={"phase": "failed"})
    s = bm.status()
    assert s["status"] == "unhealthy"
    assert s["external"] is True


def test_external_initializing_carries_current_check(monkeypatch):
    bm = _bm(
        monkeypatch,
        pid=7,
        health={"phase": "initializing", "current_check": "whisper_model"},
    )
    s = bm.status()
    assert s["status"] == "initializing"
    assert s["current_check"] == "whisper_model"
    assert s["external"] is True


def test_status_from_health_unknown_phase_is_starting():
    bm = BotManager()
    assert (
        bm._status_from_health({"phase": "weird"}, 1, external=False)["status"]
        == "starting"
    )


def test_status_from_health_missing_file_is_starting():
    bm = BotManager()
    out = bm._status_from_health(None, 1, external=True)
    assert out == {"status": "starting", "pid": 1, "checks": {}, "external": True}


# ── _find_bot_pid: pgrep parsing + cwd identity check (roborev #797) ───


class _FakeRun:
    def __init__(self, stdout):
        self.stdout = stdout


def test_find_bot_pid_returns_first_pid_whose_cwd_is_project(monkeypatch):
    import bot_manager as bm_mod

    bm = BotManager()
    monkeypatch.setattr(
        bm_mod.subprocess, "run", lambda *a, **k: _FakeRun("999\n12345\nnope\n")
    )
    # 999 is some other main.py elsewhere; 12345 is ours.
    cwds = {999: "/somewhere/else", 12345: bm_mod.PROJECT_ROOT}
    monkeypatch.setattr(bm, "_proc_cwd", lambda pid: cwds.get(pid))
    assert bm._find_bot_pid() == 12345


def test_find_bot_pid_none_when_no_cwd_matches_project(monkeypatch):
    import bot_manager as bm_mod

    bm = BotManager()
    monkeypatch.setattr(bm_mod.subprocess, "run", lambda *a, **k: _FakeRun("111\n"))
    monkeypatch.setattr(bm, "_proc_cwd", lambda pid: "/not/the/project")
    assert bm._find_bot_pid() is None


def test_find_bot_pid_swallows_subprocess_error(monkeypatch):
    import bot_manager as bm_mod

    bm = BotManager()

    def boom(*a, **k):
        raise OSError("pgrep not found")

    monkeypatch.setattr(bm_mod.subprocess, "run", boom)
    assert bm._find_bot_pid() is None
