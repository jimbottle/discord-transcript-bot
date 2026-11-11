"""Unit tests for BotManager status / external-bot detection.

BotManager() has no side effects at construction. We exercise status()
with _find_bot_pid and _read_health monkeypatched so the tests never
shell out to pgrep or read the real health file.

Regression guard: the dashboard used to report "stopped" for any bot
it didn't spawn itself (`make start`, a detached run), so it was blind
to the bot it's supposed to monitor.
"""

import os

import bot_manager as bm_mod
from bot_manager import BotManager


def _bm(monkeypatch, *, pid, health):
    bm = BotManager()
    monkeypatch.setattr(bm, "_find_bot_pid", lambda health=None: pid)
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


# ── _find_bot_pid: which process is the bot (roborev #797, -93b) ───────

PY = "/usr/bin/python3"
DECOY = "/Users/x/mcp-servers/google_workspace_mcp/main.py --tool-tier extended"


class _FakeRun:
    def __init__(self, stdout):
        self.stdout = stdout


def _procs(monkeypatch, bm, procs, pgrep=""):
    """procs: pid -> (command, cwd). pgrep: what the scan returns."""
    monkeypatch.setattr(bm_mod.subprocess, "run", lambda *a, **k: _FakeRun(pgrep))
    monkeypatch.setattr(bm, "_proc_command", lambda pid: procs.get(pid, (None,))[0])
    monkeypatch.setattr(bm, "_proc_cwd", lambda pid: procs.get(pid, (None, None))[1])


def test_runs_project_main_accepts_the_ways_the_bot_is_started():
    root = bm_mod.PROJECT_ROOT
    ok = bm_mod.BotManager._runs_project_main
    assert ok(f"{PY} main.py", root)
    assert ok(f"{PY} main.py -v", root)
    assert ok(f"{PY} ./main.py", root)
    assert ok(f"{PY} {os.path.join(root, 'main.py')}", "/somewhere/else")


def test_runs_project_main_rejects_other_main_py():
    root = bm_mod.PROJECT_ROOT
    ok = bm_mod.BotManager._runs_project_main
    # The 2026-11-10 case: an MCP server started from this directory.
    assert not ok(f"{PY} {DECOY}", root)
    assert not ok(f"{PY} /a b/Claude Extensions/workspace-mcp/main.py", root)
    assert not ok(f"{PY} main.py", "/somewhere/else")
    assert not ok(f"{PY} {os.path.join(root, 'main.py')}.bak", root)
    assert not ok(None, root)
    assert not ok(f"{PY} main.py", None)


def test_scan_skips_a_decoy_main_py_in_the_project_directory(monkeypatch):
    root = bm_mod.PROJECT_ROOT
    bm = BotManager()
    procs = {
        999: (f"{PY} main.py", "/somewhere/else"),
        44875: (f"{PY} {DECOY}", root),
        49074: (f"{PY} main.py", root),
    }
    _procs(monkeypatch, bm, procs, pgrep="999\n44875\nnope\n49074\n")
    assert bm._find_bot_pid() == 49074
    assert bm._find_bot_pid({"phase": "ready"}) == 49074


def test_scan_finds_nothing_when_only_the_decoy_runs(monkeypatch):
    bm = BotManager()
    _procs(
        monkeypatch,
        bm,
        {44875: (f"{PY} {DECOY}", bm_mod.PROJECT_ROOT)},
        pgrep="44875\n",
    )
    assert bm._find_bot_pid() is None


def test_recorded_pid_wins_over_the_scan(monkeypatch):
    root = bm_mod.PROJECT_ROOT
    bm = BotManager()
    procs = {44875: (f"{PY} main.py", root), 49074: (f"{PY} main.py", root)}
    _procs(monkeypatch, bm, procs, pgrep="44875\n49074\n")
    assert bm._find_bot_pid({"phase": "ready", "pid": 49074}) == 49074


def test_recorded_pid_that_died_is_not_replaced_by_a_scan(monkeypatch):
    # The bot crashed and left its health file; nothing else may stand
    # in for it, least of all the decoy.
    root = bm_mod.PROJECT_ROOT
    bm = BotManager()
    _procs(monkeypatch, bm, {44875: (f"{PY} {DECOY}", root)}, pgrep="44875\n")
    assert bm._find_bot_pid({"phase": "ready", "pid": 49074}) is None


def test_recorded_pid_reused_by_another_program_is_not_the_bot(monkeypatch):
    bm = BotManager()
    _procs(monkeypatch, bm, {49074: ("/usr/bin/vim notes.txt", "/tmp")})
    assert bm._find_bot_pid({"phase": "ready", "pid": 49074}) is None


def test_crashed_bot_with_stale_health_and_decoy_reads_stopped(monkeypatch):
    bm = BotManager()
    _procs(
        monkeypatch,
        bm,
        {44875: (f"{PY} {DECOY}", bm_mod.PROJECT_ROOT)},
        pgrep="44875\n",
    )
    monkeypatch.setattr(bm, "_read_health", lambda: {"phase": "ready", "pid": 49074})
    assert bm.status() == {"status": "stopped", "pid": None, "external": False}


def test_find_bot_pid_swallows_subprocess_error(monkeypatch):
    bm = BotManager()

    def boom(*a, **k):
        raise OSError("pgrep not found")

    monkeypatch.setattr(bm_mod.subprocess, "run", boom)
    assert bm._find_bot_pid() is None


def test_real_decoy_process_is_not_the_bot(tmp_path):
    """End to end with real ps/lsof: a live `python <elsewhere>/main.py`
    whose cwd is this project is not the bot."""
    import subprocess
    import sys
    import time

    decoy = tmp_path / "main.py"
    decoy.write_text("import time\ntime.sleep(30)\n")
    proc = subprocess.Popen([sys.executable, str(decoy)], cwd=bm_mod.PROJECT_ROOT)
    try:
        bm = BotManager()
        deadline = time.time() + 5
        while bm._proc_cwd(proc.pid) is None and time.time() < deadline:
            time.sleep(0.05)
        assert os.path.realpath(bm._proc_cwd(proc.pid)) == os.path.realpath(
            bm_mod.PROJECT_ROOT
        )
        assert "main.py" in bm._proc_command(proc.pid)
        assert bm._is_bot(proc.pid) is False
        assert bm._find_bot_pid({"phase": "ready", "pid": proc.pid}) is None
    finally:
        proc.kill()
        proc.wait()
