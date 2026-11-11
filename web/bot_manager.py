import json
import os
import re
import signal
import subprocess
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VENV_PYTHON = os.path.join(PROJECT_ROOT, "venv", "bin", "python")
HEALTH_STATUS_FILE = os.path.join(PROJECT_ROOT, ".logs", "health_status.json")
BOT_STATE_FILE = os.path.join(PROJECT_ROOT, ".logs", "bot_state.json")


class BotManager:
    def __init__(self):
        self._process = None
        self._stderr_log = None
        self._stderr_fh = None

    def _close_stderr_fh(self):
        if self._stderr_fh is not None:
            try:
                self._stderr_fh.close()
            except OSError:
                pass
            self._stderr_fh = None

    @property
    def running(self):
        return self._process is not None and self._process.poll() is None

    def _python_executable(self):
        if os.path.isfile(VENV_PYTHON):
            return VENV_PYTHON
        return sys.executable

    def _read_health(self):
        """Read the health status file written by the bot process."""
        try:
            with open(HEALTH_STATUS_FILE, "r") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return None

    def _proc_cwd(self, pid):
        """The working directory of `pid`, or None. lsof is on macOS &
        Linux; -Fn makes the cwd line machine-parseable."""
        try:
            out = subprocess.run(
                ["lsof", "-a", "-p", str(pid), "-d", "cwd", "-Fn"],
                capture_output=True,
                text=True,
                timeout=5,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        for line in out.stdout.splitlines():
            if line.startswith("n"):
                return line[1:]
        return None

    def _proc_command(self, pid):
        """The full command line of `pid`, or None."""
        try:
            out = subprocess.run(
                ["ps", "-o", "command=", "-p", str(pid)],
                capture_output=True,
                text=True,
                timeout=5,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return out.stdout.strip() or None

    @staticmethod
    def _runs_project_main(command, cwd):
        """True when `command`, started in `cwd`, runs THIS project's
        main.py.

        An MCP server started from this directory as
        `python /elsewhere/google_workspace_mcp/main.py` has "main.py"
        in its command line and this project as its cwd, and used to be
        reported as the bot (discord-transcript-bot-93b). So the script
        argument itself must resolve to PROJECT_ROOT/main.py.
        """
        if not command or not cwd:
            return False
        ours = os.path.realpath(os.path.join(PROJECT_ROOT, "main.py"))
        # An absolute path survives even when it contains spaces, which
        # the token split below would break apart.
        if re.search(re.escape(ours) + r"(\s|$)", command):
            return True
        for tok in command.split()[1:]:
            if os.path.basename(tok) != "main.py":
                continue
            if os.path.realpath(os.path.join(cwd, tok)) == ours:
                return True
        return False

    def _is_bot(self, pid):
        """`pid` is a live process running this project's main.py."""
        return self._runs_project_main(self._proc_command(pid), self._proc_cwd(pid))

    def _find_bot_pid(self, health=None):
        """PID of a `main.py` bot we did NOT spawn (make start / detached),
        or None.

        The bot records its own pid in the health file, so that pid is
        checked first; it only counts while it is still a process
        running this project's main.py (a crashed run leaves the file
        behind, and the pid can be reused). A health file without a pid
        (written by an older bot) falls back to scanning for one.
        """
        recorded = (health or {}).get("pid")
        if isinstance(recorded, int) and not isinstance(recorded, bool):
            return recorded if self._is_bot(recorded) else None
        try:
            out = subprocess.run(
                ["pgrep", "-f", "main.py"],
                capture_output=True,
                text=True,
                timeout=5,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        for tok in out.stdout.split():
            try:
                pid = int(tok)
            except ValueError:
                continue
            if self._is_bot(pid):
                return pid
        return None

    def _status_from_health(self, health, pid, external):
        if health is None:
            return {
                "status": "starting",
                "pid": pid,
                "checks": {},
                "external": external,
            }
        phase = health.get("phase", "starting")
        combined = {
            "ready": "ready",
            "failed": "unhealthy",
            "initializing": "initializing",
        }.get(phase, "starting")
        result = {
            "status": combined,
            "pid": pid,
            "checks": health.get("checks", {}),
            "external": external,
        }
        if combined == "initializing" and health.get("current_check"):
            result["current_check"] = health["current_check"]
        return result

    def status(self):
        if self._process is None:
            # We didn't spawn it — but it may be running externally
            # (`make start`, a detached run). Require BOTH a health file
            # AND a live process running this project's main.py (see
            # _find_bot_pid). Read health once and reuse it (no
            # double-read / TOCTOU). Skip the process lookups entirely
            # when there's no health file.
            health = self._read_health()
            pid = self._find_bot_pid(health) if health is not None else None
            if pid is not None:
                return self._status_from_health(health, pid, external=True)
            return {"status": "stopped", "pid": None, "external": False}
        rc = self._process.poll()
        if rc is None:
            return self._status_from_health(
                self._read_health(), self._process.pid, external=False
            )
        error = self._read_stderr()
        return {
            "status": "crashed",
            "pid": None,
            "exit_code": rc,
            "error": error,
            "external": False,
        }

    def _read_stderr(self):
        if self._stderr_log and os.path.isfile(self._stderr_log):
            try:
                with open(self._stderr_log, "r") as f:
                    # Return last 2000 chars to keep response reasonable
                    content = f.read()
                    return content[-2000:] if len(content) > 2000 else content
            except OSError:
                return None
        return None

    def start(self):
        if self.running:
            return False
        # Close any leftover stderr handle from a prior run
        self._close_stderr_fh()
        # Clear stale health file from any previous run
        for _stale in (HEALTH_STATUS_FILE, BOT_STATE_FILE):
            try:
                os.remove(_stale)
            except FileNotFoundError:
                pass
        self._stderr_log = os.path.join(PROJECT_ROOT, ".logs", "bot_stderr.log")
        os.makedirs(os.path.dirname(self._stderr_log), exist_ok=True)
        self._stderr_fh = open(self._stderr_log, "a")
        self._process = subprocess.Popen(
            [self._python_executable(), "main.py"],
            cwd=PROJECT_ROOT,
            stdout=subprocess.DEVNULL,
            stderr=self._stderr_fh,
        )
        return True

    def stop(self):
        if not self.running:
            return False
        self._process.send_signal(signal.SIGINT)
        try:
            self._process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self._process.kill()
        self._process = None
        self._close_stderr_fh()
        for _stale in (HEALTH_STATUS_FILE, BOT_STATE_FILE):
            try:
                os.remove(_stale)
            except FileNotFoundError:
                pass
        return True
