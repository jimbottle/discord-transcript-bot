import json
import os
import signal
import subprocess
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VENV_PYTHON = os.path.join(PROJECT_ROOT, "venv", "bin", "python")
HEALTH_STATUS_FILE = os.path.join(PROJECT_ROOT, ".logs", "health_status.json")


class BotManager:
    def __init__(self):
        self._process = None
        self._stderr_log = None

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

    def status(self):
        if self._process is None:
            return {"status": "stopped", "pid": None}
        rc = self._process.poll()
        if rc is None:
            health = self._read_health()
            if health is None:
                return {
                    "status": "starting",
                    "pid": self._process.pid,
                    "checks": {},
                }
            phase = health.get("phase", "starting")
            if phase == "ready":
                combined = "ready"
            elif phase == "failed":
                combined = "unhealthy"
            elif phase == "initializing":
                combined = "initializing"
            else:
                combined = "starting"
            result = {
                "status": combined,
                "pid": self._process.pid,
                "checks": health.get("checks", {}),
            }
            if combined == "initializing" and health.get("current_check"):
                result["current_check"] = health["current_check"]
            return result
        error = self._read_stderr()
        return {"status": "crashed", "pid": None, "exit_code": rc, "error": error}

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
        # Clear stale health file from any previous run
        try:
            os.remove(HEALTH_STATUS_FILE)
        except FileNotFoundError:
            pass
        self._stderr_log = os.path.join(PROJECT_ROOT, ".logs", "bot_stderr.log")
        os.makedirs(os.path.dirname(self._stderr_log), exist_ok=True)
        stderr_file = open(self._stderr_log, "a")
        self._process = subprocess.Popen(
            [self._python_executable(), "main.py"],
            cwd=PROJECT_ROOT,
            stdout=subprocess.DEVNULL,
            stderr=stderr_file,
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
        try:
            os.remove(HEALTH_STATUS_FILE)
        except FileNotFoundError:
            pass
        return True
