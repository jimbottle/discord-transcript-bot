import json
import logging
import os
import shutil
import subprocess
import time

import yaml

logger = logging.getLogger(__name__)

STATUS_FILE = os.path.join(os.getcwd(), ".logs", "health_status.json")


class HealthCheck:
    def __init__(self):
        self.checks = {}  # name -> {ok: bool, message: str, critical: bool}
        self._ollama_process = None

    def _write_status(self, phase: str, current_check: str = None):
        """Write current health status to the status file for the web UI."""
        os.makedirs(os.path.dirname(STATUS_FILE), exist_ok=True)
        payload = {"phase": phase, "checks": dict(self.checks)}
        if current_check:
            payload["current_check"] = current_check
        tmp = STATUS_FILE + ".tmp"
        with open(tmp, "w") as f:
            json.dump(payload, f)
        os.replace(tmp, STATUS_FILE)

    @classmethod
    def clear_status_file(cls):
        """Delete the health status file."""
        try:
            os.remove(STATUS_FILE)
        except FileNotFoundError:
            pass

    def run_all(self, autofix: bool = False, bot=None) -> dict:
        """Run all checks, return {name: {ok, message, critical}}.

        If autofix=True, checks will attempt to fix themselves before
        declaring failure (e.g. start ollama, pull models, create dirs).
        Status file is only written during autofix (startup) runs so that
        slash-command re-checks don't overwrite the startup state.

        If bot is provided (a VoloBot instance), the discord_gateway check
        will verify the bot's websocket connection is alive.
        """
        self.checks = {}
        if autofix:
            self._write_status("initializing", "env_vars")

        self._check_env_vars()
        if autofix:
            self._write_status("initializing", "ffmpeg")

        self._check_ffmpeg()
        if autofix:
            self._write_status("initializing", "opus")

        self._check_opus()
        if autofix:
            self._write_status("initializing", "whisper_model")

        self._check_whisper_model()
        if autofix:
            self._write_status("initializing", "openai_api")

        self._check_openai_api()
        if autofix:
            self._write_status("initializing", "transcripts_dir")

        self._check_transcripts_dir(autofix=autofix)
        if autofix:
            self._write_status("initializing", "log_dirs")

        self._check_log_dirs(autofix=autofix)
        if autofix:
            self._write_status("initializing", "player_map")

        self._check_player_map()
        if autofix:
            self._write_status("initializing", "ollama_server")

        self._check_ollama_server(autofix=autofix)
        if autofix:
            self._write_status("initializing", "ollama_model")

        # Only check model if server is up
        if self.checks.get("ollama_server", {}).get("ok"):
            self._check_ollama_model()
        else:
            # Non-critical: ollama is only used by /ask; bot must still start.
            self._record(
                "ollama_model",
                False,
                "Skipped — ollama server is not running",
                critical=False,
            )

        if autofix:
            self._write_status("initializing", "discord_gateway")

        self._check_discord_gateway(bot=bot)

        if autofix:
            self._write_status("ready" if self.all_ok() else "failed")
        return self.checks

    def all_ok(self) -> bool:
        """True if all critical checks pass."""
        return all(c["ok"] for c in self.checks.values() if c["critical"])

    def summary(self) -> str:
        """Human-readable summary of check results."""
        lines = []
        for name, result in self.checks.items():
            icon = (
                "PASS" if result["ok"] else ("FAIL" if result["critical"] else "WARN")
            )
            lines.append(f"[{icon}] {name}: {result['message']}")
        return "\n".join(lines)

    def failure_summary(self) -> str:
        """Summary of only failed critical checks."""
        lines = []
        for name, result in self.checks.items():
            if result["critical"] and not result["ok"]:
                lines.append(f"- {name}: {result['message']}")
        return "\n".join(lines)

    def _record(self, name: str, ok: bool, message: str, critical: bool = True):
        self.checks[name] = {"ok": ok, "message": message, "critical": critical}

    # ── ffmpeg ────────────────────────────────────────────────────────

    def _check_ffmpeg(self):
        if shutil.which("ffmpeg"):
            self._record("ffmpeg", True, "ffmpeg found on PATH")
        else:
            self._record(
                "ffmpeg", False, "ffmpeg not found on PATH (required for voice audio)"
            )

    # ── opus ──────────────────────────────────────────────────────────

    def _check_opus(self):
        try:
            import discord.opus

            if not discord.opus.is_loaded():
                discord.opus._load_default()
            # If _load_default didn't find it, try known paths (e.g. Homebrew)
            if not discord.opus.is_loaded():
                import glob as _glob

                candidates = [
                    "/opt/homebrew/lib/libopus.0.dylib",
                    "/opt/homebrew/lib/libopus.dylib",
                    "/usr/local/lib/libopus.0.dylib",
                    "/usr/local/lib/libopus.dylib",
                    "/usr/lib/libopus.so.0",
                    "/usr/lib/x86_64-linux-gnu/libopus.so.0",
                ]
                for path in candidates:
                    if os.path.exists(path):
                        try:
                            discord.opus.load_opus(path)
                            if discord.opus.is_loaded():
                                break
                        except Exception:
                            continue
            if discord.opus.is_loaded():
                self._record("opus", True, "libopus loaded")
            else:
                self._record(
                    "opus", False, "libopus not found (required for voice codec)"
                )
        except Exception as e:
            self._record("opus", False, f"libopus failed to load: {e}")

    # ── env_vars ──────────────────────────────────────────────────────

    def _check_env_vars(self):
        missing = []
        if not os.getenv("DISCORD_BOT_TOKEN"):
            missing.append("DISCORD_BOT_TOKEN")
        if os.getenv("TRANSCRIPTION_METHOD") == "openai" and not os.getenv(
            "OPENAI_API_KEY"
        ):
            missing.append("OPENAI_API_KEY")
        if missing:
            self._record("env_vars", False, f"Missing: {', '.join(missing)}")
        else:
            self._record("env_vars", True, "All required env vars set")

    # ── whisper_model ─────────────────────────────────────────────────

    def _check_whisper_model(self):
        try:
            from src.asr.selection import get_backend

            # Resolves + loads the actually-selected backend (MLX on Apple
            # Silicon, else faster-whisper) and decodes a tiny silent clip to
            # prove it can run — better coverage than the old audio_model-only
            # check. healthcheck() iterates the decode so a broken model is
            # caught (roborev #512).
            backend = get_backend()
            backend.healthcheck()
            self._record(
                "whisper_model",
                True,
                f"{backend.name}/{backend.model_id} loaded and responding",
            )
        except Exception as e:
            self._record("whisper_model", False, f"Whisper model failed: {e}")

    # ── openai_api ────────────────────────────────────────────────────

    def _check_openai_api(self):
        if os.getenv("TRANSCRIPTION_METHOD") != "openai":
            self._record(
                "openai_api",
                True,
                "Not using OpenAI transcription (skipped)",
                critical=False,
            )
            return
        try:
            from openai import OpenAI

            client = OpenAI()
            client.models.list()
            self._record("openai_api", True, "OpenAI API reachable")
        except Exception as e:
            self._record("openai_api", False, f"OpenAI API unreachable: {e}")

    # ── transcripts_dir ───────────────────────────────────────────────

    def _check_transcripts_dir(self, autofix: bool = False):
        transcript_dir = os.path.join(os.getcwd(), "transcripts")
        try:
            if autofix:
                os.makedirs(transcript_dir, exist_ok=True)
            if not os.path.isdir(transcript_dir):
                self._record(
                    "transcripts_dir", False, f"{transcript_dir} does not exist"
                )
                return
            test_file = os.path.join(transcript_dir, ".health_check")
            with open(test_file, "w") as f:
                f.write("ok")
            os.remove(test_file)
            self._record("transcripts_dir", True, f"{transcript_dir} is writable")
        except Exception as e:
            self._record("transcripts_dir", False, f"Not writable: {e}")

    # ── log_dirs ──────────────────────────────────────────────────────

    def _check_log_dirs(self, autofix: bool = False):
        dirs = [".logs/transcripts", ".logs/pdfs"]
        missing = []
        for d in dirs:
            if autofix:
                os.makedirs(d, exist_ok=True)
            if not os.path.isdir(d):
                missing.append(d)
        if missing:
            self._record(
                "log_dirs", False, f"Missing directories: {', '.join(missing)}"
            )
        else:
            self._record("log_dirs", True, "All log directories exist")

    # ── player_map ────────────────────────────────────────────────────

    def _check_player_map(self):
        player_map_path = os.getenv("PLAYER_MAP_FILE_PATH")
        if not player_map_path:
            self._record(
                "player_map",
                True,
                "PLAYER_MAP_FILE_PATH not set (optional)",
                critical=False,
            )
            return
        if not os.path.exists(player_map_path):
            self._record(
                "player_map",
                False,
                f"File not found: {player_map_path}",
                critical=False,
            )
            return
        try:
            with open(player_map_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            if data is None:
                data = {}
            if not isinstance(data, dict):
                self._record(
                    "player_map",
                    False,
                    "File is not a valid YAML mapping",
                    critical=False,
                )
                return
            self._record(
                "player_map", True, f"Loaded {len(data)} entries", critical=False
            )
        except Exception as e:
            self._record("player_map", False, f"Invalid YAML: {e}", critical=False)

    # ── ollama_server ─────────────────────────────────────────────────

    def _ollama_is_reachable(self) -> bool:
        try:
            import ollama

            ollama.list()
            return True
        except Exception:
            return False

    def _check_ollama_server(self, autofix: bool = False):
        # Non-critical: only /ask depends on ollama. Voice transcription works
        # without it, so a host without ollama installed must still be able to
        # start the bot.
        if self._ollama_is_reachable():
            self._record(
                "ollama_server", True, "Ollama server reachable", critical=False
            )
            return

        if not autofix:
            self._record(
                "ollama_server", False, "Ollama server not reachable", critical=False
            )
            return

        # Auto-fix: start ollama serve if we haven't already
        if self._ollama_process is None or self._ollama_process.poll() is not None:
            logger.info("Health autofix: starting 'ollama serve'...")
            try:
                self._ollama_process = subprocess.Popen(
                    ["ollama", "serve"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            except FileNotFoundError:
                self._record(
                    "ollama_server",
                    False,
                    "ollama binary not found on PATH (only /ask needs it)",
                    critical=False,
                )
                return
            except Exception as e:
                self._record(
                    "ollama_server",
                    False,
                    f"Failed to start ollama: {e}",
                    critical=False,
                )
                return

        # Poll for readiness (up to ~10s)
        for i in range(20):
            time.sleep(0.5)
            if self._ollama_is_reachable():
                logger.info("Health autofix: ollama server is now reachable.")
                self._record(
                    "ollama_server",
                    True,
                    "Ollama server started via autofix",
                    critical=False,
                )
                return

        self._record(
            "ollama_server",
            False,
            "Started ollama serve but it didn't become reachable in 10s",
            critical=False,
        )

    # ── ollama_model ──────────────────────────────────────────────────

    def _check_ollama_model(self):
        # Shared resolver with main.py's /ask (single source of truth;
        # empty ASK_OLLAMA_MODEL= falls back to the default) so the
        # health check always verifies whatever model /ask will use.
        from src.config.ollama_config import get_ask_model, is_ollama_pullable

        model_name = get_ask_model()
        try:
            import ollama

            models = ollama.list()
            installed = [m.model for m in models.models]
            if any(
                name == model_name or name.endswith("/" + model_name)
                for name in installed
            ):
                self._record(
                    "ollama_model", True, f"{model_name} is available", critical=False
                )
                return

            # Non-critical: /ask is the only feature that uses this model.
            # Voice transcription works without it. We deliberately do NOT
            # auto-pull on startup — these models are multi-GB and it would
            # block on_ready for several minutes.
            #
            # Do not blindly suggest `ollama pull <name>`: a Docker-namespaced
            # name can never be pulled from Ollama's registry, and this check
            # previously sent users to run exactly the command that had just
            # failed for them.
            if is_ollama_pullable(model_name):
                remedy = f"run `ollama pull {model_name}` if /ask needed"
            else:
                remedy = (
                    f"`{model_name}` is a Docker Hub name and CANNOT be pulled "
                    "from Ollama's registry — set ASK_OLLAMA_MODEL to an "
                    "Ollama model (e.g. gemma4:26b)"
                )
            self._record(
                "ollama_model",
                False,
                f"{model_name} not installed ({remedy})",
                critical=False,
            )
        except Exception as e:
            self._record(
                "ollama_model", False, f"Ollama model check failed: {e}", critical=False
            )

    # ── discord_gateway ──────────────────────────────────────────────

    def _check_discord_gateway(self, bot=None):
        if bot is None:
            self._record(
                "discord_gateway",
                True,
                "Skipped — no bot instance (standalone check)",
                critical=False,
            )
            return
        try:
            if not bot.is_ready():
                self._record(
                    "discord_gateway", False, "Bot is not ready (gateway not connected)"
                )
                return
            latency = bot.latency
            if latency == float("inf") or latency > 5.0:
                self._record(
                    "discord_gateway",
                    False,
                    f"Gateway latency too high: {latency:.2f}s",
                )
                return
            self._record(
                "discord_gateway", True, f"Connected (latency: {latency*1000:.0f}ms)"
            )
        except Exception as e:
            self._record("discord_gateway", False, f"Gateway check failed: {e}")
