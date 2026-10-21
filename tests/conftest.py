import os
import sys
from unittest.mock import MagicMock

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "web"))

# Load .env if present so the integration test can find a real
# DISCORD_BOT_TOKEN. Falls back to a placeholder so unit-only runs (CI
# without secrets, fresh checkout) still pass.
try:
    from dotenv import load_dotenv

    load_dotenv(dotenv_path=os.path.join(PROJECT_ROOT, ".env"))
except ImportError:
    pass
os.environ.setdefault("DISCORD_BOT_TOKEN", "test-placeholder")


@pytest.fixture(autouse=True)
def _fast_whisper_model(monkeypatch, request):
    """Stub the memoized ASR backend for unit tests so constructing a
    WhisperSink / running HealthCheck._check_whisper_model doesn't actually
    load a model (hundreds of MB, several seconds). Pre-populating the cache
    means selection.get_backend() returns the stub without building anything.
    Integration tests opt out via the `integration` marker — they need the
    real backend to exercise transcription end-to-end."""
    from src.asr import base, selection

    if "integration" in request.keywords:
        selection.reset_backend_cache()
        yield
        selection.reset_backend_cache()
        return

    stub = MagicMock(name="asr_backend_stub")
    stub.name = "stub"
    stub.model_id = "stub"
    stub.transcribe.return_value = base.TranscribeResult(segments=[], info=None)
    stub.healthcheck.return_value = None
    monkeypatch.setattr(selection, "_backend", stub)
    yield


@pytest.fixture(autouse=True)
def _isolate_question_log(monkeypatch, tmp_path):
    """No test may append to the real history/questions.jsonl: /ask tests
    in test_web.py go through the route that records every question."""
    monkeypatch.setenv("QUESTION_LOG_FILE", str(tmp_path / "questions.jsonl"))


@pytest.fixture(autouse=True)
def _no_real_model_server(monkeypatch):
    """No test may talk to a real local model server (this machine may run
    Ollama), and the developer's own ASK_PROVIDER/LOCAL_LLM_* must not leak
    in. Port 9 (discard) refuses at once, so a health check that probes it
    reports 'unreachable' without waiting."""
    for name in ("ASK_PROVIDER", "LOCAL_LLM_MODEL", "LOCAL_LLM_API_KEY", "GUARD_MODEL"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("LOCAL_LLM_URL", "http://127.0.0.1:9/v1")
