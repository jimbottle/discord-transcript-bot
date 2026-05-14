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
    """Replace the faster-whisper module-level audio_model with a stub for
    unit tests so HealthCheck._check_whisper_model doesn't actually load
    large-v3 (hundreds of MB, several seconds). Integration tests opt out
    by adding the `integration` marker — they need the real model only if
    they exercise transcription end-to-end."""
    if "integration" in request.keywords:
        yield
        return

    from src.sinks import whisper_sink as ws_mod

    stub = MagicMock(name="audio_model_stub")
    stub.transcribe.return_value = (iter([]), MagicMock(language="en"))
    monkeypatch.setattr(ws_mod, "audio_model", stub)
    yield
