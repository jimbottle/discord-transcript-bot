"""Single source of truth for the Ollama model used by `/ask`.

Imported by both `main.py` (the `/ask` command) and
`src/bot/health.py` (the startup model check) so the two can never
drift apart. Intentionally side-effect free at import time (no
`load_dotenv`, no logging, no network) so importing it is cheap and
safe from the health module.
"""
import os

DEFAULT_ASK_MODEL = "ai/mistral:latest"


def get_ask_model() -> str:
    """Resolve the model `/ask` should use.

    An unset *or* empty/whitespace-only ``ASK_OLLAMA_MODEL`` falls back
    to :data:`DEFAULT_ASK_MODEL`. ``.env.sample`` uses the
    empty-assignment convention (``FOO=``), so ``ASK_OLLAMA_MODEL=`` is
    a realistic footgun, not a theoretical one — a bare ``os.getenv``
    default would return ``""`` and break ``ollama.chat(model="")``.
    """
    return os.getenv("ASK_OLLAMA_MODEL", "").strip() or DEFAULT_ASK_MODEL
