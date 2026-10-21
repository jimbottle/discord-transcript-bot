"""Which model answers `/ask`, `/recap` and the dashboard: the entry point
every caller uses.

``ASK_PROVIDER``:

- ``local`` (recommended): a model on your own machine through Ollama or
  another OpenAI-compatible server (``src/llm/local.py``). Free per
  question, and transcripts stay on the machine.
- ``anthropic``: Claude through the Anthropic API (``src/llm/claude.py``),
  with ``ANTHROPIC_API_KEY``.

Unset: ``anthropic`` when an Anthropic key is configured, otherwise
``local``, so a fresh checkout is local-first and an existing API setup
keeps working. Each request goes to exactly one provider; a failure is
reported, never retried on the other one.
"""

import logging

from src.llm import claude, local
from src.llm.common import (  # noqa: F401 - the shared types, for callers
    EMPTY_MSG,
    NO_NOTES_MSG,
    ONLY_GM_NOTES_MSG,
    AskBlocked,
    AskFailed,
    AskResult,
    env,
)

logger = logging.getLogger(__name__)

LOCAL = "local"
ANTHROPIC = "anthropic"
_ALIASES = {
    "local": LOCAL,
    "ollama": LOCAL,
    "openai-compatible": LOCAL,
    "anthropic": ANTHROPIC,
    "claude": ANTHROPIC,
}

_warned = set()


def name() -> str:
    raw = env("ASK_PROVIDER").lower()
    if raw:
        chosen = _ALIASES.get(raw)
        if chosen:
            return chosen
        if raw not in _warned:
            _warned.add(raw)
            logger.warning(
                f"ASK_PROVIDER={raw!r} is not 'local' or 'anthropic'; choosing "
                "automatically"
            )
    return ANTHROPIC if claude.api_key() else LOCAL


def configured() -> bool:
    """Whether a question can be sent at all. Local is always configured
    (the server is checked when asked); Anthropic needs its key."""
    return name() == LOCAL or claude.configured()


def not_configured_message() -> str:
    return claude.NO_KEY_MSG


def model() -> str:
    return local.model() if name() == LOCAL else claude.model()


def describe() -> str:
    """One line for logs and the health check."""
    if name() == LOCAL:
        return f"local model {local.model()} at {local.url()}"
    return f"Anthropic API ({claude.model()}, effort {claude.effort()})"


def ask(transcript, question, *, header="", note_dirs=(), max_chars=None):
    backend = local if name() == LOCAL else claude
    return backend.ask(
        transcript, question, header=header, note_dirs=note_dirs, max_chars=max_chars
    )


def ask_lore(question, *, header="", note_dirs=(), include_gm=False):
    backend = local if name() == LOCAL else claude
    return backend.ask_lore(
        question, header=header, note_dirs=note_dirs, include_gm=include_gm
    )
