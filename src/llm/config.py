"""Single source of truth for `/ask` provider credentials and model ids.

Imported by the chain, by ``main.py`` and by ``src/bot/health.py`` so the
startup check always reports on the exact providers the command will
use. Intentionally side-effect free at import time (no ``load_dotenv``,
no network, no client construction) so importing it is cheap and safe
from the health module.

The two cloud providers have **separate base URLs and separate model
ids** — an OpenRouter slug like ``vendor/model`` means nothing to
Cerebras — so each tier carries its own model rather than sharing one
``MODEL`` variable.
"""

import os

DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_CEREBRAS_BASE_URL = "https://api.cerebras.ai/v1"

# Validated against the real NL grounding task in the sibling
# raylytics/louisville-open-data-expenditure-bot repo (3/3 at ~7s, the
# best of the four models benchmarked there on 2026-08-18) in its
# `:free` form. This is the paid slug of the same model: 1M context,
# which matters because a full session transcript is the prompt, and
# $0.085/M in — a long D&D transcript costs a fraction of a cent to ask
# about. Re-run that comparison before switching rather than trusting a
# model card.
DEFAULT_OPENROUTER_MODEL = "nvidia/nemotron-3-super-120b-a12b"

# The model the expenditure bot runs in production on Cerebras.
DEFAULT_CEREBRAS_MODEL = "gpt-oss-120b"


def _env(name: str) -> str:
    """Environment lookup that treats whitespace-only as unset.

    ``.env.sample`` uses the empty-assignment convention (``FOO=``), so
    a blank ``OPENROUTER_API_KEY=`` is a realistic footgun rather than a
    theoretical one — a bare ``os.getenv`` default would hand an empty
    string to the client and turn a missing key into a 401.
    """
    return os.getenv(name, "").strip()


def openrouter_key() -> str:
    return _env("OPENROUTER_API_KEY")


def cerebras_key() -> str:
    """The Cerebras key, preferring the paid one.

    ``CEREBRAS_API_KEY`` is checked as a courtesy for hosts that still
    have an old free-tier key set; Cerebras retired its always-free tier
    in July 2026, so in practice the paid key is the one that answers.
    """
    return _env("CEREBRAS_PAID_API_KEY") or _env("CEREBRAS_API_KEY")


def openrouter_model() -> str:
    return _env("OPENROUTER_MODEL") or DEFAULT_OPENROUTER_MODEL


def cerebras_model() -> str:
    return _env("CEREBRAS_MODEL") or DEFAULT_CEREBRAS_MODEL


def openrouter_base_url() -> str:
    return _env("OPENROUTER_BASE_URL") or DEFAULT_OPENROUTER_BASE_URL


def cerebras_base_url() -> str:
    return _env("CEREBRAS_BASE_URL") or DEFAULT_CEREBRAS_BASE_URL


def ollama_enabled() -> bool:
    """Whether the local last-resort tier may be used.

    On by default: the bot already health-checks Ollama and the model is
    usually warm, so it is the difference between `/ask` working and not
    when the house wifi is down mid-session. Set ``ASK_DISABLE_OLLAMA=1``
    to make `/ask` cloud-only.
    """
    return _env("ASK_DISABLE_OLLAMA").lower() not in ("1", "true", "yes", "on")


def configured_cloud_providers() -> list:
    """Names of the cloud tiers that have a key, in call order.

    Used by the health check to report what `/ask` can actually reach
    without constructing a client or spending a token.
    """
    names = []
    if openrouter_key():
        names.append("openrouter")
    if cerebras_key():
        names.append("cerebras")
    return names
