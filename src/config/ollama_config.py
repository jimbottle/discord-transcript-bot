"""Single source of truth for the Ollama model used by `/ask`.

Imported by both `main.py` (the `/ask` command) and
`src/bot/health.py` (the startup model check) so the two can never
drift apart. Intentionally side-effect free at import time (no
`load_dotenv`, no logging, no network) so importing it is cheap and
safe from the health module.
"""

import os

# The previous default, "ai/mistral:latest", was a DOCKER HUB model name
# (Docker Model Runner's `ai/` namespace) sitting in an Ollama config. Ollama's
# registry 404s it — `ollama pull ai/mistral:latest` fails with
# "manifest unknown" — so /ask could never work out of the box. Confirmed live
# on 2026-08-11: /ask returned "model 'ai/mistral:latest' not found (404)".
#
# gemma4:26b is the winner of the bake-off in the sibling local-models repo
# (dev/discord-ask-model): 6/6 human-judged correct vs mistral's 3/6 at
# comparable latency (sub-600ms median). Mistral failed in exactly the ways
# the /ask grounding prompt exists to prevent — it misattributed a death save
# to the wrong player and conflated separate threads into a hallucinated
# summary. Any replacement must be a name Ollama can actually pull; see
# health.py's _check_ollama_model, which now flags the `ai/` trap explicitly.
DEFAULT_ASK_MODEL = "gemma4:26b"

# Docker Model Runner's registry namespace. Names under it are not servable by
# Ollama's default registry, so telling a user to `ollama pull` one sends them
# in a circle — the bug this constant exists to prevent recurring.
DOCKER_HUB_NAMESPACE = "ai/"


def is_ollama_pullable(model: str) -> bool:
    """Whether ``ollama pull <model>`` could plausibly succeed.

    Narrow on purpose: it only rejects the one namespace we have verified
    Ollama's registry cannot serve. Ollama does support user-namespaced names
    (``someuser/model``, ``hf.co/org/model``), so a blanket "contains a slash"
    test would wrongly reject valid models.
    """
    return not (model or "").strip().lower().startswith(DOCKER_HUB_NAMESPACE)


def get_ask_model() -> str:
    """Resolve the model `/ask` should use.

    An unset *or* empty/whitespace-only ``ASK_OLLAMA_MODEL`` falls back
    to :data:`DEFAULT_ASK_MODEL`. ``.env.sample`` uses the
    empty-assignment convention (``FOO=``), so ``ASK_OLLAMA_MODEL=`` is
    a realistic footgun, not a theoretical one — a bare ``os.getenv``
    default would return ``""`` and break ``ollama.chat(model="")``.
    """
    return os.getenv("ASK_OLLAMA_MODEL", "").strip() or DEFAULT_ASK_MODEL
