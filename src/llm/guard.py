"""The GM check (discord-transcript-bot-zm3): a harness in front of
every /ask-style model call that refuses table-rules violations before an
answer is generated.

Blocked:
- **strategy**: tactical or strategic advice on beating a scenario,
  encounter, puzzle, negotiation or plan ("what's my best approach?").
- **optimization**: min-maxing a build, stats, gear, cyberware or spells.
- **hidden_info**: what the characters have not learned: GM secrets, NPC
  hidden motives, what is behind the door, who the traitor is.
- **prediction**: guessing where the plot is going or what the GM plans.

Allowed: recalling what happened or was said at the table, what the
characters know, summaries, who is who, and how a game rule works.

One small structured-output call (it only classifies the question, a few
hundred tokens) on the same backend that answers: with a local model, the
local model itself (``LOCAL_LLM_MODEL``, so no second model is loaded); with
the Anthropic API, Claude Haiku 5.5. ``GUARD_MODEL`` overrides either. It
runs inside both backends' ``ask`` and ``ask_lore``, so no caller can skip it.
It **fails closed**: if the check itself cannot run, the question is not
answered. The main prompts carry the same rule as a backstop.

``TABLE_GUARD=off`` disables it (for a GM-only deployment); default on.
"""

import json
import logging
import os
from dataclasses import dataclass
from typing import Optional

import anthropic

logger = logging.getLogger(__name__)

DEFAULT_GUARD_MODEL = "claude-haiku-5-5"
CATEGORIES = ("ok", "strategy", "optimization", "hidden_info", "prediction")

GUARD_SYSTEM = (
    "You are the table-rules check for a tabletop RPG assistant that answers "
    "questions about the game's session transcripts and campaign notes. "
    "Decide whether the question may be answered. The GM does not want the "
    "assistant to help players play the game or know more than their "
    "characters. Block a question when it asks for any of:\n"
    "- strategy: tactical or strategic advice for beating or handling a "
    "scenario, encounter, fight, puzzle, negotiation, heist or plan (for "
    "example 'what is my best approach', 'how do we win this', 'what should "
    "we do next');\n"
    "- optimization: min-maxing a character build, stats, skills, gear, "
    "cyberware, spells or loadout;\n"
    "- hidden_info: anything the characters have not learned in play: GM "
    "secrets, an NPC's hidden motives or allegiances, what is behind a door, "
    "who a traitor is, how a mystery resolves;\n"
    "- prediction: guessing where the plot is going or what the GM is "
    "planning.\n"
    "Allow (category ok) questions that recall or summarise what happened or "
    "was said at the table, what the characters already know, who someone or "
    "something is as established in play or the notes, and how a game rule "
    "works in general. Judge the intent, not keywords: 'what did Noah say "
    "about the debate plan' is recall and allowed; 'what's our best plan for "
    "the debate' is strategy and blocked. When unsure, allow recall and block "
    "advice. reason is one short sentence addressed to the player."
)
GM_MODE_NOTE = (
    "\nThe asker is the GM preparing the game, so hidden_info and prediction "
    "do not apply; still block strategy and optimization advice for players."
)

SCHEMA = {
    "type": "object",
    "properties": {
        "allowed": {"type": "boolean"},
        "category": {"type": "string", "enum": list(CATEGORIES)},
        "reason": {"type": "string"},
    },
    "required": ["allowed", "category", "reason"],
    "additionalProperties": False,
}

GUARD_ERROR_MSG = (
    "Couldn't run the table-rules check just now, so I'm holding that "
    "question. Try again in a moment."
)


@dataclass
class Verdict:
    allowed: bool
    category: str
    reason: str
    model: Optional[str] = None
    input_tokens: int = 0
    output_tokens: int = 0


def enabled() -> bool:
    return os.getenv("TABLE_GUARD", "on").strip().lower() not in (
        "0",
        "off",
        "false",
        "no",
    )


def model(backend: str = "anthropic") -> str:
    override = os.getenv("GUARD_MODEL", "").strip()
    if override:
        return override
    if backend == "local":
        from src.llm import local

        return local.model()
    return DEFAULT_GUARD_MODEL


def _client(api_key: str) -> anthropic.Anthropic:
    return anthropic.Anthropic(api_key=api_key, max_retries=2, timeout=30)


def build_request(question: str, *, context: str = "", gm: bool = False) -> dict:
    system = GUARD_SYSTEM + (GM_MODE_NOTE if gm else "")
    content = (f"{context}\n\n" if context else "") + "QUESTION:\n" + question
    return {
        "model": model(),
        "max_tokens": 1024,
        "system": system,
        "messages": [{"role": "user", "content": content}],
        "output_config": {
            "effort": "low",
            "format": {"type": "json_schema", "schema": SCHEMA},
        },
    }


class GuardUnavailable(Exception):
    """The check could not produce a verdict (API error, bad output)."""


def check(
    question: str,
    *,
    api_key: str = "",
    context: str = "",
    gm: bool = False,
    backend: str = "anthropic",
) -> Verdict:
    """Classify ``question`` on ``backend`` ("anthropic" or "local").
    Raises :class:`GuardUnavailable` when no verdict could be produced; the
    caller fails closed on that."""
    if backend == "local":
        from src.llm import local

        system = GUARD_SYSTEM + (GM_MODE_NOTE if gm else "")
        content = (f"{context}\n\n" if context else "") + "QUESTION:\n" + question
        used_model = model("local")
        try:
            text, used_model, tokens_in, tokens_out = local.classify(
                system, content, SCHEMA, model=used_model
            )
        except local.LocalError as e:
            raise GuardUnavailable(str(e)) from e
        return _parse(text, used_model, tokens_in, tokens_out)

    request = build_request(question, context=context, gm=gm)
    try:
        response = _client(api_key).messages.create(**request)
    except anthropic.APIError as e:
        raise GuardUnavailable(str(e)) from e
    if getattr(response, "stop_reason", None) == "refusal":
        # The classifier declining is itself a signal; do not answer.
        return Verdict(
            False, "hidden_info", "That one is off the table.", model=request["model"]
        )
    text = "".join(
        getattr(b, "text", "")
        for b in response.content
        if getattr(b, "type", None) == "text"
    )
    usage = getattr(response, "usage", None)
    return _parse(
        text,
        getattr(response, "model", None) or request["model"],
        getattr(usage, "input_tokens", 0) or 0,
        getattr(usage, "output_tokens", 0) or 0,
    )


def _parse(text: str, used_model, tokens_in: int, tokens_out: int) -> Verdict:
    """A verdict from the classifier's JSON, normalised; GuardUnavailable if
    it isn't one."""
    try:
        data = json.loads(text)
        allowed = bool(data["allowed"])
        category = str(data.get("category") or ("ok" if allowed else "strategy"))
        reason = str(data.get("reason") or "").strip()
    except (ValueError, KeyError, TypeError) as e:
        raise GuardUnavailable(f"unparseable verdict: {text[:200]!r}") from e
    if category not in CATEGORIES:
        category = "ok" if allowed else "strategy"
    if allowed and category != "ok":
        allowed = False  # a named violation wins over a contradictory flag
    if not allowed and category == "ok":
        category = "strategy"  # a block needs a blocking category (roborev #6467)
    return Verdict(
        allowed=allowed,
        category=category,
        reason=reason,
        model=used_model,
        input_tokens=tokens_in,
        output_tokens=tokens_out,
    )


BLOCK_LEADS = {
    "strategy": "The GM wouldn't want me coaching the play on that one.",
    "optimization": "No min-maxing help from me; that's yours to work out at the table.",
    "hidden_info": "That's something your characters haven't learned, so I can't say.",
    "prediction": "I won't guess where the GM is taking the story.",
}
BLOCK_TAIL = (
    "I can help with what happened at the table, who's who, or how a rule works."
)


def block_message(verdict: Verdict) -> str:
    """What the player sees: a short in-voice line per category. The
    classifier's own reason is wordy and repeats the refusal, so it goes to
    the question history (and the GM), not to the player."""
    lead = BLOCK_LEADS.get(verdict.category, BLOCK_LEADS["strategy"])
    return f"{lead} {BLOCK_TAIL}"
