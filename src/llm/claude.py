"""`/ask`, answered by Claude through the Anthropic API.

One provider, one call. This replaced a three-tier chain (OpenRouter ->
Cerebras -> local Ollama) whose entire bulk was failover between
providers; a single first-party API with the SDK's built-in retries does
not need any of it (discord-transcript-bot-gh7).

Everything here is blocking. `/ask` is an async command, so the caller
MUST run :func:`ask` off the event loop (``asyncio.to_thread``): a call
can take tens of seconds and blocking the loop that long stalls voice
receive and the gateway heartbeat, not just the command.

The transcript goes in a cached system block, so repeated questions about
the same session within the cache window are mostly cache reads.

Campaign background (``library/``: game, group and campaign notes) is retrieved on demand: the
request carries only a one-line-per-note index and a tool; the model pulls
a note's body only when the question needs it (discord-transcript-bot-kda).
"""

import logging
import os
from dataclasses import dataclass, field
from typing import Optional

import anthropic

from src.llm import campaign, guard

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-opus-5-5"
# Reasoning depth. Grounded Q&A over a transcript is not a hard problem;
# `medium` is the model's own default. Override with ASK_EFFORT.
DEFAULT_EFFORT = "medium"
# Includes the model's thinking tokens. Discord shows ~1900 characters of
# the visible answer anyway, so this is headroom for thinking, not output.
MAX_ANSWER_TOKENS = 8192
TIMEOUT_S = 120

TABLE_RULES = (
    " Table rules from the GM: do not give tactical or strategic advice on "
    "beating a scenario, do not help optimise (min-max) characters, and do "
    "not reveal or speculate about anything the characters have not learned "
    "or where the plot is going. If asked, decline in one sentence and offer "
    "to recall what happened instead."
)

SYSTEM_PROMPT = (
    "You are an assistant that answers questions about a voice chat "
    "transcript of a tabletop RPG session (the SESSION line, when present, names the game and campaign; follow its STYLE line if there is one). Be concise and direct. Answer from "
    "the transcript; if the answer isn't there, say so. When a CAMPAIGN NOTES "
    "INDEX is provided you may read campaign notes with the read_campaign_notes "
    "tool, but only when the question needs background the transcript lacks, "
    "and say when an answer comes from the notes rather than the transcript."
    + TABLE_RULES
)
# Tool rounds per question. One is the norm (the model asks for the notes it
# needs in a single call); the cap keeps a confused model from spending money.
MAX_TOOL_ROUNDS = 3

NO_KEY_MSG = "/ask isn't configured: ANTHROPIC_API_KEY is not set on the bot host."
BAD_KEY_MSG = "Anthropic rejected the API key. Check ANTHROPIC_API_KEY on the bot host."
RATE_LIMIT_MSG = "Anthropic is rate-limiting us right now. Try again in a minute."
BILLING_MSG = "The Anthropic account is out of credit. Top it up and try again."
UNREACHABLE_MSG = "Couldn't reach Anthropic (network or timeout). Try again shortly."
SERVER_ERROR_MSG = "Anthropic is having trouble right now. Try again shortly."
REFUSAL_MSG = "Claude declined to answer that one."
EMPTY_MSG = "The model returned no answer. Try rephrasing the question."
TRUNCATED_MSG = (
    "The model ran out of room before answering (it hit the token limit "
    "while reasoning). Try a narrower question, or lower ASK_EFFORT on the "
    "bot host."
)
TRUNCATION_NOTE = "\n\n_(answer cut off at the token limit)_"


def _env(name: str) -> str:
    """Environment lookup that treats whitespace-only as unset (``.env.sample``
    uses the ``FOO=`` convention, so a blank key is a realistic footgun)."""
    return os.getenv(name, "").strip()


def api_key() -> str:
    return _env("ANTHROPIC_API_KEY")


def model() -> str:
    return _env("ASK_MODEL") or DEFAULT_MODEL


def effort() -> str:
    return _env("ASK_EFFORT") or DEFAULT_EFFORT


def configured() -> bool:
    """Whether `/ask` can run at all. Key presence only: no test call, which
    would spend money on every bot start."""
    return bool(api_key())


class AskFailed(Exception):
    """`/ask` could not produce an answer. ``user_message`` is safe to post
    to Discord as-is; the original error is chained for the log."""

    def __init__(self, user_message: str, cause: Optional[BaseException] = None):
        super().__init__(user_message)
        self.user_message = user_message
        self.__cause__ = cause


class AskBlocked(Exception):
    """The GM check refused the question before any answer was generated.
    Not an AskFailed, so callers record it as ``blocked``, not a failure.
    ``user_message`` is safe to post as-is."""

    def __init__(self, user_message: str, verdict):
        super().__init__(user_message)
        self.user_message = user_message
        self.verdict = verdict


def _gate(question: str, *, context: str = "", gm: bool = False):
    """Run the GM check; raise AskBlocked, or AskFailed when it can't run
    (fail closed). Returns the verdict (None when TABLE_GUARD is off)."""
    if not guard.enabled():
        return None
    try:
        verdict = guard.check(question, api_key=api_key(), context=context, gm=gm)
    except guard.GuardUnavailable as e:
        logger.error(f"GM check unavailable, refusing to answer: {e}")
        raise AskFailed(guard.GUARD_ERROR_MSG, e) from e
    if not verdict.allowed:
        logger.info(
            f"GM check blocked a question ({verdict.category}): {verdict.reason}"
        )
        raise AskBlocked(guard.block_message(verdict), verdict)
    return verdict


@dataclass
class AskResult:
    answer: str
    model: str
    input_tokens: int = 0
    cached_tokens: int = 0
    output_tokens: int = 0
    # Campaign notes the model pulled in (ids), in order; [] when it answered
    # from the transcript alone. Token counts above are summed over rounds.
    notes_used: list = field(default_factory=list)
    rounds: int = 1
    # The GM check that let the question through (None when disabled).
    guard: Optional[object] = None


def _api_error_message(error: anthropic.APIStatusError) -> str:
    """The API's own human-readable error text. The SDK's ``e.message`` is
    ``"Error code: 400 - {<whole parsed body>}"`` — a dict dump, not
    something to post to a channel — so dig the message out of the body
    and fall back to a fixed string."""
    body = getattr(error, "body", None)
    if isinstance(body, dict):
        inner = body.get("error")
        if isinstance(inner, dict) and inner.get("message"):
            return str(inner["message"])
        if body.get("message"):
            return str(body["message"])
    return f"HTTP {getattr(error, 'status_code', '?')}"


def _client() -> anthropic.Anthropic:
    # Built per call: cheap, and it picks up a key set after startup.
    return anthropic.Anthropic(api_key=api_key(), max_retries=2, timeout=TIMEOUT_S)


def build_request(
    transcript: str, question: str, index: str = "", header: str = ""
) -> dict:
    """The exact ``messages.create`` arguments for one question. Pure, so the
    request shape is unit-testable without a client.

    With a campaign ``index`` the request gains a third system block and the
    ``read_campaign_notes`` tool; without one it is the plain two-block shape.
    The transcript keeps its own cache breakpoint either way, so editing a
    note (which changes the index) does not evict the cached transcript.
    ``header`` is the one-line session context (game, campaign, group,
    date) and is placed just before the transcript in the same cached block.
    """
    system = [
        {"type": "text", "text": SYSTEM_PROMPT},
        {
            "type": "text",
            "text": (header + "\n\n" if header else "") + "TRANSCRIPT:\n" + transcript,
            # Cached so follow-up questions about the same session re-read
            # the transcript at the cache rate instead of paying for it
            # every time.
            "cache_control": {"type": "ephemeral"},
        },
    ]
    request = {
        "model": model(),
        "max_tokens": MAX_ANSWER_TOKENS,
        "system": system,
        "messages": [{"role": "user", "content": question}],
        "output_config": {"effort": effort()},
    }
    if index:
        system.append(
            {"type": "text", "text": index, "cache_control": {"type": "ephemeral"}}
        )
        request["tools"] = [campaign.TOOL]
        # Opus 5.5 rejects forced tool_choice; "auto" plus the tool
        # description is what makes retrieval conditional on the question.
        request["tool_choice"] = {"type": "auto"}
    return request


def _create(request: dict):
    """One API call with every SDK error mapped to an :class:`AskFailed`."""
    try:
        return _client().messages.create(**request)
    except anthropic.AuthenticationError as e:
        raise AskFailed(BAD_KEY_MSG, e) from e
    except anthropic.PermissionDeniedError as e:
        raise AskFailed(BAD_KEY_MSG, e) from e
    except anthropic.RateLimitError as e:
        raise AskFailed(RATE_LIMIT_MSG, e) from e
    except anthropic.BadRequestError as e:
        detail = _api_error_message(e)
        lowered = detail.lower()
        if "credit" in lowered or "billing" in lowered:
            raise AskFailed(BILLING_MSG, e) from e
        raise AskFailed(f"Anthropic rejected the request: {detail}", e) from e
    except anthropic.APIStatusError as e:
        if e.status_code >= 500:
            raise AskFailed(SERVER_ERROR_MSG, e) from e
        raise AskFailed(f"Anthropic returned HTTP {e.status_code}.", e) from e
    except anthropic.APIConnectionError as e:  # includes APITimeoutError
        raise AskFailed(UNREACHABLE_MSG, e) from e


def _tool_results(response, notes, used: list) -> list:
    """Run every ``read_campaign_notes`` call in ``response`` and return the
    matching ``tool_result`` blocks (all of them in one user message, as the
    API expects for parallel calls)."""
    results = []
    for block in response.content:
        if getattr(block, "type", None) != "tool_use":
            continue
        raw = getattr(block, "input", None) or {}
        ids = raw.get("ids") if isinstance(raw, dict) else None
        if block.name != campaign.TOOL_NAME or not isinstance(ids, list):
            text, is_error = f"Unknown tool call: {block.name}", True
        else:
            text, is_error = campaign.read_notes(ids, notes)
            used.extend(i for i in ids if i in {n.id for n in notes})
        result = {"type": "tool_result", "tool_use_id": block.id, "content": text}
        if is_error:
            result["is_error"] = True
        results.append(result)
    return results


def ask(transcript: str, question: str, *, header: str = "", note_dirs=()) -> AskResult:
    """Answer ``question`` from ``transcript``. Blocking. Raises
    :class:`AskFailed` with a Discord-ready message on any failure.

    ``header`` and ``note_dirs`` come from the session's tag
    (src/sessions.py, src/library.py); without them the question is
    answered from the transcript alone."""
    if not configured():
        raise AskFailed(NO_KEY_MSG)
    verdict = _gate(question, context=header.split("\n")[0] if header else "")
    notes = campaign.load_notes(note_dirs)
    request = build_request(
        transcript, question, campaign.index_text(notes), header=header
    )
    result = _run(request, notes)
    result.guard = verdict
    return result


LORE_SYSTEM_PROMPT = (
    "You answer questions about a tabletop RPG campaign for its players and "
    "GM, using the campaign's reference notes. The CAMPAIGN line names the "
    "game, campaign and group; follow its STYLE line if there is one. Read "
    "the notes you need with the read_campaign_notes tool (request every "
    "relevant note in one call) and answer only from them. If the notes "
    "don't cover the question, say so plainly rather than inventing "
    "campaign facts; you may add general knowledge of the game's published "
    "rules or setting if you label it as such. Be concise and direct." + TABLE_RULES
)
NO_NOTES_MSG = (
    "This campaign has no reference notes yet, so there is nothing to answer from."
)


def build_lore_request(question: str, index: str, header: str = "") -> dict:
    """The ``messages.create`` arguments for a notes-only (Lore page)
    question. No transcript: the header and the note index are the whole
    cached context, and the tool is the only way to the note bodies."""
    return {
        "model": model(),
        "max_tokens": MAX_ANSWER_TOKENS,
        "system": [
            {"type": "text", "text": LORE_SYSTEM_PROMPT},
            {
                "type": "text",
                "text": (header + "\n\n" if header else "") + index,
                "cache_control": {"type": "ephemeral"},
            },
        ],
        "messages": [{"role": "user", "content": question}],
        "output_config": {"effort": effort()},
        "tools": [campaign.TOOL],
        "tool_choice": {"type": "auto"},
    }


def ask_lore(
    question: str, *, header: str = "", note_dirs=(), include_gm: bool = False
) -> AskResult:
    """Answer a campaign question from reference notes alone (dashboard
    Lore page; deliberately not wired to any Discord command). Blocking.
    ``include_gm`` admits ``audience: gm`` notes for this one question."""
    if not configured():
        raise AskFailed(NO_KEY_MSG)
    notes = campaign.load_notes(note_dirs, include_gm=include_gm)
    if not notes:
        raise AskFailed(NO_NOTES_MSG)
    verdict = _gate(
        question, context=header.split("\n")[0] if header else "", gm=include_gm
    )
    result = _run(
        build_lore_request(question, campaign.index_text(notes), header), notes
    )
    result.guard = verdict
    return result


def _run(request: dict, notes: list) -> AskResult:
    """The call/tool loop shared by :func:`ask` and :func:`ask_lore`."""
    totals = {"input_tokens": 0, "cache_read_input_tokens": 0, "output_tokens": 0}
    used: list = []
    rounds = 0
    while True:
        rounds += 1
        response = _create(request)
        usage = getattr(response, "usage", None)
        for key in totals:
            totals[key] += getattr(usage, key, 0) or 0
        stop_reason = getattr(response, "stop_reason", None)
        if stop_reason != "tool_use":
            break
        if rounds >= MAX_TOOL_ROUNDS:
            logger.warning(
                f"/ask still calling tools after {rounds} rounds; answering "
                "without the extra notes"
            )
            # Resend with the tool withheld so the model has to answer from
            # what it already has (the content so far, notes included).
            request["messages"] = request["messages"] + [
                {"role": "assistant", "content": response.content},
                {
                    "role": "user",
                    "content": _tool_results(response, notes, used)
                    + [{"type": "text", "text": "Answer now with what you have."}],
                },
            ]
            request["tool_choice"] = {"type": "none"}
            rounds += 1
            response = _create(request)
            usage = getattr(response, "usage", None)
            for key in totals:
                totals[key] += getattr(usage, key, 0) or 0
            stop_reason = getattr(response, "stop_reason", None)
            break
        # Hand the assistant turn back verbatim (thinking blocks included:
        # the API needs them unchanged) followed by the tool results.
        request["messages"] = request["messages"] + [
            {"role": "assistant", "content": response.content},
            {"role": "user", "content": _tool_results(response, notes, used)},
        ]

    if stop_reason == "refusal":
        raise AskFailed(REFUSAL_MSG)
    answer = "".join(
        getattr(block, "text", "")
        for block in response.content
        if getattr(block, "type", None) == "text"
    ).strip()
    if stop_reason == "max_tokens":
        # The budget covers thinking too, so this usually means the model
        # reasoned past MAX_ANSWER_TOKENS. Say so rather than posting a
        # silently clipped answer or a misleading "try rephrasing".
        logger.warning(
            f"/ask hit max_tokens={MAX_ANSWER_TOKENS} on {request['model']} "
            f"(effort {effort()}); answer {'truncated' if answer else 'empty'}"
        )
        if not answer:
            raise AskFailed(TRUNCATED_MSG)
        answer += TRUNCATION_NOTE
    if not answer:
        raise AskFailed(EMPTY_MSG)
    seen = set()
    notes_used = [i for i in used if not (i in seen or seen.add(i))]
    return AskResult(
        answer=answer,
        model=getattr(response, "model", None) or request["model"],
        input_tokens=totals["input_tokens"],
        cached_tokens=totals["cache_read_input_tokens"],
        output_tokens=totals["output_tokens"],
        notes_used=notes_used,
        rounds=rounds,
    )
