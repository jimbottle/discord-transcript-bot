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
"""

import logging
import os
from dataclasses import dataclass
from typing import Optional

import anthropic

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-opus-5-5"
# Reasoning depth. Grounded Q&A over a transcript is not a hard problem;
# `medium` is the model's own default. Override with ASK_EFFORT.
DEFAULT_EFFORT = "medium"
# Includes the model's thinking tokens. Discord shows ~1900 characters of
# the visible answer anyway, so this is headroom for thinking, not output.
MAX_ANSWER_TOKENS = 8192
TIMEOUT_S = 120

SYSTEM_PROMPT = (
    "You are an assistant that answers questions about a voice chat "
    "transcript of a D&D game. Be concise and direct. Only reference what's "
    "actually in the transcript. If the answer isn't in the transcript, say so."
)

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


@dataclass
class AskResult:
    answer: str
    model: str
    input_tokens: int = 0
    cached_tokens: int = 0
    output_tokens: int = 0


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


def build_request(transcript: str, question: str) -> dict:
    """The exact ``messages.create`` arguments for one question. Pure, so the
    request shape is unit-testable without a client."""
    return {
        "model": model(),
        "max_tokens": MAX_ANSWER_TOKENS,
        "system": [
            {"type": "text", "text": SYSTEM_PROMPT},
            {
                "type": "text",
                "text": "TRANSCRIPT:\n" + transcript,
                # Cached so follow-up questions about the same session re-read
                # the transcript at the cache rate instead of paying for it
                # every time.
                "cache_control": {"type": "ephemeral"},
            },
        ],
        "messages": [{"role": "user", "content": question}],
        "output_config": {"effort": effort()},
    }


def ask(transcript: str, question: str) -> AskResult:
    """Answer ``question`` from ``transcript``. Blocking. Raises
    :class:`AskFailed` with a Discord-ready message on any failure."""
    if not configured():
        raise AskFailed(NO_KEY_MSG)
    request = build_request(transcript, question)
    try:
        response = _client().messages.create(**request)
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

    stop_reason = getattr(response, "stop_reason", None)
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
    usage = getattr(response, "usage", None)
    return AskResult(
        answer=answer,
        model=getattr(response, "model", None) or request["model"],
        input_tokens=getattr(usage, "input_tokens", 0) or 0,
        cached_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
        output_tokens=getattr(usage, "output_tokens", 0) or 0,
    )
