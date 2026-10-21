"""The local backend for `/ask` (``ASK_PROVIDER=local``, the recommended
setup): a model on your own machine, through any OpenAI-compatible server.

Ollama is the default (``http://localhost:11434/v1``); LM Studio, llama.cpp's
``llama-server`` and vLLM speak the same API, so pointing ``LOCAL_LLM_URL``
at them works too. No API key, no per-question cost, and the transcript never
leaves the machine.

Measured on an Apple Silicon Mac with ``gemma4:26b`` under Ollama 0.40
(2026-10-21): the first question about a session reads the whole transcript
(about 4 minutes for 45k tokens); later questions reuse the server's cache
of that prefix and take seconds. So the transcript is the stable start of
the prompt and everything that varies (the notes index, the length limit,
the question) comes after it, and a live session that grows at the end only
costs the new lines. The GM check is a separate small call and does not
evict that cache.

Context length is a server setting, not a request one: Ollama ignores a
per-request ``num_ctx`` on this endpoint. If the server's window is smaller
than the transcript, it drops the start of the prompt without an error, so
:func:`_run` compares the reported prompt size with the text it sent and
says so in the answer (raise ``OLLAMA_CONTEXT_LENGTH``; see README).

Everything here is blocking; callers run it off the event loop, as with the
Anthropic backend.
"""

import json
import logging
import re

import openai

from src.llm import campaign, guard
from src.llm.common import (
    EMPTY_MSG,
    LORE_SYSTEM_PROMPT,
    MAX_TOOL_ROUNDS,
    NO_NOTES_MSG,
    ONLY_GM_NOTES_MSG,
    SYSTEM_PROMPT,
    TRUNCATION_NOTE,
    AskBlocked,
    AskFailed,
    AskResult,
    dedupe,
    env,
    length_instruction,
)

logger = logging.getLogger(__name__)

DEFAULT_URL = "http://localhost:11434/v1"
# The model this backend was tested with (tools, JSON output, 256k context).
# Smaller tool-capable models work too, with weaker answers; see README.
DEFAULT_MODEL = "gemma4:26b"
# Generous: a first question over a long session is minutes of prompt
# reading on a laptop, and Discord keeps a deferred reply open for 15.
DEFAULT_TIMEOUT_S = 840
MAX_ANSWER_TOKENS = 8192
# How much of the session the local model reads. Measured 2026-10-21 on
# Ollama 0.40 / gemma4:26b / Apple Silicon: ~2.9 characters per token once
# Discord ids are stripped, prompt reading ~135-190 tokens/s, so 100k
# characters (~35k tokens) is ~4 minutes for a first question. The whole
# 2026-10-20 session (~68k tokens without ids, 123k with) ran past the
# timeout and then crashed Ollama's Metal backend at 80k tokens. 0 = no cap.
DEFAULT_MAX_TRANSCRIPT_CHARS = 100_000
# Fewer characters per token than this would be unusual, so a reported
# prompt under half of len(text)/CHARS_PER_TOKEN means the server cut it.
CHARS_PER_TOKEN = 4
CONTEXT_NOTE = (
    "\n\n_(the local model read only part of the transcript: its context "
    "window is smaller than this session. Raise it on the model server, for "
    "Ollama `OLLAMA_CONTEXT_LENGTH`; see README.)_"
)

_THINK = re.compile(r"<think>.*?</think>\s*", re.DOTALL)

TRIM_MARK = (
    "[The start of this session is omitted: the local model reads only the "
    "most recent part. Say so if the question needs earlier events.]\n"
)


def server_failed_msg() -> str:
    return (
        "The local model server failed while answering (on a long session "
        "this is usually the GPU running out of memory; its log says why). "
        f"Restart the model (`ollama stop {model()}`) and try again; a smaller "
        "LOCAL_LLM_MAX_TRANSCRIPT_CHARS or a smaller model avoids it."
    )


def trimmed_note(kept_lines: int, total_lines: int) -> str:
    return (
        f"\n\n_(local model: answered from the most recent {kept_lines} of "
        f"{total_lines} transcript lines. LOCAL_LLM_MAX_TRANSCRIPT_CHARS sets "
        "how much it reads.)_"
    )


def url() -> str:
    return env("LOCAL_LLM_URL") or DEFAULT_URL


def model() -> str:
    return env("LOCAL_LLM_MODEL") or DEFAULT_MODEL


def max_transcript_chars() -> int:
    raw = env("LOCAL_LLM_MAX_TRANSCRIPT_CHARS")
    try:
        n = int(raw) if raw else DEFAULT_MAX_TRANSCRIPT_CHARS
    except ValueError:
        return DEFAULT_MAX_TRANSCRIPT_CHARS
    return max(n, 0)


def fit_transcript(transcript: str, budget: int):
    """``(text, kept_lines, total_lines)``: the most recent part of
    ``transcript`` that fits ``budget`` characters, cut at a line boundary.

    The cut moves in steps of a quarter of the budget rather than line by
    line, so while a live session grows the kept text keeps the same start
    for a while and stays a prefix the server can reuse from its cache. A
    cut that slid with every new line would change the prefix each time and
    make every question re-read everything."""
    lines = transcript.splitlines(keepends=True)
    total = len(lines)
    if budget <= 0 or len(transcript) <= budget:
        return transcript, total, total
    step = max(budget // 4, 1)
    over = len(transcript) - budget
    drop = -(-over // step) * step  # round up to the next step
    cut, seen = 0, 0
    while cut < total - 1 and seen < drop:  # always keep the newest line
        seen += len(lines[cut])
        cut += 1
    kept = lines[cut:]
    return TRIM_MARK + "".join(kept), len(kept), total


def timeout_s() -> float:
    try:
        return float(env("LOCAL_LLM_TIMEOUT_S") or DEFAULT_TIMEOUT_S)
    except ValueError:
        return DEFAULT_TIMEOUT_S


def unreachable_msg() -> str:
    return (
        f"Couldn't reach the local model server at {url()}. Is Ollama (or "
        "your LOCAL_LLM_URL server) running on the bot host?"
    )


def missing_model_msg() -> str:
    return (
        f"The local model server has no model named `{model()}`. Run "
        f"`ollama pull {model()}` on the bot host, or set LOCAL_LLM_MODEL."
    )


def timeout_msg() -> str:
    return (
        f"The local model took longer than {timeout_s():.0f}s. The first "
        "question about a long session is the slow one; try again, or use a "
        "smaller model."
    )


TRUNCATED_MSG = (
    "The local model ran out of room before answering (it hit the token "
    "limit). Try a narrower question."
)


class LocalError(Exception):
    """A request to the local server failed; ``failure`` is the Discord-ready
    :class:`AskFailed` for it."""

    def __init__(self, failure: AskFailed):
        super().__init__(failure.user_message)
        self.failure = failure


def _client() -> openai.OpenAI:
    # Servers that want a key (vLLM, a proxied LM Studio) read LOCAL_LLM_API_KEY;
    # Ollama ignores it, but the SDK requires a non-empty value.
    return openai.OpenAI(
        base_url=url(),
        api_key=env("LOCAL_LLM_API_KEY") or "local",
        timeout=timeout_s(),
        max_retries=0,  # a retry would redo minutes of prompt reading
    )


def _failure(e: Exception) -> AskFailed:
    if isinstance(e, openai.APITimeoutError):
        return AskFailed(timeout_msg(), e)
    if isinstance(e, openai.APIConnectionError):
        return AskFailed(unreachable_msg(), e)
    if isinstance(e, openai.NotFoundError):
        return AskFailed(missing_model_msg(), e)
    if isinstance(e, openai.APIStatusError):
        detail = _status_detail(e)
        if "not found" in detail.lower() and "model" in detail.lower():
            return AskFailed(missing_model_msg(), e)
        return AskFailed(
            f"The local model server returned HTTP {e.status_code}: {detail}", e
        )
    return AskFailed(f"The local model request failed: {e}", e)


def _status_detail(e) -> str:
    body = getattr(e, "body", None)
    if isinstance(body, dict):
        inner = body.get("error")
        if isinstance(inner, dict) and inner.get("message"):
            return str(inner["message"])
        if isinstance(inner, str):
            return inner
        if body.get("message"):
            return str(body["message"])
    return str(getattr(e, "message", "") or e)[:200]


def _check_not_broken(response) -> None:
    """Ollama answers HTTP 200 with no content, no finish reason and no
    tokens once its GPU backend has failed (seen 2026-10-21: a Metal
    command-buffer error at 80k tokens, after which every request came back
    empty). That is a server failure, not an empty answer."""
    choices = getattr(response, "choices", None) or []
    choice = choices[0] if choices else None
    message = getattr(choice, "message", None)
    usage = getattr(response, "usage", None)
    if choice is None or (
        getattr(choice, "finish_reason", None) is None
        and not (getattr(message, "content", None) or "").strip()
        and not getattr(message, "tool_calls", None)
        and not (getattr(usage, "completion_tokens", 0) or 0)
    ):
        raise LocalError(AskFailed(server_failed_msg()))


def _create(**kwargs):
    try:
        return _client().chat.completions.create(**kwargs)
    except openai.OpenAIError as e:
        raise LocalError(_failure(e)) from e


def status() -> dict:
    """For the health check: is the server up and does it have the model?
    One cheap local GET; never raises."""
    info = {"url": url(), "model": model(), "reachable": False, "has_model": False}
    try:
        client = openai.OpenAI(
            base_url=url(),
            api_key=env("LOCAL_LLM_API_KEY") or "local",
            timeout=3,
            max_retries=0,  # a startup probe; the SDK's backoff would stall it
        )
        ids = [m.id for m in client.models.list()]
    except Exception as e:  # noqa: BLE001 - reported, not raised
        info["error"] = str(e)[:200]
        return info
    info["reachable"] = True
    wanted = model()
    # Ollama lists "name:tag"; a bare name means ":latest".
    info["has_model"] = wanted in ids or (
        ":" not in wanted and f"{wanted}:latest" in ids
    )
    return info


def classify(system: str, content: str, schema: dict, *, model: str):
    """One JSON-schema call for the GM check. Returns ``(text, model,
    input_tokens, output_tokens)``; raises :class:`LocalError`."""
    response = _create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": content},
        ],
        max_tokens=1024,
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "verdict", "schema": schema, "strict": True},
        },
    )
    _check_not_broken(response)
    usage = getattr(response, "usage", None)
    return (
        _text(response.choices[0].message),
        getattr(response, "model", None) or model,
        getattr(usage, "prompt_tokens", 0) or 0,
        getattr(usage, "completion_tokens", 0) or 0,
    )


def _gate(question: str, *, context: str = "", gm: bool = False):
    """The GM check on the local model; fails closed like the Anthropic one."""
    if not guard.enabled():
        return None
    try:
        verdict = guard.check(question, context=context, gm=gm, backend="local")
    except guard.GuardUnavailable as e:
        logger.error(f"GM check unavailable, refusing to answer: {e}")
        cause = e.__cause__
        if isinstance(cause, LocalError):
            # Server down or model missing: say that, it won't fix itself.
            raise cause.failure from e
        raise AskFailed(guard.GUARD_ERROR_MSG, e) from e
    if not verdict.allowed:
        logger.info(
            f"GM check blocked a question ({verdict.category}): {verdict.reason}"
        )
        raise AskBlocked(guard.block_message(verdict), verdict)
    return verdict


def _tool():
    return {
        "type": "function",
        "function": {
            "name": campaign.TOOL_NAME,
            "description": campaign.TOOL["description"],
            "parameters": campaign.TOOL["input_schema"],
        },
    }


def build_messages(
    transcript: str, question: str, *, index: str = "", header: str = "", max_chars=None
) -> list:
    """The chat messages for one question. Pure, for tests. The system
    message is the prompt, header and transcript only, so it is a stable
    prefix the server can keep cached between questions."""
    system = (
        SYSTEM_PROMPT
        + "\n\n"
        + (header + "\n\n" if header else "")
        + "TRANSCRIPT:\n"
        + transcript
    )
    parts = []
    if index:
        parts.append(index)
    if max_chars:
        parts.append(length_instruction(max_chars))
    parts.append("QUESTION:\n" + question)
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": "\n\n".join(parts)},
    ]


def build_lore_messages(question: str, index: str, header: str = "") -> list:
    return [
        {
            "role": "system",
            "content": LORE_SYSTEM_PROMPT
            + "\n\n"
            + (header + "\n\n" if header else "")
            + index,
        },
        {"role": "user", "content": question},
    ]


def ask(
    transcript: str, question: str, *, header: str = "", note_dirs=(), max_chars=None
) -> AskResult:
    """Answer ``question`` from ``transcript`` with the local model. Blocking.
    Raises :class:`AskFailed` (Discord-ready) or :class:`AskBlocked`."""
    try:
        verdict = _gate(question, context=header.split("\n")[0] if header else "")
        notes = campaign.load_notes(note_dirs)
        index = campaign.index_text(notes)
        fitted, kept, total = fit_transcript(transcript, max_transcript_chars())
        messages = build_messages(
            fitted, question, index=index, header=header, max_chars=max_chars
        )
        result = _run(messages, notes, tools=[_tool()] if index else None)
    except LocalError as e:
        raise e.failure from e
    if kept < total:
        result.answer += trimmed_note(kept, total)
    result.guard = verdict
    return result


def ask_lore(
    question: str, *, header: str = "", note_dirs=(), include_gm: bool = False
) -> AskResult:
    """A notes-only question (dashboard Lore page) on the local model."""
    notes = campaign.load_notes(note_dirs, include_gm=include_gm)
    if not notes:
        if not include_gm and campaign.load_notes(note_dirs, include_gm=True):
            raise AskFailed(ONLY_GM_NOTES_MSG)
        raise AskFailed(NO_NOTES_MSG)
    try:
        verdict = _gate(
            question, context=header.split("\n")[0] if header else "", gm=include_gm
        )
        messages = build_lore_messages(question, campaign.index_text(notes), header)
        result = _run(messages, notes, tools=[_tool()])
    except LocalError as e:
        raise e.failure from e
    result.guard = verdict
    return result


def _text(message) -> str:
    # Reasoning models on Ollama return their thinking in a separate field;
    # servers that inline it use <think> tags. Never post either.
    return _THINK.sub("", getattr(message, "content", None) or "").strip()


def _tool_messages(message, notes, used: list) -> list:
    """The assistant turn with its tool calls, then one ``tool`` message per
    call with the requested notes (or an error the model can read)."""
    calls = getattr(message, "tool_calls", None) or []
    out = [
        {
            "role": "assistant",
            "content": getattr(message, "content", None) or "",
            "tool_calls": [
                {
                    "id": c.id,
                    "type": "function",
                    "function": {
                        "name": c.function.name,
                        "arguments": c.function.arguments,
                    },
                }
                for c in calls
            ],
        }
    ]
    known = {n.id for n in notes}
    for c in calls:
        try:
            args = json.loads(c.function.arguments or "{}")
        except (TypeError, ValueError):
            args = None
        ids = args.get("ids") if isinstance(args, dict) else None
        if c.function.name != campaign.TOOL_NAME or not isinstance(ids, list):
            text = f"Error: unknown tool call or bad arguments: {c.function.name}"
        else:
            text, is_error = campaign.read_notes(ids, notes)
            if is_error:
                text = "Error: " + text
            used.extend(i for i in ids if i in known)
        out.append({"role": "tool", "tool_call_id": c.id, "content": text})
    return out


def _run(messages: list, notes: list, *, tools=None) -> AskResult:
    """The call/tool loop. ``messages`` is extended in place."""
    sent_chars = sum(len(m["content"]) for m in messages)
    totals = {"in": 0, "cached": 0, "out": 0}
    used: list = []
    first_prompt_tokens = None
    rounds = 0

    def call(with_tools):
        nonlocal first_prompt_tokens
        kwargs = {
            "model": model(),
            "messages": messages,
            "max_tokens": MAX_ANSWER_TOKENS,
        }
        if with_tools:
            kwargs["tools"] = with_tools
        response = _create(**kwargs)
        _check_not_broken(response)
        usage = getattr(response, "usage", None)
        prompt = getattr(usage, "prompt_tokens", 0) or 0
        if first_prompt_tokens is None:
            first_prompt_tokens = prompt
        totals["in"] += prompt
        totals["out"] += getattr(usage, "completion_tokens", 0) or 0
        details = getattr(usage, "prompt_tokens_details", None)
        totals["cached"] += getattr(details, "cached_tokens", 0) or 0
        return response

    while True:
        rounds += 1
        response = call(tools)
        choice = response.choices[0]
        message = choice.message
        if not (tools and getattr(message, "tool_calls", None)):
            break
        messages.extend(_tool_messages(message, notes, used))
        if rounds >= MAX_TOOL_ROUNDS:
            logger.warning(
                f"/ask still calling tools after {rounds} rounds; answering "
                "without the extra notes"
            )
            messages.append(
                {"role": "user", "content": "Answer now with what you have."}
            )
            rounds += 1
            response = call(None)
            choice = response.choices[0]
            message = choice.message
            break

    answer = _text(message)
    if getattr(choice, "finish_reason", None) == "length":
        logger.warning(f"/ask hit max_tokens={MAX_ANSWER_TOKENS} on {model()}")
        if not answer:
            raise AskFailed(TRUNCATED_MSG)
        answer += TRUNCATION_NOTE
    if not answer:
        raise AskFailed(EMPTY_MSG)
    if first_prompt_tokens and first_prompt_tokens * 2 < sent_chars / CHARS_PER_TOKEN:
        logger.warning(
            f"Local server reported {first_prompt_tokens} prompt tokens for "
            f"{sent_chars} characters: it truncated the prompt (context window "
            "too small)."
        )
        answer += CONTEXT_NOTE
    return AskResult(
        answer=answer,
        model=getattr(response, "model", None) or model(),
        input_tokens=totals["in"],
        cached_tokens=totals["cached"],
        output_tokens=totals["out"],
        notes_used=dedupe(used),
        rounds=rounds,
    )
