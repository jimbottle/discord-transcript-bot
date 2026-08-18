"""The `/ask` provider chain: OpenRouter -> Cerebras paid -> local Ollama.

Ported from the failover logic in the sibling
``raylytics/louisville-open-data-expenditure-bot`` repo
(``analytics_agent._call_with_retry``), with one structural difference:
that bot has two tiers and this one has three, so the "try the next
provider" step is a loop over tiers rather than a single ``fallback_fn``.

Failure kinds and what each earns:

* **out of credit (402) / daily allowance spent** — the next tier, now.
  No amount of waiting fixes a spent key, and the tier is latched out
  for :data:`PRIMARY_RECHECK_SECONDS` so the next fifteen minutes of
  questions skip a round trip that is known to fail.
* **empty completion** — the next tier, now. It will not fill itself in.
* **unknown model (404)** — the next tier, now, and latched: a retired
  slug does not come back within one question.
* **ordinary rate limit (429)** — one more attempt against the same
  tier (per-minute windows do clear), then the next tier.
* **anything else** — logged and passed to the next tier. With three
  tiers configured, a transport hiccup on one provider should not cost
  the asker their answer.

Every call here is blocking. `/ask` is an async command, so the caller
MUST run :func:`ask` off the event loop (``asyncio.to_thread``) — a
three-tier chain can occupy tens of seconds, and blocking the loop that
long stalls voice receive and the heartbeat, not just the command.
"""

import logging
import re
import time
from dataclasses import dataclass
from typing import Callable, List

from src.config import ollama_config
from src.llm import config
from src.llm.errors import (
    AllProvidersFailed,
    EmptyCompletionError,
    is_daily_cap_error,
    is_model_not_found_error,
    is_quota_error,
    is_rate_limit_error,
    TruncatedCompletionError,
)

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are an assistant that answers questions about a voice chat "
    "transcript. Be concise and direct. Only reference what's actually in "
    "the transcript. If the answer isn't in the transcript, say so."
)

# Caps latency and cost. Discord truncates around 1900 characters anyway,
# and temperature=0 keeps answers deterministic and transcript-grounded,
# mirroring the local bench harness the Ollama model was chosen with.
#
# The cloud cap is deliberately higher than the local one. Both cloud
# defaults are reasoning-capable, and OpenAI-compatible endpoints bill
# reasoning tokens against the SAME completion budget as visible content
# — so a 512-token cap can be spent entirely on thinking, returning empty
# content with finish_reason="length". That reads as an empty completion,
# which fails the tier over, and the paid providers would quietly hand
# every question down to local Ollama. The local tier keeps 512 because
# it runs with think=False and has no reasoning budget to fund.
MAX_ANSWER_TOKENS_CLOUD = 2048
MAX_ANSWER_TOKENS_LOCAL = 512
TEMPERATURE = 0

# One extra attempt against the same tier, for ordinary rate limits only.
MAX_ATTEMPTS_PER_TIER = 2
RETRY_BASE_DELAY = 2.0

# How long a tier stays latched out after a failure that cannot clear on
# its own. Healing is automatic: once the window lapses the tier is tried
# again, so credit added or a slug restored recovers without a restart.
PRIMARY_RECHECK_SECONDS = 900

DISCORD_QUOTA_MSG = (
    "The `/ask` language-model account is out of credit, so the provider is "
    "refusing new questions. This won't clear on its own — someone needs to "
    "top the account up. Transcription is unaffected."
)
DISCORD_DAILY_CAP_MSG = (
    "`/ask` has used up today's allowance from its language-model provider "
    "and no fallback could pick it up. Nothing is wrong with your question — "
    "the allowance resets at midnight UTC."
)
DISCORD_UNAVAILABLE_MSG = (
    "`/ask` couldn't reach any language-model provider just now. "
    "Transcription is unaffected — try again in a minute."
)
DISCORD_UNCONFIGURED_MSG = (
    "`/ask` has no language-model provider configured. Set "
    "`OPENROUTER_API_KEY` or `CEREBRAS_PAID_API_KEY` in `.env`, or run a "
    "local Ollama, then restart the bot."
)

# Tiers known to be unusable, name -> the time they were latched out.
_latched_out: dict = {}


@dataclass
class AskResult:
    """A successful answer plus which tier actually produced it."""

    answer: str
    provider: str
    model: str
    used_fallback: bool


@dataclass
class _Tier:
    name: str
    model: str
    call: Callable[[str, str], str]


def _is_latched(name: str) -> bool:
    entry = _latched_out.get(name)
    return entry is not None and (time.time() - entry[0]) < PRIMARY_RECHECK_SECONDS


def _latch(name: str, error: Exception = None) -> None:
    """Latch a tier out, remembering the error that caused it.

    The reason is kept, not just the timestamp, because a latched tier is
    skipped on later questions and would otherwise vanish from the
    message-selection logic: an out-of-credit primary would answer
    "out of credit" for the first question and then "try again in a
    minute" for the next fifteen minutes, which is advice that cannot
    come true.
    """
    logger.info("Latching out /ask tier '%s' for %ds", name, PRIMARY_RECHECK_SECONDS)
    _latched_out[name] = (time.time(), error)


def _latched_error(name: str):
    """The error a tier was latched out for, if it is still latched."""
    if not _is_latched(name):
        return None
    return _latched_out[name][1]


def reset_latches() -> None:
    """Clear all latch state. For tests and for an explicit re-check."""
    _latched_out.clear()


def _messages(transcript: str, question: str) -> list:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Here is the transcript:\n\n{transcript}\n\nQuestion: {question}",
        },
    ]


def _openai_compatible_call(
    base_url: str, api_key: str, model: str, extra_headers: dict = None
) -> Callable[[str, str], str]:
    """Build a callable that asks one OpenAI-compatible provider.

    ``max_retries=0`` because this module owns the retry policy; the
    SDK's own ladder would silently multiply the wait before our
    classifier ever sees the error.
    """

    def _call(transcript: str, question: str) -> str:
        import httpx
        import openai

        client = openai.OpenAI(
            api_key=api_key,
            base_url=base_url,
            max_retries=0,
            timeout=httpx.Timeout(45.0, connect=10.0),
            default_headers=extra_headers or None,
        )
        response = client.chat.completions.create(
            model=model,
            messages=_messages(transcript, question),
            max_tokens=MAX_ANSWER_TOKENS_CLOUD,
            temperature=TEMPERATURE,
        )
        choices = getattr(response, "choices", None) or []
        choice = choices[0] if choices else None
        content = getattr(getattr(choice, "message", None), "content", None)
        if (content or "").strip():
            return content

        # An empty answer that hit the token ceiling is our configuration
        # failing, not the provider: it is reported separately so it shows
        # up as "raise MAX_ANSWER_TOKENS_CLOUD" in the log rather than
        # hiding inside a generic empty-completion fallover.
        if getattr(choice, "finish_reason", None) == "length":
            raise TruncatedCompletionError(
                f"{model} hit the {MAX_ANSWER_TOKENS_CLOUD}-token cap without "
                "emitting an answer (likely spent on reasoning tokens)"
            )
        raise EmptyCompletionError(f"{model} returned an empty completion")

    return _call


def _ollama_call(model: str) -> Callable[[str, str], str]:
    def _call(transcript: str, question: str) -> str:
        import ollama

        # think=False disables native thinking (Gemma 4 defaults it on);
        # the answer cleaner strips inline <think> for models that
        # ignore the flag.
        kwargs = dict(
            model=model,
            messages=_messages(transcript, question),
            options={
                "num_predict": MAX_ANSWER_TOKENS_LOCAL,
                "temperature": TEMPERATURE,
            },
        )
        try:
            response = ollama.chat(think=False, **kwargs)
        except TypeError as e:
            # An ollama client older than 0.5.x rejects think=. Previously
            # this surfaced to the asker as "upgrade your ollama client";
            # since the answer cleaner strips inline <think> anyway, just
            # drop the kwarg and answer the question. Matched on the exact
            # message so an unrelated TypeError still propagates.
            if "unexpected keyword argument 'think'" not in str(e):
                raise
            logger.warning(
                "ollama client predates the think= parameter; "
                "falling back to an untyped call (upgrade with "
                "`pip install -U 'ollama>=0.5.1'`)"
            )
            response = ollama.chat(**kwargs)
        content = (response.get("message") or {}).get("content")
        if not (content or "").strip():
            raise EmptyCompletionError(f"{model} returned an empty completion")
        return content

    return _call


def build_tiers() -> List[_Tier]:
    """The tiers `/ask` may use, in call order, skipping unconfigured ones."""
    tiers = []
    if config.openrouter_key():
        tiers.append(
            _Tier(
                "openrouter",
                config.openrouter_model(),
                _openai_compatible_call(
                    config.openrouter_base_url(),
                    config.openrouter_key(),
                    config.openrouter_model(),
                    # OpenRouter attributes traffic to the app named here.
                    {
                        "HTTP-Referer": "https://github.com/jimbottle/discord-transcript-bot",
                        "X-Title": "Discord Transcript Bot",
                    },
                ),
            )
        )
    if config.cerebras_key():
        tiers.append(
            _Tier(
                "cerebras",
                config.cerebras_model(),
                _openai_compatible_call(
                    config.cerebras_base_url(),
                    config.cerebras_key(),
                    config.cerebras_model(),
                ),
            )
        )
    if config.ollama_enabled():
        model = ollama_config.get_ask_model()
        tiers.append(_Tier("ollama", model, _ollama_call(model)))
    return tiers


def _user_message_for(error: Exception) -> str:
    if error is None:
        return DISCORD_UNAVAILABLE_MSG
    if is_quota_error(error):
        return DISCORD_QUOTA_MSG
    if is_daily_cap_error(error):
        return DISCORD_DAILY_CAP_MSG
    return DISCORD_UNAVAILABLE_MSG


def _retry_delay(error: Exception, attempt: int) -> float:
    """Honour a provider-supplied 'retry in 3.2s' hint, else back off."""
    match = re.search(r"retry in ([\d.]+)s", str(error))
    if match:
        try:
            return min(float(match.group(1)), 10.0)
        except ValueError:
            pass
    return RETRY_BASE_DELAY * attempt


def ask(transcript: str, question: str, sleep=time.sleep) -> AskResult:
    """Ask the transcript question, walking the tiers until one answers.

    ``sleep`` is injectable so tests can exercise the rate-limit ladder
    without spending real seconds.

    Raises :class:`AllProvidersFailed` when every configured tier
    declines; its ``user_message`` is ready to send to Discord.
    """
    tiers = build_tiers()
    if not tiers:
        raise AllProvidersFailed(
            "No /ask provider is configured", DISCORD_UNCONFIGURED_MSG
        )

    # A tier latched out by an earlier question is skipped — unless every
    # tier is latched, in which case the latches are stale or the outage
    # is total and we would rather try than refuse outright.
    usable = [t for t in tiers if not _is_latched(t.name)]
    if not usable:
        logger.info("All /ask tiers were latched out; clearing and retrying")
        reset_latches()
        usable = tiers

    errors_by_tier = {}
    last_error = None

    for index, tier in enumerate(usable):
        for attempt in range(1, MAX_ATTEMPTS_PER_TIER + 1):
            try:
                answer = tier.call(transcript, question)
                if index > 0:
                    logger.info("/ask served by fallback tier '%s'", tier.name)
                return AskResult(
                    answer=answer,
                    provider=tier.name,
                    model=tier.model,
                    used_fallback=index > 0,
                )
            except Exception as e:  # noqa: BLE001 — classified just below
                last_error = e
                errors_by_tier.setdefault(tier.name, e)

                exhausted = is_quota_error(e) or is_daily_cap_error(e)
                missing_model = is_model_not_found_error(e)
                empty = isinstance(e, EmptyCompletionError)
                rate_limited = is_rate_limit_error(e)

                # Conditions a retry cannot fix: stop spending time on
                # this tier and let the next one have the question.
                if exhausted or missing_model:
                    logger.warning(
                        "/ask tier '%s' unusable (%s): %s",
                        tier.name,
                        "exhausted" if exhausted else "unknown model",
                        e,
                    )
                    _latch(tier.name, e)
                    break
                if empty:
                    if isinstance(e, TruncatedCompletionError):
                        # Named explicitly, because this one is OUR bug to
                        # fix, not the provider's. It is also the only
                        # place it becomes visible: the chain fails over
                        # to the next tier and returns successfully, so
                        # last_error never reaches anyone unless the
                        # truncating tier happened to be the last tried.
                        # Without this line, a paid tier quietly handing
                        # every question to local Ollama looks exactly
                        # like a provider returning junk.
                        logger.warning(
                            "/ask tier '%s' spent its %d-token budget without "
                            "answering — raise MAX_ANSWER_TOKENS_CLOUD if this "
                            "repeats: %s",
                            tier.name,
                            MAX_ANSWER_TOKENS_CLOUD,
                            e,
                        )
                    else:
                        logger.warning(
                            "/ask tier '%s' returned an empty answer: %s",
                            tier.name,
                            e,
                        )
                    break
                if not rate_limited:
                    logger.error("/ask tier '%s' failed: %s", tier.name, e)
                    break

                # Ordinary rate limit: one more go at this tier, since a
                # per-minute window really does clear, then move on.
                if attempt >= MAX_ATTEMPTS_PER_TIER:
                    logger.warning(
                        "/ask tier '%s' still rate limited after %d attempts",
                        tier.name,
                        attempt,
                    )
                    break
                delay = _retry_delay(e, attempt)
                logger.info(
                    "/ask tier '%s' rate limited, retrying in %.1fs",
                    tier.name,
                    delay,
                )
                sleep(delay)

    # The user-facing message comes from the FIRST tier's failure in the
    # configured order — if OpenRouter is out of credit and the local
    # Ollama simply is not installed, "the account is out of credit" is
    # the useful thing to say, not "ollama is not running".
    #
    # Walk the full tier list, not just the ones tried this call: a tier
    # skipped because it was latched out still carries the error it was
    # latched for. Without that, question 1 says "out of credit" and
    # questions 2..N for the next fifteen minutes say "try again in a
    # minute" — advice that cannot come true, since nothing clears
    # without a top-up.
    governing_error = None
    for tier in tiers:
        candidate = errors_by_tier.get(tier.name) or _latched_error(tier.name)
        if candidate is not None:
            governing_error = candidate
            break

    raise AllProvidersFailed(
        f"All {len(usable)} /ask tier(s) failed; last error: {last_error}",
        _user_message_for(governing_error),
        last_error=last_error,
    )
