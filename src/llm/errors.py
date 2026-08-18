"""Failure classification for the `/ask` provider chain.

Deliberately duck-typed rather than keyed off ``openai`` exception
subclasses: the chain spans two OpenAI-compatible HTTP providers *and*
the ollama client, whose errors share no base class. Classifying on
``status_code`` plus the response text means one set of predicates
covers all three, and the unit tests can exercise them without
constructing SDK-specific exception objects.
"""


class EmptyCompletionError(RuntimeError):
    """A provider returned a completion with no content.

    Its own class (not a bare ``ValueError``) so the chain can fail the
    call over to the next provider instead of surfacing it as a bug:
    an empty reply will not fill itself in on a retry against the same
    model.
    """


class AllProvidersFailed(RuntimeError):
    """Every configured tier declined to answer.

    ``user_message`` is the Discord-facing text. It is chosen from the
    *first* tier's failure, not the last: if OpenRouter is out of credit
    and the local Ollama fallback is simply not installed, the useful
    thing to tell the asker is "the account is out of credit", not
    "ollama is not running".
    """

    def __init__(self, message: str, user_message: str, last_error: Exception = None):
        super().__init__(message)
        self.user_message = user_message
        self.last_error = last_error


def _status_code(e: Exception):
    """HTTP status carried by an exception, if it has one."""
    for attr in ("status_code", "status"):
        code = getattr(e, attr, None)
        if isinstance(code, int):
            return code
    response = getattr(e, "response", None)
    code = getattr(response, "status_code", None)
    return code if isinstance(code, int) else None


def is_quota_error(e: Exception) -> bool:
    """Billing exhaustion on the provider account — only money fixes it.

    Distinct from :func:`is_rate_limit_error`: a 429 clears by itself in
    a minute, this does not. Checked BEFORE the rate-limit branch
    because OpenAI-compatible providers report the same condition as a
    429 carrying ``insufficient_quota``.
    """
    if _status_code(e) == 402:
        return True
    s = str(e).lower()
    return any(
        k in s
        for k in ("payment_required", "insufficient_quota", "insufficient_credit")
    )


def is_daily_cap_error(e: Exception) -> bool:
    """A per-day allowance is spent, as opposed to a per-minute burst.

    Arrives as a 429, but unlike an ordinary rate limit it does not
    clear in a few seconds — it resets at midnight UTC. Walking the
    retry ladder just stalls the asker before the same failure, so this
    routes onto the quota path (cross to the next tier immediately).
    """
    s = str(e).lower()
    return "free-models-per-day" in s or ("per-day" in s and "free" in s)


def is_rate_limit_error(e: Exception) -> bool:
    """An ordinary throughput limit that a short wait can clear."""
    if _status_code(e) == 429:
        return True
    s = str(e).lower()
    return "rate limit" in s or "rate_limit" in s or "too many requests" in s


def is_model_not_found_error(e: Exception) -> bool:
    """The configured model id is not servable by this provider.

    Retrying cannot help and neither can waiting: the slug is wrong or
    retired, so the tier is skipped and the next one gets the question.
    """
    if _status_code(e) == 404:
        return True
    s = str(e).lower()
    return "model_not_found" in s or "does not exist" in s or "manifest unknown" in s
