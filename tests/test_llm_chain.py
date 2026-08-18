"""Provider-chain behaviour for `/ask`.

Every test drives the chain through injected fake tiers or patched
config, so nothing here touches OpenRouter, Cerebras or a local Ollama.
"""

import pytest

from src.llm import chain, config
from src.llm.errors import (
    AllProvidersFailed,
    EmptyCompletionError,
    TruncatedCompletionError,
    is_daily_cap_error,
    is_model_not_found_error,
    is_quota_error,
    is_rate_limit_error,
)


class _HTTPError(Exception):
    """Stand-in for an SDK error carrying an HTTP status."""

    def __init__(self, status_code, message=""):
        super().__init__(message or f"HTTP {status_code}")
        self.status_code = status_code


@pytest.fixture(autouse=True)
def _clear_latches():
    chain.reset_latches()
    yield
    chain.reset_latches()


@pytest.fixture
def no_env(monkeypatch):
    """A clean provider environment — no keys, no overrides."""
    for var in (
        "OPENROUTER_API_KEY",
        "OPENROUTER_MODEL",
        "OPENROUTER_BASE_URL",
        "CEREBRAS_PAID_API_KEY",
        "CEREBRAS_API_KEY",
        "CEREBRAS_MODEL",
        "CEREBRAS_BASE_URL",
        "ASK_DISABLE_OLLAMA",
    ):
        monkeypatch.delenv(var, raising=False)


def _tier(name, behaviour):
    """A tier whose call runs ``behaviour()`` and returns its value."""
    return chain._Tier(name, f"{name}-model", lambda t, q: behaviour())


def _run_with_tiers(monkeypatch, tiers):
    monkeypatch.setattr(chain, "build_tiers", lambda: tiers)
    return chain.ask("transcript", "question", sleep=lambda _: None)


# ── error classification ──────────────────────────────────────────────


@pytest.mark.parametrize(
    "error",
    [
        _HTTPError(402),
        Exception("payment_required"),
        Exception("Your account has insufficient_quota"),
        Exception("insufficient_credit for this request"),
    ],
)
def test_quota_errors_are_recognised(error):
    assert is_quota_error(error)


def test_rate_limit_is_not_mistaken_for_a_quota_error():
    """A 429 clears by itself; a 402 does not. Conflating them would either
    latch a tier out for 15 minutes over a one-minute spike, or tell the
    user to add credit when nothing is wrong with the account."""
    burst = _HTTPError(429, "rate limit exceeded")
    assert is_rate_limit_error(burst)
    assert not is_quota_error(burst)


def test_daily_cap_is_distinguished_from_an_ordinary_rate_limit():
    capped = Exception("Rate limit exceeded: free-models-per-day")
    assert is_daily_cap_error(capped)
    assert not is_daily_cap_error(Exception("rate limit exceeded"))


def test_model_not_found_is_recognised():
    assert is_model_not_found_error(_HTTPError(404))
    assert is_model_not_found_error(Exception("model_not_found"))
    # Ollama's phrasing for a model it cannot pull.
    assert is_model_not_found_error(Exception("manifest unknown"))


# ── failover ──────────────────────────────────────────────────────────


def test_primary_answers_and_no_fallback_is_reported(monkeypatch):
    result = _run_with_tiers(
        monkeypatch,
        [
            _tier("openrouter", lambda: "the answer"),
            _tier("cerebras", lambda: "unused"),
        ],
    )
    assert result.answer == "the answer"
    assert result.provider == "openrouter"
    assert result.used_fallback is False


def test_quota_error_crosses_to_the_next_tier_immediately(monkeypatch):
    calls = []

    def primary():
        calls.append("primary")
        raise _HTTPError(402, "payment_required")

    result = _run_with_tiers(
        monkeypatch,
        [_tier("openrouter", primary), _tier("cerebras", lambda: "fallback answer")],
    )
    assert result.answer == "fallback answer"
    assert result.provider == "cerebras"
    assert result.used_fallback is True
    # No retry ladder: waiting cannot refill a spent account.
    assert calls == ["primary"]


def test_daily_cap_crosses_over_without_retrying(monkeypatch):
    calls = []

    def primary():
        calls.append("primary")
        raise Exception("Rate limit exceeded: free-models-per-day")

    result = _run_with_tiers(
        monkeypatch,
        [_tier("openrouter", primary), _tier("cerebras", lambda: "fallback")],
    )
    assert result.provider == "cerebras"
    assert calls == ["primary"]


def test_ordinary_rate_limit_retries_the_same_tier_once_first(monkeypatch):
    """A per-minute window really does clear, so the primary is worth one
    more attempt before paying the fallback."""
    calls = []

    def primary():
        calls.append("primary")
        if len(calls) == 1:
            raise _HTTPError(429, "rate limit exceeded")
        return "recovered"

    result = _run_with_tiers(
        monkeypatch,
        [_tier("openrouter", primary), _tier("cerebras", lambda: "fallback")],
    )
    assert result.answer == "recovered"
    assert result.provider == "openrouter"
    assert calls == ["primary", "primary"]


def test_persistent_rate_limit_falls_through_after_its_retry(monkeypatch):
    calls = []

    def primary():
        calls.append("primary")
        raise _HTTPError(429, "rate limit exceeded")

    result = _run_with_tiers(
        monkeypatch,
        [_tier("openrouter", primary), _tier("cerebras", lambda: "fallback")],
    )
    assert result.provider == "cerebras"
    assert len(calls) == chain.MAX_ATTEMPTS_PER_TIER


def test_empty_completion_fails_over_rather_than_surfacing(monkeypatch):
    def primary():
        raise EmptyCompletionError("no content")

    result = _run_with_tiers(
        monkeypatch,
        [_tier("openrouter", primary), _tier("cerebras", lambda: "fallback")],
    )
    assert result.provider == "cerebras"


def test_chain_walks_all_three_tiers_to_the_local_one(monkeypatch):
    result = _run_with_tiers(
        monkeypatch,
        [
            _tier("openrouter", lambda: (_ for _ in ()).throw(_HTTPError(402))),
            _tier("cerebras", lambda: (_ for _ in ()).throw(_HTTPError(402))),
            _tier("ollama", lambda: "local answer"),
        ],
    )
    assert result.answer == "local answer"
    assert result.provider == "ollama"
    assert result.used_fallback is True


# ── latching ──────────────────────────────────────────────────────────


def test_exhausted_tier_is_skipped_on_the_next_question(monkeypatch):
    """The whole point of the latch: a second question must not pay
    another doomed round trip to a tier known to be out of credit."""
    primary_calls = []

    def primary():
        primary_calls.append(1)
        raise _HTTPError(402, "payment_required")

    tiers = [_tier("openrouter", primary), _tier("cerebras", lambda: "fallback")]
    monkeypatch.setattr(chain, "build_tiers", lambda: tiers)

    chain.ask("t", "q1", sleep=lambda _: None)
    chain.ask("t", "q2", sleep=lambda _: None)

    assert len(primary_calls) == 1


def test_ordinary_rate_limit_does_not_latch_the_tier(monkeypatch):
    """Latching on a 429 would route 15 minutes of traffic to the billed
    fallback over a one-minute spike."""
    primary_calls = []

    def primary():
        primary_calls.append(1)
        raise _HTTPError(429, "rate limit exceeded")

    tiers = [_tier("openrouter", primary), _tier("cerebras", lambda: "fallback")]
    monkeypatch.setattr(chain, "build_tiers", lambda: tiers)

    chain.ask("t", "q1", sleep=lambda _: None)
    chain.ask("t", "q2", sleep=lambda _: None)

    assert len(primary_calls) == 2 * chain.MAX_ATTEMPTS_PER_TIER


def test_all_tiers_latched_clears_rather_than_refusing(monkeypatch):
    """Stale latches must not make /ask permanently answer 'unavailable'."""
    chain._latch("openrouter")
    tiers = [_tier("openrouter", lambda: "recovered")]
    result = _run_with_tiers(monkeypatch, tiers)
    assert result.answer == "recovered"


# ── user-facing failure messages ──────────────────────────────────────


def test_out_of_credit_message_is_the_one_that_mentions_money(monkeypatch):
    tiers = [_tier("openrouter", lambda: (_ for _ in ()).throw(_HTTPError(402)))]
    with pytest.raises(AllProvidersFailed) as excinfo:
        _run_with_tiers(monkeypatch, tiers)
    assert excinfo.value.user_message == chain.DISCORD_QUOTA_MSG


def test_daily_cap_message_says_it_resets_not_that_money_is_needed(monkeypatch):
    """DISCORD_QUOTA_MSG asks for a top-up and would be a lie here: a daily
    allowance comes back at midnight on its own."""
    capped = Exception("free-models-per-day limit reached")
    tiers = [_tier("openrouter", lambda: (_ for _ in ()).throw(capped))]
    with pytest.raises(AllProvidersFailed) as excinfo:
        _run_with_tiers(monkeypatch, tiers)
    assert excinfo.value.user_message == chain.DISCORD_DAILY_CAP_MSG


def test_message_comes_from_the_first_tier_not_the_last(monkeypatch):
    """OpenRouter out of credit + Ollama not installed should tell the user
    about the credit, not about ollama."""
    tiers = [
        _tier("openrouter", lambda: (_ for _ in ()).throw(_HTTPError(402))),
        _tier("ollama", lambda: (_ for _ in ()).throw(Exception("connection refused"))),
    ]
    with pytest.raises(AllProvidersFailed) as excinfo:
        _run_with_tiers(monkeypatch, tiers)
    assert excinfo.value.user_message == chain.DISCORD_QUOTA_MSG


def test_no_provider_configured_is_its_own_message(monkeypatch, no_env):
    monkeypatch.setenv("ASK_DISABLE_OLLAMA", "1")
    with pytest.raises(AllProvidersFailed) as excinfo:
        chain.ask("t", "q", sleep=lambda _: None)
    assert excinfo.value.user_message == chain.DISCORD_UNCONFIGURED_MSG


# ── tier construction from environment ────────────────────────────────


def test_tiers_are_built_in_documented_order(monkeypatch, no_env):
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
    monkeypatch.setenv("CEREBRAS_PAID_API_KEY", "cb-key")
    assert [t.name for t in chain.build_tiers()] == [
        "openrouter",
        "cerebras",
        "ollama",
    ]


def test_unconfigured_cloud_tiers_are_skipped(monkeypatch, no_env):
    assert [t.name for t in chain.build_tiers()] == ["ollama"]


def test_ollama_tier_can_be_disabled(monkeypatch, no_env):
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
    monkeypatch.setenv("ASK_DISABLE_OLLAMA", "1")
    assert [t.name for t in chain.build_tiers()] == ["openrouter"]


def test_paid_cerebras_key_is_preferred_over_a_legacy_free_one(monkeypatch, no_env):
    """Cerebras retired its free tier in July 2026; an old free key answers
    402 on every call, so the paid key must win when both are present."""
    monkeypatch.setenv("CEREBRAS_API_KEY", "old-free-key")
    monkeypatch.setenv("CEREBRAS_PAID_API_KEY", "paid-key")
    assert config.cerebras_key() == "paid-key"


def test_blank_env_assignment_is_treated_as_unset(monkeypatch, no_env):
    """.env.sample ships `OPENROUTER_API_KEY=`, so a blank value is a real
    case — it must not become an empty-string key and a 401."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "   ")
    assert config.openrouter_key() == ""
    assert "openrouter" not in config.configured_cloud_providers()


def test_model_defaults_are_provider_specific(monkeypatch, no_env):
    """The two providers speak different slugs; sharing one MODEL var would
    send an OpenRouter slug to Cerebras, which means nothing to it."""
    assert config.openrouter_model() == config.DEFAULT_OPENROUTER_MODEL
    assert config.cerebras_model() == config.DEFAULT_CEREBRAS_MODEL
    assert config.openrouter_model() != config.cerebras_model()


# ── provider adapters ─────────────────────────────────────────────────
#
# The tests above inject synthetic tiers, which leaves the two functions
# that actually speak to a provider untested. These drive the real
# adapter bodies against fake SDK modules.


class _FakeMessage:
    def __init__(self, content):
        self.content = content


class _FakeChoice:
    def __init__(self, content, finish_reason="stop"):
        self.message = _FakeMessage(content)
        self.finish_reason = finish_reason


class _FakeResponse:
    def __init__(self, choices):
        self.choices = choices


class _FakeCompletions:
    def __init__(self, response):
        self._response = response
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        if isinstance(self._response, Exception):
            raise self._response
        return self._response


class _FakeOpenAI:
    """Captures constructor kwargs so the retry/header policy is assertable."""

    last_init = None

    def __init__(self, **kwargs):
        type(self).last_init = kwargs
        self.chat = type("_Chat", (), {})()
        self.chat.completions = self._completions

    @classmethod
    def factory(cls, completions):
        return type("_Bound", (cls,), {"_completions": completions})


def _install_fake_openai(monkeypatch, response):
    """Patch the openai module the adapter imports at call time."""
    import openai

    completions = _FakeCompletions(response)
    fake_cls = _FakeOpenAI.factory(completions)
    monkeypatch.setattr(openai, "OpenAI", fake_cls)
    return fake_cls, completions


def test_cloud_adapter_returns_the_completion_text(monkeypatch):
    _install_fake_openai(monkeypatch, _FakeResponse([_FakeChoice("the answer")]))
    call = chain._openai_compatible_call("https://x/v1", "key", "some-model")
    assert call("transcript", "question") == "the answer"


def test_cloud_adapter_disables_sdk_retries_and_sends_headers(monkeypatch):
    """max_retries=0 matters: the SDK's own ladder would multiply the wait
    before this module's classifier ever saw the error."""
    fake_cls, completions = _install_fake_openai(
        monkeypatch, _FakeResponse([_FakeChoice("ok")])
    )
    call = chain._openai_compatible_call(
        "https://x/v1", "key", "some-model", {"X-Title": "t"}
    )
    call("transcript", "question")

    assert fake_cls.last_init["max_retries"] == 0
    assert fake_cls.last_init["api_key"] == "key"
    assert fake_cls.last_init["base_url"] == "https://x/v1"
    assert fake_cls.last_init["default_headers"] == {"X-Title": "t"}
    assert completions.kwargs["model"] == "some-model"
    assert completions.kwargs["max_tokens"] == chain.MAX_ANSWER_TOKENS_CLOUD


def test_cloud_adapter_raises_on_no_choices(monkeypatch):
    _install_fake_openai(monkeypatch, _FakeResponse([]))
    call = chain._openai_compatible_call("https://x/v1", "key", "m")
    with pytest.raises(EmptyCompletionError):
        call("transcript", "question")


def test_cloud_adapter_raises_on_whitespace_only_content(monkeypatch):
    _install_fake_openai(monkeypatch, _FakeResponse([_FakeChoice("   ")]))
    call = chain._openai_compatible_call("https://x/v1", "key", "m")
    with pytest.raises(EmptyCompletionError):
        call("transcript", "question")


def test_cloud_adapter_reports_truncation_distinctly(monkeypatch):
    """A reasoning model that spends the whole budget thinking returns
    empty content with finish_reason='length'. Reporting that as a plain
    empty completion is how a paid tier silently falls through to local."""
    _install_fake_openai(
        monkeypatch, _FakeResponse([_FakeChoice("", finish_reason="length")])
    )
    call = chain._openai_compatible_call("https://x/v1", "key", "m")
    with pytest.raises(TruncatedCompletionError):
        call("transcript", "question")


def test_truncation_still_fails_over_like_an_empty_completion():
    """It is a subclass on purpose: distinct in the log, same in the chain."""
    assert issubclass(TruncatedCompletionError, EmptyCompletionError)


# ── local adapter ─────────────────────────────────────────────────────


class _FakeOllama:
    def __init__(self, content="local answer", raise_on_think=None):
        self.content = content
        self.raise_on_think = raise_on_think
        self.calls = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        if "think" in kwargs and self.raise_on_think is not None:
            raise self.raise_on_think
        return {"message": {"content": self.content}}


def _install_fake_ollama(monkeypatch, fake):
    import ollama

    monkeypatch.setattr(ollama, "chat", fake.chat)
    return fake


def test_local_adapter_returns_content_and_disables_thinking(monkeypatch):
    fake = _install_fake_ollama(monkeypatch, _FakeOllama())
    assert chain._ollama_call("m")("transcript", "question") == "local answer"
    assert fake.calls[0]["think"] is False
    assert fake.calls[0]["options"]["num_predict"] == chain.MAX_ANSWER_TOKENS_LOCAL


def test_local_adapter_raises_on_empty_content(monkeypatch):
    _install_fake_ollama(monkeypatch, _FakeOllama(content="  "))
    with pytest.raises(EmptyCompletionError):
        chain._ollama_call("m")("transcript", "question")


def test_old_ollama_client_gets_the_answer_not_an_upgrade_notice(monkeypatch):
    """Clients older than 0.5.x reject think=. Since the answer cleaner
    strips inline <think> anyway, drop the kwarg and answer the question
    rather than failing the tier."""
    fake = _install_fake_ollama(
        monkeypatch,
        _FakeOllama(
            raise_on_think=TypeError(
                "chat() got an unexpected keyword argument 'think'"
            )
        ),
    )
    assert chain._ollama_call("m")("transcript", "question") == "local answer"
    assert len(fake.calls) == 2
    assert "think" not in fake.calls[1]


def test_unrelated_type_error_still_propagates(monkeypatch):
    """The downgrade is matched on the exact message so a genuine bug in
    the call is not silently retried and swallowed."""
    _install_fake_ollama(
        monkeypatch, _FakeOllama(raise_on_think=TypeError("something else entirely"))
    )
    with pytest.raises(TypeError, match="something else entirely"):
        chain._ollama_call("m")("transcript", "question")


# ── latch remembers why ───────────────────────────────────────────────


def test_latched_reason_survives_into_later_questions(monkeypatch):
    """Regression for the message degrading over the latch window: with
    the primary latched out on a 402 and no other tier able to answer,
    question 2 must still say 'out of credit', not 'try again in a
    minute' — nothing clears without a top-up."""
    tiers = [
        _tier("openrouter", lambda: (_ for _ in ()).throw(_HTTPError(402))),
        _tier("ollama", lambda: (_ for _ in ()).throw(Exception("connection refused"))),
    ]
    monkeypatch.setattr(chain, "build_tiers", lambda: tiers)

    with pytest.raises(AllProvidersFailed) as first:
        chain.ask("t", "q1", sleep=lambda _: None)
    assert first.value.user_message == chain.DISCORD_QUOTA_MSG

    with pytest.raises(AllProvidersFailed) as second:
        chain.ask("t", "q2", sleep=lambda _: None)
    assert second.value.user_message == chain.DISCORD_QUOTA_MSG


def test_truncation_is_named_in_the_log_not_hidden_as_an_empty_reply(
    monkeypatch, caplog
):
    """The distinct class only pays off if it reaches a log line: the chain
    fails over and returns successfully, so last_error never surfaces
    unless the truncating tier was the last one tried. Without this, a
    paid tier quietly handing every question to local Ollama is
    indistinguishable from a provider returning junk."""
    truncated = TruncatedCompletionError("hit the 2048-token cap")
    tiers = [
        _tier("openrouter", lambda: (_ for _ in ()).throw(truncated)),
        _tier("ollama", lambda: "local answer"),
    ]
    with caplog.at_level("WARNING"):
        result = _run_with_tiers(monkeypatch, tiers)

    assert result.provider == "ollama"
    messages = " ".join(r.getMessage() for r in caplog.records)
    assert "MAX_ANSWER_TOKENS_CLOUD" in messages
    assert str(chain.MAX_ANSWER_TOKENS_CLOUD) in messages


def test_ordinary_empty_reply_logs_without_the_token_advice(monkeypatch, caplog):
    """The opposite case must not tell the operator to raise a cap that is
    not the problem."""
    tiers = [
        _tier("openrouter", lambda: (_ for _ in ()).throw(EmptyCompletionError("nil"))),
        _tier("ollama", lambda: "local answer"),
    ]
    with caplog.at_level("WARNING"):
        _run_with_tiers(monkeypatch, tiers)

    messages = " ".join(r.getMessage() for r in caplog.records)
    assert "MAX_ANSWER_TOKENS_CLOUD" not in messages
    assert "empty answer" in messages
