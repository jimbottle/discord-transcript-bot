"""`/ask` via the Anthropic API (src/llm/claude.py): request shape, result
extraction, and every failure class mapped to a Discord-safe message. No
network: the client factory is replaced with a fake."""

from types import SimpleNamespace

import anthropic
import pytest

try:
    import httpx2 as httpx
except ImportError:  # older SDKs
    import httpx

from src.llm import claude


@pytest.fixture(autouse=True)
def _key(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.delenv("ASK_MODEL", raising=False)
    monkeypatch.delenv("ASK_EFFORT", raising=False)


def _response(
    text="The party fought a troll.", stop_reason="end_turn", model="claude-x"
):
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text=text)],
        stop_reason=stop_reason,
        model=model,
        usage=SimpleNamespace(
            input_tokens=120, cache_read_input_tokens=100, output_tokens=9
        ),
    )


class _FakeClient:
    def __init__(self, result=None, error=None):
        self.calls = []
        self._result = result
        self._error = error
        self.messages = self

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self._error is not None:
            raise self._error
        return self._result


def _use(monkeypatch, client):
    monkeypatch.setattr(claude, "_client", lambda: client)
    return client


def _status_error(cls, status, message="nope"):
    """An SDK error built the way the SDK builds it: the parsed JSON body on
    ``body`` and a ``"Error code: N - {dict}"`` dump on ``message``."""
    req = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    body = {
        "type": "error",
        "error": {"type": "invalid_request_error", "message": message},
    }
    resp = httpx.Response(status, request=req, json=body)
    return cls(f"Error code: {status} - {body}", response=resp, body=body)


# ── configuration ────────────────────────────────────────────────────


def test_configured_requires_a_non_blank_key(monkeypatch):
    assert claude.configured() is True
    monkeypatch.setenv("ANTHROPIC_API_KEY", "   ")
    assert claude.configured() is False


def test_model_and_effort_env_overrides(monkeypatch):
    assert claude.model() == claude.DEFAULT_MODEL
    assert claude.effort() == claude.DEFAULT_EFFORT
    monkeypatch.setenv("ASK_MODEL", "claude-sonnet-5-5")
    monkeypatch.setenv("ASK_EFFORT", "low")
    assert claude.model() == "claude-sonnet-5-5"
    assert claude.effort() == "low"


def test_ask_without_key_fails_before_any_call(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    client = _use(monkeypatch, _FakeClient(_response()))
    with pytest.raises(claude.AskFailed) as ei:
        claude.ask("t", "q")
    assert ei.value.user_message == claude.NO_KEY_MSG
    assert client.calls == []


# ── request shape ────────────────────────────────────────────────────


def test_request_puts_transcript_in_a_cached_system_block():
    req = claude.build_request("Alice: hi\nBob: hello", "who spoke?")
    assert req["model"] == claude.DEFAULT_MODEL
    assert req["max_tokens"] == claude.MAX_ANSWER_TOKENS
    assert req["output_config"] == {"effort": claude.DEFAULT_EFFORT}
    assert req["messages"] == [{"role": "user", "content": "who spoke?"}]
    base, transcript = req["system"]
    assert base["text"] == claude.SYSTEM_PROMPT and "cache_control" not in base
    assert transcript["text"].endswith("Alice: hi\nBob: hello")
    assert transcript["cache_control"] == {"type": "ephemeral"}


def test_request_shape_is_accepted_by_the_sdk_signature():
    """Every key build_request emits must be a real messages.create
    parameter, or the SDK raises TypeError at call time."""
    import inspect

    params = inspect.signature(anthropic.Anthropic().messages.create).parameters
    for key in claude.build_request("t", "q"):
        assert key in params, f"messages.create has no parameter {key!r}"


# ── results ──────────────────────────────────────────────────────────


def test_ask_returns_text_model_and_usage(monkeypatch):
    client = _use(monkeypatch, _FakeClient(_response()))
    result = claude.ask("transcript", "what happened?")
    assert result.answer == "The party fought a troll."
    assert result.model == "claude-x"
    assert (result.input_tokens, result.cached_tokens, result.output_tokens) == (
        120,
        100,
        9,
    )
    assert client.calls[0]["messages"][0]["content"] == "what happened?"


def test_ask_concatenates_text_blocks_and_ignores_thinking(monkeypatch):
    resp = _response()
    resp.content = [
        SimpleNamespace(type="thinking", thinking="..."),
        SimpleNamespace(type="text", text="Part one. "),
        SimpleNamespace(type="text", text="Part two."),
    ]
    _use(monkeypatch, _FakeClient(resp))
    assert claude.ask("t", "q").answer == "Part one. Part two."


def test_refusal_is_reported_not_posted(monkeypatch):
    _use(monkeypatch, _FakeClient(_response(stop_reason="refusal")))
    with pytest.raises(claude.AskFailed) as ei:
        claude.ask("t", "q")
    assert ei.value.user_message == claude.REFUSAL_MSG


def test_empty_answer_is_a_failure(monkeypatch):
    _use(monkeypatch, _FakeClient(_response(text="   ")))
    with pytest.raises(claude.AskFailed) as ei:
        claude.ask("t", "q")
    assert ei.value.user_message == claude.EMPTY_MSG


# ── failures map to Discord-safe messages ───────────────────────────


@pytest.mark.parametrize(
    "error, expected",
    [
        (_status_error(anthropic.AuthenticationError, 401), claude.BAD_KEY_MSG),
        (_status_error(anthropic.PermissionDeniedError, 403), claude.BAD_KEY_MSG),
        (_status_error(anthropic.RateLimitError, 429), claude.RATE_LIMIT_MSG),
        (_status_error(anthropic.InternalServerError, 503), claude.SERVER_ERROR_MSG),
        (
            anthropic.APIConnectionError(
                request=httpx.Request("POST", "https://api.anthropic.com")
            ),
            claude.UNREACHABLE_MSG,
        ),
        (
            anthropic.APITimeoutError(
                request=httpx.Request("POST", "https://api.anthropic.com")
            ),
            claude.UNREACHABLE_MSG,
        ),
    ],
)
def test_api_errors_become_user_messages(monkeypatch, error, expected):
    _use(monkeypatch, _FakeClient(error=error))
    with pytest.raises(claude.AskFailed) as ei:
        claude.ask("t", "q")
    assert ei.value.user_message == expected
    assert ei.value.__cause__ is error


def test_out_of_credit_is_named_as_billing(monkeypatch):
    err = _status_error(
        anthropic.BadRequestError,
        400,
        "Your credit balance is too low to access the Anthropic API.",
    )
    _use(monkeypatch, _FakeClient(error=err))
    with pytest.raises(claude.AskFailed) as ei:
        claude.ask("t", "q")
    assert ei.value.user_message == claude.BILLING_MSG


def test_other_bad_request_carries_the_api_message_not_the_dict_dump(monkeypatch):
    err = _status_error(anthropic.BadRequestError, 400, "prompt is too long")
    _use(monkeypatch, _FakeClient(error=err))
    with pytest.raises(claude.AskFailed) as ei:
        claude.ask("t", "q")
    assert ei.value.user_message == "Anthropic rejected the request: prompt is too long"
    assert "{" not in ei.value.user_message, "no raw body dump in the channel"


def test_bad_request_without_a_json_body_falls_back_to_status(monkeypatch):
    req = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    resp = httpx.Response(400, request=req, text="<html>gateway</html>")
    err = anthropic.BadRequestError(
        "<html>gateway</html>", response=resp, body="<html>gateway</html>"
    )
    _use(monkeypatch, _FakeClient(error=err))
    with pytest.raises(claude.AskFailed) as ei:
        claude.ask("t", "q")
    assert ei.value.user_message == "Anthropic rejected the request: HTTP 400"


# ── max_tokens ───────────────────────────────────────────────────────


def test_max_tokens_with_text_appends_a_truncation_note(monkeypatch, caplog):
    import logging

    _use(
        monkeypatch,
        _FakeClient(_response(text="Half an ans", stop_reason="max_tokens")),
    )
    with caplog.at_level(logging.WARNING, logger="src.llm.claude"):
        result = claude.ask("t", "q")
    assert result.answer == "Half an ans" + claude.TRUNCATION_NOTE
    assert any("max_tokens" in r.message for r in caplog.records)


def test_max_tokens_with_no_text_names_the_budget(monkeypatch):
    _use(monkeypatch, _FakeClient(_response(text="", stop_reason="max_tokens")))
    with pytest.raises(claude.AskFailed) as ei:
        claude.ask("t", "q")
    assert ei.value.user_message == claude.TRUNCATED_MSG
