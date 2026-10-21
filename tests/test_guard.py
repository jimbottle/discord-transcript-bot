"""The GM check itself (src/llm/guard.py): request shape and verdict
parsing, with a fake client. Live behaviour on real questions is checked
by hand (see the commit); these pin the contract."""

import json
from types import SimpleNamespace

import anthropic
import pytest

from src.llm import guard


class _Client:
    def __init__(self, response=None, error=None):
        self.calls, self.messages = [], self
        self._response, self._error = response, error

    def create(self, **kw):
        self.calls.append(kw)
        if self._error:
            raise self._error
        return self._response


def _verdict(data, stop_reason="end_turn"):
    return SimpleNamespace(
        content=[
            SimpleNamespace(
                type="text", text=json.dumps(data) if isinstance(data, dict) else data
            )
        ],
        stop_reason=stop_reason,
        model="claude-haiku-5-5",
        usage=SimpleNamespace(input_tokens=300, output_tokens=30),
    )


def _use(monkeypatch, client):
    monkeypatch.setattr(guard, "_client", lambda key: client)
    return client


def test_request_is_a_small_structured_classification(monkeypatch):
    monkeypatch.delenv("GUARD_MODEL", raising=False)
    req = guard.build_request("q?", context="SESSION: Game: Cyberpunk RED")
    assert req["model"] == "claude-haiku-5-5"
    assert req["output_config"]["effort"] == "low"
    assert req["output_config"]["format"]["schema"] == guard.SCHEMA
    assert (
        req["messages"][0]["content"] == "SESSION: Game: Cyberpunk RED\n\nQUESTION:\nq?"
    )
    assert "GM preparing" not in req["system"]
    assert "GM preparing" in guard.build_request("q", gm=True)["system"]


def test_guard_model_override(monkeypatch):
    monkeypatch.setenv("GUARD_MODEL", "claude-opus-5-5")
    assert guard.build_request("q")["model"] == "claude-opus-5-5"


def test_allowed_verdict(monkeypatch):
    _use(
        monkeypatch,
        _Client(_verdict({"allowed": True, "category": "ok", "reason": ""})),
    )
    v = guard.check("what did noah say?", api_key="k")
    assert v.allowed and v.category == "ok" and v.input_tokens == 300


def test_blocked_verdict(monkeypatch):
    _use(
        monkeypatch,
        _Client(
            _verdict(
                {
                    "allowed": False,
                    "category": "strategy",
                    "reason": "That asks for tactics.",
                }
            )
        ),
    )
    v = guard.check("best approach?", api_key="k")
    assert (v.allowed, v.category) == (False, "strategy")
    msg = guard.block_message(v)
    assert msg == guard.BLOCK_LEADS["strategy"] + " " + guard.BLOCK_TAIL
    assert v.reason == "That asks for tactics."  # kept for the history, not shown


def test_named_violation_wins_over_a_contradictory_allowed_flag(monkeypatch):
    _use(
        monkeypatch,
        _Client(_verdict({"allowed": True, "category": "optimization", "reason": "r"})),
    )
    assert guard.check("q", api_key="k").allowed is False


def test_refusal_by_the_classifier_blocks(monkeypatch):
    _use(monkeypatch, _Client(_verdict("", stop_reason="refusal")))
    assert guard.check("q", api_key="k").allowed is False


@pytest.mark.parametrize("text", ["not json", "[1]", '{"category": "ok"}'])
def test_unparseable_verdict_is_unavailable(monkeypatch, text):
    _use(monkeypatch, _Client(_verdict(text)))
    with pytest.raises(guard.GuardUnavailable):
        guard.check("q", api_key="k")


def test_api_error_is_unavailable(monkeypatch):
    err = (
        anthropic.APIConnectionError(request=None)
        if False
        else anthropic.APIError("down", request=None, body=None)
    )
    _use(monkeypatch, _Client(error=err))
    with pytest.raises(guard.GuardUnavailable):
        guard.check("q", api_key="k")


def test_enabled_by_default(monkeypatch):
    monkeypatch.delenv("TABLE_GUARD", raising=False)
    assert guard.enabled()
    monkeypatch.setenv("TABLE_GUARD", "off")
    assert not guard.enabled()
