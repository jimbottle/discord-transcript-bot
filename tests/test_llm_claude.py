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
def _key(monkeypatch, tmp_path):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.delenv("ASK_MODEL", raising=False)
    monkeypatch.delenv("ASK_EFFORT", raising=False)
    monkeypatch.delenv("CAMPAIGN_GM_NOTES", raising=False)
    # The GM check is its own API call; these tests are about the answer
    # path, so it allows everything here (tests/test_guard.py covers it).
    monkeypatch.setattr(
        claude.guard,
        "check",
        lambda q, **kw: claude.guard.Verdict(True, "ok", "", model="guard-stub"),
    )


def _response(
    text="The party fought a troll.",
    stop_reason="end_turn",
    model="claude-x",
    content=None,
):
    return SimpleNamespace(
        content=(
            content
            if content is not None
            else [SimpleNamespace(type="text", text=text)]
        ),
        stop_reason=stop_reason,
        model=model,
        usage=SimpleNamespace(
            input_tokens=120, cache_read_input_tokens=100, output_tokens=9
        ),
    )


def _tool_call(ids, call_id="tu_1", name=claude.campaign.TOOL_NAME):
    return _response(
        stop_reason="tool_use",
        content=[
            SimpleNamespace(type="thinking", thinking="hmm"),
            SimpleNamespace(type="tool_use", id=call_id, name=name, input={"ids": ids}),
        ],
    )


class _FakeClient:
    """``result`` may be a list: one response per call, in order."""

    def __init__(self, result=None, error=None):
        self.calls = []
        self._results = list(result) if isinstance(result, list) else [result]
        self._error = error
        self.messages = self

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self._error is not None:
            raise self._error
        return self._results.pop(0) if len(self._results) > 1 else self._results[0]


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


# ── campaign notes: index in the prompt, bodies only on request ──────────


NOTE = """---
id: npcs
title: Named NPCs
summary: Who the recurring NPCs are.
keywords: [NPC, Lilith Kane]
---
Lilith Kane is the fixer who runs Little Europe.
"""


@pytest.fixture
def notes_dir(tmp_path, monkeypatch):
    d = tmp_path / "notes"
    d.mkdir()
    (d / "npcs.md").write_text(NOTE, encoding="utf-8")
    return d


def test_request_without_notes_has_no_tools():
    req = claude.build_request("t", "q")
    assert "tools" not in req and "tool_choice" not in req
    assert len(req["system"]) == 2


def test_request_with_notes_adds_a_cached_index_block_and_the_tool(notes_dir):
    notes = claude.campaign.load_notes([notes_dir])
    index = claude.campaign.index_text(notes)
    req = claude.build_request("t", "q", index)
    assert req["system"][1]["text"].startswith("TRANSCRIPT:")
    assert req["system"][1]["cache_control"] == {"type": "ephemeral"}
    assert req["system"][2]["text"] == index
    assert "npcs — Named NPCs: Who the recurring NPCs are. [NPC, Lilith Kane]" in index
    assert "Lilith Kane is the fixer" not in index  # bodies are not in the prompt
    assert req["tools"] == [claude.campaign.TOOL]
    assert req["tool_choice"] == {"type": "auto"}


def test_ask_without_notes_sends_no_tools(monkeypatch):
    client = _use(monkeypatch, _FakeClient(_response()))
    claude.ask("t", "q")
    assert "tools" not in client.calls[0]


def test_ask_answers_in_one_round_when_the_model_does_not_call_the_tool(
    monkeypatch, notes_dir
):
    client = _use(monkeypatch, _FakeClient(_response("From the transcript.")))
    result = claude.ask("t", "q", note_dirs=[notes_dir])
    assert len(client.calls) == 1
    assert result.answer == "From the transcript."
    assert result.notes_used == []
    assert result.rounds == 1


def test_ask_feeds_requested_notes_back_and_sums_usage(monkeypatch, notes_dir):
    client = _use(
        monkeypatch,
        _FakeClient([_tool_call(["npcs"]), _response("Kane runs Little Europe.")]),
    )
    result = claude.ask("t", "who is kane?", note_dirs=[notes_dir])
    assert len(client.calls) == 2
    second = client.calls[1]["messages"]
    # user question, the assistant turn verbatim (thinking block included),
    # then one user message holding the tool result
    assert [m["role"] for m in second] == ["user", "assistant", "user"]
    assert second[1]["content"][0].type == "thinking"
    (tool_result,) = second[2]["content"]
    assert tool_result["type"] == "tool_result"
    assert tool_result["tool_use_id"] == "tu_1"
    assert "Lilith Kane is the fixer" in tool_result["content"]
    assert "is_error" not in tool_result
    assert result.answer == "Kane runs Little Europe."
    assert result.notes_used == ["npcs"]
    assert result.rounds == 2
    assert result.input_tokens == 240 and result.cached_tokens == 200
    assert result.output_tokens == 18


def test_unknown_note_id_is_an_error_result_not_a_crash(monkeypatch, notes_dir):
    client = _use(
        monkeypatch,
        _FakeClient([_tool_call(["nope"]), _response("Can't say.")]),
    )
    result = claude.ask("t", "q", note_dirs=[notes_dir])
    (tool_result,) = client.calls[1]["messages"][2]["content"]
    assert tool_result["is_error"] is True
    assert "Unknown note id(s): nope" in tool_result["content"]
    assert "Available: npcs" in tool_result["content"]
    assert result.notes_used == []


def test_tool_rounds_are_capped(monkeypatch, notes_dir):
    looping = _tool_call(["npcs"])
    client = _use(
        monkeypatch,
        _FakeClient([looping, looping, looping, _response("Finally.")]),
    )
    result = claude.ask("t", "q", note_dirs=[notes_dir])
    # MAX_TOOL_ROUNDS calls that asked for tools, then one forced answer
    assert len(client.calls) == claude.MAX_TOOL_ROUNDS + 1
    assert client.calls[-1]["tool_choice"] == {"type": "none"}
    assert result.answer == "Finally."
    assert result.notes_used == ["npcs"]


def test_gm_notes_are_hidden_unless_enabled(tmp_path, monkeypatch):
    d = tmp_path / "notes"
    d.mkdir()
    (d / "npcs.md").write_text(NOTE, encoding="utf-8")
    (d / "secrets.md").write_text(
        NOTE.replace("id: npcs", "id: secrets").replace(
            "---\nLilith", "audience: gm\n---\nLilith"
        ),
        encoding="utf-8",
    )
    assert [n.id for n in claude.campaign.load_notes([d])] == ["npcs"]
    monkeypatch.setenv("CAMPAIGN_GM_NOTES", "1")
    assert [n.id for n in claude.campaign.load_notes([d])] == ["npcs", "secrets"]


def test_session_header_goes_before_the_transcript_in_the_cached_block():
    req = claude.build_request(
        "[19:00] a: hi", "q", header="SESSION: Game: Cyberpunk RED"
    )
    block = req["system"][1]
    assert block["text"].startswith(
        "SESSION: Game: Cyberpunk RED\n\nTRANSCRIPT:\n[19:00]"
    )
    assert block["cache_control"] == {"type": "ephemeral"}


def test_ask_passes_header_through(monkeypatch):
    client = _use(monkeypatch, _FakeClient(_response()))
    claude.ask("t", "q", header="SESSION: X")
    assert client.calls[0]["system"][1]["text"].startswith("SESSION: X")


# ── ask_lore: notes only, dashboard only ─────────────────────────────────


def test_lore_request_has_no_transcript_and_always_offers_the_tool():
    req = claude.build_lore_request("who?", "INDEX", header="CAMPAIGN: X")
    assert req["system"][0]["text"] == claude.LORE_SYSTEM_PROMPT
    assert req["system"][1]["text"] == "CAMPAIGN: X\n\nINDEX"
    assert req["system"][1]["cache_control"] == {"type": "ephemeral"}
    assert "TRANSCRIPT" not in str(req["system"])
    assert req["tools"] == [claude.campaign.TOOL]
    assert req["tool_choice"] == {"type": "auto"}


def test_ask_lore_reads_notes_through_the_tool(monkeypatch, notes_dir):
    client = _use(
        monkeypatch,
        _FakeClient([_tool_call(["npcs"]), _response("Kane runs Little Europe.")]),
    )
    r = claude.ask_lore("who is kane?", note_dirs=[notes_dir])
    assert r.answer == "Kane runs Little Europe." and r.notes_used == ["npcs"]
    assert (
        "Lilith Kane is the fixer"
        in client.calls[1]["messages"][2]["content"][0]["content"]
    )


def test_ask_lore_without_notes_fails_before_any_call(monkeypatch, tmp_path):
    client = _use(monkeypatch, _FakeClient(_response()))
    with pytest.raises(claude.AskFailed) as e:
        claude.ask_lore("q", note_dirs=[tmp_path])
    assert e.value.user_message == claude.NO_NOTES_MSG
    assert client.calls == []


def test_ask_lore_gm_notes_only_when_asked(monkeypatch, notes_dir):
    (notes_dir / "secrets.md").write_text(
        NOTE.replace("id: npcs", "id: secrets").replace(
            "---\nLilith", "audience: gm\n---\nLilith"
        ),
        encoding="utf-8",
    )
    client = _use(monkeypatch, _FakeClient(_response("x")))
    claude.ask_lore("q", note_dirs=[notes_dir])
    assert "secrets" not in client.calls[0]["system"][1]["text"]
    claude.ask_lore("q", note_dirs=[notes_dir], include_gm=True)
    assert "secrets" in client.calls[1]["system"][1]["text"]


def test_lore_is_not_exposed_to_discord():
    """The Lore Q&A is dashboard-only: no bot code may reach ask_lore."""
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    for path in [root / "main.py", *(root / "src" / "bot").glob("*.py")]:
        assert "ask_lore" not in path.read_text(encoding="utf-8"), path


# ── the GM check gates every answer ──────────────────────────────────────


def _blocking(monkeypatch, category="strategy"):
    seen = []

    def check(q, **kw):
        seen.append((q, kw))
        return claude.guard.Verdict(False, category, "That's tactics.", model="g")

    monkeypatch.setattr(claude.guard, "check", check)
    return seen


def test_blocked_question_never_reaches_the_answer_model(monkeypatch):
    seen = _blocking(monkeypatch)
    client = _use(monkeypatch, _FakeClient(_response()))
    with pytest.raises(claude.AskBlocked) as e:
        claude.ask(
            "t",
            "what is my best approach to beating this scenario?",
            header="SESSION: Game: Cyberpunk RED\nSTYLE: x",
        )
    assert client.calls == []
    assert e.value.verdict.category == "strategy"
    assert "coaching the play" in e.value.user_message
    assert e.value.verdict.reason == "That's tactics."
    # the guard sees the question and the SESSION line, not the STYLE line
    assert seen[0][1]["context"] == "SESSION: Game: Cyberpunk RED"
    assert not isinstance(e.value, claude.AskFailed)


def test_lore_is_gated_too_and_gm_mode_is_passed(monkeypatch, notes_dir):
    seen = _blocking(monkeypatch, "optimization")
    client = _use(monkeypatch, _FakeClient(_response()))
    with pytest.raises(claude.AskBlocked):
        claude.ask_lore("best cyberware build?", note_dirs=[notes_dir], include_gm=True)
    assert client.calls == [] and seen[0][1]["gm"] is True


def test_guard_unavailable_fails_closed(monkeypatch):
    def broken(q, **kw):
        raise claude.guard.GuardUnavailable("boom")

    monkeypatch.setattr(claude.guard, "check", broken)
    client = _use(monkeypatch, _FakeClient(_response()))
    with pytest.raises(claude.AskFailed) as e:
        claude.ask("t", "q")
    assert e.value.user_message == claude.guard.GUARD_ERROR_MSG
    assert client.calls == []


def test_guard_can_be_switched_off(monkeypatch):
    _blocking(monkeypatch)
    monkeypatch.setenv("TABLE_GUARD", "off")
    _use(monkeypatch, _FakeClient(_response("fine")))
    r = claude.ask("t", "q")
    assert r.answer == "fine" and r.guard is None


def test_allowed_verdict_is_attached_to_the_result(monkeypatch):
    _use(monkeypatch, _FakeClient(_response()))
    assert claude.ask("t", "q").guard.model == "guard-stub"


def test_both_system_prompts_carry_the_table_rules():
    assert claude.TABLE_RULES in claude.SYSTEM_PROMPT
    assert claude.TABLE_RULES in claude.LORE_SYSTEM_PROMPT


def test_ask_lore_with_only_gm_notes_says_so(monkeypatch, tmp_path):
    # roborev #6462: used to claim the campaign had no notes at all
    (tmp_path / "secrets.md").write_text(
        "---\nid: secrets\ntitle: S\nsummary: s\naudience: gm\n---\nTaka.\n"
    )
    client = _use(monkeypatch, _FakeClient(_response("x")))
    with pytest.raises(claude.AskFailed) as e:
        claude.ask_lore("q", note_dirs=[tmp_path])
    assert e.value.user_message == claude.ONLY_GM_NOTES_MSG
    assert client.calls == []
    assert claude.ask_lore("q", note_dirs=[tmp_path], include_gm=True).answer == "x"


@pytest.mark.parametrize(
    "error, expected",
    [
        (lambda: _status_error(anthropic.AuthenticationError, 401), "BAD_KEY_MSG"),
        (
            lambda: _status_error(
                anthropic.BadRequestError, 400, "Your credit balance is too low"
            ),
            "BILLING_MSG",
        ),
        (lambda: anthropic.APIConnectionError(request=None), None),
    ],
    ids=["bad-key", "billing", "connection"],
)
def test_guard_api_errors_keep_their_specific_message(monkeypatch, error, expected):
    # roborev #6467: the GM check runs first, so its API errors must read
    # like the answer call's (bad key, no credit), not "try again".
    err = error()

    def broken(q, **kw):
        raise claude.guard.GuardUnavailable(str(err)) from err

    monkeypatch.setattr(claude.guard, "check", broken)
    client = _use(monkeypatch, _FakeClient(_response()))
    with pytest.raises(claude.AskFailed) as e:
        claude.ask("t", "q")
    want = getattr(claude, expected) if expected else claude.guard.GUARD_ERROR_MSG
    assert e.value.user_message == want
    assert client.calls == []  # still fails closed
