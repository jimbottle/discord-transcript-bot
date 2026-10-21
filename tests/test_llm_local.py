"""The local backend (src/llm/local.py) and the provider choice
(src/llm/provider.py), with a fake OpenAI-compatible client. The live
behaviour against Ollama is measured by hand (see the commit)."""

import json
from types import SimpleNamespace

import httpx
import openai
import pytest

from src.llm import claude, guard, local, provider

REAL_GUARD_CHECK = guard.check


def _message(content="", tool_calls=None):
    return SimpleNamespace(content=content, tool_calls=tool_calls)


def _response(content="", tool_calls=None, finish="stop", prompt=1000, out=20):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(message=_message(content, tool_calls), finish_reason=finish)
        ],
        usage=SimpleNamespace(
            prompt_tokens=prompt, completion_tokens=out, prompt_tokens_details=None
        ),
        model="gemma4:26b",
    )


def _call(ids, call_id="call_1", name=None):
    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(
            name=name or local.campaign.TOOL_NAME, arguments=json.dumps({"ids": ids})
        ),
    )


class _Fake:
    def __init__(self, *responses, error=None):
        self.calls = []
        self._responses = list(responses)
        self._error = error
        self.chat = SimpleNamespace(completions=self)

    def create(self, **kw):
        self.calls.append(json.loads(json.dumps(kw, default=str)))
        if self._error:
            raise self._error
        return self._responses.pop(0)


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("ASK_PROVIDER", "local")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("CAMPAIGN_GM_NOTES", raising=False)
    monkeypatch.setattr(
        guard, "check", lambda q, **kw: guard.Verdict(True, "ok", "", model="stub")
    )


def _use(monkeypatch, fake):
    monkeypatch.setattr(local, "_client", lambda: fake)
    return fake


@pytest.fixture
def notes_dir(tmp_path):
    d = tmp_path / "notes"
    d.mkdir()
    (d / "npcs.md").write_text(
        "---\nid: npcs\ntitle: NPCs\nsummary: Who is who\nkeywords: [fixie]\n---\n"
        "Fixie is a courier boss.\n"
    )
    return d


# ── provider choice ────────────────────────────────────────────────────


def test_unset_provider_is_local_without_a_key(monkeypatch):
    monkeypatch.delenv("ASK_PROVIDER")
    assert provider.name() == "local"
    assert provider.configured() is True


def test_unset_provider_keeps_an_existing_anthropic_setup(monkeypatch):
    monkeypatch.delenv("ASK_PROVIDER")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    assert provider.name() == "anthropic"


def test_explicit_local_wins_over_a_key(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    assert provider.name() == "local"


@pytest.mark.parametrize("raw", ["Ollama", "LOCAL", " local "])
def test_local_aliases(monkeypatch, raw):
    monkeypatch.setenv("ASK_PROVIDER", raw)
    assert provider.name() == "local"


def test_anthropic_without_a_key_is_not_configured(monkeypatch):
    monkeypatch.setenv("ASK_PROVIDER", "claude")
    assert provider.name() == "anthropic" and provider.configured() is False
    assert "ANTHROPIC_API_KEY" in provider.not_configured_message()


def test_unknown_provider_falls_back_to_automatic(monkeypatch):
    monkeypatch.setenv("ASK_PROVIDER", "gpt")
    assert provider.name() == "local"


def test_provider_routes_to_the_chosen_backend(monkeypatch):
    seen = []
    monkeypatch.setattr(local, "ask", lambda *a, **k: seen.append("local"))
    monkeypatch.setattr(claude, "ask", lambda *a, **k: seen.append("anthropic"))
    provider.ask("t", "q")
    monkeypatch.setenv("ASK_PROVIDER", "anthropic")
    provider.ask("t", "q")
    assert seen == ["local", "anthropic"]


def test_settings_and_defaults(monkeypatch):
    monkeypatch.delenv("LOCAL_LLM_URL")
    assert local.url() == "http://localhost:11434/v1"
    assert local.model() == "gemma4:26b"
    monkeypatch.setenv("LOCAL_LLM_MODEL", "qwen3:14b")
    monkeypatch.setenv("LOCAL_LLM_TIMEOUT_S", "nonsense")
    assert local.model() == "qwen3:14b"
    assert local.timeout_s() == local.DEFAULT_TIMEOUT_S
    assert "qwen3:14b" in provider.describe()


# ── request shape ──────────────────────────────────────────────────────


def test_transcript_is_a_stable_prefix_and_the_question_comes_last():
    """The server caches the prompt prefix between questions, so nothing
    that varies may come before the transcript."""
    a = local.build_messages("LINES", "Who?", index="INDEX", header="SESSION: x")
    b = local.build_messages("LINES", "Why?", index="INDEX2", max_chars=1500)
    assert a[0] == {"role": "system", "content": a[0]["content"]}
    assert a[0]["content"].endswith("TRANSCRIPT:\nLINES")
    assert "SESSION: x" in a[0]["content"]
    user = a[1]["content"]
    assert user.index("INDEX") < user.index("QUESTION:\nWho?")
    assert "OUTPUT LIMIT" in b[1]["content"] and "OUTPUT LIMIT" not in b[0]["content"]


def test_ask_returns_the_answer_and_usage(monkeypatch):
    fake = _use(monkeypatch, _Fake(_response("They went to the docks.")))
    r = local.ask("[20:00:00] A (B) [1]: docks", "Where?")
    assert r.answer == "They went to the docks."
    assert r.model == "gemma4:26b" and r.input_tokens == 1000 and r.rounds == 1
    assert "tools" not in fake.calls[0]


def test_inline_thinking_is_never_posted(monkeypatch):
    _use(monkeypatch, _Fake(_response("<think>hmm, docks</think>\nThe docks.")))
    assert local.ask("t", "Where?").answer == "The docks."


def test_notes_are_read_through_the_tool(monkeypatch, notes_dir):
    fake = _use(
        monkeypatch,
        _Fake(
            _response("", [_call(["npcs"])], finish="tool_calls"),
            _response("Fixie runs couriers."),
        ),
    )
    r = local.ask("t", "Who is Fixie?", note_dirs=[notes_dir])
    assert r.answer == "Fixie runs couriers." and r.notes_used == ["npcs"]
    assert r.rounds == 2 and r.input_tokens == 2000
    second = fake.calls[1]["messages"]
    assert second[-2]["role"] == "assistant" and second[-2]["tool_calls"]
    assert second[-1]["role"] == "tool" and "courier boss" in second[-1]["content"]
    assert fake.calls[0]["tools"][0]["function"]["name"] == local.campaign.TOOL_NAME


def test_unknown_note_or_bad_call_is_an_error_the_model_reads(monkeypatch, notes_dir):
    fake = _use(
        monkeypatch,
        _Fake(
            _response(
                "", [_call(["nope"]), _call([], "c2", name="other")], "tool_calls"
            ),
            _response("Not in the notes."),
        ),
    )
    local.ask("t", "Who?", note_dirs=[notes_dir])
    tool_msgs = [m for m in fake.calls[1]["messages"] if m["role"] == "tool"]
    assert len(tool_msgs) == 2 and all(
        m["content"].startswith("Error") for m in tool_msgs
    )


def test_tool_rounds_are_capped_then_answered_without_tools(monkeypatch, notes_dir):
    calls = [_response("", [_call(["npcs"], f"c{i}")], "tool_calls") for i in range(3)]
    fake = _use(monkeypatch, _Fake(*calls, _response("Best I can say.")))
    r = local.ask("t", "Who?", note_dirs=[notes_dir])
    assert r.answer == "Best I can say." and r.rounds == 4
    assert "tools" not in fake.calls[-1]


def test_truncated_and_empty_answers(monkeypatch):
    _use(monkeypatch, _Fake(_response("Partial", finish="length")))
    assert local.ask("t", "q").answer.endswith(local.TRUNCATION_NOTE)
    _use(monkeypatch, _Fake(_response("", finish="length")))
    with pytest.raises(local.AskFailed) as e:
        local.ask("t", "q")
    assert e.value.user_message == local.TRUNCATED_MSG
    _use(monkeypatch, _Fake(_response("   ")))
    with pytest.raises(local.AskFailed) as e:
        local.ask("t", "q")
    assert e.value.user_message == local.EMPTY_MSG


def test_a_server_that_cut_the_prompt_is_named_in_the_answer(monkeypatch):
    """Ollama with a small context window drops the start of the prompt
    without an error; the reported prompt size gives it away."""
    monkeypatch.setenv("LOCAL_LLM_MAX_TRANSCRIPT_CHARS", "0")  # send it all
    transcript = "[20:00:00] A (B) [1]: " + "word " * 20000  # ~100k chars
    _use(monkeypatch, _Fake(_response("An answer.", prompt=4096)))
    assert local.ask(transcript, "q").answer.endswith(local.CONTEXT_NOTE)
    _use(monkeypatch, _Fake(_response("An answer.", prompt=30000)))
    assert local.ask(transcript, "q").answer == "An answer."


# ── failures ───────────────────────────────────────────────────────────

_REQ = httpx.Request("POST", "http://localhost:11434/v1/chat/completions")


def _status(cls, code, body=None):
    return cls("x", response=httpx.Response(code, request=_REQ), body=body)


@pytest.mark.parametrize(
    "error, needle",
    [
        (openai.APIConnectionError(request=_REQ), "Is Ollama"),
        (openai.APITimeoutError(request=_REQ), "took longer than"),
        (
            _status(
                openai.NotFoundError,
                404,
                {"error": {"message": "model 'gemma4:26b' not found"}},
            ),
            "ollama pull gemma4:26b",
        ),
        (_status(openai.NotFoundError, 404, {"error": "404 page not found"}), "/v1"),
        (
            _status(
                openai.BadRequestError,
                400,
                {"error": {"message": 'model "x" not found, try pulling it first'}},
            ),
            "ollama pull",
        ),
        (
            _status(openai.InternalServerError, 500, {"error": "out of memory"}),
            "HTTP 500: out of memory",
        ),
    ],
)
def test_server_errors_become_user_messages(monkeypatch, error, needle):
    _use(monkeypatch, _Fake(error=error))
    with pytest.raises(local.AskFailed) as e:
        local.ask("t", "q")
    assert needle in e.value.user_message


def test_lore_without_notes_fails_before_any_call(monkeypatch, tmp_path):
    fake = _use(monkeypatch, _Fake())
    with pytest.raises(local.AskFailed) as e:
        local.ask_lore("q", note_dirs=[tmp_path])
    assert e.value.user_message == local.NO_NOTES_MSG and fake.calls == []


def test_lore_reads_notes(monkeypatch, notes_dir):
    fake = _use(
        monkeypatch,
        _Fake(
            _response("", [_call(["npcs"])], "tool_calls"), _response("A courier boss.")
        ),
    )
    r = local.ask_lore("Who is Fixie?", note_dirs=[notes_dir])
    assert r.answer == "A courier boss." and r.notes_used == ["npcs"]
    assert "TRANSCRIPT" not in fake.calls[0]["messages"][0]["content"]


# ── the GM check on the local model ────────────────────────────────────


def test_guard_runs_on_the_local_model_with_a_json_schema(monkeypatch):
    monkeypatch.setattr(guard, "check", REAL_GUARD_CHECK)  # not the autouse stub
    fake = _use(
        monkeypatch,
        _Fake(
            _response(
                json.dumps(
                    {"allowed": False, "category": "strategy", "reason": "advice"}
                )
            )
        ),
    )
    with pytest.raises(local.AskBlocked) as e:
        local.ask("t", "What's our best plan?")
    assert e.value.verdict.category == "strategy"
    sent = fake.calls[0]
    assert sent["model"] == "gemma4:26b"
    assert sent["response_format"]["json_schema"]["schema"] == guard.SCHEMA
    assert len(fake.calls) == 1  # no answer generated


def test_guard_fails_closed_and_names_a_down_server(monkeypatch):
    monkeypatch.setattr(guard, "check", REAL_GUARD_CHECK)
    _use(monkeypatch, _Fake(error=openai.APIConnectionError(request=_REQ)))
    with pytest.raises(local.AskFailed) as e:
        local.ask("t", "What happened?")
    assert "Is Ollama" in e.value.user_message


def test_guard_unparseable_output_fails_closed(monkeypatch):
    monkeypatch.setattr(guard, "check", REAL_GUARD_CHECK)
    _use(monkeypatch, _Fake(_response("sure, allowed!")))
    with pytest.raises(local.AskFailed) as e:
        local.ask("t", "What happened?")
    assert e.value.user_message == guard.GUARD_ERROR_MSG


def test_guard_model_follows_the_backend(monkeypatch):
    monkeypatch.setenv("LOCAL_LLM_MODEL", "qwen3:14b")
    assert guard.model("local") == "qwen3:14b"
    assert guard.model("anthropic") == guard.DEFAULT_GUARD_MODEL
    monkeypatch.setenv("GUARD_MODEL", "tiny")
    assert guard.model("local") == "tiny"


# ── health probe ───────────────────────────────────────────────────────


def test_status_reports_a_closed_port_quickly():
    st = local.status()  # conftest points LOCAL_LLM_URL at port 9
    assert st["reachable"] is False and "error" in st


def test_status_matches_the_model_name_and_latest_tag(monkeypatch):
    models = SimpleNamespace(
        list=lambda: [
            SimpleNamespace(id="gemma4:26b"),
            SimpleNamespace(id="llama3:latest"),
        ]
    )
    monkeypatch.setattr(
        local.openai, "OpenAI", lambda **kw: SimpleNamespace(models=models)
    )
    assert local.status()["has_model"] is True
    monkeypatch.setenv("LOCAL_LLM_MODEL", "llama3")
    assert local.status()["has_model"] is True
    monkeypatch.setenv("LOCAL_LLM_MODEL", "mistral")
    assert local.status()["has_model"] is False


# ── transcript budget and a broken server ──────────────────────────────


def _lines(n, width=50):
    return "".join(f"[8:{i % 60:02d} PM] A (B): {'x' * width} {i}\n" for i in range(n))


def test_short_transcripts_are_sent_whole():
    t = _lines(10)
    assert local.fit_transcript(t, 100_000) == (t, 10, 10)
    assert local.fit_transcript(t, 0) == (t, 10, 10)  # 0 = no cap


def test_long_transcripts_keep_the_most_recent_lines_within_budget():
    t = _lines(2000)
    text, kept, total = local.fit_transcript(t, 20_000)
    assert total == 2000 and kept < total
    assert text.startswith(local.TRIM_MARK)
    assert len(text) - len(local.TRIM_MARK) <= 20_000
    assert text.endswith(t.splitlines(keepends=True)[-1])  # newest line kept


def test_the_cut_holds_still_while_a_live_session_grows():
    """So the kept text stays a cached prefix: adding lines one at a time
    moves the start only once per quarter-budget step, not on every line."""
    t = _lines(2000)
    starts = set()
    for i in range(200):  # ~14,000 characters of new talk, line by line
        t += _lines(1)
        starts.add(local.fit_transcript(t, 20_000)[0].splitlines()[1])
    # a 5,000-character step: about 3 moves (a sliding cut would make 200)
    assert len(starts) <= 4


def test_one_line_longer_than_the_budget_is_still_sent():
    text, kept, total = local.fit_transcript("x" * 50 + "\n" + "y" * 500, 100)
    assert kept == 1 and total == 2 and text.endswith("y" * 500)


def test_ask_trims_and_says_so(monkeypatch):
    monkeypatch.setenv("LOCAL_LLM_MAX_TRANSCRIPT_CHARS", "5000")
    fake = _use(monkeypatch, _Fake(_response("Recent stuff.", prompt=2000)))
    r = local.ask(_lines(500), "What happened?")
    assert "most recent" in r.answer and "of 500 transcript lines" in r.answer
    system = fake.calls[0]["messages"][0]["content"]
    assert local.TRIM_MARK in system and len(system) < 8000


def test_max_transcript_setting(monkeypatch):
    assert local.max_transcript_chars() == local.DEFAULT_MAX_TRANSCRIPT_CHARS
    monkeypatch.setenv("LOCAL_LLM_MAX_TRANSCRIPT_CHARS", "0")
    assert local.max_transcript_chars() == 0
    monkeypatch.setenv("LOCAL_LLM_MAX_TRANSCRIPT_CHARS", "lots")
    assert local.max_transcript_chars() == local.DEFAULT_MAX_TRANSCRIPT_CHARS


def _broken():
    return SimpleNamespace(
        choices=[SimpleNamespace(message=_message("", None), finish_reason=None)],
        usage=SimpleNamespace(
            prompt_tokens=0, completion_tokens=0, prompt_tokens_details=None
        ),
        model="",
    )


def test_an_empty_200_is_a_server_failure_not_an_empty_answer(monkeypatch):
    _use(monkeypatch, _Fake(_broken()))
    with pytest.raises(local.AskFailed) as e:
        local.ask("t", "q")
    assert "ollama stop gemma4:26b" in e.value.user_message


def test_a_broken_server_during_the_gm_check_says_so(monkeypatch):
    monkeypatch.setattr(guard, "check", REAL_GUARD_CHECK)
    _use(monkeypatch, _Fake(_broken()))
    with pytest.raises(local.AskFailed) as e:
        local.ask("t", "What happened?")
    assert "server failed" in e.value.user_message


def test_malformed_tool_arguments_are_an_error_the_model_reads(monkeypatch, notes_dir):
    """roborev #6541: local servers don't enforce the schema."""
    bad = SimpleNamespace(
        id="c1",
        function=SimpleNamespace(
            name=local.campaign.TOOL_NAME,
            arguments=json.dumps({"ids": [{"id": "npcs"}]}),
        ),
    )
    fake = _use(
        monkeypatch,
        _Fake(_response("", [bad], "tool_calls"), _response("Couldn't read that.")),
    )
    r = local.ask("t", "Who?", note_dirs=[notes_dir])
    assert r.answer == "Couldn't read that." and r.notes_used == []
    tool = [m for m in fake.calls[1]["messages"] if m["role"] == "tool"][0]
    assert tool["content"].startswith("Error")
