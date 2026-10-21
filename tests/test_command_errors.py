"""Slash-command failures always reach the user (discord-transcript-bot-8gy)."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import discord
import pytest

from src.bot import command_errors as ce
from src.bot.volo_bot import VoloBot


def _http(cls, status):
    resp = SimpleNamespace(status=status, reason="x")
    return cls(resp, "nope")


@pytest.mark.parametrize(
    "error, needle",
    [
        (asyncio.TimeoutError(), "didn't answer in time"),
        (TimeoutError(), "didn't answer in time"),
        (discord.CheckFailure(), "can't use that command"),
        (_http(discord.Forbidden, 403), "permission"),
        (_http(discord.HTTPException, 500), "HTTP 500"),
        (discord.ClientException("Already connected"), "Already connected"),
        (discord.ClientException(""), "Discord client error"),
        (RuntimeError("boom"), "bot logs"),
    ],
)
def test_user_message_is_never_empty(error, needle):
    msg = ce.user_message(error)
    assert needle in msg and msg.strip()


def test_user_message_unwraps_the_invoke_error():
    wrapped = SimpleNamespace(original=asyncio.TimeoutError())
    assert "in time" in ce.user_message(wrapped)


def _ctx(done):
    return SimpleNamespace(
        command=SimpleNamespace(qualified_name="health"),
        response=SimpleNamespace(is_done=lambda: done),
        respond=AsyncMock(),
        followup=SimpleNamespace(send=AsyncMock()),
    )


def test_report_responds_when_nothing_was_sent_yet():
    ctx = _ctx(done=False)
    asyncio.run(ce.report(ctx, RuntimeError("x")))
    ctx.respond.assert_awaited_once()
    ctx.followup.send.assert_not_awaited()


def test_report_follows_up_after_a_defer():
    ctx = _ctx(done=True)
    asyncio.run(ce.report(ctx, RuntimeError("x")))
    ctx.followup.send.assert_awaited_once()
    assert ctx.followup.send.call_args.kwargs["ephemeral"] is True


def test_report_never_raises_when_sending_fails():
    ctx = _ctx(done=True)
    ctx.followup.send.side_effect = discord.ClientException("gone")
    asyncio.run(ce.report(ctx, RuntimeError("x")))  # must not raise


def test_volo_bot_routes_command_errors_to_report(monkeypatch):
    seen = []

    async def fake_report(ctx, error):
        seen.append((ctx, error))

    monkeypatch.setattr(ce, "report", fake_report)
    err = RuntimeError("x")
    asyncio.run(VoloBot.on_application_command_error(MagicMock(), "ctx", err))
    assert seen == [("ctx", err)]
