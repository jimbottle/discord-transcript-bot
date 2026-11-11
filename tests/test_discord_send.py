"""/ask follow-up visibility (src/utils/discord_send.py, roborev #6486).
Only the first follow-up after a deferred response inherits its
visibility; every send must carry ephemeral explicitly."""

import asyncio
import pathlib
import re

from src.utils.discord_send import send_followups


class _Followup:
    def __init__(self, fail_on=None):
        self.sent, self.fail_on = [], fail_on

    async def send(self, content, **kw):
        if self.fail_on is not None and len(self.sent) == self.fail_on:
            raise RuntimeError("discord down")
        self.sent.append((content, kw))


class _Ctx:
    def __init__(self, **kw):
        self.followup = _Followup(**kw)


def test_every_part_of_a_private_answer_is_ephemeral():
    ctx = _Ctx()
    asyncio.run(send_followups(ctx, ["one", "two", "three"], ephemeral=True))
    assert [c for c, _ in ctx.followup.sent] == ["one", "two", "three"]
    assert all(kw == {"ephemeral": True} for _, kw in ctx.followup.sent)


def test_public_answers_stay_public():
    ctx = _Ctx()
    asyncio.run(send_followups(ctx, ["a", "b"], ephemeral=False))
    assert all(kw == {"ephemeral": False} for _, kw in ctx.followup.sent)


def test_a_failed_send_stops_and_raises():
    ctx = _Ctx(fail_on=1)
    try:
        asyncio.run(send_followups(ctx, ["a", "b", "c"], ephemeral=True))
        raise AssertionError("expected the send failure to propagate")
    except RuntimeError:
        pass
    assert [c for c, _ in ctx.followup.sent] == ["a"]


def test_ask_never_sends_a_followup_directly():
    """Every /ask follow-up must go through send_followups so none can
    silently default to public."""
    main = (pathlib.Path(__file__).resolve().parent.parent / "main.py").read_text()
    start = main.index("async def _run_ask(")
    end = main.index("@bot.slash_command(", start)
    assert not re.search(r"ctx\.followup\.send\(", main[start:end])


def test_earlier_sessions_are_read_off_the_event_loop_after_the_defer():
    """roborev #7119: building the earlier-session notes reads every earlier
    transcript of the campaign. In /ask that must happen after ctx.defer and
    inside the function handed to asyncio.to_thread, never on the event loop
    (no slash-command harness, so this checks the source order)."""
    main = (pathlib.Path(__file__).resolve().parent.parent / "main.py").read_text()
    assert main.count("earlier_session_notes(") == 1
    defer = main.index("await ctx.defer(ephemeral=private)")
    worker = main.index("def _answer():", defer)
    build = main.index("sessions.earlier_session_notes(session_file)")
    handoff = main.index("await asyncio.to_thread(_answer)")
    assert defer < worker < build < handoff
