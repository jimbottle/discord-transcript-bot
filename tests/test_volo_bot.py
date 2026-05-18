"""Unit tests for VoloBot.end_recording_session.

VoloBot is a discord.Bot subclass and can't be cheaply constructed in a
unit test, but end_recording_session only touches guild_is_recording and
delegates to stop_recording / cleanup_sink — so we exercise the real
method as an unbound function with a lightweight fake `self`.

Regression guard for the live-test bug (2026-05-18): /disconnect while
recording left guild_is_recording stuck True, so a reconnect + /scribe
failed with "Already recording" and the sink leaked.
"""
from types import SimpleNamespace
from unittest.mock import MagicMock

from src.bot.volo_bot import VoloBot


def _fake_self(recording_map):
    return SimpleNamespace(
        guild_is_recording=recording_map,
        stop_recording=MagicMock(),
        cleanup_sink=MagicMock(),
    )


def test_end_recording_session_tears_down_when_recording():
    fake = _fake_self({42: True})
    ctx = SimpleNamespace(guild_id=42)

    VoloBot.end_recording_session(fake, ctx)

    fake.stop_recording.assert_called_once_with(ctx)
    fake.cleanup_sink.assert_called_once_with(ctx)
    assert fake.guild_is_recording[42] is False, \
        "recording flag must be cleared so a later /scribe doesn't see 'Already recording'"


def test_end_recording_session_is_noop_when_not_recording():
    fake = _fake_self({})
    ctx = SimpleNamespace(guild_id=99)

    VoloBot.end_recording_session(fake, ctx)

    fake.stop_recording.assert_not_called()
    fake.cleanup_sink.assert_not_called()
    assert fake.guild_is_recording == {}


def test_end_recording_session_noop_when_flag_explicitly_false():
    fake = _fake_self({7: False})
    ctx = SimpleNamespace(guild_id=7)

    VoloBot.end_recording_session(fake, ctx)

    fake.stop_recording.assert_not_called()
    fake.cleanup_sink.assert_not_called()
