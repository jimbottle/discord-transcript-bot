"""Unit tests for VoloBot.end_recording_session.

VoloBot is a discord.Bot subclass and can't be cheaply constructed in a
unit test, but end_recording_session only touches guild_is_recording and
delegates to stop_recording / cleanup_sink — so we exercise the real
method as an unbound function with a lightweight fake `self`.

Regression guard for the live-test bug (2026-05-18): /disconnect while
recording left guild_is_recording stuck True, so a reconnect + /scribe
failed with "Already recording" and the sink leaked.
"""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import MagicMock

from discord.sinks.errors import RecordingException

from src.bot import volo_bot as vb_mod
from src.bot.volo_bot import VoloBot


class _FakeVC:
    def __init__(self, recording=True, raise_on_stop=None):
        self._recording = recording
        self._raise = raise_on_stop
        self.stopped = False

    def is_recording(self):
        return self._recording

    def stop_recording(self):
        if self._raise:
            raise self._raise
        self.stopped = True


def _ctx_with_vc(guild_id, vc):
    return SimpleNamespace(
        guild_id=guild_id,
        guild=SimpleNamespace(voice_client=vc),
    )


def _fake_self(recording_map):
    return SimpleNamespace(
        guild_is_recording=recording_map,
        stop_recording=MagicMock(),
        cleanup_sink=MagicMock(),
        # end_recording_session writes runtime state at the end; stub it
        # (real VoloBot has the internally-guarded method).
        _write_runtime_state=MagicMock(),
    )


def test_end_recording_session_tears_down_when_recording():
    fake = _fake_self({42: True})
    ctx = SimpleNamespace(guild_id=42)

    asyncio.run(VoloBot.end_recording_session(fake, ctx))

    fake.stop_recording.assert_called_once_with(ctx)
    fake.cleanup_sink.assert_called_once_with(ctx)
    assert (
        fake.guild_is_recording[42] is False
    ), "recording flag must be cleared so a later /scribe doesn't see 'Already recording'"


def test_end_recording_session_is_noop_when_not_recording():
    fake = _fake_self({})
    ctx = SimpleNamespace(guild_id=99)

    asyncio.run(VoloBot.end_recording_session(fake, ctx))

    fake.stop_recording.assert_not_called()
    fake.cleanup_sink.assert_not_called()
    assert fake.guild_is_recording == {}


def test_end_recording_session_noop_when_flag_explicitly_false():
    fake = _fake_self({7: False})
    ctx = SimpleNamespace(guild_id=7)

    asyncio.run(VoloBot.end_recording_session(fake, ctx))

    fake.stop_recording.assert_not_called()
    fake.cleanup_sink.assert_not_called()


# ── teardown resilience (roborev #784 + live /stop|/disconnect bugs) ───


def test_stop_recording_swallows_recording_exception():
    """vc exists but isn't actually recording -> Pycord raises
    RecordingException; stop_recording must swallow it and continue."""
    vc = _FakeVC(
        recording=True, raise_on_stop=RecordingException("You are not recording")
    )
    fake = SimpleNamespace(guild_is_recording={5: True}, guild_whisper_message_tasks={})
    VoloBot.stop_recording(fake, _ctx_with_vc(5, vc))
    assert fake.guild_is_recording[5] is False


def test_stop_recording_skips_when_not_recording():
    vc = _FakeVC(recording=False)
    fake = SimpleNamespace(guild_is_recording={5: True}, guild_whisper_message_tasks={})
    VoloBot.stop_recording(fake, _ctx_with_vc(5, vc))
    assert vc.stopped is False  # stop_recording() not invoked
    assert fake.guild_is_recording[5] is False


def test_stop_recording_no_vc_is_safe():
    fake = SimpleNamespace(guild_is_recording={}, guild_whisper_message_tasks={})
    VoloBot.stop_recording(fake, _ctx_with_vc(5, None))  # must not raise


def test_end_recording_session_resilient_when_stop_raises():
    """A failing stop_recording must not leave the guild stuck recording
    or propagate into the /stop|/disconnect command (interaction 404)."""
    fake = SimpleNamespace(
        guild_is_recording={9: True},
        stop_recording=MagicMock(side_effect=RuntimeError("boom")),
        cleanup_sink=MagicMock(),
        _write_runtime_state=MagicMock(),
    )
    ctx = SimpleNamespace(guild_id=9)
    asyncio.run(VoloBot.end_recording_session(fake, ctx))  # must not raise
    assert fake.guild_is_recording[9] is False
    fake.cleanup_sink.assert_called_once_with(ctx)


def test_close_and_clean_sink_robust_when_stop_thread_raises():
    sink = SimpleNamespace(
        stop_voice_thread=MagicMock(side_effect=RuntimeError("thread boom")),
        close=MagicMock(),
    )
    fake = SimpleNamespace(guild_whisper_sinks={3: sink})
    VoloBot._close_and_clean_sink_for_guild(fake, 3)  # must not raise
    assert 3 not in fake.guild_whisper_sinks, "sink popped even on failure"
    sink.close.assert_called_once()  # close still attempted


# ── _write_runtime_state: dashboard runtime-state file (bot-state) ─────
# Best-effort snapshot written on lifecycle transitions, including the
# hardened /stop|/disconnect teardown path — it must NEVER raise.


def _vc(channel_name):
    return SimpleNamespace(channel=SimpleNamespace(name=channel_name))


def _write_self(**kw):
    base = dict(
        guild_to_helper={},
        guild_whisper_sinks={},
        guild_is_recording={},
        _started_at=1000.0,
        get_guild=lambda gid: SimpleNamespace(name="personal"),
    )
    base.update(kw)
    return SimpleNamespace(**base)


def test_write_runtime_state_connected_and_recording(tmp_path, monkeypatch):
    monkeypatch.setattr(vb_mod, "BOT_STATE_FILE", str(tmp_path / "bot_state.json"))
    fake = _write_self(
        guild_to_helper={123: SimpleNamespace(vc=_vc("voice-1"))},
        guild_whisper_sinks={123: SimpleNamespace(session_file="/x/2026-05-18_18.txt")},
        guild_is_recording={123: True},
    )
    VoloBot._write_runtime_state(fake)
    state = json.loads((tmp_path / "bot_state.json").read_text())
    assert state["started_at"] == 1000.0
    assert "updated_at" in state
    assert state["guilds"] == [
        {
            "guild_id": 123,
            "guild": "personal",
            "channel": "voice-1",
            "recording": True,
            "session_file": "2026-05-18_18.txt",
        }
    ]


def test_write_runtime_state_no_guilds(tmp_path, monkeypatch):
    monkeypatch.setattr(vb_mod, "BOT_STATE_FILE", str(tmp_path / "bot_state.json"))
    VoloBot._write_runtime_state(_write_self())  # must not raise
    state = json.loads((tmp_path / "bot_state.json").read_text())
    assert state["guilds"] == []
    assert state["started_at"] == 1000.0


def test_write_runtime_state_skips_bad_guild_keeps_good(tmp_path, monkeypatch):
    monkeypatch.setattr(vb_mod, "BOT_STATE_FILE", str(tmp_path / "bot_state.json"))

    class _Boom:
        @property
        def vc(self):
            raise RuntimeError("stale helper")

    fake = _write_self(
        guild_to_helper={1: _Boom(), 2: SimpleNamespace(vc=_vc("ok"))},
        guild_is_recording={2: False},
    )
    VoloBot._write_runtime_state(fake)  # must not raise
    state = json.loads((tmp_path / "bot_state.json").read_text())
    gids = [g["guild_id"] for g in state["guilds"]]
    assert gids == [2]  # bad guild skipped, good one kept


def test_write_runtime_state_swallows_write_failure(tmp_path, monkeypatch):
    # Outer guard: a serialization/IO failure must be swallowed (this
    # runs in the hardened teardown path).
    monkeypatch.setattr(vb_mod, "BOT_STATE_FILE", str(tmp_path / "bot_state.json"))
    monkeypatch.setattr(
        vb_mod.json, "dump", lambda *a, **k: (_ for _ in ()).throw(OSError("disk full"))
    )
    VoloBot._write_runtime_state(_write_self())  # must not raise
    assert not (tmp_path / "bot_state.json").exists()
    # roborev #812: a failed write must not leave an orphaned .tmp.
    assert not (tmp_path / "bot_state.json.tmp").exists()
