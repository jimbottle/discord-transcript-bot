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
import os
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import yaml
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


def _vc(channel_name, members=None):
    return SimpleNamespace(
        channel=SimpleNamespace(name=channel_name, members=members or [])
    )


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
            "output_age": None,  # session_file path doesn't exist on disk
            "queue_depth": 0,  # fake sink has no voice_queue
            "stalled": False,
            "members": [],
        }
    ]


class _FakeQueue:
    def __init__(self, depth):
        self._depth = depth

    def qsize(self):
        return self._depth


def _fake_sink(session_file, queue_depth):
    return SimpleNamespace(
        session_file=session_file, voice_queue=_FakeQueue(queue_depth)
    )


def _aged_session(tmp_path, age_seconds):
    f = tmp_path / "session.txt"
    f.write_text("x", encoding="utf-8")
    t = time.time() - age_seconds
    os.utime(f, (t, t))
    return str(f)


def test_write_runtime_state_detects_stall(tmp_path, monkeypatch):
    # Recording + a backed-up queue + a session file that stopped growing =
    # the wedged-pipeline case the heartbeat must flag (today's silent death).
    monkeypatch.setattr(vb_mod, "BOT_STATE_FILE", str(tmp_path / "bot_state.json"))
    sess = _aged_session(tmp_path, vb_mod.STALL_OUTPUT_AGE_S + 60)
    fake = _write_self(
        guild_to_helper={5: SimpleNamespace(vc=_vc("voice"))},
        guild_whisper_sinks={5: _fake_sink(sess, vb_mod.STALL_QUEUE_DEPTH + 5)},
        guild_is_recording={5: True},
    )
    stalled = VoloBot._write_runtime_state(fake)
    assert stalled == [5]
    g = json.loads((tmp_path / "bot_state.json").read_text())["guilds"][0]
    assert g["stalled"] is True
    assert g["queue_depth"] >= vb_mod.STALL_QUEUE_DEPTH
    assert g["output_age"] > vb_mod.STALL_OUTPUT_AGE_S


def test_write_runtime_state_quiet_room_not_stalled(tmp_path, monkeypatch):
    # Old session file but a shallow queue = nobody's talking, not a stall.
    monkeypatch.setattr(vb_mod, "BOT_STATE_FILE", str(tmp_path / "bot_state.json"))
    sess = _aged_session(tmp_path, vb_mod.STALL_OUTPUT_AGE_S + 60)
    fake = _write_self(
        guild_to_helper={5: SimpleNamespace(vc=_vc("voice"))},
        guild_whisper_sinks={5: _fake_sink(sess, 0)},
        guild_is_recording={5: True},
    )
    assert VoloBot._write_runtime_state(fake) == []
    g = json.loads((tmp_path / "bot_state.json").read_text())["guilds"][0]
    assert g["stalled"] is False


def test_write_runtime_state_not_recording_never_stalls(tmp_path, monkeypatch):
    # A deep queue + old file but recording False (e.g. just /stopped) is
    # not a stall — there's no active session to wedge.
    monkeypatch.setattr(vb_mod, "BOT_STATE_FILE", str(tmp_path / "bot_state.json"))
    sess = _aged_session(tmp_path, vb_mod.STALL_OUTPUT_AGE_S + 60)
    fake = _write_self(
        guild_to_helper={5: SimpleNamespace(vc=_vc("voice"))},
        guild_whisper_sinks={5: _fake_sink(sess, vb_mod.STALL_QUEUE_DEPTH + 5)},
        guild_is_recording={5: False},
    )
    assert VoloBot._write_runtime_state(fake) == []


def test_write_runtime_state_includes_voice_members(tmp_path, monkeypatch):
    # The dashboard's roster editor reads `members` to offer naming people
    # who are present on the call (even if they haven't spoken).
    monkeypatch.setattr(vb_mod, "BOT_STATE_FILE", str(tmp_path / "bot_state.json"))
    members = [
        SimpleNamespace(id=7, name="ed", display_name="Ed"),
        SimpleNamespace(id=8, name="cody", display_name="Cody B"),
    ]
    fake = _write_self(
        guild_to_helper={123: SimpleNamespace(vc=_vc("voice-1", members=members))},
        guild_is_recording={123: True},
    )
    VoloBot._write_runtime_state(fake)
    state = json.loads((tmp_path / "bot_state.json").read_text())
    assert state["guilds"][0]["members"] == [
        {"id": 7, "name": "ed", "display_name": "Ed"},
        {"id": 8, "name": "cody", "display_name": "Cody B"},
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


# ── upsert_player_entry: name someone on the call (add/update mapping) ─


def test_upsert_player_entry_in_memory_only_when_no_file(monkeypatch):
    monkeypatch.setattr(vb_mod, "PLAYER_MAP_FILE_PATH", None)
    shared = {}  # the dict a running WhisperSink would hold
    fake = SimpleNamespace(player_map=shared)
    persisted = VoloBot.upsert_player_entry(fake, 42, "Ed", "Volo")
    assert persisted is False
    # mutated IN PLACE (int key) so the live session sees it
    assert shared is fake.player_map
    assert shared[42] == {"player": "Ed", "character": "Volo"}


def test_upsert_player_entry_persists_and_preserves_others(tmp_path, monkeypatch):
    pm = tmp_path / "player_map.yml"
    pm.write_text(
        yaml.dump({999: {"player": "Existing", "character": "Keep"}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(vb_mod, "PLAYER_MAP_FILE_PATH", str(pm))
    fake = SimpleNamespace(player_map={})
    persisted = VoloBot.upsert_player_entry(fake, 7, "Cody", "Jim")
    assert persisted is True
    on_disk = yaml.safe_load(pm.read_text())
    assert on_disk[7] == {"player": "Cody", "character": "Jim"}
    assert on_disk[999] == {"player": "Existing", "character": "Keep"}  # preserved
    assert fake.player_map[7] == {"player": "Cody", "character": "Jim"}


def test_upsert_player_entry_updates_existing(tmp_path, monkeypatch):
    pm = tmp_path / "player_map.yml"
    monkeypatch.setattr(vb_mod, "PLAYER_MAP_FILE_PATH", str(pm))
    fake = SimpleNamespace(player_map={7: {"player": "old", "character": "old"}})
    VoloBot.upsert_player_entry(fake, "7", "New", "NewChar")  # str id coerced
    assert fake.player_map[7] == {"player": "New", "character": "NewChar"}
    assert yaml.safe_load(pm.read_text())[7] == {
        "player": "New",
        "character": "NewChar",
    }


def test_upsert_player_entry_refuses_non_dict_yaml(tmp_path, monkeypatch):
    """roborev #824: valid-but-non-mapping roster YAML must not be
    clobbered; the in-memory change still applies for the session."""
    pm = tmp_path / "player_map.yml"
    pm.write_text(yaml.dump(["not", "a", "mapping"]), encoding="utf-8")
    monkeypatch.setattr(vb_mod, "PLAYER_MAP_FILE_PATH", str(pm))
    fake = SimpleNamespace(player_map={})
    with pytest.raises(ValueError):
        VoloBot.upsert_player_entry(fake, 7, "A", "B")
    assert fake.player_map[7] == {"player": "A", "character": "B"}  # live
    assert yaml.safe_load(pm.read_text()) == ["not", "a", "mapping"]  # untouched


def test_upsert_player_entry_atomic_no_orphan_on_write_failure(tmp_path, monkeypatch):
    """roborev #824: a mid-write failure must not corrupt the roster or
    leave a .tmp orphan (atomic tmp + os.replace)."""
    pm = tmp_path / "player_map.yml"
    pm.write_text(
        yaml.dump({1: {"player": "Keep", "character": "Keep"}}), encoding="utf-8"
    )
    monkeypatch.setattr(vb_mod, "PLAYER_MAP_FILE_PATH", str(pm))
    monkeypatch.setattr(
        vb_mod.yaml, "dump", lambda *a, **k: (_ for _ in ()).throw(OSError("disk full"))
    )
    fake = SimpleNamespace(player_map={})
    with pytest.raises(OSError):
        VoloBot.upsert_player_entry(fake, 7, "X", "Y")
    assert not (tmp_path / "player_map.yml.tmp").exists()  # no orphan
    # original roster intact — os.replace never happened
    assert yaml.safe_load(pm.read_text()) == {
        1: {"player": "Keep", "character": "Keep"}
    }
    assert fake.player_map[7] == {"player": "X", "character": "Y"}  # in-memory applied
