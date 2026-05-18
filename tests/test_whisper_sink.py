"""Tests for WhisperSink file-handle lifecycle.

These cover the roborev #514 fixes: lazy file open, race-safe close, and
no empty file created when a sink is constructed but never used.
"""

import os
import threading
import wave
from unittest.mock import MagicMock

from src.sinks.whisper_sink import Speaker, WhisperSink


def _make_sink(tmp_path, monkeypatch):
    """Construct a WhisperSink with cwd pointed at a temp dir so the
    transcripts/ side effect lands there."""
    monkeypatch.chdir(tmp_path)
    sink = WhisperSink(
        transcript_queue=MagicMock(),
        loop=MagicMock(),
        data_length=50000,
        max_speakers=10,
        transcriber_type="local",
    )
    return sink


def test_constructed_sink_does_not_create_file(tmp_path, monkeypatch):
    sink = _make_sink(tmp_path, monkeypatch)
    assert not os.path.exists(sink.session_file), \
        "session file should not exist until first write"
    sink.close()


def test_lazy_open_creates_file_on_first_write(tmp_path, monkeypatch):
    sink = _make_sink(tmp_path, monkeypatch)
    fh = sink._get_session_fh()
    assert fh is not None
    fh.write("hello\n")
    fh.flush()
    assert os.path.exists(sink.session_file)
    sink.close()


def test_get_session_fh_returns_none_after_close(tmp_path, monkeypatch):
    sink = _make_sink(tmp_path, monkeypatch)
    sink._get_session_fh()  # open it
    sink.close()
    assert sink._get_session_fh() is None, \
        "after close, lazy-open must refuse to re-open the file"


def test_session_fh_writable_during_stop_drain(tmp_path, monkeypatch):
    """Drain-on-stop fix: a transcription that finishes during /stop
    teardown (running already False because stop_voice_thread ran, but
    the sink not yet close()d) must still open/write the per-session
    .txt. Previously _get_session_fh() returned None whenever
    `running` was False, so a session whose only speech transcribed
    just after /stop got NO .txt file (the JSON log still had it)."""
    sink = _make_sink(tmp_path, monkeypatch)
    sink.running = False  # what stop_voice_thread() sets; close() not called yet
    fh = sink._get_session_fh()
    assert fh is not None, "drain-window write must still get a file handle"
    fh.write("drained line\n")
    fh.flush()
    assert os.path.exists(sink.session_file)
    sink.close()
    assert sink._get_session_fh() is None, \
        "once close() finalizes the file, further writes are refused"


def test_close_is_idempotent(tmp_path, monkeypatch):
    sink = _make_sink(tmp_path, monkeypatch)
    sink._get_session_fh()
    sink.close()
    # Second close must not raise. running is already False, fh is None.
    sink.close()


# ── write() normalization for the DAVE voice-receive API ──────────────
# Pycord's fix/voice-rec-2 (DAVE) branch delivers a VoiceData object
# (decrypted+decoded) and passes the User/Member, not raw bytes + int.
# Live test on 2026-05-18 hit `TypeError: object of type 'VoiceData'
# has no len()`. These guard the normalization back to (bytes, int-id).

class _FakeUser:
    def __init__(self, uid):
        self.id = uid


class _FakeVoiceData:
    def __init__(self, pcm, source):
        self.pcm = pcm
        self.source = source


def test_write_normalizes_voicedata_and_user(tmp_path, monkeypatch):
    sink = _make_sink(tmp_path, monkeypatch)
    vd = _FakeVoiceData(pcm=b"\x01\x02\x03\x04", source=_FakeUser(4242))
    sink.write(vd, vd.source)
    item = sink.voice_queue.get_nowait()
    assert item[0] == 4242, "user must be the int id, not the User object"
    assert item[1] == b"\x01\x02\x03\x04", "must queue the decoded PCM bytes"
    assert isinstance(item[2], float)
    sink.close()


def test_write_backward_compatible_with_bytes_and_int(tmp_path, monkeypatch):
    """getattr fallbacks keep the old (bytes, int) contract working."""
    sink = _make_sink(tmp_path, monkeypatch)
    sink.write(b"\xaa\xbb", 999)
    item = sink.voice_queue.get_nowait()
    assert item[0] == 999
    assert item[1] == b"\xaa\xbb"
    sink.close()


def test_write_drops_unattributed_or_empty(tmp_path, monkeypatch):
    sink = _make_sink(tmp_path, monkeypatch)
    sink.write(_FakeVoiceData(pcm=b"abc", source=None), None)
    sink.write(_FakeVoiceData(pcm=b"", source=_FakeUser(1)), _FakeUser(1))
    assert sink.voice_queue.empty(), \
        "frames with no source or no audio must be dropped"
    sink.close()


def test_write_trims_to_data_length(tmp_path, monkeypatch):
    sink = _make_sink(tmp_path, monkeypatch)
    sink.data_length = 4
    vd = _FakeVoiceData(pcm=b"0123456789", source=_FakeUser(7))
    sink.write(vd, vd.source)
    item = sink.voice_queue.get_nowait()
    assert item[1] == b"6789", "oversized buffer keeps only the last data_length bytes"
    sink.close()


def test_transcribe_writes_discord_pcm_wav_header(tmp_path, monkeypatch):
    """roborev #782 (LOW): transcribe() now sources the PCM format from
    discord.opus.Decoder instead of the removed self.vc.decoder. Assert
    the emitted WAV header is Discord's fixed 48kHz / 16-bit / stereo so
    the value-equivalence of the swap is guarded (no whisper model
    loaded — transcribe_audio is stubbed to capture the header)."""
    sink = _make_sink(tmp_path, monkeypatch)
    captured = {}

    def _capture(wav_io):
        wav_io.seek(0)
        with wave.open(wav_io, "rb") as w:
            captured["channels"] = w.getnchannels()
            captured["sampwidth"] = w.getsampwidth()
            captured["framerate"] = w.getframerate()
        return ""

    monkeypatch.setattr(sink, "transcribe_audio", _capture)
    spk = Speaker(user=1, player="p", character="c", data=b"\x00\x01" * 480)
    sink.transcribe(spk)

    assert captured == {"channels": 2, "sampwidth": 2, "framerate": 48000}
    sink.close()


def test_stop_voice_thread_safe_when_vc_none(tmp_path, monkeypatch):
    """roborev #784 / live bug: stop_voice_thread's finally referenced
    self.vc.channel.guild.id, which AttributeErrors after /disconnect
    (self.vc is None, no voice_thread started) and crashed
    /stop|/disconnect via cleanup_sink. Must be no-op-safe."""
    sink = _make_sink(tmp_path, monkeypatch)
    assert sink.vc is None  # _make_sink never starts recording
    sink.stop_voice_thread()  # must not raise
    sink.close()


def test_stop_voice_thread_uses_bounded_join(tmp_path, monkeypatch):
    """Regression: an unbounded voice_thread.join() blocked the asyncio
    event loop during /stop teardown, so a concurrent interaction (an
    impatient second /stop) got 'The application did not respond'. The
    join must be time-bounded so teardown can't wedge the loop forever."""
    sink = _make_sink(tmp_path, monkeypatch)

    class _FakeThread:
        def __init__(self):
            self.join_timeout = "NOT CALLED"

        def join(self, timeout=None):
            self.join_timeout = timeout

        def is_alive(self):
            return False

    ft = _FakeThread()
    sink.voice_thread = ft
    sink.stop_voice_thread()  # must not raise / hang
    assert ft.join_timeout == sink.JOIN_TIMEOUT_S, \
        "join() must be called with a finite timeout, not unbounded"
    sink.close()


def test_concurrent_lazy_open_only_opens_once(tmp_path, monkeypatch):
    """Multiple executor workers may race to first-write. The lock must
    ensure exactly one file handle is opened."""
    sink = _make_sink(tmp_path, monkeypatch)
    handles = []

    def grab():
        handles.append(sink._get_session_fh())

    threads = [threading.Thread(target=grab) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # All threads got the same handle (the one stored on the sink).
    assert all(h is sink._session_fh for h in handles)
    assert len({id(h) for h in handles}) == 1
    sink.close()
