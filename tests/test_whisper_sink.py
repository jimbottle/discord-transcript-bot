"""Tests for WhisperSink file-handle lifecycle.

These cover the roborev #514 fixes: lazy file open, race-safe close, and
no empty file created when a sink is constructed but never used.
"""

import io
import os
import threading
import time
import wave
from concurrent.futures import Future
from types import SimpleNamespace
from unittest.mock import MagicMock

import src.sinks.whisper_sink as ws
from src.asr.base import NormalizedSegment, TranscribeResult
from src.sinks.whisper_sink import DEFAULT_INITIAL_PROMPT, Speaker, WhisperSink


def _fake_backend(segments):
    """A stand-in transcription backend whose transcribe() returns the given
    segments wrapped in a TranscribeResult — replaces sink._backend so the
    sink's accuracy filters can be exercised without loading a model."""
    return SimpleNamespace(
        transcribe=lambda *a, **k: TranscribeResult(segments=list(segments), info=None)
    )


def _wav_bytesio(seconds=0.5):
    """A valid Discord-format (48kHz/16-bit/stereo) silent WAV in memory,
    long enough to pass transcribe_audio's >0.1s length guard."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(48000)
        w.writeframes(b"\x00\x00\x00\x00" * int(48000 * seconds))
    buf.seek(0)
    return buf


def _done_future(value=None, exc=None):
    """A pre-completed Future, as add_done_callback would hand to the
    commit callback."""
    f = Future()
    if exc is not None:
        f.set_exception(exc)
    else:
        f.set_result(value)
    return f


def _make_sink(tmp_path, monkeypatch, player_map=None):
    """Construct a WhisperSink with cwd pointed at a temp dir so the
    transcripts/ side effect lands there."""
    monkeypatch.chdir(tmp_path)
    sink = WhisperSink(
        transcript_queue=MagicMock(),
        loop=MagicMock(),
        data_length=50000,
        max_speakers=10,
        transcriber_type="local",
        player_map=player_map or {},
    )
    return sink


def test_constructed_sink_does_not_create_file(tmp_path, monkeypatch):
    sink = _make_sink(tmp_path, monkeypatch)
    assert not os.path.exists(
        sink.session_file
    ), "session file should not exist until first write"
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
    assert (
        sink._get_session_fh() is None
    ), "after close, lazy-open must refuse to re-open the file"


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
    assert (
        sink._get_session_fh() is None
    ), "once close() finalizes the file, further writes are refused"


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
    assert sink.voice_queue.empty(), "frames with no source or no audio must be dropped"
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
    assert (
        ft.join_timeout == sink.JOIN_TIMEOUT_S
    ), "join() must be called with a finite timeout, not unbounded"
    sink.close()


def test_stop_voice_thread_warns_and_returns_when_thread_wont_die(
    tmp_path, monkeypatch, caplog
):
    """roborev #788 (LOW): cover the is_alive()==True branch — a thread
    that doesn't exit within the timeout must be abandoned (logged
    warning) and stop_voice_thread() must still return promptly, not
    hang. The thread is a daemon so abandoning it is safe."""
    import logging

    sink = _make_sink(tmp_path, monkeypatch)

    class _StuckThread:
        def join(self, timeout=None):
            return  # returns immediately (timeout elapsed) but...

        def is_alive(self):
            return True  # ...still alive

    sink.voice_thread = _StuckThread()
    with caplog.at_level(logging.WARNING):
        sink.stop_voice_thread()  # must return, not hang
    assert any(
        "did not exit" in r.message for r in caplog.records
    ), "a non-exiting voice thread must log a warning"
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


# ── Roster -> initial_prompt proper-noun biasing (discord-transcript-bot-cul) ──


def test_initial_prompt_includes_roster_names(tmp_path, monkeypatch):
    pm = {
        1: {"player": "Reiko Tanaka", "character": "Gus"},
        2: {"player": "Steve Calderon", "character": "Johan"},
    }
    sink = _make_sink(tmp_path, monkeypatch, player_map=pm)
    p = sink.initial_prompt
    assert p.startswith(DEFAULT_INITIAL_PROMPT)
    for name in ("Gus", "Johan", "Reiko Tanaka", "Steve Calderon"):
        assert name in p, f"{name} should be biased into the prompt"
    # ALL character names precede ALL player names (global, not per-entry):
    # a later entry's character must outrank an earlier entry's player so
    # the char-budget truncation keeps the higher-value (spoken-most) names.
    assert max(p.index("Gus"), p.index("Johan")) < min(
        p.index("Reiko Tanaka"), p.index("Steve Calderon")
    )
    sink.close()


def test_initial_prompt_defaults_without_roster(tmp_path, monkeypatch):
    sink = _make_sink(tmp_path, monkeypatch, player_map={})
    assert sink.initial_prompt == DEFAULT_INITIAL_PROMPT
    sink.close()


def test_initial_prompt_dedupes_case_insensitively(tmp_path, monkeypatch):
    pm = {
        1: {"player": "Sam", "character": "Gus"},
        2: {"player": "sam", "character": "Gus"},  # both dup of entry 1
    }
    sink = _make_sink(tmp_path, monkeypatch, player_map=pm)
    assert sink.initial_prompt.count("Gus") == 1
    assert sink.initial_prompt.lower().count("sam") == 1
    sink.close()


def test_initial_prompt_skips_malformed_and_empty_entries(tmp_path, monkeypatch):
    pm = {
        1: "not-a-dict",
        2: {"player": "", "character": None},
        3: {"character": "Noah"},
    }
    sink = _make_sink(tmp_path, monkeypatch, player_map=pm)
    assert "Noah" in sink.initial_prompt
    sink.close()


def test_initial_prompt_respects_char_budget(tmp_path, monkeypatch):
    pm = {
        i: {
            "player": f"PlayerNameNumber{i:03d}",
            "character": f"CharacterNameNumber{i:03d}",
        }
        for i in range(200)
    }
    sink = _make_sink(tmp_path, monkeypatch, player_map=pm)
    assert len(sink.initial_prompt) <= sink.MAX_INITIAL_PROMPT_CHARS
    assert sink.initial_prompt.startswith(DEFAULT_INITIAL_PROMPT)
    # Truncated whole-name: ends cleanly with a period, no dangling comma.
    assert sink.initial_prompt.endswith(".")
    assert not sink.initial_prompt.rstrip(".").endswith(",")
    sink.close()


# ── Prompt-echo hallucination guard ──────────────────────────────────────
# Whisper regurgitates the roster names from initial_prompt on near-silent
# audio (an 85ms clip transcribed as "Sovereign Lord GM says,"). These guard
# the drop-the-echo filter that kills that leak while keeping real speech.


def test_prompt_echo_drops_bare_and_filler_suffixed_name(tmp_path, monkeypatch):
    pm = {1: {"player": "Sovereign Lord GM", "character": "Noah"}}
    sink = _make_sink(tmp_path, monkeypatch, player_map=pm)
    assert "Sovereign Lord GM" in sink._prompt_names
    assert sink._drop_if_prompt_echo(" Sovereign Lord GM") == ""
    assert sink._drop_if_prompt_echo("Sovereign Lord GM says,") == ""
    assert sink._is_prompt_echo("sovereign lord gm SAID.") is True
    sink.close()


def test_prompt_echo_keeps_real_speech(tmp_path, monkeypatch):
    pm = {1: {"player": "Sovereign Lord GM", "character": "Noah"}}
    sink = _make_sink(tmp_path, monkeypatch, player_map=pm)
    # A name embedded in a real sentence (or with real trailing content) is
    # kept — only a whole-utterance echo is dropped.
    for keep in (
        "Noah, watch out!",
        "I cast fireball",
        "Sovereign Lord GM says we should go now",
    ):
        assert sink._drop_if_prompt_echo(keep) == keep
    sink.close()


def test_prompt_echo_noop_without_roster(tmp_path, monkeypatch):
    sink = _make_sink(tmp_path, monkeypatch, player_map={})
    assert sink._prompt_names == []
    assert sink._is_prompt_echo("Sovereign Lord GM says,") is False
    assert (
        sink._drop_if_prompt_echo("Sovereign Lord GM says,")
        == "Sovereign Lord GM says,"
    )
    sink.close()


def test_transcribe_audio_drops_prompt_echo_end_to_end(tmp_path, monkeypatch):
    pm = {1: {"player": "Sovereign Lord GM", "character": "Noah"}}
    sink = _make_sink(tmp_path, monkeypatch, player_map=pm)
    monkeypatch.setattr(sink, "check_audio_length", lambda _f: 1.0)  # pass length gate
    seg = SimpleNamespace(text=" Sovereign Lord GM says,")
    sink._backend = _fake_backend([seg])
    assert sink.transcribe_audio(io.BytesIO()) == ""
    sink.close()


def test_prompt_echo_drops_multiple_concatenated_names(tmp_path, monkeypatch):
    """The leak isn't only single names: "Sovereign Lord GM, Rahul Patch
    Sarker." is several prompt names run together and must also drop."""
    pm = {
        1: {"player": "Sovereign Lord GM", "character": "Noah"},
        2: {"player": "Rahul Patch Sarker", "character": "William"},
    }
    sink = _make_sink(tmp_path, monkeypatch, player_map=pm)
    assert sink._drop_if_prompt_echo("Sovereign Lord GM, Rahul Patch Sarker.") == ""
    # comma-separated names (how the prompt lists them) drop via punctuation
    # normalization — no article/conjunction filler needed
    assert sink._is_prompt_echo("Noah, William.") is True
    # a real sentence that happens to contain names (with an article between
    # them) is kept — "and"/"the" are NOT fillers, so this survives
    assert sink._drop_if_prompt_echo("Noah and William") != ""
    assert sink._drop_if_prompt_echo("Noah hands William the gun") != ""
    sink.close()


def test_known_whisper_artifact_dropped(tmp_path, monkeypatch):
    sink = _make_sink(tmp_path, monkeypatch, player_map={})
    assert sink._drop_if_prompt_echo("Subtitles by the Amara.org community") == ""
    assert sink._drop_if_prompt_echo("Thank you for watching!") == ""
    assert sink._is_hallucination_phrase("please subscribe") is True
    # a real sentence that merely contains "thank you" is kept (exact-match only)
    assert (
        sink._drop_if_prompt_echo("thank you for the potion")
        == "thank you for the potion"
    )
    sink.close()


class _Seg:
    """Minimal stand-in for a faster-whisper Segment with the quality
    metrics the hallucination filter inspects."""

    def __init__(
        self, text, no_speech_prob=0.0, avg_logprob=0.0, compression_ratio=1.0
    ):
        self.text = text
        self.no_speech_prob = no_speech_prob
        self.avg_logprob = avg_logprob
        self.compression_ratio = compression_ratio


def test_accept_segment_filters_by_metrics(tmp_path, monkeypatch):
    sink = _make_sink(tmp_path, monkeypatch, player_map={})
    keep = _Seg("real speech here", no_speech_prob=0.1, avg_logprob=-0.3)
    assert sink._accept_segment(keep) is True
    assert sink._accept_segment(_Seg("   ")) is False  # empty
    assert sink._accept_segment(_Seg("x", no_speech_prob=0.95)) is False  # silence
    assert sink._accept_segment(_Seg("x", avg_logprob=-1.5)) is False  # low confidence
    assert sink._accept_segment(_Seg("la la la", compression_ratio=3.0)) is False  # rep
    assert (
        sink._accept_segment(_Seg("Subtitles by the Amara.org community")) is False
    )  # known artifact
    sink.close()


def test_transcribe_audio_keeps_only_good_segments(tmp_path, monkeypatch):
    sink = _make_sink(tmp_path, monkeypatch, player_map={})
    monkeypatch.setattr(sink, "check_audio_length", lambda _f: 1.0)
    segs = [
        _Seg(" Real line.", no_speech_prob=0.1, avg_logprob=-0.3),
        _Seg(" Subtitles by the Amara.org community", avg_logprob=-0.3),
        _Seg(" la la la la", compression_ratio=3.5),
        _Seg(" Another real bit.", no_speech_prob=0.1, avg_logprob=-0.2),
    ]
    sink._backend = _fake_backend(segs)
    assert sink.transcribe_audio(io.BytesIO()) == " Real line. Another real bit."
    sink.close()


def test_biasing_disabled_via_env(tmp_path, monkeypatch):
    monkeypatch.setattr(ws, "BIAS_PROMPT_WITH_ROSTER", False)
    pm = {1: {"player": "Sovereign Lord GM", "character": "Noah"}}
    sink = _make_sink(tmp_path, monkeypatch, player_map=pm)
    assert sink.initial_prompt == DEFAULT_INITIAL_PROMPT
    assert sink._prompt_names == []
    sink.close()


def test_env_flag_parsing():
    # unset -> default truthy; explicit falsy tokens (any case) -> False
    assert ws._env_flag(None) is True
    assert ws._env_flag("1") is True
    assert ws._env_flag("true") is True
    assert ws._env_flag("yes") is True
    assert ws._env_flag("anything") is True
    for falsy in ("", "0", "false", "FALSE", "No", " no "):
        assert ws._env_flag(falsy) is False, falsy


# ── Decode params + batched CPU inference reach the model
#    (discord-transcript-bot-std / discord-transcript-bot-ob3) ──


def test_transcribe_audio_forwards_configured_params_to_backend(tmp_path, monkeypatch):
    """beam_size/best_of are read from module-level config (not hardcoded) and,
    with the biased initial_prompt and VAD settings, forwarded to the selected
    backend. (batch_size is a faster-whisper-only knob and is covered in the
    backend's own test.)"""
    sink = _make_sink(
        tmp_path, monkeypatch, player_map={1: {"player": "Sam", "character": "Gus"}}
    )
    captured = {}

    def fake_transcribe(audio, **kwargs):
        captured.update(kwargs)
        return TranscribeResult(
            segments=[NormalizedSegment(text="hello there")], info=None
        )

    sink._backend = SimpleNamespace(transcribe=fake_transcribe)
    monkeypatch.setattr(ws, "WHISPER_BEAM_SIZE", 7)
    monkeypatch.setattr(ws, "WHISPER_BEST_OF", 9)

    out = sink.transcribe_audio(_wav_bytesio(0.5))
    assert out == "hello there"
    assert captured["beam_size"] == 7
    assert captured["best_of"] == 9
    assert captured["initial_prompt"] == sink.initial_prompt
    assert captured["vad_filter"] is True
    assert captured["no_speech_threshold"] == 0.6
    sink.close()


# ── Parallel transcription with in-order commit (discord-transcript-bot-hin) ──
# Transcriptions run concurrently on the executor, but results must be
# WRITTEN in the order their segments were submitted: a chunk that finishes
# early waits behind earlier, still-running chunks so the transcript never
# posts out of chronological order.


def _spk(uid):
    return Speaker(user=uid, player=f"p{uid}", character=f"c{uid}", data=b"\x00")


def test_results_commit_in_submission_order(tmp_path, monkeypatch):
    sink = _make_sink(tmp_path, monkeypatch)
    written = []
    monkeypatch.setattr(
        sink,
        "write_transcription_log",
        lambda spk, text: written.append((spk.user, text)),
    )
    s0, s1, s2 = _spk(0), _spk(1), _spk(2)

    # seq 2 finishes FIRST — it must not post while 0 and 1 are outstanding.
    sink._on_transcribed(2, s2, _done_future("two"))
    assert written == [], "a later segment must wait for earlier ones"

    # seq 0 arrives — commits, but seq 1 still blocks seq 2.
    sink._on_transcribed(0, s0, _done_future("zero"))
    assert written == [(0, "zero")]

    # seq 1 arrives — now 1 then the buffered 2 flush in order.
    sink._on_transcribed(1, s1, _done_future("one"))
    assert written == [(0, "zero"), (1, "one"), (2, "two")]
    sink.close()


def test_failed_future_still_advances_commit_pointer(tmp_path, monkeypatch):
    """A segment whose transcription raised must commit as empty text and
    advance the pointer, so one bad chunk can't wedge every later result."""
    sink = _make_sink(tmp_path, monkeypatch)
    written = []
    monkeypatch.setattr(
        sink,
        "write_transcription_log",
        lambda spk, text: written.append((spk.user, text)),
    )
    sink._on_transcribed(0, _spk(0), _done_future(exc=RuntimeError("boom")))
    assert written == [(0, "")], "failed future commits empty, doesn't wedge"
    sink._on_transcribed(1, _spk(1), _done_future("one"))
    assert written == [(0, ""), (1, "one")]
    sink.close()


def test_submit_failure_does_not_wedge_commit_pointer(tmp_path, monkeypatch):
    """If executor.submit raises after a seq is allocated, that seq must be
    committed empty so _next_commit doesn't wedge — otherwise every later
    result silently stops posting (total commit loss)."""
    sink = _make_sink(tmp_path, monkeypatch)
    sink.SILENCE_GAP_S = 1000
    sink.MAX_SEGMENT_S = 0.0
    written = []
    monkeypatch.setattr(
        sink,
        "write_transcription_log",
        lambda spk, text: written.append((spk.user, text)),
    )

    class _BoomExecutor:
        def submit(self, *a, **k):
            raise RuntimeError("executor is shut down")

    monkeypatch.setattr(sink, "executor", _BoomExecutor())

    t = time.time()
    sink.voice_queue.put_nowait([9, b"aa", t])
    sink.voice_queue.put_nowait([9, b"bb", t])  # new_bytes > 1

    _run_insert_voice(sink, until=lambda: bool(written))
    assert written == [(9, "")], "a failed submit must commit its seq empty"
    assert sink._next_commit == 1, "commit pointer must advance past the failed seq"

    # A later normal result still commits — the pointer was not wedged.
    sink._on_transcribed(1, _spk(1), _done_future("ok"))
    assert written == [(9, ""), (1, "ok")]
    sink.close()


def _run_insert_voice(sink, until, timeout=2.0):
    """Run insert_voice in a thread until `until()` is true (or timeout),
    then stop it and join."""
    sink.running = True
    th = threading.Thread(target=sink.insert_voice, daemon=True)
    th.start()
    deadline = time.time() + timeout
    while not until() and time.time() < deadline:
        time.sleep(0.02)
    sink.running = False
    th.join(timeout=2)
    return th


def test_buffer_force_flushed_past_duration_cap(tmp_path, monkeypatch):
    """A speaker who never pauses (silence gap never fires) must still be
    flushed once the buffer spans MAX_SEGMENT_S, bounding latency."""
    sink = _make_sink(tmp_path, monkeypatch)
    sink.SILENCE_GAP_S = 1000  # silence-based flush effectively disabled
    sink.MAX_SEGMENT_S = 0.0  # any buffered speaker is immediately "too long"
    submitted = []
    monkeypatch.setattr(
        sink, "transcribe", lambda spk: submitted.append(spk.user) or ""
    )

    t = time.time()
    sink.voice_queue.put_nowait([5, b"aa", t])
    sink.voice_queue.put_nowait([5, b"bb", t])  # new_bytes > 1

    _run_insert_voice(sink, until=lambda: bool(submitted))
    assert submitted == [
        5
    ], "buffer must be force-flushed by the duration cap, not only on silence"
    sink.close()


def test_drain_continues_while_a_transcription_is_in_flight(tmp_path, monkeypatch):
    """The snowball fix: the drain loop must NOT block on future.result().
    While one transcription is stuck running, freshly-arriving audio must
    still be drained off voice_queue (bounded depth)."""
    sink = _make_sink(tmp_path, monkeypatch)
    sink.SILENCE_GAP_S = 1000
    sink.MAX_SEGMENT_S = 0.0
    started = threading.Event()
    release = threading.Event()

    def slow_transcribe(spk):
        started.set()
        release.wait(5)
        return ""

    monkeypatch.setattr(sink, "transcribe", slow_transcribe)

    t = time.time()
    sink.voice_queue.put_nowait([1, b"aa", t])
    sink.voice_queue.put_nowait([1, b"bb", t])  # submitted, then blocks

    sink.running = True
    th = threading.Thread(target=sink.insert_voice, daemon=True)
    th.start()
    try:
        assert started.wait(2), "first transcription should have started"
        # Pour in more audio while that transcription is blocked.
        for _ in range(50):
            sink.voice_queue.put_nowait([2, b"xx", time.time()])
        deadline = time.time() + 2
        while not sink.voice_queue.empty() and time.time() < deadline:
            time.sleep(0.02)
        assert (
            sink.voice_queue.empty()
        ), "voice_queue must keep draining while a transcription is in flight"
    finally:
        release.set()
        sink.running = False
        th.join(timeout=2)
    sink.close()
