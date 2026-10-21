import asyncio
import io
import json
import logging
import os
import re
import threading
import time
import wave
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from queue import Queue
from typing import List

from discord.opus import Decoder
from discord.sinks.core import Filters, Sink, default_filters
from openai import OpenAI

from src.asr import selection
from src.session_capture import SessionCapture

# Model id + device/precision + engine selection now live in src/asr/selection.py
# (configurable via WHISPER_MODEL / ASR_BACKEND; default large-v3, accuracy-first).
WHISPER_LANGUAGE = "en"  # language hint for every engine; no auto-detect

# Decode params. beam_size was 10 / best_of 3 — unusually high and a major
# CPU cost for marginal accuracy; 5 is faster-whisper's own default. Made
# configurable so the accuracy/speed knee can be A/B-tuned on real session
# audio (discord-transcript-bot-61z). discord-transcript-bot-std.
WHISPER_BEAM_SIZE = int(os.getenv("WHISPER_BEAM_SIZE", "5"))
WHISPER_BEST_OF = int(os.getenv("WHISPER_BEST_OF", "5"))


def transcribe_workers(raw=None) -> int:
    """How many transcriptions run at once (discord-transcript-bot-4uk).

    Measured 2026-10-21 on 96 clips of the 2026-10-20 session
    (scripts/bench_concurrency.py, MLX large-v3, M-series GPU), two runs:
    1.45-1.65 clips/s at every count from 1 to 8, within the 9% spread
    between two 1-worker runs; texts identical and no backend errors. So
    concurrent MLX decoding is safe but the GPU serialises it, and more
    workers only add threads and memory. 2 keeps a slow clip from holding
    up the next. On faster-whisper (CPU) each call already uses every core.
    Invalid values fall back to 2.
    """
    raw = os.getenv("TRANSCRIBE_WORKERS", "2") if raw is None else raw
    try:
        n = int(raw)
    except (TypeError, ValueError):
        return 2
    return n if n >= 1 else 2


# WHISPER_BATCH_SIZE (BatchedInferencePipeline throughput, discord-transcript-bot-ob3)
# is a faster-whisper-only knob and is read in src/asr/selection.py where that
# backend is constructed.

# Per-segment hallucination filter. On near-silent / noisy audio Whisper
# emits high-confidence garbage (YouTube-caption ghosts like "Subtitles by
# the Amara.org community", repeated tokens, and — with roster biasing — the
# injected names). faster-whisper exposes per-segment quality metrics; drop a
# segment that looks like a hallucination rather than concatenating it blindly.
# Conservative defaults to avoid dropping real speech; every drop is logged
# with its metrics so the thresholds can be calibrated on real audio
# (discord-transcript-bot-61z). Env-overridable.
WHISPER_DROP_NO_SPEECH_PROB = float(os.getenv("WHISPER_DROP_NO_SPEECH_PROB", "0.8"))
WHISPER_DROP_AVG_LOGPROB = float(os.getenv("WHISPER_DROP_AVG_LOGPROB", "-1.0"))
WHISPER_DROP_COMPRESSION_RATIO = float(
    os.getenv("WHISPER_DROP_COMPRESSION_RATIO", "2.4")
)


def _env_flag(raw, default="1"):
    """Parse a truthy/falsy env string. Falsy: '', '0', 'false', 'no' (any
    case); everything else (incl. unset -> default) is truthy."""
    return (raw if raw is not None else default).strip().lower() not in (
        "",
        "0",
        "false",
        "no",
    )


# Roster proper-noun biasing of initial_prompt is an unvalidated accuracy
# lever that also causes prompt-echo hallucinations; gate it so it can be
# disabled (or scoped) without a code change. Truthy values enable it.
BIAS_PROMPT_WITH_ROSTER = _env_flag(os.getenv("BIAS_PROMPT_WITH_ROSTER"))

# Persist the per-speaker audio this session transcribes, so a live game
# doubles as reference-data capture for the A/B harness. OFF by default: it
# costs ~192 KB per second of captured speech and normal sessions don't want
# it. Turn on for a session you intend to use as bake-off reference data
# (discord-transcript-bot-3dn). See src/session_capture.py.
CAPTURE_SESSION_AUDIO = _env_flag(os.getenv("CAPTURE_SESSION_AUDIO"), default="0")

# Notorious Whisper silence-hallucination phrases (YouTube caption training
# ghosts). Matched case-insensitively against the normalized segment text as a
# precise backstop to the metric filter. Kept lowercase + punctuation-free.
_HALLUCINATION_PHRASES = frozenset(
    {
        "thank you for watching",
        "thanks for watching",
        "thank you for watching this video",
        "subtitles by the amara org community",
        "transcription by the amara org community",
        "please subscribe",
        "like and subscribe",
        "see you in the next video",
    }
)

# Whisper's initial_prompt biases spelling/vocabulary. It is end-weighted and
# capped near 224 tokens, so we keep this base hint short and append the
# session's roster proper nouns (character/player names) AFTER it — see
# WhisperSink._build_initial_prompt. discord-transcript-bot-cul.
DEFAULT_INITIAL_PROMPT = "You are writing the transcriptions for a D&D game."

logger = logging.getLogger(__name__)

_now = time.time  # Speaker's default timestamp; the parameter shadows the module

# The transcription engine (faster-whisper CPU/CUDA, or MLX-Whisper on Apple
# Silicon) is selected at runtime and memoized in src/asr/selection.py. It is
# resolved lazily on first use — NOT at import — so importing this module is
# cheap and tests don't load a real model. The chosen backend normalizes its
# output to NormalizedSegment so _accept_segment + the prompt-echo filter below
# work regardless of engine. discord-transcript-bot-d8g.


class Speaker:
    """One person's audio since they last went quiet.

    The sink keeps one of these per user who is currently talking; it
    collects the PCM chunks Discord delivers for them and the span they
    cover. When the person pauses (or the span grows too long) the whole
    thing is handed to the transcriber as a single segment and a fresh
    Speaker starts on their next chunk.

    :param user: Discord user id.
    :param player: name from the roster, or None.
    :param character: character name from the roster, or None.
    :param data: the first PCM chunk.
    :param time: when that chunk arrived (defaults to now).
    """

    __slots__ = (
        "user",
        "player",
        "character",
        "data",
        "first_word",
        "last_word",
        "chunk_count",
        "seq",
        "clip_name",
    )

    def __init__(self, user: int, player: str, character: str, data, time=None):
        # ``time`` is resolved here rather than as a default argument, which
        # would be evaluated once at import.
        arrived = _now() if time is None else time
        self.user = user
        self.player = player
        self.character = character
        self.data = [data]  # PCM chunks, in arrival order
        self.first_word = arrived
        self.last_word = arrived
        self.chunk_count = 1
        # Submission sequence, assigned by insert_voice when this segment is
        # handed to the executor. Used to name the captured clip so the
        # capture directory sorts in the same order as the manifest.
        self.seq = None
        # Basename of this segment's captured audio, set by transcribe() when
        # capture is enabled and read when the transcription is committed, so
        # the manifest pairs each clip with its text.
        self.clip_name = None

    def add_chunk(self, pcm: bytes, arrived: float) -> None:
        self.data.append(pcm)
        self.chunk_count += 1
        self.last_word = arrived

    @property
    def span_s(self) -> float:
        """Seconds between the first and the latest chunk."""
        return self.last_word - self.first_word


def pcm_to_wav(pcm: bytes) -> io.BytesIO:
    """Wrap raw Discord PCM (48 kHz, 16-bit, stereo) in a WAV container,
    rewound to the start, ready for the transcriber or the capture."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as out:
        out.setnchannels(Decoder.CHANNELS)
        out.setsampwidth(Decoder.SAMPLE_SIZE // Decoder.CHANNELS)
        out.setframerate(Decoder.SAMPLING_RATE)
        out.writeframes(pcm)
    buf.seek(0)
    return buf


def wav_seconds(buf) -> float:
    """Duration of the WAV data in ``buf``; leaves it rewound."""
    buf.seek(0)
    with wave.open(buf, "rb") as src:
        seconds = src.getnframes() / float(src.getframerate())
    buf.seek(0)
    return seconds


# Shorter clips than this are not worth a decode and never carry a word.
MIN_CLIP_S = 0.1


class WhisperSink(Sink):
    """Pycord recording sink that turns each person's speech into text.

    Pycord calls ``write`` with every decoded audio chunk and the user it
    came from. The chunks go onto ``voice_queue``; the sink's own thread
    (``insert_voice``) sorts them into per-user ``Speaker`` buffers, and
    when a person pauses their buffer is handed to a worker thread for
    transcription. Results are committed in submission order to the
    per-session ``transcripts/<ts>.txt`` and the JSON transcript log.

    :param transcript_queue: asyncio queue the loop side may consume;
        ``None`` is put on it at close.
    :param loop: the bot's event loop.
    :param transcriber_type: ``"local"`` (the engine chosen by
        ``src.asr.selection``) or ``"openai"`` (the Whisper API).
    :param filters: Pycord ``Sink`` filters; the defaults when omitted.
    :param player_map: Discord user id -> ``{"player", "character"}``, shared
        by reference with the bot so roster changes apply mid-session.
    :param data_length: cap on one incoming chunk, in bytes; a larger chunk
        keeps only its tail (Discord can deliver a long block of silence
        in one go after a pause).
    :param max_speakers: how many people may be buffered at once; negative
        means no limit.
    """

    def __init__(
        self,
        transcript_queue: asyncio.Queue,
        loop: asyncio.AbstractEventLoop,
        transcriber_type="local",
        *,
        filters=None,
        player_map=None,
        data_length=50000,
        max_speakers=-1,
    ):
        self.queue = transcript_queue
        self.loop = loop
        self.filters = default_filters if filters is None else filters
        Filters.__init__(self, **self.filters)
        self.data_length = data_length
        self.max_speakers = max_speakers
        self.transcriber_type = transcriber_type
        if transcriber_type == "openai":
            self.client = OpenAI()
        else:
            # Resolve the local transcription backend once (memoized across
            # sinks + the health check so the model loads a single time).
            self._backend = selection.get_backend()
        self.vc = None  # set by Pycord when recording starts
        # Pycord's Sink.cleanup() iterates this; we never put anything in it
        # because audio is handled as it arrives, not buffered per file.
        self.audio_data = {}
        self.running = True  # cleared to stop the voice thread
        self.speakers: List[Speaker] = []  # people with buffered audio
        self.voice_queue = Queue()  # (user_id, pcm, arrival time) from write()
        self.executor = ThreadPoolExecutor(max_workers=transcribe_workers())
        self.player_map = {} if player_map is None else player_map
        # Biased Whisper prompt built once from the roster's proper nouns;
        # player_map is fixed for this sink's lifetime. discord-transcript-bot-cul.
        # _prompt_names is the exact name list put in the prompt, used to
        # detect when Whisper regurgitates one as a hallucination.
        self.initial_prompt, self._prompt_names = self._build_initial_prompt()

        # Ordered-commit state. Transcriptions run in parallel on the
        # executor, but their results are written in the order the segments
        # were *submitted* — a chunk that finishes early waits behind
        # earlier, still-running chunks. This is submission order, not
        # strict cross-speaker chronology (see _record_result for the
        # caveat). `_submit_seq` is assigned only on the (single)
        # insert_voice thread; `_next_commit` and `_pending_results` are
        # touched only under `_commit_lock`.
        self._submit_seq = 0
        self._next_commit = 0
        self._pending_results = {}
        self._commit_lock = threading.Lock()
        # Detached transcription futures that have not completed yet. The
        # insert_voice loop never waits on them (that is the snowball fix),
        # so teardown must: a segment submitted just before /stop is still
        # running when the voice thread exits, and close() would otherwise
        # finalize the .txt before its text arrives (discord-transcript-bot-abh).
        # Counted down at the END of each done-callback, after the result
        # has been committed — not when the future finishes. CPython's
        # Future.set_result wakes concurrent.futures.wait() BEFORE it runs
        # the done-callbacks, so waiting on the futures themselves can
        # return while _on_transcribed is still writing (roborev #6410).
        self._inflight = 0
        self._inflight_cv = threading.Condition()

        # Per-session transcript file. Path is fixed at construction so all
        # utterances from this sink land in the same file, but the handle is
        # opened lazily on first write — so a sink that's constructed but
        # never used won't leave an empty file on disk.
        transcript_dir = os.path.join(os.getcwd(), "transcripts")
        os.makedirs(transcript_dir, exist_ok=True)
        session_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.session_file = os.path.join(transcript_dir, f"{session_time}.txt")
        # Reference-data capture, sharing the transcript's timestamp so clips
        # and the .txt for one session are trivially correlated. None unless
        # CAPTURE_SESSION_AUDIO is set; the directory itself is created lazily
        # on the first clip.
        self.capture = (
            SessionCapture(
                os.path.join(os.getcwd(), "captures", session_time),
                session_info=self._capture_session_info(session_time),
            )
            if CAPTURE_SESSION_AUDIO
            else None
        )
        if self.capture is not None:
            # Say it up front. The directory is created lazily on the first
            # clip, so without this the operator gets no confirmation capture
            # is armed until someone speaks — and a session recorded with it
            # silently OFF is only discovered after the game, when the
            # reference data isn't there. discord-transcript-bot-tr2.
            logger.info(
                "Session audio capture ENABLED — per-speaker clips will be "
                f"written to {self.capture.directory}"
            )
        self._session_fh = None
        self._session_fh_lock = threading.Lock()
        # Distinct from `running`: `running` going False just stops the
        # insert_voice loop accepting new audio, but a transcription
        # submitted before /stop can still finish during teardown (the
        # stop_voice_thread join waits for it) and must still reach the
        # per-session .txt. Only once the file is finalized in close()
        # do we refuse further writes (the roborev #514 race guard).
        self._closed = False

    def _capture_session_info(self, session_time):
        """The engine/prompt settings that produced this session's draft
        transcriptions, recorded next to the clips.

        The roster-biased ``initial_prompt`` is the important one: it's the
        bot's main proper-noun accuracy lever, it's rebuilt from a roster that
        may change before anyone runs the bake-off, and a config scored
        WITHOUT it understates the bot's real accuracy on names. Recording it
        here lets the bake-off measure like-for-like.
        """
        backend = getattr(self, "_backend", None)
        return {
            "session": session_time,
            "transcriber_type": self.transcriber_type,
            "backend": getattr(backend, "name", None),
            "model": getattr(backend, "model_id", None),
            "language": WHISPER_LANGUAGE,
            "initial_prompt": self.initial_prompt,
            "roster_names_in_prompt": list(self._prompt_names),
            "beam_size": WHISPER_BEAM_SIZE,
            "best_of": WHISPER_BEST_OF,
            "no_speech_threshold": 0.6,
            "vad_parameters": dict(min_silence_duration_ms=150, threshold=0.8),
        }

    def start_voice_thread(self, on_exception=None):
        """Start the voice thread. ``on_exception(exc)`` is called, on that
        thread, if insert_voice dies with an exception.

        This used to assign the process-wide ``threading.excepthook``, so
        every sink replaced every other's handler (and every unrelated
        thread's) and nothing ever restored it (discord-transcript-bot-7g8).
        The handler is now scoped to this sink's own thread.
        """
        logger.debug(f"Voice thread starting for guild {self._guild_id()}.")
        self._on_thread_exception = on_exception
        self.voice_thread = threading.Thread(
            target=self._voice_thread_main, name="whisper-sink-voice", daemon=True
        )
        self.voice_thread.start()

    def _guild_id(self):
        """The recording guild's id for log lines, or None once the voice
        client is gone (this must never raise on the teardown path)."""
        try:
            return self.vc.channel.guild.id
        except AttributeError:
            return None

    def _voice_thread_main(self):
        try:
            self.insert_voice()
        except Exception as e:  # noqa: BLE001 - reported, never re-raised
            logger.exception(f"Voice thread died: {e}")
            handler = getattr(self, "_on_thread_exception", None)
            if handler is not None:
                try:
                    handler(e)
                except Exception as handler_err:  # noqa: BLE001
                    logger.error(
                        f"Voice thread exception handler failed: {handler_err}"
                    )

    # Upper bound for joining the voice thread. It should exit within a
    # couple of seconds of self.running going False (one insert_voice
    # loop iteration plus the final flush), so this is generous; it
    # exists only so a wedged loop can't block teardown forever.
    JOIN_TIMEOUT_S = 15
    # Upper bound for the in-flight transcriptions to finish after the
    # voice thread has exited. The final flush can submit one segment per
    # speaker (each ≤ MAX_SEGMENT_S of audio), so this covers a full
    # table's worth of trailing speech on the GPU backend with margin.
    DRAIN_TIMEOUT_S = 60

    def stop_voice_thread(self):
        """Stop the voice thread (it flushes every buffered speaker on the
        way out) and wait for the transcriptions that flush submitted."""
        self.running = False
        try:
            thread = getattr(self, "voice_thread", None)
            if thread is not None:
                thread.join(timeout=self.JOIN_TIMEOUT_S)
                if thread.is_alive():
                    logger.warning(
                        "Voice thread did not exit within "
                        f"{self.JOIN_TIMEOUT_S}s; abandoning it (daemon "
                        "thread, will not block process exit)."
                    )
            self._wait_for_inflight()
        except Exception as e:  # noqa: BLE001 - teardown must finish
            logger.error(f"Stopping the voice thread failed: {e}")
        finally:
            # self.vc (and its .channel) can already be torn down after a
            # /disconnect, so resolving the guild id here must never raise
            # — this runs on the teardown path and an AttributeError would
            # propagate up through cleanup_sink and crash /stop|/disconnect.
            guild_id = None
            try:
                guild_id = self.vc.channel.guild.id
            except AttributeError:
                pass
            logger.debug(f"A sink thread was stopped for guild {guild_id}.")

    def _wait_for_inflight(self):
        """Block (bounded) until every detached transcription has been
        COMMITTED — its done-callback has run to completion — so the text is
        in the session file before close() finalizes it. Runs on the
        teardown path; must never raise."""
        with self._inflight_cv:
            pending = self._inflight
            if not pending:
                return
            logger.info(
                f"Waiting for {pending} in-flight transcription(s) to finish..."
            )
            drained = self._inflight_cv.wait_for(
                lambda: self._inflight == 0, timeout=self.DRAIN_TIMEOUT_S
            )
            remaining = self._inflight
        if not drained:
            logger.warning(
                f"{remaining} transcription(s) still running after "
                f"{self.DRAIN_TIMEOUT_S}s; their text will reach the JSON log "
                "but not the per-session .txt."
            )

    # Conservative character budget for initial_prompt. Whisper's prompt is
    # ~224-token capped; at ~4 chars/token that's ~900 chars, so this leaves
    # generous headroom for the base hint plus as many roster names as fit.
    MAX_INITIAL_PROMPT_CHARS = 600

    def _build_initial_prompt(self):
        """Build a Whisper initial_prompt that biases spelling toward this
        session's proper nouns — the character and player names from the
        roster — so names like 'Arasaka' or a character 'Johan' transcribe
        correctly instead of as common-word homophones. Falls back to the
        generic D&D hint when there's no roster. Length-bounded because the
        prompt is end-weighted and ~224-token capped; names are appended
        after the base hint (the high-value position) and truncated whole.
        """
        if not BIAS_PROMPT_WITH_ROSTER:
            # Biasing disabled (BIAS_PROMPT_WITH_ROSTER=0): generic hint only,
            # no names injected, so no names to echo.
            return DEFAULT_INITIAL_PROMPT, []
        entries = [e for e in (self.player_map or {}).values() if isinstance(e, dict)]
        names = []
        seen = set()
        # Two passes — ALL character names first, then ALL player names —
        # not interleaved per entry. Characters are spoken most, so when the
        # char budget truncates the list they must take priority globally: a
        # later entry's character outranks an earlier entry's player. Both
        # deduped case-insensitively.
        for key in ("character", "player"):
            for entry in entries:
                name = (entry.get(key) or "").strip()
                if name and name.lower() not in seen:
                    seen.add(name.lower())
                    names.append(name)

        if not names:
            return DEFAULT_INITIAL_PROMPT, []

        prefix = DEFAULT_INITIAL_PROMPT + " Names: "
        kept = []
        for name in names:
            candidate = ", ".join(kept + [name])
            # +1 for the trailing period.
            if len(prefix) + len(candidate) + 1 > self.MAX_INITIAL_PROMPT_CHARS:
                break
            kept.append(name)

        if not kept:
            return DEFAULT_INITIAL_PROMPT, []
        return prefix + ", ".join(kept) + ".", kept

    # Verb connectives Whisper tacks onto a regurgitated name ("Sovereign
    # Lord GM says,"). A name plus only these is still an echo. Deliberately
    # excludes articles/conjunctions (and/the/a): the prompt lists names
    # comma-separated, so the echo pattern is "Name, Name" (handled by
    # punctuation normalization), and including bare articles would over-drop
    # real clipped speech beginning with a short name ("Will the...").
    _ECHO_FILLERS = frozenset({"says", "said", "say", "speaking", "speaks"})

    @staticmethod
    def _normalize_echo(text):
        """Lowercase, strip punctuation, collapse whitespace — so prompt
        names and transcribed text compare on the same footing."""
        return " ".join(re.sub(r"[^\w\s]", " ", text.lower()).split())

    def _is_prompt_echo(self, text):
        """True when a transcription reduces to NOTHING BUT roster names from
        the biasing prompt (plus trivial fillers like "says").

        Whisper regurgitates its initial_prompt on near-silent/garbage audio,
        leaking the biased names as fake speech — a bare name ("Sovereign Lord
        GM"), a name + filler ("Sovereign Lord GM says,"), OR several names run
        together ("Sovereign Lord GM, Rahul Patch Sarker."). Strip every prompt
        name and filler from the text; if nothing real is left and at least one
        name was present, it's an echo.

        Deliberately conservative: a line that merely *mentions* a name with
        real content ("Noah, watch out") keeps that content, so it survives.
        """
        norm = self._normalize_echo(text)
        if not norm:
            return False
        name_norms = sorted(
            (n for n in (self._normalize_echo(n) for n in self._prompt_names) if n),
            key=len,
            reverse=True,  # strip longest names first so multi-word names go whole
        )
        if not name_norms:
            return False
        stripped = norm
        matched_a_name = False
        for n in name_norms:
            new = re.sub(rf"\b{re.escape(n)}\b", " ", stripped)
            if new != stripped:
                matched_a_name = True
                stripped = new
        leftover = [w for w in stripped.split() if w not in self._ECHO_FILLERS]
        return matched_a_name and not leftover

    def _is_hallucination_phrase(self, text):
        """True when the whole utterance is a known Whisper silence-artifact
        (YouTube caption ghost like "Subtitles by the Amara.org community")."""
        return self._normalize_echo(text) in _HALLUCINATION_PHRASES

    def _drop_if_prompt_echo(self, text):
        """Return text, or "" (silence downstream) when the WHOLE utterance is
        a roster-name prompt-echo or a known Whisper hallucination phrase."""
        if self._is_prompt_echo(text):
            logger.info(f"Dropping prompt-echo hallucination: {text!r}")
            return ""
        if self._is_hallucination_phrase(text):
            logger.info(f"Dropping known Whisper artifact: {text!r}")
            return ""
        return text

    def _accept_segment(self, seg):
        """Whether to keep one faster-whisper segment, or drop it as a
        hallucination based on its quality metrics. Drops are logged with the
        metrics so the thresholds can be calibrated on real audio."""
        text = (getattr(seg, "text", "") or "").strip()
        if not text:
            return False
        nsp = getattr(seg, "no_speech_prob", 0.0) or 0.0
        alp = getattr(seg, "avg_logprob", 0.0) or 0.0
        cr = getattr(seg, "compression_ratio", 1.0) or 1.0
        reason = None
        if cr > WHISPER_DROP_COMPRESSION_RATIO:
            reason = f"repetition (compression_ratio={cr:.2f})"
        elif alp < WHISPER_DROP_AVG_LOGPROB:
            reason = f"low-confidence (avg_logprob={alp:.2f})"
        elif nsp > WHISPER_DROP_NO_SPEECH_PROB:
            reason = f"silence (no_speech_prob={nsp:.2f})"
        elif self._is_hallucination_phrase(text):
            reason = "known-artifact"
        if reason:
            logger.info(f"Dropping hallucinated segment [{reason}]: {text!r}")
            return False
        return True

    def _transcribe_openai(self, wav) -> str:
        """One Whisper API call; the prompt biases it toward the roster."""
        response = self.client.audio.transcriptions.create(
            file=("segment.wav", wav),
            model="whisper-1",
            language=WHISPER_LANGUAGE,
            prompt=self.initial_prompt,
        )
        logger.info(f"OpenAI transcription: {response.text}")
        return response.text

    def _transcribe_local(self, wav) -> str:
        """The selected local engine (faster-whisper CPU/CUDA, or MLX-Whisper
        on Apple Silicon). It returns NormalizedSegments carrying the quality
        metrics _accept_segment inspects, so the hallucination filtering is
        the same whichever engine is underneath."""
        result = self._backend.transcribe(
            wav,
            language=WHISPER_LANGUAGE,
            beam_size=WHISPER_BEAM_SIZE,
            best_of=WHISPER_BEST_OF,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=150, threshold=0.8),
            no_speech_threshold=0.6,
            initial_prompt=self.initial_prompt,
        )
        # Keep only the segments that pass the per-segment filter
        # (silence / low confidence / repetition / known artifact) rather
        # than everything the engine emitted.
        text = "".join(s.text for s in result.segments if self._accept_segment(s))
        logger.info(f"Transcription: {text}")
        return text

    def transcribe_audio(self, wav) -> str:
        """Text for one WAV buffer, or ``""`` when it is too short, the
        engine failed, or the result was a prompt echo. Never raises: a bad
        segment must not take the worker down."""
        try:
            if wav_seconds(wav) <= MIN_CLIP_S:
                return ""
            if self.transcriber_type == "openai":
                text = self._transcribe_openai(wav)
            else:
                text = self._transcribe_local(wav)
            # Whole-utterance backstop for prompt echoes and artifacts that
            # slipped past the per-segment metrics.
            return self._drop_if_prompt_echo(text)
        except Exception as e:  # noqa: BLE001
            logger.error(f"Transcription failed for a segment: {e}")
            return ""

    def transcribe(self, speaker: Speaker) -> str:
        """Transcribe everything buffered for ``speaker``. Runs on a worker
        thread. Discord PCM is always 48 kHz 16-bit stereo, so the WAV
        header is fixed (``pcm_to_wav``)."""
        wav = pcm_to_wav(b"".join(speaker.data))

        # Persist the EXACT bytes about to be transcribed, before doing it, so
        # a reference clip survives even if transcription then fails.
        # getvalue() doesn't move the stream position. Runs on an executor
        # thread, so SessionCapture is thread-safe. The except is deliberate
        # belt-and-braces on top of SessionCapture swallowing its own errors:
        # capture is an optional side feature and no bug in it may ever cost a
        # live session its transcription.
        if self.capture is not None:
            try:
                speaker.clip_name = self.capture.save_clip(
                    wav.getvalue(), speaker.seq, speaker
                )
            except Exception as e:  # noqa: BLE001
                logger.warning(f"Session audio capture failed for this segment: {e}")

        return self.transcribe_audio(wav)

    # A speaker's buffer is submitted for transcription once they've gone
    # quiet for this long...
    SILENCE_GAP_S = 1.5
    # ...or once it has spanned this long even without a gap, so a speaker
    # who never pauses (a heavy combat scene) can't accumulate a
    # multi-minute segment that takes minutes to transcribe and stalls the
    # whole pipeline. Bounds worst-case latency. See
    # discord-transcript-bot-hin.
    MAX_SEGMENT_S = 30
    # Idle nap for the insert_voice loop so it doesn't busy-spin a core
    # when there's no audio to drain or flush.
    IDLE_SLEEP_S = 0.1

    def insert_voice(self):
        while self.running:
            try:
                self._drain_voice_queue()
                # Submit every speaker that has gone quiet OR whose buffer has
                # grown past MAX_SEGMENT_S. Each is handed to the executor and
                # DETACHED: we never call future.result() here, so this loop
                # keeps draining the voice_queue while transcriptions run in
                # parallel. The speaker is removed from the active list on
                # submit (so it isn't double-submitted); new audio from the
                # same user starts a fresh segment. A done-callback commits
                # the result in submission order (see _on_transcribed).
                now = time.time()
                for speaker in self.speakers[:]:
                    paused = (now - speaker.last_word) >= self.SILENCE_GAP_S
                    if paused or speaker.span_s >= self.MAX_SEGMENT_S:
                        self._submit_speaker(speaker)
            except Exception as e:  # noqa: BLE001 - the loop must keep going
                logger.error(f"Voice loop iteration failed: {e}")
            finally:
                # Idle nap so this loop doesn't busy-spin. In `finally`, NOT at
                # the end of the try: an exception raised each pass would
                # otherwise skip the sleep and hot-loop a core while flooding
                # the log — for a bug whose only symptom is heat, on the
                # thread that keeps transcription flowing. Measurement says
                # this isn't what caused the 100%-core spin in
                # discord-transcript-bot-309 (that is Pycord's receive loop;
                # this loop measured 0.0%), so this is defense in depth.
                time.sleep(self.IDLE_SLEEP_S)
        self._flush_on_stop()

    def _flush_on_stop(self):
        """Final flush when the loop exits on /stop or /disconnect: whoever
        was mid-sentence has a buffer that never reached SILENCE_GAP_S, and
        Pycord drains its decoders into write() during stop_recording(), so
        the queue can hold audio too. Submit all of it; stop_voice_thread
        then waits for the futures (discord-transcript-bot-abh). Guarded:
        this runs on the voice thread's way out and must not raise."""
        try:
            self._drain_voice_queue()
            for speaker in self.speakers[:]:
                self._submit_speaker(speaker)
        except Exception as e:
            logger.error(f"Error flushing buffered audio on stop: {e}")

    def _drain_voice_queue(self):
        """Move ALL pending audio into per-speaker buffers. Must never block
        on a transcription (see the detached submit in _submit_speaker) or
        the queue backs up and segments snowball."""
        while not self.voice_queue.empty():
            user_id, pcm, arrived = self.voice_queue.get()
            speaker = next((s for s in self.speakers if s.user == user_id), None)
            if speaker is not None:
                speaker.add_chunk(pcm, arrived)
            elif self.max_speakers < 0 or len(self.speakers) <= self.max_speakers:
                names = self.player_map.get(user_id, {})
                self.speakers.append(
                    Speaker(
                        user_id,
                        names.get("player"),
                        names.get("character"),
                        pcm,
                        arrived,
                    )
                )

    def _submit_speaker(self, speaker):
        """Hand one speaker's buffer to the executor (detached) and remove it
        from the active list. Only the insert_voice thread calls this."""
        if speaker.chunk_count <= 1:
            return
        self.speakers.remove(speaker)
        seq = self._submit_seq
        self._submit_seq += 1
        # Tag the segment with its submission order before it goes to the
        # executor, so a captured clip's filename sorts the same way its
        # manifest line will.
        speaker.seq = seq
        # Counted before submit so teardown can never observe "nothing in
        # flight" between the submit and the callback registration.
        with self._inflight_cv:
            self._inflight += 1
        try:
            future = self.executor.submit(self.transcribe, speaker)
            future.add_done_callback(
                lambda fut, s=seq, spk=speaker: self._on_transcribed(s, spk, fut)
            )
        except Exception as e:
            # submit/callback registration failed after seq was allocated
            # (e.g. a shut-down executor). No future will ever fire
            # _on_transcribed(seq, …), so commit this seq empty now —
            # otherwise _next_commit wedges on it permanently and every
            # later result stops committing — and release the count.
            logger.error(f"Failed to submit transcription for {speaker.user}: {e}")
            self._record_result(seq, speaker, "")
            self._inflight_done()

    def _inflight_done(self):
        with self._inflight_cv:
            self._inflight -= 1
            self._inflight_cv.notify_all()

    def _on_transcribed(self, seq, speaker, future):
        """Done-callback for a detached transcription future. Runs on an
        executor thread. A failed future is committed as empty text so one
        bad segment can't wedge the commit pointer forever.

        Must not touch self.speakers (owned by the insert_voice thread); it
        only reads the speaker object handed to it and writes the result.
        """
        try:
            try:
                transcription = future.result()
            except Exception as e:
                logger.warning(f"Error in transcription future for {speaker.user}: {e}")
                transcription = ""
            self._record_result(seq, speaker, transcription)
        finally:
            # Only now has the text reached the session file; this is what
            # _wait_for_inflight waits on.
            self._inflight_done()

    def _record_result(self, seq, speaker, transcription):
        """Store a result under its submission sequence, then commit every
        result whose turn has come IN SUBMISSION ORDER — a segment that
        finished early waits behind earlier, still-running segments.

        Note: this is submission order, not strict cross-speaker
        chronology. A long speaker force-flushed by MAX_SEGMENT_S can be
        submitted after a shorter, later-but-already-silent speaker, so the
        per-session .txt line order can differ slightly from true start
        time. The embedded `begin`/`date` fields in each JSON record remain
        authoritative — consumers needing exact chronology should sort by
        those rather than relying on file line order.

        Called by the done-callback and by the submit-failure path (so an
        allocated seq that never produced a future is still committed-empty
        and can't wedge the pointer).
        """
        with self._commit_lock:
            self._pending_results[seq] = (speaker, transcription)
            while self._next_commit in self._pending_results:
                spk, text = self._pending_results.pop(self._next_commit)
                self._next_commit += 1
                try:
                    self.write_transcription_log(spk, text)
                except Exception as e:
                    logger.error(f"Error writing transcription log: {e}")

    @staticmethod
    def _split_stamp(unix_time):
        """``("YYYY-MM-DD", "HH:MM:SS.mmm")`` for a Unix timestamp."""
        stamp = datetime.fromtimestamp(unix_time)
        return stamp.strftime("%Y-%m-%d"), stamp.strftime("%H:%M:%S.%f")[:-3]

    def write_transcription_log(self, speaker, transcription):
        """Commit one transcribed segment: a JSON record on the
        ``transcription`` logger (the per-day log the dashboard reads), the
        capture manifest line when capture is on, and a human-readable line
        in the per-session ``.txt``. The record's keys are a file format
        (``src.transcript_lines``, ``web/transcript_service.py``)."""
        date, begin = self._split_stamp(speaker.first_word)
        _, end = self._split_stamp(speaker.last_word)
        record = {
            "date": date,
            "begin": begin,
            "end": end,
            "user_id": speaker.user,
            "player": speaker.player,
            "character": speaker.character,
            "event_source": "Discord",
            "data": transcription,
        }
        logging.getLogger("transcription").info(json.dumps(record))

        # Pair this segment's captured clip with the text the engine produced,
        # as the draft a human corrects into reference data. Deliberately
        # before the per-session .txt write below, which early-returns on a
        # closed file handle — the manifest line must not be lost to that.
        if self.capture is not None:
            try:
                self.capture.record(
                    getattr(speaker, "clip_name", None),
                    speaker,
                    transcription,
                    begin=begin,
                )
            except Exception as e:
                logger.warning(f"Session capture manifest write failed: {e}")

        # Write human-readable line to per-session transcript file
        if transcription and transcription.strip():
            player = speaker.player or "Unknown"
            character = speaker.character or "Unknown"
            timestamp = datetime.fromtimestamp(speaker.first_word).strftime("%H:%M:%S")
            line = f"[{timestamp}] {player} ({character}) [{speaker.user}]: {transcription.strip()}\n"
            fh = self._get_session_fh()
            if fh is None:
                return
            try:
                fh.write(line)
                fh.flush()
            except (ValueError, OSError) as e:
                logger.warning(f"Failed to write per-session transcript line: {e}")

    def _get_session_fh(self):
        """Lazily open the per-session transcript file. Returns None only
        once the sink is fully closed — NOT merely when `running` is
        False — so a transcription that finishes during /stop teardown
        (its future was submitted before /stop; the stop_voice_thread
        join blocks until it completes) still lands in the .txt. After
        close() finalizes the file, further writes are refused (the
        roborev #514 race guard against a late executor task re-opening
        a freshly-closed handle).
        """
        with self._session_fh_lock:
            if self._closed:
                return None
            if self._session_fh is None:
                try:
                    self._session_fh = open(self.session_file, "a", encoding="utf-8")
                except OSError as e:
                    logger.warning(f"Failed to open per-session transcript file: {e}")
                    return None
            return self._session_fh

    @Filters.container
    def write(self, data, user):
        """Pycord's entry point: one decoded audio chunk from one user.
        Queued for the voice thread; nothing is decoded here."""
        # Empty buffers (b"") — no audio, or a frame not yet attributable
        # to a user — are dropped below, not queued. Discord's first
        # non-empty buffer after a silence gap can be a large block of
        # silent audio; that case is bounded by the
        # `len(pcm) > self.data_length` trim further down, not dropped.

        # Pycord's DAVE-capable voice receive (2.7+, the fix/voice-rec-2
        # pin) hands us a VoiceData object (already decrypted + decoded)
        # instead of raw PCM bytes, and passes the User/Member as `user`
        # instead of the integer ID. Normalize both back to the
        # (bytes, int-id) contract the queue / player_map / transcript
        # line still expect. The getattr fallbacks keep this working if
        # a future Pycord reverts to the old bytes/int signature.
        pcm = getattr(data, "pcm", data)
        user_id = getattr(user, "id", user)

        # data.source is None until the SSRC->user mapping resolves;
        # that audio can't be attributed to a speaker, so drop it.
        if user_id is None or not pcm:
            return

        # Keep the tail: that is where the speech is when Discord delivers
        # a long silent block followed by words.
        if len(pcm) > self.data_length:
            pcm = pcm[-self.data_length :]
        self.voice_queue.put_nowait((user_id, pcm, time.time()))

    def close(self):
        """Finalize the sink: tell the loop side we are done (``None`` on
        the queue), release Pycord's buffers, close the capture and the
        session file, and shut the worker pool down. Never raises."""
        logger.debug(f"Closing the sink for guild {self._guild_id()}.")
        self.running = False
        self.queue.put_nowait(None)
        super().cleanup()
        if self.capture is not None:
            self.capture.close()
        with self._session_fh_lock:
            # Set under the lock and before clearing the handle so any
            # write racing close() either completes first or is cleanly
            # refused — never writes to a half-closed handle.
            self._closed = True
            try:
                if self._session_fh is not None:
                    self._session_fh.close()
                    self._session_fh = None
            except OSError:
                pass
        # Release the worker threads: each /scribe builds a fresh sink with
        # its own pool, and an un-shut-down pool keeps its (up to 8) idle
        # workers alive for the life of the process. wait=False — anything
        # still running was already given DRAIN_TIMEOUT_S by
        # stop_voice_thread; it may finish on its own but can no longer
        # reach the finalized .txt. Guarded: close() must never raise.
        try:
            self.executor.shutdown(wait=False)
        except Exception as e:  # noqa: BLE001
            logger.debug(f"executor shutdown skipped: {e}")
