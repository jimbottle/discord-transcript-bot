import asyncio
import io
import json
import logging
import os
import threading
import time
import wave
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from queue import Queue
from typing import List

import speech_recognition as sr
import torch
from discord.opus import Decoder
from discord.sinks.core import Filters, Sink, default_filters
from faster_whisper import WhisperModel
from openai import OpenAI

WHISPER_MODEL = "large-v3"
WHISPER_LANGUAGE = "en"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Set the model to evaluation mode (important for inference)
logger = logging.getLogger(__name__)

if DEVICE == "cuda":
    gpu_ram = torch.cuda.get_device_properties(0).total_memory / 1024**3
    if gpu_ram < 5.0:
        logger.warning("GPU has less than 5GB of RAM. Switching to CPU.")
        DEVICE = "cpu"

# Precision: int8 on CPU is ~3-4x faster and far lighter on RAM than the
# old float32 default, with negligible accuracy loss for speech; float16 is
# the GPU sweet spot. float32 on CPU could not keep up with a busy
# multi-speaker session and ballooned RSS to ~9.6 GB — see
# discord-transcript-bot-hin.
WHISPER__PRECISION = "int8" if DEVICE == "cpu" else "float16"

audio_model = WhisperModel(
    WHISPER_MODEL, device=DEVICE, compute_type=WHISPER__PRECISION
)


class Speaker:
    """
    A class to store the audio data and transcription for each user.
    """

    def __init__(self, user: int, player: str, character: str, data, time=time.time()):
        self.user = user
        self.player = player
        self.character = character
        self.data = [data]
        self.first_word = time
        self.last_word = time
        self.new_bytes = 1


class WhisperSink(Sink):
    """A sink for discord that takes audio in a voice channel and transcribes it for each user.

    Uses faster whisper for transcription. can be swapped out for other audio transcription libraries pretty easily.

    :param transcript_queue: The queue to send the transcription output to
    :param filters: Some discord thing I'm not sure about
    :param data_length: The amount of data to save when user is silent but their mic is still active

    :param max_speakers: The amount of users to transcribe when all speakers are talking at once.
    """

    def __init__(
        self,
        transcript_queue: asyncio.Queue,
        loop: asyncio.AbstractEventLoop,
        transcriber_type="local",
        *,
        filters=None,
        player_map={},
        data_length=50000,
        max_speakers=-1,
    ):
        self.queue = transcript_queue
        self.transcription_output_queue = asyncio.Queue()
        self.loop = loop

        if filters is None:
            filters = default_filters
        self.filters = filters
        Filters.__init__(self, **self.filters)
        self.data_length = data_length
        self.max_speakers = max_speakers
        self.transcriber_type = transcriber_type
        if transcriber_type == "openai":
            self.client = OpenAI()
        self.vc = None
        self.audio_data = {}
        self.running = True
        self.speakers: List[Speaker] = []
        self.voice_queue = Queue()
        self.executor = ThreadPoolExecutor(max_workers=8)  # TODO: Adjust this
        self.player_map = player_map

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

        # Per-session transcript file. Path is fixed at construction so all
        # utterances from this sink land in the same file, but the handle is
        # opened lazily on first write — so a sink that's constructed but
        # never used won't leave an empty file on disk.
        transcript_dir = os.path.join(os.getcwd(), "transcripts")
        os.makedirs(transcript_dir, exist_ok=True)
        session_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.session_file = os.path.join(transcript_dir, f"{session_time}.txt")
        self._session_fh = None
        self._session_fh_lock = threading.Lock()
        # Distinct from `running`: `running` going False just stops the
        # insert_voice loop accepting new audio, but a transcription
        # submitted before /stop can still finish during teardown (the
        # stop_voice_thread join waits for it) and must still reach the
        # per-session .txt. Only once the file is finalized in close()
        # do we refuse further writes (the roborev #514 race guard).
        self._closed = False

    def start_voice_thread(self, on_exception=None):
        def thread_exception_hook(args):
            logger.debug(
                f"""Exception in voice thread: {args} Likely disconnected while listening."""
            )

        logger.debug(
            f"Starting whisper sink thread for guild {self.vc.channel.guild.id}."
        )
        self.voice_thread = threading.Thread(
            target=self.insert_voice, args=(), daemon=True
        )

        if on_exception:
            threading.excepthook = on_exception
        else:
            threading.excepthook = thread_exception_hook

        self.voice_thread.start()

    # Upper bound for joining the voice thread. It should exit within a
    # couple of seconds of self.running going False (one insert_voice
    # loop iteration), so this is generous; it exists only so a wedged
    # transcription future can't block teardown forever.
    JOIN_TIMEOUT_S = 15

    def stop_voice_thread(self):
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
        except Exception as e:
            logger.error(f"Unexpected error during thread join: {e}")
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

    def check_audio_length(self, temp_file):
        # Ensure the BytesIO is at the start
        temp_file.seek(0)

        # Open the BytesIO object as a WAV file
        with wave.open(temp_file, "rb") as wave_file:
            frames = wave_file.getnframes()
            frame_rate = wave_file.getframerate()
            duration = frames / float(frame_rate)
        return duration

    def transcribe_audio(self, temp_file):
        try:
            # Ensure that the audio is long enough to transcribe. If not, return an empty string
            if self.check_audio_length(temp_file) <= 0.1:
                return ""

            if self.transcriber_type == "openai":
                temp_file.seek(0)
                openai_transcription = self.client.audio.transcriptions.create(
                    file=("foobar.wav", temp_file),
                    model="whisper-1",
                    language=WHISPER_LANGUAGE,
                )
                logger.info(f"OpenAI Transcription: {openai_transcription.text}")
                return openai_transcription.text
            else:
                # The whisper model
                temp_file.seek(0)
                segments, info = audio_model.transcribe(
                    temp_file,
                    language=WHISPER_LANGUAGE,
                    beam_size=10,
                    best_of=3,
                    vad_filter=True,
                    vad_parameters=dict(min_silence_duration_ms=150, threshold=0.8),
                    no_speech_threshold=0.6,
                    initial_prompt="You are writing the transcriptions for a D&D game.",
                )

                segments = list(segments)
                result = ""
                for segment in segments:
                    result += segment.text

                logger.info(f"Transcription: {result}")
                return result
        except Exception as e:
            logger.error(f"Error transcribing audio: {e}")
            return ""

    def transcribe(self, speaker: Speaker):
        # Discord voice PCM is a fixed format (48kHz, 16-bit, stereo).
        # The new voice-receive client no longer exposes self.vc.decoder,
        # so read the constants off discord.opus.Decoder directly.
        audio_data = sr.AudioData(
            bytes().join(speaker.data),
            Decoder.SAMPLING_RATE,
            Decoder.SAMPLE_SIZE // Decoder.CHANNELS,
        )

        wav_data = io.BytesIO(audio_data.get_wav_data())

        wav_io = io.BytesIO()
        with wave.open(wav_io, "wb") as wave_writer:
            wave_writer.setnchannels(Decoder.CHANNELS)
            wave_writer.setsampwidth(Decoder.SAMPLE_SIZE // Decoder.CHANNELS)
            wave_writer.setframerate(Decoder.SAMPLING_RATE)
            wave_writer.writeframes(wav_data.getvalue())

        wav_io.seek(0)
        # Check if the audio is long enough to transcribe, else return empty string

        transcription = self.transcribe_audio(wav_io)

        return transcription

    def get_transcriptions(self):
        """Retrieve all transcriptions from the queue, format them to only include data, begin, and user_id."""
        transcriptions = []
        while not self.transcription_queue.empty():
            log_message = self.transcription_queue.get_nowait()

            # Assuming log_message is a dictionary (or string in JSON format)
            if isinstance(log_message, str):
                log_message = json.loads(
                    log_message
                )  # Convert from string to dictionary if needed

            # Extract only the desired fields from the log message
            begin = log_message.get("begin", "Unknown begin")
            user_id = log_message.get("user_id", "Unknown user")
            data = log_message.get("data", "")

            # Format the transcription entry with only the relevant fields
            formatted_entry = (
                f"Begin: {begin}\n"
                f"User ID: {user_id}\n"
                f"Data: {data}\n"
                "-------------------------\n"
            )

            # Add the formatted entry to the transcription list
            transcriptions.append(formatted_entry)

        return transcriptions

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
                # Drain ALL pending audio into per-speaker buffers. This must
                # never block on a transcription — see the detached submit
                # below — or the queue backs up and segments snowball.
                while not self.voice_queue.empty():
                    item = self.voice_queue.get()
                    # Find or create a speaker
                    speaker = next(
                        (s for s in self.speakers if s.user == item[0]), None
                    )
                    if speaker:
                        speaker.data.append(item[1])
                        speaker.new_bytes += 1
                        speaker.last_word = item[2]
                    elif (
                        self.max_speakers < 0 or len(self.speakers) <= self.max_speakers
                    ):
                        user_id = item[0]
                        user_map = self.player_map.get(user_id, {})
                        player = user_map.get("player")
                        character = user_map.get("character")
                        self.speakers.append(
                            Speaker(user_id, player, character, item[1], item[2])
                        )

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
                    silent = (now - speaker.last_word) >= self.SILENCE_GAP_S
                    too_long = (
                        speaker.last_word - speaker.first_word
                    ) >= self.MAX_SEGMENT_S
                    if not (silent or too_long):
                        continue
                    if speaker.new_bytes <= 1:
                        continue
                    self.speakers.remove(speaker)
                    seq = self._submit_seq
                    self._submit_seq += 1
                    try:
                        future = self.executor.submit(self.transcribe, speaker)
                        future.add_done_callback(
                            lambda fut, s=seq, spk=speaker: self._on_transcribed(
                                s, spk, fut
                            )
                        )
                    except Exception as e:
                        # submit/callback registration failed after seq was
                        # allocated (e.g. a shut-down executor). No future will
                        # ever fire _on_transcribed(seq, …), so commit this seq
                        # empty now — otherwise _next_commit wedges on it
                        # permanently and every later result stops committing.
                        logger.error(
                            f"Failed to submit transcription for {speaker.user}: {e}"
                        )
                        self._record_result(seq, speaker, "")

                # Idle nap so this loop doesn't busy-spin when idle.
                time.sleep(self.IDLE_SLEEP_S)
            except Exception as e:
                logger.error(f"Error in insert_voice: {e}")

    def _on_transcribed(self, seq, speaker, future):
        """Done-callback for a detached transcription future. Runs on an
        executor thread. A failed future is committed as empty text so one
        bad segment can't wedge the commit pointer forever.

        Must not touch self.speakers (owned by the insert_voice thread); it
        only reads the speaker object handed to it and writes the result.
        """
        try:
            transcription = future.result()
        except Exception as e:
            logger.warning(f"Error in transcription future for {speaker.user}: {e}")
            transcription = ""
        self._record_result(seq, speaker, transcription)

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

    def check_speaker_timeouts(self, current_speaker, transcription):

        # Copy the list to avoid modification during iteration
        for speaker in self.speakers[:]:
            if current_speaker.user == speaker.user:
                self.write_transcription_log(speaker, transcription)
                self.speakers.remove(speaker)

    def write_transcription_log(self, speaker, transcription):
        # Convert first_word and last_word Unix timestamps to datetime
        first_word_time = datetime.fromtimestamp(speaker.first_word).strftime(
            "%Y-%m-%d %H:%M:%S.%f"
        )[:-3]
        last_word_time = datetime.fromtimestamp(speaker.last_word).strftime(
            "%Y-%m-%d %H:%M:%S.%f"
        )[:-3]
        # Prepare the log data as a dictionary
        log_data = {
            "date": first_word_time[:10],  # Date (from first_word)
            "begin": first_word_time[11:],  # First word time (HH:MM:SS.ss)
            "end": last_word_time[11:],  # Last word time (HH:MM:SS.ss)
            "user_id": speaker.user,  # User ID
            "player": speaker.player,
            "character": speaker.character,
            "event_source": "Discord",  # Event source
            "data": transcription,  # Transcription text
        }

        # Convert the log data to JSON
        log_message = json.dumps(log_data)

        # Get the transcription logger
        transcription_logger = logging.getLogger("transcription")
        # Log the message
        transcription_logger.info(log_message)
        # Place into queue for processing
        self.transcription_output_queue.put_nowait(log_message)

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
        """Gets audio data from discord for each user talking"""
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

        if len(pcm) > self.data_length:
            pcm = pcm[-self.data_length :]
        write_time = time.time()
        # Send bytes to be transcribed
        self.voice_queue.put_nowait([user_id, pcm, write_time])

    def close(self):
        logger.debug("Closing whisper sink.")
        self.running = False
        self.queue.put_nowait(None)
        super().cleanup()
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
