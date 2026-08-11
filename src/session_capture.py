"""Persist the per-speaker audio a live session transcribes, so playing a
game doubles as reference-data capture for the A/B harness.

WhisperSink normally transcribes each speaker's buffer and throws the audio
away. That makes the gating human step of the accuracy epic
(discord-transcript-bot-3dn) impossible to satisfy from a normal session:
there is nothing on disk to score a config against. With capture enabled the
sink writes the EXACT wav bytes it fed the backend, plus a draft manifest in
the format scripts/ab_transcribe.py consumes, pre-filled with the machine
transcription so the human step is *correcting* a draft rather than typing
hours of audio from scratch.

The draft manifest is deliberately named ``manifest.draft.jsonl``: its
``reference`` fields are MACHINE output, not ground truth. Scoring a config
against un-corrected machine text measures nothing (a config would score ~0%
WER against its own output). The human renames it to ``manifest.jsonl`` once
the text is corrected; ab_transcribe warns loudly if handed a draft.

Everything here is best-effort and MUST NOT break a live session: every
public method swallows its own errors, and a hard failure disables capture
for the rest of the session rather than propagating into the transcription
path. discord-transcript-bot-h7j.
"""

import io
import json
import logging
import os
import re
import threading
import wave

logger = logging.getLogger(__name__)

MANIFEST_NAME = "manifest.draft.jsonl"
README_NAME = "README.md"
SESSION_INFO_NAME = "session_info.json"

# Discord voice is 48kHz/16-bit/stereo = ~192 KB per second of captured
# speech, so a long session with several talkative players runs to a few GB.
# Stop capturing past this ceiling rather than filling the disk mid-game;
# transcription is unaffected either way. Override with CAPTURE_MAX_GB.
DEFAULT_MAX_GB = 20.0

_UNSAFE = re.compile(r"[^A-Za-z0-9_-]+")


def _safe_name(value, fallback):
    """A filesystem-safe token for a clip filename. Player/character names
    are arbitrary user input (spaces, slashes, emoji), so squash anything
    that isn't alphanumeric and fall back when nothing usable is left."""
    token = _UNSAFE.sub("-", str(value or "")).strip("-")
    return token[:40] or str(fallback)


README_TEXT = """# Session audio capture

Per-speaker clips from a live session, captured so this session can serve as
reference data for the transcription A/B harness (discord-transcript-bot-3dn).

Each `.wav` is exactly what the bot fed the transcription engine for one
speaker's utterance: 48kHz/16-bit/stereo, one file per submitted segment.

## `manifest.draft.jsonl` is NOT ground truth

The `reference` field of every line holds the **machine transcription** — the
very thing the bake-off is supposed to be judged against. Scoring configs
against it would compare Whisper to itself and report a meaninglessly low
WER.

## Turning this into reference data

1. Open `manifest.draft.jsonl`. Each line is one clip, in the order the clips
   were submitted (so the file and a sorted directory listing agree):

       {"audio": "0007_Gus.wav", "reference": "I cast fireball",
        "player": "Gus", "character": "Thorin", "user_id": 254468368969629697,
        "begin": "19:30:45"}

2. Play each clip and **correct the `reference` text** to what was actually
   said — especially character and player names, which are the whole point of
   the proper-noun accuracy work. Only `audio` and `reference` are read by the
   harness; the other fields are context to help you review.
3. Lines with `"needs_reference": true` had their transcription dropped
   (silence, or the hallucination filter). Either type in the real words or
   delete the line. Do not leave an empty `reference` — an empty reference
   scores every hypothesis as fully wrong and skews the corpus WER.
4. Save it as `manifest.jsonl` (dropping `.draft`) once corrected.

## Running the bake-off

    python scripts/ab_transcribe.py --manifest captures/<this-dir>/manifest.jsonl

That prints per-config WER and real-time factor, which settles the open
`WHISPER_MODEL` / `WHISPER_BEAM_SIZE` decisions (discord-transcript-bot-d6j)
and unblocks the Parakeet (-mni) and VAD (-1s7) bake-offs.

`session_info.json` records the engine settings and the exact roster-biased
`initial_prompt` this session ran under. Set that same `initial_prompt` on the
configs you score — the harness defaults to the generic hint, which
understates the bot's real accuracy on names.

You do not have to correct every clip. A few hundred well-chosen seconds
spanning all speakers, including noisy/overlapping moments and plenty of
spoken names, is worth more than hours of clean solo narration.
"""


class SessionCapture:
    """Writes per-speaker clips + a draft manifest for one session.

    Thread-safety: ``save_clip`` is called from executor threads (several
    transcriptions run in parallel) and ``record`` from the commit thread, so
    all shared state is under one lock. The directory is created lazily on the
    first clip, so an enabled-but-silent session leaves nothing behind —
    matching the sink's lazy transcript-file behavior.
    """

    def __init__(self, directory, max_bytes=None, session_info=None):
        self.directory = directory
        # Engine/prompt settings this session ran under, written alongside the
        # clips. Without it a later bake-off can't reproduce the conditions
        # that produced the draft — in particular the roster-biased
        # initial_prompt, which is the bot's main proper-noun accuracy lever
        # and is rebuilt from a roster that may since have changed. Scoring
        # configs without that prompt understates the bot's real accuracy on
        # names (see the caveat in scripts/ab_transcribe.py).
        self.session_info = session_info or {}
        if max_bytes is None:
            try:
                max_bytes = float(os.getenv("CAPTURE_MAX_GB", DEFAULT_MAX_GB)) * 1024**3
            except (TypeError, ValueError):
                max_bytes = DEFAULT_MAX_GB * 1024**3
        self.max_bytes = max_bytes
        self._lock = threading.Lock()
        self._started = False
        self._disabled = False
        self._closed = False
        self.clips = 0
        self.bytes_written = 0
        self.audio_seconds = 0.0

    # -- internals -------------------------------------------------------

    def _disable(self, reason):
        """Turn capture off for the rest of the session. Called under lock."""
        if not self._disabled:
            self._disabled = True
            logger.warning(f"Session audio capture disabled: {reason}")

    def _ensure_dir(self):
        """Create the capture dir + README on first use. Under lock."""
        if self._started:
            return True
        os.makedirs(self.directory, exist_ok=True)
        readme = os.path.join(self.directory, README_NAME)
        try:
            with open(readme, "w", encoding="utf-8") as fh:
                fh.write(README_TEXT)
        except OSError as e:
            # A missing README doesn't invalidate the capture — keep going.
            logger.warning(f"Could not write capture README: {e}")
        if self.session_info:
            try:
                info_path = os.path.join(self.directory, SESSION_INFO_NAME)
                with open(info_path, "w", encoding="utf-8") as fh:
                    json.dump(self.session_info, fh, indent=2, ensure_ascii=False)
            except (OSError, TypeError, ValueError) as e:
                logger.warning(f"Could not write capture session info: {e}")
        self._started = True
        logger.info(f"Session audio capture writing to {self.directory}")
        return True

    @staticmethod
    def _wav_seconds(wav_bytes):
        with wave.open(io.BytesIO(wav_bytes), "rb") as w:
            rate = w.getframerate()
            return (w.getnframes() / float(rate)) if rate else 0.0

    # -- public API ------------------------------------------------------

    def save_clip(self, wav_bytes, seq, speaker):
        """Write one speaker segment's wav bytes. Returns the clip's basename
        (recorded later alongside its transcription), or None if capture is
        off/failed. Never raises."""
        try:
            with self._lock:
                if self._disabled or self._closed or not wav_bytes:
                    return None
                if self.bytes_written + len(wav_bytes) > self.max_bytes:
                    self._disable(
                        f"disk cap reached ({self.max_bytes / 1024**3:.1f} GB); "
                        f"{self.clips} clips kept. Transcription continues."
                    )
                    return None
                self._ensure_dir()
                who = _safe_name(
                    getattr(speaker, "player", None)
                    or getattr(speaker, "character", None),
                    getattr(speaker, "user", "unknown"),
                )
                # Zero-padded seq keeps the directory sorted in submission
                # order, which matches the manifest's line order. Fall back to
                # the clip counter when a caller has no sequence number, so two
                # segments from one speaker can never overwrite each other.
                ordinal = self.clips if seq is None else int(seq)
                name = f"{ordinal:04d}_{who}.wav"
                path = os.path.join(self.directory, name)
                with open(path, "wb") as fh:
                    fh.write(wav_bytes)
                self.clips += 1
                self.bytes_written += len(wav_bytes)
                try:
                    self.audio_seconds += self._wav_seconds(wav_bytes)
                except (wave.Error, EOFError):
                    pass
                return name
        except (OSError, ValueError) as e:
            with self._lock:
                self._disable(f"clip write failed ({e})")
            return None

    def record(self, clip_name, speaker, text, begin=None):
        """Append one manifest line pairing a saved clip with the machine
        transcription that becomes the human's starting draft. Never raises."""
        if not clip_name:
            return
        try:
            entry = {
                "audio": clip_name,
                "reference": (text or "").strip(),
                "player": getattr(speaker, "player", None),
                "character": getattr(speaker, "character", None),
                "user_id": getattr(speaker, "user", None),
            }
            if begin:
                entry["begin"] = begin
            if not entry["reference"]:
                # Dropped by the hallucination filter or genuinely silent.
                # Flag it so the human fills it in or deletes the line rather
                # than leaving an empty reference to skew the corpus WER.
                entry["needs_reference"] = True
            line = json.dumps(entry, ensure_ascii=False) + "\n"
            with self._lock:
                if self._disabled or self._closed:
                    return
                self._ensure_dir()
                path = os.path.join(self.directory, MANIFEST_NAME)
                with open(path, "a", encoding="utf-8") as fh:
                    fh.write(line)
        except (OSError, ValueError, TypeError) as e:
            with self._lock:
                self._disable(f"manifest write failed ({e})")

    def summary(self):
        """One-line human summary for the log / the /stop reply."""
        with self._lock:
            if not self._started:
                return "no audio captured"
            return (
                f"{self.clips} clips ({self.audio_seconds / 60:.1f} min audio, "
                f"{self.bytes_written / 1024**2:.0f} MB) in {self.directory}"
            )

    def close(self):
        """Finalize. Idempotent; never raises."""
        with self._lock:
            self._closed = True
        if self._started:
            logger.info(f"Session audio capture complete: {self.summary()}")
