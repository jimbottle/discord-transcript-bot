"""The bot's own log on disk: ``.logs/bot.log``, rotated.

Run from a terminal (``make start``), the bot logged only to stdout, so the
heartbeat's CPU warning and the UDP keep-alive warning that
discord-transcript-bot-309 needs were gone once the terminal closed.
Transcript lines are left out: they have their own per-day JSON log.
"""

import logging
import os
from logging.handlers import RotatingFileHandler

MAX_BYTES = 5 * 1024 * 1024
BACKUPS = 5


class _NotTranscripts(logging.Filter):
    def filter(self, record):
        return not record.name.startswith("transcription")


def install(directory=".logs", level=logging.INFO, fmt=None, datefmt=None):
    """Attach the rotating handler to the root logger; returns it. Never
    raises: an unwritable .logs/ must not stop the bot."""
    try:
        os.makedirs(directory, exist_ok=True)
        handler = RotatingFileHandler(
            os.path.join(directory, "bot.log"),
            maxBytes=MAX_BYTES,
            backupCount=BACKUPS,
            encoding="utf-8",
        )
    except OSError as e:
        logging.getLogger(__name__).warning("bot.log disabled: %s", e)
        return None
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(fmt, datefmt))
    handler.addFilter(_NotTranscripts())
    logging.getLogger().addHandler(handler)
    return handler
