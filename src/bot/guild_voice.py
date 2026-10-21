"""Per-guild voice connection record.

``VoloBot.guild_to_helper`` maps a guild id to one of these. It remembers the
voice client the bot joined with so the slash commands (``/scribe``,
``/stop``, ``/disconnect``) can find it again, and nothing more: it holds no
behaviour of its own, and clearing it is how a guild is marked disconnected.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class GuildVoice:
    """The voice client the bot holds in one guild, or none after leaving."""

    __slots__ = ("bot", "guild_id", "vc")

    def __init__(self, bot, guild_id=None, vc=None):
        self.bot = bot
        self.guild_id = guild_id
        self.vc = vc

    @property
    def connected(self) -> bool:
        return self.vc is not None

    def set_vc(self, voice_client) -> None:
        """Replace the cached voice client. ``None`` means the bot left.

        Note the cache can go stale after a voice reconnect, so callers that
        disconnect should also look at ``guild.voice_client``
        (``src.utils.voice.disconnect_targets``)."""
        self.vc = voice_client
        if voice_client is None:
            logger.debug("Guild %s: voice client cleared.", self.guild_id)

    def clear(self) -> None:
        """Forget both the guild and the client; used on ``/disconnect``."""
        self.set_vc(None)
        self.guild_id = None

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        state = "connected" if self.connected else "idle"
        return f"GuildVoice(guild_id={self.guild_id}, {state})"
