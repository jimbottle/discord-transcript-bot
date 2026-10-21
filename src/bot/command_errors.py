"""What a slash command tells the user when it fails (discord-transcript-bot-8gy).

Without a handler, an exception inside a command reaches only the log: the
user sees "The application did not respond", or a "thinking..." that never
ends after a defer. VoloBot.on_application_command_error sends every such
failure through ``report``.
"""

import asyncio
import logging

import discord

logger = logging.getLogger(__name__)


def _unwrap(error):
    # Pycord wraps the command's own exception.
    return getattr(error, "original", None) or error


def user_message(error) -> str:
    """A short, actionable message for ``error``. Never empty: ``str()`` of
    a TimeoutError is "", which used to produce "Failed to connect: "."""
    err = _unwrap(error)
    if isinstance(err, discord.CheckFailure):
        return "You can't use that command here."
    if isinstance(err, (asyncio.TimeoutError, TimeoutError)):
        return "Discord didn't answer in time. Try again in a moment."
    if isinstance(err, discord.Forbidden):
        return (
            "I don't have permission to do that here. Check the bot's role "
            "permissions for this channel."
        )
    if isinstance(err, discord.HTTPException):
        return f"Discord rejected the request (HTTP {err.status}). Try again."
    if isinstance(err, discord.ClientException):
        # Voice errors and others alike (InteractionResponded, ...).
        detail = str(err).strip().rstrip(".")
        return f"Discord client error: {detail}." if detail else "Discord client error."
    return "Something went wrong running that command. The details are in the bot logs."


async def report(ctx, error) -> None:
    """Log ``error`` with its traceback and tell the user. Never raises."""
    err = _unwrap(error)
    name = getattr(getattr(ctx, "command", None), "qualified_name", "?")
    logger.error(
        "/%s failed: %s", name, err, exc_info=(type(err), err, err.__traceback__)
    )
    msg = user_message(err)
    try:
        if ctx.response.is_done():
            await ctx.followup.send(msg, ephemeral=True)
        else:
            await ctx.respond(msg, ephemeral=True)
    except Exception as send_err:  # noqa: BLE001 - nothing left to tell
        logger.error("could not report /%s failure to the user: %s", name, send_err)
