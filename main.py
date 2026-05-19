import asyncio
import logging
import os
import time
from datetime import datetime

import discord
import ollama
import yaml
from dotenv import load_dotenv

from src.bot.helper import BotHelper
from src.config.cliargs import CLIArgs
from src.config.ollama_config import get_ask_model
from src.utils.answer import clamp_message, clean_ollama_answer
from src.utils.commandline import CommandLine
from src.utils.pdf_generator import pdf_generator
from src.utils.voice import disconnect_targets

load_dotenv()
DISCORD_BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN")
PLAYER_MAP_FILE_PATH = os.getenv("PLAYER_MAP_FILE_PATH")
# Model used by /ask. Resolved via the shared resolver (empty/unset env
# → default) so main.py and src/bot/health.py can never disagree about
# which model /ask actually uses.
ASK_OLLAMA_MODEL = get_ask_model()
# Discord hard-rejects messages longer than this many characters.
DISCORD_MESSAGE_LIMIT = 2000

logger = logging.getLogger()  # root logger


def configure_logging():
    logging.getLogger("discord").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    logging.getLogger("faster_whisper").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    # Ensure the directory exists
    log_directory = ".logs/transcripts"
    pdf_directory = ".logs/pdfs"
    os.makedirs(log_directory, exist_ok=True)
    os.makedirs(pdf_directory, exist_ok=True)

    # Get the current date for the log file name
    current_date = datetime.now().strftime("%Y-%m-%d")
    log_filename = os.path.join(log_directory, f"{current_date}-transcription.log")

    # Custom logging format (date with milliseconds, message)
    log_format = "%(asctime)s %(name)s: %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S.%f"[:-3]  # Trim to milliseconds

    if CLIArgs.verbose:
        logger.setLevel(logging.DEBUG)
        logging.basicConfig(level=logging.DEBUG, format=log_format, datefmt=date_format)
    else:
        logger.setLevel(logging.INFO)
        logging.basicConfig(level=logging.INFO, format=log_format, datefmt=date_format)

    # Set up the transcription logger
    transcription_logger = logging.getLogger("transcription")
    transcription_logger.setLevel(logging.INFO)

    # File handler for transcription logs (append mode)
    file_handler = logging.FileHandler(log_filename, mode="a")
    file_handler.setLevel(logging.INFO)

    # Custom formatter WITHOUT the automatic timestamp
    file_handler.setFormatter(
        logging.Formatter(
            "%(message)s"  # Only log the custom message, no automatic timestamp
        )
    )

    # Add the handler to the transcription logger
    transcription_logger.addHandler(file_handler)


if __name__ == "__main__":
    args = CommandLine.read_command_line()
    CLIArgs.update_from_args(args)

    configure_logging()
    loop = asyncio.get_event_loop()

    from src.bot.volo_bot import VoloBot

    bot = VoloBot(loop)

    @bot.event
    async def on_voice_state_update(member, before, after):
        if member.id == bot.user.id:
            # If the bot left the "before" channel
            if after.channel is None:
                guild_id = before.channel.guild.id
                helper = bot.guild_to_helper.get(guild_id, None)
                if helper:
                    helper.set_vc(None)
                    bot.guild_to_helper.pop(guild_id, None)

                bot._close_and_clean_sink_for_guild(guild_id)

    @bot.slash_command(name="connect", description="Join your voice channel.")
    async def connect(ctx: discord.context.ApplicationContext):
        if bot._is_ready is False:
            await ctx.respond(
                "System's still booting up, choom. Try again in a sec.", ephemeral=True
            )
            return
        await ctx.defer()
        # Quick gateway sanity check — full health was verified at startup.
        # Avoid running heavy checks (whisper transcription) in a thread here
        # as GIL contention can starve the gateway heartbeat.
        if not bot.is_ready() or bot.latency == float("inf") or bot.latency > 5.0:
            await ctx.followup.send(
                "Gateway connection is unstable. Try again in a moment, or run `/health` for details.",
            )
            return
        author_vc = ctx.author.voice
        if not author_vc:
            await ctx.followup.send(
                "You're not in a voice channel. Jack in first, then call me."
            )
            return
        # check if we are already connected or mid-connection
        guild_id = ctx.guild_id
        if bot.guild_to_helper.get(guild_id, None):
            await ctx.followup.send(
                "Already connected on this server. One channel at a time."
            )
            return
        if guild_id in bot._connecting_guilds:
            await ctx.followup.send("Already connecting. Hold on.")
            return
        bot._connecting_guilds.add(guild_id)
        try:
            # Clean up any ghost voice client from a crashed previous session
            existing_vc = ctx.guild.voice_client
            if existing_vc:
                logger.warning(f"Cleaning up ghost voice client for guild {guild_id}")
                try:
                    await existing_vc.disconnect(force=True)
                except Exception:
                    pass
                await asyncio.sleep(1)

            vc = await author_vc.channel.connect(reconnect=False, timeout=15)
            if not vc.is_connected():
                try:
                    await vc.disconnect(force=True)
                except Exception:
                    pass
                try:
                    await ctx.guild.change_voice_state(channel=None)
                except Exception:
                    pass
                await ctx.followup.send(
                    "Voice connection failed — could not establish a stable connection. Try again."
                )
                return
            helper = bot.guild_to_helper.get(guild_id, BotHelper(bot))
            helper.guild_id = guild_id
            helper.set_vc(vc)
            bot.guild_to_helper[guild_id] = helper
            bot._write_runtime_state()  # connected (best-effort, never raises)
            await ctx.followup.send(
                "Jacked in. Connected to the voice channel and standing by."
            )
            await ctx.guild.change_voice_state(
                channel=author_vc.channel, self_mute=True
            )
        except Exception as e:
            # Force-leave at the gateway level. The bot appears in the
            # channel as soon as OP 4 is sent (before voice WS connects),
            # so we must send channel=None to leave even if VoiceClient
            # was never fully constructed.
            remaining_vc = ctx.guild.voice_client
            if remaining_vc:
                try:
                    await remaining_vc.disconnect(force=True)
                except Exception:
                    pass
            try:
                await ctx.guild.change_voice_state(channel=None)
            except Exception:
                pass
            bot.guild_to_helper.pop(guild_id, None)
            await ctx.followup.send(f"Failed to connect to voice: {e}")
        finally:
            bot._connecting_guilds.discard(guild_id)

    @bot.slash_command(name="scribe", description="Start transcribing voice to text.")
    async def ink(ctx: discord.context.ApplicationContext):
        await ctx.trigger_typing()
        connect_command = next(
            (cmd for cmd in ctx.bot.application_commands if cmd.name == "connect"), None
        )
        if not connect_command:
            connect_text = "`/connect`"
        else:
            connect_text = f"</connect:{connect_command.id}>"
        if not bot.guild_to_helper.get(ctx.guild_id, None):
            await ctx.respond(
                f"Not connected yet. Use {connect_text} first.", ephemeral=True
            )
            return
        # check if we are already scribing
        if bot.guild_is_recording.get(ctx.guild_id, False):
            await ctx.respond(
                "Already recording. Can't run two taps at once.", ephemeral=True
            )
            return
        bot.start_recording(ctx)
        await ctx.respond(
            "Recording. Every word in this channel is being transcribed in realtime.",
            ephemeral=False,
        )

    @bot.slash_command(name="stop", description="Stop transcribing.")
    async def stop(ctx: discord.context.ApplicationContext):
        guild_id = ctx.guild_id
        helper = bot.guild_to_helper.get(guild_id, None)
        if not helper:
            await ctx.respond(
                "Not connected to a channel. Nothing to stop.", ephemeral=True
            )
            return

        bot_vc = helper.vc

        if not bot_vc:
            await ctx.respond(
                "Not connected to a channel. Nothing to stop.", ephemeral=True
            )
            return

        if not bot.guild_is_recording.get(guild_id, False):
            await ctx.respond(
                "Not recording right now. Nothing to kill.", ephemeral=True
            )
            return

        # Teardown (voice-thread join, sink close) can take >3s. Defer
        # first so the interaction token stays valid (~15 min) instead
        # of expiring mid-teardown and 404ing as "The application did
        # not respond". Guard the teardown so we always confirm.
        await ctx.defer()
        teardown_ok = True
        # Independent try blocks: a get_transcription failure must NOT
        # skip the actual teardown (that would leave guild_is_recording
        # stuck True and leak the sink — the exact class this path
        # exists to prevent).
        try:
            await bot.get_transcription(ctx)
        except Exception as e:
            logger.error(f"/stop get_transcription error: {e}")
            teardown_ok = False
        try:
            # end_recording_session keeps vc.stop_recording() on the
            # loop thread (thread-safe after-callback scheduling; carries
            # a bounded ~ms/≤5s router join) and offloads the long
            # voice-thread join. See end_recording_session's TRADEOFF
            # docstring for why the loop isn't fully freed.
            await bot.end_recording_session(ctx)
        except Exception as e:
            logger.error(f"/stop end_recording_session error: {e}")
            teardown_ok = False
        # Guarantee the flag clears even if teardown raised, so a later
        # /scribe can't see "Already recording".
        bot.guild_is_recording.pop(guild_id, None)
        if teardown_ok:
            await ctx.followup.send(
                "Recording stopped. Data saved. Standing by for the next run."
            )
        else:
            await ctx.followup.send(
                "Recording stopped, but cleanup hit an error — check the logs. "
                "If the bot seems stuck, try `/disconnect`."
            )

    @bot.slash_command(name="disconnect", description="Leave the voice channel.")
    async def disconnect(ctx: discord.context.ApplicationContext):
        guild_id = ctx.guild_id
        id_exists = bot.guild_to_helper.get(guild_id, None)
        if not id_exists:
            await ctx.respond(
                "Not connected to anything on this server.", ephemeral=True
            )
            return

        helper = bot.guild_to_helper[guild_id]
        bot_vc = helper.vc

        if not bot_vc:
            await ctx.respond(
                "Lost the connection somehow. Try reconnecting.", ephemeral=True
            )
            return

        # Teardown + voice disconnect can take >3s; defer so the
        # interaction token stays valid instead of expiring and 404ing
        # as "The application did not respond".
        await ctx.defer()

        # Archive/teardown any active recording BEFORE disconnecting so
        # the transcript is flushed and guild_is_recording is cleared.
        # Independent try blocks so a get_transcription failure can't
        # skip the actual teardown, and neither can skip the disconnect.
        teardown_ok = True
        if bot.guild_is_recording.get(guild_id, False):
            try:
                await bot.get_transcription(ctx)
            except Exception as e:
                logger.error(f"/disconnect get_transcription error: {e}")
                teardown_ok = False
            try:
                # Keeps vc.stop_recording() on the loop thread (safe
                # after-callback scheduling; bounded ~ms/≤5s router
                # join) and offloads the long voice-thread join. See
                # end_recording_session's TRADEOFF docstring.
                await bot.end_recording_session(ctx)
            except Exception as e:
                logger.error(f"/disconnect end_recording_session error: {e}")
                teardown_ok = False

        # helper.vc can be a STALE voice client after a voice reconnect,
        # so disconnecting it alone leaves the bot in the channel while
        # still reporting success. Disconnect the live client and any
        # others bound to this guild too (see disconnect_targets).
        for vc in disconnect_targets(
            ctx.guild.voice_client if ctx.guild else None,
            bot_vc,
            list(bot.voice_clients),
            guild_id,
        ):
            try:
                await vc.disconnect(force=True)
            except Exception as e:
                logger.error(f"/disconnect: error disconnecting a voice client: {e}")

        helper.guild_id = None
        helper.set_vc(None)
        bot.guild_to_helper.pop(guild_id, None)
        # Clear the flag even if it was stale-True, so a later /scribe
        # after reconnecting starts clean.
        bot.guild_is_recording.pop(guild_id, None)
        bot._write_runtime_state()  # disconnected (best-effort, never raises)

        if teardown_ok:
            await ctx.followup.send(
                "Disconnected. Session archived. Catch you on the next one, chooms."
            )
        else:
            await ctx.followup.send(
                "Disconnected, but session cleanup hit an error — check the logs."
            )

    @bot.slash_command(
        name="generate_pdf", description="Export the transcript as a PDF."
    )
    async def generate_pdf(ctx: discord.context.ApplicationContext):
        guild_id = ctx.guild_id
        helper = bot.guild_to_helper.get(guild_id, None)
        if not helper:
            await ctx.respond("Not connected. Nothing to export.", ephemeral=True)
            return
        transcription = await bot.get_transcription(ctx)
        if not transcription:
            await ctx.respond(
                "No transcript data yet. Start a recording first.", ephemeral=True
            )
            return
        pdf_file_path = await pdf_generator(transcription)
        # Send the PDF as an attachment
        if os.path.exists(pdf_file_path):
            try:
                with open(pdf_file_path, "rb") as f:
                    discord_file = discord.File(
                        f, filename=f"session_transcription.pdf"
                    )
                    await ctx.respond(
                        "Dossier compiled. Here's your transcript.", file=discord_file
                    )
            finally:
                os.remove(pdf_file_path)
        else:
            await ctx.respond("PDF generation failed. Check the logs.", ephemeral=True)

    @bot.slash_command(
        name="update_player_map",
        description="Sync player names with the server roster.",
    )
    async def update_player_map(ctx: discord.context.ApplicationContext):
        if bot.guild_is_recording.get(ctx.guild_id, False):
            await ctx.respond(
                "Can't update the roster while recording. Stop the session first.",
                ephemeral=True,
            )
            return
        try:
            await bot.update_player_map(ctx)
            await ctx.respond("Roster synced. All player handles are up to date.")
        except Exception as e:
            await ctx.respond(f"Roster sync failed:\n{e}", ephemeral=True)
            raise e

    async def _run_ask(
        ctx: discord.context.ApplicationContext,
        question: str,
        server: str | None,
        *,
        public: bool,
    ):
        """Shared implementation for /ask and /ask-public.

        `public` controls only the visibility of the deferred response
        and the final answer: False → ephemeral (only the asker sees
        it), True → posted to the whole channel. The early validation
        responses below stay ephemeral in both cases (a "no transcript
        yet" / "not connected" message shouldn't be broadcast).
        """
        # Resolve the guild_id — either from the guild context or by finding shared guilds in DMs
        if ctx.guild_id:
            guild_id = ctx.guild_id
        else:
            # DM context: find guilds with active transcripts where the user is a member
            active_guilds = []
            for gid, sink in bot.guild_whisper_sinks.items():
                if not (
                    hasattr(sink, "session_file") and os.path.exists(sink.session_file)
                ):
                    continue
                guild = bot.get_guild(gid)
                if not guild:
                    continue
                try:
                    await guild.fetch_member(ctx.author.id)
                    active_guilds.append(gid)
                except discord.NotFound:
                    continue

            if not active_guilds:
                await ctx.respond(
                    "No active transcripts found in any of your shared servers.",
                    ephemeral=True,
                )
                return

            if len(active_guilds) == 1:
                guild_id = active_guilds[0]
            elif server:
                # Match by server name (case-insensitive)
                match = next(
                    (
                        gid
                        for gid in active_guilds
                        if bot.get_guild(gid)
                        and bot.get_guild(gid).name.lower() == server.lower()
                    ),
                    None,
                )
                if not match:
                    names = ", ".join(
                        f"`{bot.get_guild(gid).name}`"
                        for gid in active_guilds
                        if bot.get_guild(gid)
                    )
                    await ctx.respond(
                        f"No matching server found. Active transcripts in: {names}",
                        ephemeral=True,
                    )
                    return
                guild_id = match
            else:
                names = ", ".join(
                    f"`{bot.get_guild(gid).name}`"
                    for gid in active_guilds
                    if bot.get_guild(gid)
                )
                await ctx.respond(
                    f"Multiple servers have active transcripts: {names}\n"
                    "Use the `server` option to pick one.",
                    ephemeral=True,
                )
                return

        whisper_sink = bot.guild_whisper_sinks.get(guild_id, None)

        # Try to read the session transcript file
        transcript_text = None
        if (
            whisper_sink
            and hasattr(whisper_sink, "session_file")
            and os.path.exists(whisper_sink.session_file)
        ):
            with open(whisper_sink.session_file, "r", encoding="utf-8") as f:
                transcript_text = f.read()

        if not transcript_text or not transcript_text.strip():
            await ctx.respond(
                "No transcript data in the current session yet. Start recording first.",
                ephemeral=True,
            )
            return

        # ephemeral=not public: /ask is private to the asker,
        # /ask-public posts the answer to the whole channel. The
        # followups below inherit this ephemerality from the defer.
        await ctx.defer(ephemeral=not public)

        try:
            response = ollama.chat(
                model=ASK_OLLAMA_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an assistant that answers questions about a voice chat transcript. Be concise and direct. Only reference what's actually in the transcript. If the answer isn't in the transcript, say so.",
                    },
                    {
                        "role": "user",
                        "content": f"Here is the transcript:\n\n{transcript_text}\n\nQuestion: {question}",
                    },
                ],
                # /ask wants a short grounded answer, not chain-of-thought.
                # think=False disables native thinking (Gemma 4 defaults it
                # on); clean_ollama_answer strips inline <think> for models
                # that ignore it. num_predict caps latency — Discord
                # truncates at ~1900 chars anyway. temperature=0 keeps
                # answers deterministic and transcript-grounded AND mirrors
                # the local-models bench harness (bench.py runs think=False,
                # num_predict=512, temperature=0) so a model that scores
                # well there behaves the same here.
                think=False,
                options={"num_predict": 512, "temperature": 0},
            )
            answer = clean_ollama_answer(response["message"]["content"])

            if not answer:
                await ctx.followup.send(
                    "The model returned no answer (it may have produced only "
                    "internal reasoning). Try rephrasing, or set "
                    "`ASK_OLLAMA_MODEL` to a non-reasoning model."
                )
                return

            # clean_ollama_answer only caps the answer; a long /ask
            # question can still push the composed message past Discord's
            # hard limit. clamp_message is the unit-tested final guard.
            message = clamp_message(
                f"**Q:** {question}\n\n{answer}", DISCORD_MESSAGE_LIMIT
            )
            await ctx.followup.send(message)
        except Exception as e:
            # An ollama client older than 0.5.x rejects the think= kwarg
            # with `TypeError: chat() got an unexpected keyword argument
            # 'think'`. Special-case ONLY that exact message so an
            # unrelated TypeError elsewhere in the block isn't
            # misattributed to the client version.
            if isinstance(
                e, TypeError
            ) and "unexpected keyword argument 'think'" in str(e):
                logger.error(f"Ollama client error: {e}")
                await ctx.followup.send(
                    "The Ollama Python client is too old for `/ask` "
                    "(needs the `think` parameter). Upgrade it: "
                    "`pip install -U 'ollama>=0.5.1'`.\n"
                    f"`{e}`"
                )
            else:
                logger.error(f"Ollama error: {e}")
                await ctx.followup.send(
                    f"Failed to query the model. Make sure Ollama is running.\n`{e}`"
                )

    @bot.slash_command(
        name="ask",
        description="Ask about the transcript — answer is private to you.",
        contexts={
            discord.InteractionContextType.guild,
            discord.InteractionContextType.bot_dm,
        },
    )
    async def ask(
        ctx: discord.context.ApplicationContext,
        question: discord.Option(str, description="Your question about the transcript"),
        server: discord.Option(
            str,
            description="Server name (only needed in DMs if you share multiple servers)",
            required=False,
            default=None,
        ),
    ):
        await _run_ask(ctx, question, server, public=False)

    @bot.slash_command(
        name="ask-public",
        description="Ask about the transcript — answer is posted to the whole channel.",
        contexts={
            discord.InteractionContextType.guild,
            discord.InteractionContextType.bot_dm,
        },
    )
    async def ask_public(
        ctx: discord.context.ApplicationContext,
        question: discord.Option(str, description="Your question about the transcript"),
        server: discord.Option(
            str,
            description="Server name (only needed in DMs if you share multiple servers)",
            required=False,
            default=None,
        ),
    ):
        await _run_ask(ctx, question, server, public=True)

    @bot.slash_command(name="help", description="Show available commands.")
    async def help(ctx: discord.context.ApplicationContext):
        embed_fields = [
            discord.EmbedField(
                name="/connect", value="Join your voice channel.", inline=True
            ),
            discord.EmbedField(
                name="/disconnect", value="Leave the voice channel.", inline=True
            ),
            discord.EmbedField(
                name="/scribe", value="Start transcribing.", inline=True
            ),
            discord.EmbedField(name="/stop", value="Stop transcribing.", inline=True),
            discord.EmbedField(
                name="/ask",
                value="Ask about the transcript (private to you).",
                inline=True,
            ),
            discord.EmbedField(
                name="/ask-public",
                value="Ask about the transcript (answer shown to everyone).",
                inline=True,
            ),
            discord.EmbedField(
                name="/generate_pdf", value="Export transcript as PDF.", inline=True
            ),
            discord.EmbedField(
                name="/update_player_map",
                value="Sync player names with roster.",
                inline=True,
            ),
            discord.EmbedField(
                name="/health", value="Show system health status.", inline=True
            ),
            discord.EmbedField(name="/help", value="Show this menu.", inline=True),
        ]

        embed = discord.Embed(
            title="// TRANSCRIPT-BOT",
            description="""Realtime voice-to-text. All comms logged.""",
            color=discord.Color.from_rgb(0, 255, 136),
            fields=embed_fields,
        )

        await ctx.respond(embed=embed, ephemeral=True)

    @bot.slash_command(name="health", description="Show system health status.")
    async def health(ctx: discord.context.ApplicationContext):
        await ctx.defer(ephemeral=True)
        # Reuse the bot's existing HealthCheck instance so any ollama process
        # it spawned at startup stays tracked, and to avoid re-loading the
        # whisper model on every /health invocation.
        await asyncio.to_thread(bot.health.run_all, autofix=False, bot=bot)
        status = (
            "All systems operational."
            if bot.health.all_ok()
            else "Critical checks failing."
        )
        await ctx.followup.send(f"**{status}**\n```\n{bot.health.summary()}\n```")

    try:
        loop.run_until_complete(bot.start(DISCORD_BOT_TOKEN))
    except KeyboardInterrupt:
        logger.info("^C received, shutting down...")
        asyncio.run(bot.stop_and_cleanup())
    finally:
        # Close all connections
        loop.run_until_complete(bot.close_consumers())

        tasks = [t for t in asyncio.all_tasks(loop) if not t.done()]
        for task in tasks:
            task.cancel()
        loop.run_until_complete(asyncio.gather(*tasks, return_exceptions=True))

        # Close the loop
        loop.run_until_complete(bot.close())
        loop.close()
