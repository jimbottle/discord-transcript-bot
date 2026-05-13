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
from src.utils.commandline import CommandLine
from src.utils.pdf_generator import pdf_generator

load_dotenv()
DISCORD_BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN")
PLAYER_MAP_FILE_PATH = os.getenv("PLAYER_MAP_FILE_PATH")

logger = logging.getLogger()  # root logger


def configure_logging():
    logging.getLogger('discord').setLevel(logging.WARNING)
    logging.getLogger('asyncio').setLevel(logging.WARNING)
    logging.getLogger('faster_whisper').setLevel(logging.WARNING)
    logging.getLogger('httpx').setLevel(logging.WARNING)
    logging.getLogger('httpcore').setLevel(logging.WARNING)

    # Ensure the directory exists
    log_directory = '.logs/transcripts'
    pdf_directory = '.logs/pdfs'
    os.makedirs(log_directory, exist_ok=True)
    os.makedirs(pdf_directory, exist_ok=True)

    # Get the current date for the log file name
    current_date = datetime.now().strftime('%Y-%m-%d')
    log_filename = os.path.join(log_directory, f"{current_date}-transcription.log")

    # Custom logging format (date with milliseconds, message)
    log_format = '%(asctime)s %(name)s: %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S.%f'[:-3]  # Trim to milliseconds

    if CLIArgs.verbose:
        logger.setLevel(logging.DEBUG)
        logging.basicConfig(level=logging.DEBUG,
                            format=log_format,
                            datefmt=date_format)
    else:
        logger.setLevel(logging.INFO)
        logging.basicConfig(level=logging.INFO,
                            format=log_format,
                            datefmt=date_format)

    # Set up the transcription logger
    transcription_logger = logging.getLogger('transcription')
    transcription_logger.setLevel(logging.INFO)

    # File handler for transcription logs (append mode)
    file_handler = logging.FileHandler(log_filename, mode='a')
    file_handler.setLevel(logging.INFO)

    # Custom formatter WITHOUT the automatic timestamp
    file_handler.setFormatter(logging.Formatter(
        '%(message)s'  # Only log the custom message, no automatic timestamp
    ))

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
            await ctx.respond("System's still booting up, choom. Try again in a sec.", ephemeral=True)
            return
        await ctx.defer()
        # Quick gateway sanity check — full health was verified at startup.
        # Avoid running heavy checks (whisper transcription) in a thread here
        # as GIL contention can starve the gateway heartbeat.
        if not bot.is_ready() or bot.latency == float('inf') or bot.latency > 5.0:
            await ctx.followup.send(
                "Gateway connection is unstable. Try again in a moment, or run `/health` for details.",
            )
            return
        author_vc = ctx.author.voice
        if not author_vc:
            await ctx.followup.send("You're not in a voice channel. Jack in first, then call me.")
            return
        # check if we are already connected or mid-connection
        guild_id = ctx.guild_id
        if bot.guild_to_helper.get(guild_id, None):
            await ctx.followup.send("Already connected on this server. One channel at a time.")
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
                await ctx.followup.send("Voice connection failed — could not establish a stable connection. Try again.")
                return
            helper = bot.guild_to_helper.get(guild_id, BotHelper(bot))
            helper.guild_id = guild_id
            helper.set_vc(vc)
            bot.guild_to_helper[guild_id] = helper
            await ctx.followup.send("Jacked in. Connected to the voice channel and standing by.")
            await ctx.guild.change_voice_state(channel=author_vc.channel, self_mute=True)
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
        connect_command = next((cmd for cmd in ctx.bot.application_commands if cmd.name == "connect"), None)
        if not connect_command:
            connect_text = "`/connect`"
        else:
            connect_text = f"</connect:{connect_command.id}>"
        if not bot.guild_to_helper.get(ctx.guild_id, None):
            await ctx.respond(f"Not connected yet. Use {connect_text} first.", ephemeral=True)
            return
        # check if we are already scribing
        if bot.guild_is_recording.get(ctx.guild_id, False):
            await ctx.respond("Already recording. Can't run two taps at once.", ephemeral=True)
            return
        bot.start_recording(ctx)
        await ctx.respond("Recording. Every word in this channel is being transcribed in realtime.", ephemeral=False)

    @bot.slash_command(name="stop", description="Stop transcribing.")
    async def stop(ctx: discord.context.ApplicationContext):
        guild_id = ctx.guild_id
        helper = bot.guild_to_helper.get(guild_id, None)
        if not helper:
            await ctx.respond("Not connected to a channel. Nothing to stop.", ephemeral=True)
            return

        bot_vc = helper.vc

        if not bot_vc:
            await ctx.respond("Not connected to a channel. Nothing to stop.", ephemeral=True)
            return

        if not bot.guild_is_recording.get(guild_id, False):
            await ctx.respond("Not recording right now. Nothing to kill.", ephemeral=True)
            return

        await ctx.trigger_typing()

        if bot.guild_is_recording.get(guild_id, False):
            await bot.get_transcription(ctx)
            bot.stop_recording(ctx)
            bot.guild_is_recording[guild_id] = False
            await ctx.respond("Recording stopped. Data saved. Standing by for the next run.", ephemeral=False)
            bot.cleanup_sink(ctx)

    @bot.slash_command(name="disconnect", description="Leave the voice channel.")
    async def disconnect(ctx: discord.context.ApplicationContext):
        guild_id = ctx.guild_id
        id_exists = bot.guild_to_helper.get(guild_id, None)
        if not id_exists:
            await ctx.respond("Not connected to anything on this server.", ephemeral=True)
            return

        helper = bot.guild_to_helper[guild_id]
        bot_vc = helper.vc

        if not bot_vc:
            await ctx.respond("Lost the connection somehow. Try reconnecting.", ephemeral=True)
            return

        await ctx.trigger_typing()
        await bot_vc.disconnect()
        helper.guild_id = None
        helper.set_vc(None)
        bot.guild_to_helper.pop(guild_id, None)

        await ctx.respond("Disconnected. Session archived. Catch you on the next one, chooms.", ephemeral=False)

    @bot.slash_command(name="generate_pdf", description="Export the transcript as a PDF.")
    async def generate_pdf(ctx: discord.context.ApplicationContext):
        guild_id = ctx.guild_id
        helper = bot.guild_to_helper.get(guild_id, None)
        if not helper:
            await ctx.respond("Not connected. Nothing to export.", ephemeral=True)
            return
        transcription = await bot.get_transcription(ctx)
        if not transcription:
            await ctx.respond("No transcript data yet. Start a recording first.", ephemeral=True)
            return
        pdf_file_path = await pdf_generator(transcription)
        # Send the PDF as an attachment
        if os.path.exists(pdf_file_path):
            try:
                with open(pdf_file_path, "rb") as f:
                    discord_file = discord.File(f, filename=f"session_transcription.pdf")
                    await ctx.respond("Dossier compiled. Here's your transcript.", file=discord_file)
            finally:
                os.remove(pdf_file_path)
        else:
            await ctx.respond("PDF generation failed. Check the logs.", ephemeral=True)


    @bot.slash_command(name="update_player_map", description="Sync player names with the server roster.")
    async def update_player_map(ctx: discord.context.ApplicationContext):
        if bot.guild_is_recording.get(ctx.guild_id, False):
            await ctx.respond("Can't update the roster while recording. Stop the session first.", ephemeral=True)
            return
        try:
            await bot.update_player_map(ctx)
            await ctx.respond("Roster synced. All player handles are up to date.")
        except Exception as e:
            await ctx.respond(f"Roster sync failed:\n{e}", ephemeral=True)
            raise e


    @bot.slash_command(
        name="ask",
        description="Ask a question about the current transcript.",
        contexts={discord.InteractionContextType.guild, discord.InteractionContextType.bot_dm},
    )
    async def ask(ctx: discord.context.ApplicationContext,
                  question: discord.Option(str, description="Your question about the transcript"),
                  server: discord.Option(str, description="Server name (only needed in DMs if you share multiple servers)", required=False, default=None)):

        # Resolve the guild_id — either from the guild context or by finding shared guilds in DMs
        if ctx.guild_id:
            guild_id = ctx.guild_id
        else:
            # DM context: find guilds with active transcripts where the user is a member
            active_guilds = []
            for gid, sink in bot.guild_whisper_sinks.items():
                if not (hasattr(sink, 'session_file') and os.path.exists(sink.session_file)):
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
                    "No active transcripts found in any of your shared servers.", ephemeral=True)
                return

            if len(active_guilds) == 1:
                guild_id = active_guilds[0]
            elif server:
                # Match by server name (case-insensitive)
                match = next(
                    (gid for gid in active_guilds
                     if bot.get_guild(gid) and bot.get_guild(gid).name.lower() == server.lower()),
                    None,
                )
                if not match:
                    names = ", ".join(
                        f"`{bot.get_guild(gid).name}`" for gid in active_guilds if bot.get_guild(gid))
                    await ctx.respond(
                        f"No matching server found. Active transcripts in: {names}", ephemeral=True)
                    return
                guild_id = match
            else:
                names = ", ".join(
                    f"`{bot.get_guild(gid).name}`" for gid in active_guilds if bot.get_guild(gid))
                await ctx.respond(
                    f"Multiple servers have active transcripts: {names}\n"
                    "Use the `server` option to pick one.", ephemeral=True)
                return

        whisper_sink = bot.guild_whisper_sinks.get(guild_id, None)

        # Try to read the session transcript file
        transcript_text = None
        if whisper_sink and hasattr(whisper_sink, 'session_file') and os.path.exists(whisper_sink.session_file):
            with open(whisper_sink.session_file, "r", encoding="utf-8") as f:
                transcript_text = f.read()

        if not transcript_text or not transcript_text.strip():
            await ctx.respond("No transcript data in the current session yet. Start recording first.", ephemeral=True)
            return

        await ctx.defer()

        try:
            response = ollama.chat(
                model="ai/mistral:latest",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an assistant that answers questions about a voice chat transcript. Be concise and direct. Only reference what's actually in the transcript. If the answer isn't in the transcript, say so."
                    },
                    {
                        "role": "user",
                        "content": f"Here is the transcript:\n\n{transcript_text}\n\nQuestion: {question}"
                    }
                ]
            )
            answer = response['message']['content']

            # Discord has a 2000 char limit
            if len(answer) > 1900:
                answer = answer[:1900] + "\n\n...(truncated)"

            await ctx.followup.send(f"**Q:** {question}\n\n{answer}")
        except Exception as e:
            logger.error(f"Ollama error: {e}")
            await ctx.followup.send(f"Failed to query the model. Make sure Ollama is running.\n`{e}`")

    @bot.slash_command(name="help", description="Show available commands.")
    async def help(ctx: discord.context.ApplicationContext):
        embed_fields = [
            discord.EmbedField(
                name="/connect", value="Join your voice channel.", inline=True),
            discord.EmbedField(
                name="/disconnect", value="Leave the voice channel.", inline=True),
            discord.EmbedField(
                name="/scribe", value="Start transcribing.", inline=True),
            discord.EmbedField(
                name="/stop", value="Stop transcribing.", inline=True),
            discord.EmbedField(
                name="/ask", value="Ask about the transcript.", inline=True),
            discord.EmbedField(
                name="/generate_pdf", value="Export transcript as PDF.", inline=True),
            discord.EmbedField(
                name="/update_player_map", value="Sync player names with roster.", inline=True),
            discord.EmbedField(
                name="/health", value="Show system health status.", inline=True),
            discord.EmbedField(
                name="/help", value="Show this menu.", inline=True),
        ]

        embed = discord.Embed(title="// TRANSCRIPT-BOT",
                              description="""Realtime voice-to-text. All comms logged.""",
                              color=discord.Color.from_rgb(0, 255, 136),
                              fields=embed_fields)

        await ctx.respond(embed=embed, ephemeral=True)

    @bot.slash_command(name="health", description="Show system health status.")
    async def health(ctx: discord.context.ApplicationContext):
        await ctx.defer(ephemeral=True)
        # Reuse the bot's existing HealthCheck instance so any ollama process
        # it spawned at startup stays tracked, and to avoid re-loading the
        # whisper model on every /health invocation.
        await asyncio.to_thread(bot.health.run_all, autofix=False, bot=bot)
        status = "All systems operational." if bot.health.all_ok() else "Critical checks failing."
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
