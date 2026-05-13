import asyncio
import json
import logging
import os
import time
from collections import defaultdict

import discord
import yaml

from src.bot.health import HealthCheck
from src.sinks.whisper_sink import WhisperSink

TRANSCRIPTION_METHOD = os.getenv("TRANSCRIPTION_METHOD")
PLAYER_MAP_FILE_PATH = os.getenv("PLAYER_MAP_FILE_PATH")


logger = logging.getLogger(__name__)

class VoloBot(discord.Bot):
    def __init__(self, loop):

        super().__init__(command_prefix="!", loop=loop,
                         activity=discord.CustomActivity(name='Monitoring voice subnets'))
        self.guild_to_helper = {}
        self.guild_is_recording = {}
        self.guild_whisper_sinks = {}
        self.guild_whisper_message_tasks = {}
        self.player_map = {}
        self._is_ready = False
        self._gateway_latency = None
        self._last_interaction_time = None
        self._connecting_guilds = set()
        self.health = HealthCheck()
        if TRANSCRIPTION_METHOD == "openai":
            self.transcriber_type = "openai"
        else:
            self.transcriber_type = "local"
        if PLAYER_MAP_FILE_PATH:
            with open(PLAYER_MAP_FILE_PATH, "r", encoding="utf-8") as file:
                self.player_map = yaml.safe_load(file) or {}

    

    async def on_ready(self):
        logger.info(f"Logged in as {self.user} to Discord.")

        # on_ready can fire multiple times (Discord reconnection).
        # Only skip if the bot previously passed all health checks.
        if self._is_ready:
            logger.info("Reconnected — skipping health checks (already passed).")
            return

        # Unconditionally send a voice-state disconnect for every guild.
        # After a hard kill, Discord's voice servers may still hold a ghost
        # session, but the NEW gateway session won't populate guild.me.voice
        # with the old state — so we can't rely on checking me.voice.channel.
        # Sending channel=None is a no-op if we're not in voice, but clears
        # any ghost if we are. Run concurrently so N guilds don't serialize
        # the bot's readiness behind N×sleep.
        async def _ghost_cleanup(guild):
            logger.debug(f"Sending voice-state disconnect for guild {guild.name} (ghost cleanup)")
            try:
                await guild.change_voice_state(channel=None)
            except Exception as e:
                logger.error(f"Failed to send voice-state disconnect for {guild.name}: {e}")

        if self.guilds:
            await asyncio.gather(*(_ghost_cleanup(g) for g in self.guilds), return_exceptions=True)

        # Run startup health checks in a thread so blocking autofix
        # (ollama serve poll, model pull) doesn't freeze the event loop.
        await asyncio.to_thread(self.health.run_all, autofix=True, bot=self)
        for line in self.health.summary().splitlines():
            logger.info(f"Health: {line}")
        if not self.health.all_ok():
            logger.error("Critical health checks failed — bot will not accept commands until resolved.")
            return

        self._gateway_latency = self.latency
        self._is_ready = True

        # Sync slash commands globally (needed for DM-capable commands like /ask)
        await self.sync_commands()
        logger.info("Synced global commands.")
        # Also sync to each guild for instant guild-level updates
        for guild in self.guilds:
            await self.sync_commands(guild_ids=[guild.id])
            logger.info(f"Synced commands to guild {guild.name} ({guild.id})")


    async def on_application_command(self, ctx):
        self._last_interaction_time = time.time()

    async def close_consumers(self):
        if hasattr(self, 'consumer_manager'):
            await self.consumer_manager.close()

    def _close_and_clean_sink_for_guild(self, guild_id: int):
        whisper_sink: WhisperSink | None = self.guild_whisper_sinks.get(
            guild_id, None)

        if whisper_sink:
            logger.debug(f"Stopping whisper sink, requested by {guild_id}.")
            whisper_sink.stop_voice_thread()
            del self.guild_whisper_sinks[guild_id]
            whisper_sink.close()

    
    def start_recording(self, ctx: discord.context.ApplicationContext):
        """
        Start recording audio from the voice channel. Create a whisper sink
        and start sending transcripts to the queue.

        Since this is a critical function, this is where we should handle
        subscription checks and limits.
        """
        try:
            self.start_whisper_sink(ctx)
            self.guild_is_recording[ctx.guild_id] = True
        except Exception as e:
            logger.error(f"Error starting whisper sink: {e}")

    def start_whisper_sink(self, ctx: discord.context.ApplicationContext):
        guild_voice_sink = self.guild_whisper_sinks.get(ctx.guild_id, None)
        if guild_voice_sink:
            logger.debug(
                f"Sink is already active for guild {ctx.guild_id}.")
            return

        async def on_stop_record_callback(sink: WhisperSink, ctx):
            logger.debug(
                f"{ctx.channel.guild.id} -> on_stop_record_callback")
            self._close_and_clean_sink_for_guild(ctx.guild_id)

        transcript_queue = asyncio.Queue()

        whisper_sink = WhisperSink(
            transcript_queue,
            self.loop,
            data_length=50000,
            max_speakers=10,
            transcriber_type=self.transcriber_type,
            player_map=self.player_map,
        )

        self.guild_to_helper[ctx.guild_id].vc.start_recording(
            whisper_sink, on_stop_record_callback, ctx)

        def on_thread_exception(e):
            logger.warning(
                f"Whisper sink thread exception for guild {ctx.guild_id}. Retry in 5 seconds...\n{e}")
            self._close_and_clean_sink_for_guild(ctx.guild_id)

            # retry in 5 seconds
            self.loop.call_later(5, self.start_recording, ctx)

        whisper_sink.start_voice_thread(on_exception=on_thread_exception)

        self.guild_whisper_sinks[ctx.guild_id] = whisper_sink

    def stop_recording(self, ctx: discord.context.ApplicationContext):
        vc = ctx.guild.voice_client
        if vc:
            self.guild_is_recording[ctx.guild_id] = False
            vc.stop_recording()
        guild_id = ctx.guild_id
        whisper_message_task = self.guild_whisper_message_tasks.get(
            guild_id, None)
        if whisper_message_task:
            logger.debug("Cancelling whisper message task.")
            whisper_message_task.cancel()
            del self.guild_whisper_message_tasks[guild_id]

    def cleanup_sink(self, ctx: discord.context.ApplicationContext):
        guild_id = ctx.guild_id
        self._close_and_clean_sink_for_guild(guild_id)

    async def get_transcription(self, ctx: discord.context.ApplicationContext):
        # Get the transcription queue
        if not (self.guild_whisper_sinks.get(ctx.guild_id)):
            return
        whisper_sink = self.guild_whisper_sinks[ctx.guild_id]
        transcriptions = []
        if whisper_sink is None:
            return
    
        transcriptions_queue = whisper_sink.transcription_output_queue
        while not transcriptions_queue.empty():
            transcriptions.append(await transcriptions_queue.get())
        return transcriptions

    async def update_player_map(self, ctx: discord.context.ApplicationContext):
        # Load existing file to preserve entries from other guilds and manual edits
        existing_map = {}
        if PLAYER_MAP_FILE_PATH:
            try:
                with open(PLAYER_MAP_FILE_PATH, "r", encoding="utf-8") as file:
                    existing_map = yaml.safe_load(file) or {}
            except FileNotFoundError:
                pass

        # Only add new members that don't already have entries (preserve manual edits)
        for member in ctx.guild.members:
            if member.id not in existing_map:
                existing_map[member.id] = {
                    "player": member.name,
                    "character": member.display_name
                }

        self.player_map = existing_map
        logger.info(f"{str(self.player_map)}")
        if PLAYER_MAP_FILE_PATH:
            with open(PLAYER_MAP_FILE_PATH, "w", encoding="utf-8") as file:
                yaml.dump(self.player_map, file, default_flow_style=False, allow_unicode=True)

    async def stop_and_cleanup(self):
        try:
            # Disconnect from all voice channels first so Discord knows we left
            for vc in self.voice_clients:
                try:
                    await vc.disconnect(force=True)
                except Exception:
                    pass
            for sink in self.guild_whisper_sinks.values():
                sink.close()
                sink.stop_voice_thread()
                logger.debug(
                    f"Stopped whisper sink for guild {sink.vc.channel.guild.id} in cleanup.")
            self.guild_whisper_sinks.clear()
            self.guild_to_helper.clear()
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
        finally:
            logger.info("Cleanup completed.")
    