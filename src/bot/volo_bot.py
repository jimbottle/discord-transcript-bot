import asyncio
import json
import logging
import os
import time
from collections import defaultdict

import discord
import yaml
from discord.sinks.errors import RecordingException

from src.bot.health import HealthCheck
from src.sinks.whisper_sink import WhisperSink

TRANSCRIPTION_METHOD = os.getenv("TRANSCRIPTION_METHOD")
PLAYER_MAP_FILE_PATH = os.getenv("PLAYER_MAP_FILE_PATH")


logger = logging.getLogger(__name__)


class VoloBot(discord.Bot):
    def __init__(self, loop):

        super().__init__(
            command_prefix="!",
            loop=loop,
            activity=discord.CustomActivity(name="Monitoring voice subnets"),
        )
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
            logger.debug(
                f"Sending voice-state disconnect for guild {guild.name} (ghost cleanup)"
            )
            try:
                await guild.change_voice_state(channel=None)
            except Exception as e:
                logger.error(
                    f"Failed to send voice-state disconnect for {guild.name}: {e}"
                )

        if self.guilds:
            await asyncio.gather(
                *(_ghost_cleanup(g) for g in self.guilds), return_exceptions=True
            )

        # Run startup health checks in a thread so blocking autofix
        # (ollama serve poll, model pull) doesn't freeze the event loop.
        await asyncio.to_thread(self.health.run_all, autofix=True, bot=self)
        for line in self.health.summary().splitlines():
            logger.info(f"Health: {line}")
        if not self.health.all_ok():
            logger.error(
                "Critical health checks failed — bot will not accept commands until resolved."
            )
            return

        self._gateway_latency = self.latency
        self._is_ready = True

        # Sync slash commands ONCE, globally. Command definitions only
        # change when the code changes — not on every restart — and
        # global commands persist on Discord's side between runs, so a
        # restart does not require re-registering them.
        #
        # The previous per-guild sync loop re-registered commands for
        # every guild on every startup. With frequent restarts that
        # exhausts Discord's command-sync rate limit, and because
        # interaction ACKs share the same REST client, the global 429
        # backoff makes the bot miss the 3-second ACK deadline —
        # surfacing to the user as "The application did not respond" on
        # commands like /connect. One global sync is sufficient and
        # cheap; guild-scoped re-sync is not worth that failure mode.
        await self.sync_commands()
        logger.info("Synced global commands.")

    async def on_application_command(self, ctx):
        self._last_interaction_time = time.time()

    async def close_consumers(self):
        if hasattr(self, "consumer_manager"):
            await self.consumer_manager.close()

    def _close_and_clean_sink_for_guild(self, guild_id: int):
        # Pop first so the sink is untracked even if a teardown step
        # raises — otherwise a failure here leaks the sink AND a later
        # /scribe sees a stale entry. Each step is independently guarded
        # so one failure can't skip the next or propagate into the
        # /stop|/disconnect command.
        whisper_sink: WhisperSink | None = self.guild_whisper_sinks.pop(guild_id, None)
        if not whisper_sink:
            return

        logger.debug(f"Stopping whisper sink, requested by {guild_id}.")
        try:
            whisper_sink.stop_voice_thread()
        except Exception as e:
            logger.error(f"Error stopping voice thread for {guild_id}: {e}")
        try:
            whisper_sink.close()
        except Exception as e:
            logger.error(f"Error closing whisper sink for {guild_id}: {e}")

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
            logger.debug(f"Sink is already active for guild {ctx.guild_id}.")
            return

        async def on_stop_record_callback(sink: WhisperSink, ctx):
            logger.debug(f"{ctx.channel.guild.id} -> on_stop_record_callback")
            # Pycord schedules this on the loop; _close_and_clean_sink
            # does the blocking voice-thread join, so offload it or it
            # freezes the event loop. Idempotent (pop-first) with the
            # explicit cleanup_sink in end_recording_session.
            await asyncio.to_thread(self._close_and_clean_sink_for_guild, ctx.guild_id)

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
            whisper_sink, on_stop_record_callback, ctx
        )

        def on_thread_exception(e):
            logger.warning(
                f"Whisper sink thread exception for guild {ctx.guild_id}. Retry in 5 seconds...\n{e}"
            )
            self._close_and_clean_sink_for_guild(ctx.guild_id)

            # retry in 5 seconds
            self.loop.call_later(5, self.start_recording, ctx)

        whisper_sink.start_voice_thread(on_exception=on_thread_exception)

        self.guild_whisper_sinks[ctx.guild_id] = whisper_sink

    def stop_recording(self, ctx: discord.context.ApplicationContext):
        vc = ctx.guild.voice_client
        if vc:
            self.guild_is_recording[ctx.guild_id] = False
            # vc can exist without an active recorder: a stale-True flag,
            # or Pycord recreated the reader after a voice reconnect.
            # vc.stop_recording() raises RecordingException then; guard
            # and swallow so teardown (and the /stop|/disconnect command)
            # continues instead of aborting mid-cleanup.
            try:
                if vc.is_recording():
                    vc.stop_recording()
            except RecordingException as e:
                logger.debug(f"stop_recording: nothing to stop ({e}); continuing.")
        guild_id = ctx.guild_id
        whisper_message_task = self.guild_whisper_message_tasks.get(guild_id, None)
        if whisper_message_task:
            logger.debug("Cancelling whisper message task.")
            whisper_message_task.cancel()
            del self.guild_whisper_message_tasks[guild_id]

    def cleanup_sink(self, ctx: discord.context.ApplicationContext):
        guild_id = ctx.guild_id
        self._close_and_clean_sink_for_guild(guild_id)

    async def end_recording_session(self, ctx: discord.context.ApplicationContext):
        """Tear down an active recording for ctx's guild: stop the voice
        recording, clear the recording flag, and close/remove the sink.

        Shared by /stop and /disconnect. Before this existed, /disconnect
        skipped the teardown, so disconnecting while recording left
        guild_is_recording stuck True — a later /scribe (even after
        reconnecting) failed with "Already recording" — and leaked the
        sink's file handle + voice thread. No-op if not recording.

        Must be awaited on the event loop. stop_recording() runs
        vc.stop_recording(), which schedules the async stop callback via
        loop.create_task() — that scheduling is NOT thread-safe off the
        loop, so stop_recording() stays on the loop thread.

        TRADEOFF (not a full offload): Pycord's AudioReader._stop() also
        runs packet_router.join(timeout=5) + _drain_all_decoders()
        synchronously before that create_task. So a bounded blocking
        segment stays on the loop here — typically ~ms (PacketRouter.stop
        sets the end flag and notifies the waiter, and the drain just
        queues buffered packets via WhisperSink.write), worst case 5s if
        the router thread is wedged. This is deliberate: moving it off
        the loop reintroduces the non-thread-safe create_task. The
        genuinely long part (cleanup_sink → voice-thread join, ~Whisper
        latency, can be 10s+) IS offloaded via asyncio.to_thread so a
        concurrent interaction can still be ACKed during normal teardown.
        """
        guild_id = ctx.guild_id
        if not self.guild_is_recording.get(guild_id, False):
            return
        # Each step is independently guarded and the flag is always
        # cleared, so a failure in one step can't leave the guild stuck
        # "recording" or propagate into the /stop|/disconnect command
        # (which would 404 the interaction and look like the bot died).
        try:
            # On the loop thread: safe create_task scheduling, but also a
            # bounded (~ms, ≤5s worst-case) packet_router join — see the
            # method docstring's TRADEOFF note.
            self.stop_recording(ctx)
        except Exception as e:
            logger.error(f"end_recording_session: stop_recording failed: {e}")
        self.guild_is_recording[guild_id] = False
        try:
            await asyncio.to_thread(self.cleanup_sink, ctx)
        except Exception as e:
            logger.error(f"end_recording_session: cleanup_sink failed: {e}")

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
                    "character": member.display_name,
                }

        self.player_map = existing_map
        logger.info(f"{str(self.player_map)}")
        if PLAYER_MAP_FILE_PATH:
            with open(PLAYER_MAP_FILE_PATH, "w", encoding="utf-8") as file:
                yaml.dump(
                    self.player_map, file, default_flow_style=False, allow_unicode=True
                )

    async def stop_and_cleanup(self):
        try:
            # Disconnect from all voice channels first so Discord knows we left
            for vc in self.voice_clients:
                try:
                    await vc.disconnect(force=True)
                except Exception:
                    pass
            for sink in self.guild_whisper_sinks.values():
                # stop_voice_thread() BEFORE close(): the join drains an
                # in-flight transcription while the file is still
                # writable (running False, _closed False). close() then
                # sets _closed and finalizes. Reversing this (the old
                # order) drops the last transcription on the shutdown
                # path — roborev #791.
                sink.stop_voice_thread()
                sink.close()
                gid = None
                try:
                    gid = sink.vc.channel.guild.id
                except AttributeError:
                    pass
                logger.debug(f"Stopped whisper sink for guild {gid} in cleanup.")
            self.guild_whisper_sinks.clear()
            self.guild_to_helper.clear()
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
        finally:
            logger.info("Cleanup completed.")
