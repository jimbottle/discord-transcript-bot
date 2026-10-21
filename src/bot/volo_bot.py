import asyncio
import json
import logging
import os
import time
from collections import defaultdict
from typing import Optional

import discord
import yaml
from discord.sinks.errors import RecordingException

from src import library, player_map_store, sessions
from src.bot import command_errors
from src.bot.health import HealthCheck
from src.sinks.whisper_sink import WhisperSink

TRANSCRIPTION_METHOD = os.getenv("TRANSCRIPTION_METHOD")
PLAYER_MAP_FILE_PATH = os.getenv("PLAYER_MAP_FILE_PATH")

# Runtime state the web dashboard reads for true connected-guild /
# recording / uptime (vs. its file-freshness heuristic). Written
# best-effort on lifecycle transitions; a write failure must never
# affect the bot. Same .logs dir + cwd convention as health_status.json.
BOT_STATE_FILE = os.path.join(os.getcwd(), ".logs", "bot_state.json")

# Liveness heartbeat / stall watchdog. _write_runtime_state only fires on
# lifecycle transitions, so without a periodic write the dashboard can't
# tell a live-but-quiet bot from a dead one — a wedged sink once left
# bot_state.json 115 min stale during a real session while it showed
# "recording: true". The heartbeat rewrites state on a fixed cadence (so
# `updated_at` is a real liveness signal) and flags a recording guild whose
# transcript output has stalled while audio keeps backing up in the queue.
HEARTBEAT_INTERVAL_S = 30
# A recording guild is "stalled" when its session file hasn't grown in this
# long AND audio is piling up in the sink's queue (the unambiguous wedged-
# pipeline case — not just a quiet room, where the queue stays shallow).
STALL_OUTPUT_AGE_S = 120
STALL_QUEUE_DEPTH = 25
# CPU spin detector (discord-transcript-bot-309): process CPU% is sampled
# between state writes and published as `cpu_pct`. High CPU while
# transcription is legitimately busy is normal (the CPU backend uses every
# core), so the heartbeat only warns when the process is hot AND idle: no
# audio queued and no transcript output for IDLE_OUTPUT_AGE_S.
HIGH_CPU_PCT = 90
IDLE_OUTPUT_AGE_S = 60
# Minimum window for a CPU% sample; state writes on rapid lifecycle
# transitions would otherwise produce noisy sub-second readings.
CPU_SAMPLE_MIN_WINDOW_S = 1.0

logger = logging.getLogger(__name__)


def _sample_cpu_pct(obj):
    """Process CPU% since ``obj``'s previous sample, or None until a window
    of at least CPU_SAMPLE_MIN_WINDOW_S has elapsed. Shorter gaps return the
    previous reading and keep the baseline, so a burst of lifecycle writes
    can't reset the measurement. Module-level (not a method) so it works on
    any object that can hold two attributes — the bot, or a test stand-in."""
    now_cpu, now_wall = time.process_time(), time.monotonic()
    prev = getattr(obj, "_cpu_sample", None)
    if prev is None:
        obj._cpu_sample = (now_cpu, now_wall)
        return None
    window = now_wall - prev[1]
    if window < CPU_SAMPLE_MIN_WINDOW_S:
        return getattr(obj, "_last_cpu_pct", None)
    pct = round((now_cpu - prev[0]) / window * 100.0, 1)
    obj._cpu_sample = (now_cpu, now_wall)
    obj._last_cpu_pct = pct
    return pct


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
        # guild_id -> the .txt of the guild's latest recording; outlives the
        # sink so /generate_pdf works after /stop and /disconnect.
        self.guild_last_session_file = {}
        # guild_id -> token of a pending voice-thread restart; /stop and
        # /disconnect cancel it (cancel_voice_recovery).
        self._voice_recovery = {}
        self.guild_whisper_message_tasks = {}
        self.player_map = {}
        self._is_ready = False
        self._heartbeat_task = None
        self._gateway_latency = None
        self._last_interaction_time = None
        # Process start, set once at construction (NOT in on_ready —
        # on_ready re-fires on every gateway reconnect, which would make
        # the dashboard uptime jump). Accurate from the first snapshot.
        self._started_at = time.time()
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

        # Run startup health checks in a thread so the blocking parts (the
        # whisper model load and its test decode) don't freeze the event loop.
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
        self._write_runtime_state()  # ready, not yet connected

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

        # Start the liveness heartbeat once the bot is ready. The _is_ready
        # short-circuit at the top of on_ready keeps a gateway reconnect
        # from spawning a second loop.
        self._start_heartbeat()

    def _start_heartbeat(self):
        if self._heartbeat_task and not self._heartbeat_task.done():
            return
        self._heartbeat_task = self.loop.create_task(self._heartbeat_loop())

    async def _heartbeat_loop(self):
        """Refresh bot_state.json on a fixed cadence so the dashboard can
        tell a live-but-quiet bot from a dead one, and log a loud warning
        when a recording guild's pipeline has stalled (audio queuing but
        nothing transcribed). Fully guarded — a heartbeat failure is a
        dashboard nicety and must never crash the bot.
        """
        while True:
            try:
                await asyncio.sleep(HEARTBEAT_INTERVAL_S)
                for gid in self._write_runtime_state():
                    logger.warning(
                        f"Recording STALLED for guild {gid}: audio is queuing "
                        f"but no transcript output in >{STALL_OUTPUT_AGE_S}s — "
                        "the sink may be wedged. Try /stop then /scribe, or "
                        "restart the bot."
                    )
                cpu_alert = getattr(self, "_cpu_alert", None)
                if cpu_alert:
                    logger.warning(cpu_alert)
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001 - heartbeat must not die loudly
                logger.debug(f"heartbeat iteration failed (ignored): {e}")

    async def on_application_command(self, ctx):
        self._last_interaction_time = time.time()

    async def on_application_command_error(self, ctx, error):
        """Any exception a slash command doesn't handle itself: log it and
        tell the user, instead of leaving Discord's "did not respond"."""
        await command_errors.report(ctx, error)

    async def on_voice_state_update(self, member, before, after):
        """Two jobs. (1) If the BOT itself left a channel (kicked, moved
        out, or a reconnect that failed), tear down that guild's helper,
        sink and recording flag so a later /connect + /scribe starts clean
        instead of failing with "Already recording". (2) Refresh the
        dashboard's live voice roster when someone joins or leaves a
        channel the bot is connected to.

        This is the ONLY handler for the event: registering another with
        ``@bot.event`` in main.py replaces this method outright (that is
        how discord.Client.event works), which is how the roster refresh
        silently stopped running (discord-transcript-bot-xnj). Add more
        handlers with ``@bot.listen()``.

        Best-effort and fully guarded — a failure here must never affect
        voice. The roster part only rewrites the small state JSON (same
        atomic path as every other transition) and only when the change
        actually touches one of our connected channels, so unrelated
        server-wide voice churn doesn't thrash the file.
        """
        try:
            if (
                self.user is not None
                and member.id == self.user.id
                and after.channel is None
                and before.channel is not None
            ):
                await self._on_bot_left_voice(before.channel.guild.id)
        except Exception as e:  # noqa: BLE001 - never fatal
            logger.error(f"on_voice_state_update: bot-left cleanup failed: {e}")
        try:
            connected = {
                getattr(getattr(h, "vc", None), "channel", None)
                for h in dict(self.guild_to_helper).values()
            }
            connected.discard(None)
            if before.channel in connected or after.channel in connected:
                self._write_runtime_state()
        except Exception as e:  # noqa: BLE001 - dashboard nicety, never fatal
            logger.debug(f"on_voice_state_update state refresh skipped: {e}")

    async def _on_bot_left_voice(self, guild_id):
        """Forget the guild's voice client and tear down its sink after the
        bot left a channel by any route other than /disconnect (which does
        this itself). The sink teardown joins the voice thread and waits
        for in-flight transcriptions, so it is offloaded off the loop."""
        helper = self.guild_to_helper.pop(guild_id, None)
        if helper:
            helper.set_vc(None)
        self.guild_is_recording.pop(guild_id, None)
        await asyncio.to_thread(self._close_and_clean_sink_for_guild, guild_id)
        self._write_runtime_state()

    async def close_consumers(self):
        if hasattr(self, "consumer_manager"):
            await self.consumer_manager.close()

    def _write_runtime_state(self):
        """Best-effort snapshot of connected guilds / recording / uptime
        for the web dashboard. ENTIRELY guarded: this runs on lifecycle
        transitions including the hardened /stop|/disconnect teardown
        path, so a failure here must never propagate. Written atomically
        (tmp + os.replace) so the 3s dashboard poll never sees a torn
        file. Each guild is extracted independently — one bad guild
        can't blank the whole file.

        Returns the list of guild ids detected as STALLED this write (a
        recording guild whose output has stopped while audio backs up) so
        the heartbeat caller can log/alert. Always returns a list, even on
        the guarded failure path.
        """
        stalled_gids = []
        try:
            now = time.time()
            guilds = []
            for gid, helper in dict(self.guild_to_helper).items():
                try:
                    g = self.get_guild(gid)
                    vc = getattr(helper, "vc", None)
                    channel_obj = getattr(vc, "channel", None)
                    channel = getattr(channel_obj, "name", None)
                    sink = self.guild_whisper_sinks.get(gid)
                    session = getattr(sink, "session_file", None)
                    recording = bool(self.guild_is_recording.get(gid, False))
                    # Stall signals, both read-only off the existing sink so
                    # this doesn't touch whisper_sink: how long since the
                    # session file last grew, and how much audio is waiting
                    # in the sink's queue. A deep queue with a stale file is
                    # the wedged-pipeline case (today's silent 45-min death);
                    # a quiet room keeps the queue shallow.
                    output_age = None
                    if session:
                        try:
                            output_age = now - os.path.getmtime(session)
                        except OSError:
                            output_age = None
                    queue_depth = 0
                    vq = getattr(sink, "voice_queue", None)
                    if vq is not None:
                        try:
                            queue_depth = vq.qsize()
                        except Exception:  # noqa: BLE001
                            queue_depth = 0
                    stalled = bool(
                        recording
                        and queue_depth >= STALL_QUEUE_DEPTH
                        and output_age is not None
                        and output_age > STALL_OUTPUT_AGE_S
                    )
                    if stalled:
                        stalled_gids.append(gid)
                    guilds.append(
                        {
                            "guild_id": gid,
                            "guild": getattr(g, "name", None),
                            "channel": channel,
                            "recording": recording,
                            "session_file": (
                                os.path.basename(session) if session else None
                            ),
                            "output_age": (
                                round(output_age, 1) if output_age is not None else None
                            ),
                            "queue_depth": queue_depth,
                            "stalled": stalled,
                            # Current voice-channel roster so the dashboard
                            # can offer to name people who are present but
                            # haven't spoken yet (read-only; bot is never
                            # signalled by the resulting edit).
                            "members": [
                                {
                                    "id": getattr(m, "id", None),
                                    "name": getattr(m, "name", None),
                                    "display_name": getattr(m, "display_name", None),
                                }
                                for m in (getattr(channel_obj, "members", None) or [])
                            ],
                        }
                    )
                except Exception as e:  # noqa: BLE001 - never fail a write
                    logger.debug(f"runtime-state: skipped guild {gid}: {e}")
            # Hot-and-idle detection (discord-transcript-bot-309): a core
            # pegged while nothing is queued and nothing has been produced
            # for a while is a spinning thread, not work. Busy rooms keep
            # output_age small, so a CPU backend at 400% mid-game never trips.
            cpu_pct = _sample_cpu_pct(self)
            idle = all(
                g["queue_depth"] == 0
                and (g["output_age"] is None or g["output_age"] >= IDLE_OUTPUT_AGE_S)
                for g in guilds
            )
            self._cpu_alert = None
            if cpu_pct is not None and cpu_pct >= HIGH_CPU_PCT and idle:
                self._cpu_alert = (
                    f"Process CPU at {cpu_pct:.0f}% while transcription is idle "
                    f"(nothing queued, no output for ≥{IDLE_OUTPUT_AGE_S}s) — a "
                    f"thread is spinning. Run `kill -USR1 {os.getpid()}` to write "
                    "a ranked stack dump to .logs/thread_dump-*.txt "
                    "(discord-transcript-bot-309)."
                )
            state = {
                "updated_at": time.time(),
                "started_at": self._started_at,
                "cpu_pct": cpu_pct,
                "guilds": guilds,
            }
            os.makedirs(os.path.dirname(BOT_STATE_FILE), exist_ok=True)
            tmp = BOT_STATE_FILE + ".tmp"
            try:
                with open(tmp, "w", encoding="utf-8") as f:
                    json.dump(state, f)
                os.replace(tmp, BOT_STATE_FILE)
            except Exception:
                # Don't leave an orphaned .tmp behind if json.dump or
                # replace fails — re-raise to the outer guard after.
                try:
                    os.remove(tmp)
                except OSError:
                    pass
                raise
        except Exception as e:  # noqa: BLE001 - dashboard nicety, never fatal
            logger.debug(f"runtime-state write failed (ignored): {e}")
        return stalled_gids

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

    def start_recording(self, ctx: discord.context.ApplicationContext) -> bool:
        """
        Start recording audio from the voice channel. Create a whisper sink
        and start sending transcripts to the queue.

        Returns True when recording is on, False when the sink could not be
        started — the caller must tell the user, because nothing is being
        transcribed (discord-transcript-bot-vjm). Never raises.

        Since this is a critical function, this is where we should handle
        subscription checks and limits.
        """
        ok = False
        try:
            self.start_whisper_sink(ctx)
            self.guild_is_recording[ctx.guild_id] = True
            ok = True
        except Exception as e:
            logger.error(f"Error starting whisper sink: {e}")
            # Undo whatever got started. If vc.start_recording succeeded and
            # the voice thread then failed to start, Pycord's recorder would
            # keep feeding a sink nobody drains, and every later /scribe
            # would hit "Already recording" until a /disconnect
            # (roborev #6411). stop_recording guards the not-recording case.
            try:
                self.stop_recording(ctx)
            except Exception as stop_err:  # noqa: BLE001 - best-effort undo
                logger.error(f"Error undoing a failed recording start: {stop_err}")
            # A half-built sink (constructed but vc.start_recording raised)
            # must not linger as "already active" for the next /scribe.
            self._close_and_clean_sink_for_guild(ctx.guild_id)
        self._write_runtime_state()  # recording state changed
        return ok

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

        self._tag_session(ctx, whisper_sink)
        self.guild_last_session_file[ctx.guild_id] = whisper_sink.session_file

        # Registered before vc.start_recording so a failure there (e.g.
        # RecordingException) leaves a sink start_recording can find and
        # close, rather than an orphaned executor + open capture.
        self.guild_whisper_sinks[ctx.guild_id] = whisper_sink
        self.guild_to_helper[ctx.guild_id].vc.start_recording(
            whisper_sink, on_stop_record_callback, ctx
        )

        def on_thread_exception(e):
            # Runs on the dying voice thread. Everything else (the teardown,
            # which joins this very thread, and the restart) belongs on the
            # loop, and only run_coroutine_threadsafe may hand it there.
            asyncio.run_coroutine_threadsafe(
                self._recover_voice_thread(ctx, whisper_sink, e), self.loop
            )

        whisper_sink.start_voice_thread(on_exception=on_thread_exception)

    VOICE_RESTART_DELAY_S = 5

    def cancel_voice_recovery(self, guild_id) -> bool:
        """Cancel a pending voice-thread restart; True if there was one.
        Between its teardown and its restart the guild looks not-recording,
        so without this a /stop there was answered "Not recording" and then
        overridden 5s later (roborev #6513)."""
        return self._voice_recovery.pop(guild_id, None) is not None

    async def _recover_voice_thread(self, ctx, sink, error) -> bool:
        """The voice thread died: end that recording, wait, start a new one,
        and say so in the channel either way. Returns whether recording
        restarted. Does nothing if the recording was already stopped or
        replaced, since then nobody is waiting on this sink."""
        guild_id = ctx.guild_id
        if self.guild_whisper_sinks.get(guild_id) is not sink:
            return False
        logger.warning(
            f"Voice thread for guild {guild_id} died ({error}); restarting "
            f"the recording in {self.VOICE_RESTART_DELAY_S}s."
        )
        token = object()
        self._voice_recovery[guild_id] = token
        try:
            await self.end_recording_session(ctx)
        except Exception as e:  # noqa: BLE001 - still try to restart
            logger.error(f"Teardown after a voice thread failure failed: {e}")
        await asyncio.sleep(self.VOICE_RESTART_DELAY_S)
        # /stop, /disconnect or a new /scribe in the meantime: leave it alone.
        if self._voice_recovery.get(guild_id) is not token:
            return False
        self._voice_recovery.pop(guild_id, None)
        if guild_id not in self.guild_to_helper or self.guild_is_recording.get(
            guild_id
        ):
            return False
        ok = self.start_recording(ctx)
        notice = (
            "Transcription hit an error and restarted. A few seconds may be missing."
            if ok
            else "Transcription hit an error and could not restart. Run `/scribe` again."
        )
        try:
            await ctx.channel.send(notice)
        except Exception as e:  # noqa: BLE001 - the restart matters more
            logger.warning(f"Could not post the restart notice: {e}")
        return ok

    def _tag_session(self, ctx, whisper_sink) -> None:
        """Write the recording's group/game/campaign/session sidecar
        (src/sessions.py) next to its transcript. Best-effort: a missing or
        broken library.yml leaves the recording untagged, never unrecorded."""
        try:
            helper = self.guild_to_helper.get(ctx.guild_id)
            channel = getattr(getattr(helper, "vc", None), "channel", None)
            meta = sessions.build_meta(
                library.load(),
                guild_id=ctx.guild_id,
                guild=getattr(ctx.guild, "name", None),
                channel_id=getattr(channel, "id", None),
                channel=getattr(channel, "name", None),
            )
            sessions.write_meta(whisper_sink.session_file, meta)
            whisper_sink.session_meta = meta
            if meta.campaign:
                logger.info(
                    f"Session tagged {meta.key} (group={meta.group}, game={meta.game})"
                )
            else:
                logger.warning(
                    f"Recording in guild {ctx.guild_id} matches no group in "
                    "library/library.yml; it is untagged (scripts/tag_session.py "
                    "can tag it later)."
                )
        except Exception as e:  # noqa: BLE001 - tagging must never block /scribe
            logger.warning(f"Could not tag session: {e}")

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
        self._write_runtime_state()  # recording stopped (still connected)

    def session_file_for_export(self, guild_id) -> Optional[str]:
        """The recording /generate_pdf exports for ``guild_id``: the one
        this process last started there, else (after a restart) the newest
        tagged with the guild on disk. None when there is none."""
        path = self.guild_last_session_file.get(guild_id)
        if path:
            if os.path.exists(path):
                return path
            # The sink creates the .txt on the first line, so a recording
            # nobody has spoken in yet has only its .meta.json. Exporting the
            # previous session instead would be wrong (roborev #6511). With
            # neither file the recording was deleted: fall back to disk.
            if sessions.meta_path(path).exists():
                return None
        found = sessions.latest_for_guild(
            guild_id, os.path.join(os.getcwd(), "transcripts")
        )
        return str(found) if found else None

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

    def upsert_player_entry(self, user_id, player, character):
        """Add or update one Discord-user → player/character mapping.

        Mutates self.player_map IN PLACE — an already-running
        WhisperSink holds that same dict, so a person added mid-call is
        attributed correctly for the rest of the session immediately
        (this is the whole point: name someone who's on the call now).
        Persists by merging into the on-disk file so other guilds' and
        hand-edited entries are preserved.

        Returns True if saved to disk, False if no PLAYER_MAP_FILE_PATH
        is configured (in-memory only — lost on restart). Raises only on
        a real file IO/parse error so the caller can report it.
        """
        user_id = int(user_id)
        self.player_map[user_id] = {
            "player": player,
            "character": character,
        }  # live for the running session
        if not PLAYER_MAP_FILE_PATH:
            return False
        # Disk read / non-dict guard / atomic merge-write lives in the
        # shared player_map_store so the web dashboard's roster editor
        # persists identically. A non-mapping file raises ValueError (the
        # in-memory entry above still applies); the /add_player caller
        # surfaces it. Mid-call writes can race a live session — the store
        # writes atomically (tmp + os.replace) so the roster is never torn.
        player_map_store.upsert(PLAYER_MAP_FILE_PATH, user_id, player, character)
        return True

    async def stop_and_cleanup(self):
        try:
            # Stop the liveness heartbeat so it doesn't rewrite state mid-
            # teardown or outlive the shutdown.
            task = getattr(self, "_heartbeat_task", None)
            if task is not None:
                task.cancel()
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
