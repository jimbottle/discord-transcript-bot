import asyncio
import io
import logging
import os
import time
from datetime import datetime

import discord
import yaml
from dotenv import load_dotenv

from src.bot import command_errors
from src.bot.helper import BotHelper
from src.config.cliargs import CLIArgs
from src import library, question_log, sessions
from src.llm.provider import AskBlocked, AskFailed, ask as llm_ask
from src.session_capture import scribe_notice
from src.utils.answer import answer_budget, clean_answer, discord_messages
from src.utils.discord_send import send_followups
from src.utils.commandline import CommandLine
from src.utils.pdf_generator import pdf_generator
from src.transcript_lines import parse_transcript_text
from src.utils import bot_log, timefmt
from src.utils.voice import disconnect_targets

load_dotenv()
DISCORD_BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN")
PLAYER_MAP_FILE_PATH = os.getenv("PLAYER_MAP_FILE_PATH")
# /ask is answered by a local model or the Anthropic API (src/llm/provider.py);
# the choice and the model are resolved there so main.py and
# src/bot/health.py cannot disagree.
# Discord hard-rejects messages longer than this many characters.
DISCORD_MESSAGE_LIMIT = 2000
# Discord's upload limit for a server without boosts (10 MB).
TRANSCRIPT_ATTACHMENT_LIMIT = 10 * 1024 * 1024
# /recap is /ask with this question (discord-transcript-bot-csk).
RECAP_QUESTION = (
    "Recap this session for the players: the main events in order, the "
    "decisions the party made, the NPCs they met, and the threads left open."
)

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

    # The bot's own log, kept on disk so a problem in a session can be
    # diagnosed after the terminal is gone (discord-transcript-bot-309).
    bot_log.install(
        level=logging.DEBUG if CLIArgs.verbose else logging.INFO,
        fmt=log_format,
        datefmt=date_format,
    )

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

    # `kill -USR1 <pid>` writes a sampled all-thread stack dump to .logs/ —
    # the no-root way to name the thread behind a CPU spin
    # (discord-transcript-bot-309). Must run on the main thread.
    from src.utils.thread_dump import install_signal_handler

    install_signal_handler(directory=".logs")

    # Guarded workarounds for the pinned Pycord voice-receive branch; each
    # skips itself with a warning if upstream has changed shape.
    from src.bot.pycord_patches import apply_all as apply_pycord_patches

    apply_pycord_patches()

    from src.bot.volo_bot import VoloBot

    bot = VoloBot(loop)
    # Overwrite any previous run's bot_state.json immediately so the
    # dashboard never shows the old run's guild / recording / (growing)
    # uptime during the restart→on_ready window. Best-effort, never raises.
    bot._write_runtime_state()

    # NOTE: do not register on_voice_state_update here with @bot.event —
    # that REPLACES VoloBot.on_voice_state_update (bot-left cleanup +
    # dashboard roster refresh). Use @bot.listen() for any extra handler.

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
            # Cosmetic. A failure here must not tear down the connection we
            # just reported as working (the except below would).
            try:
                await ctx.guild.change_voice_state(
                    channel=author_vc.channel, self_mute=True
                )
            except Exception as e:
                logger.warning(f"/connect: self-mute failed: {e}")
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
            logger.exception("/connect failed")
            await ctx.followup.send(
                f"Couldn't join voice. {command_errors.user_message(e)}"
            )
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
        if not bot.start_recording(ctx):
            await ctx.respond(
                "Couldn't start recording — the transcriber failed to start. "
                "Check the bot logs, then `/disconnect`, `/connect` and try again.",
                ephemeral=True,
            )
            return
        message = (
            "Recording. Every word in this channel is being transcribed in realtime."
        )
        # Non-ephemeral so every participant sees it, not just whoever ran the
        # command — they have no other way to know audio is being kept.
        notice = scribe_notice(bot.guild_whisper_sinks.get(ctx.guild_id))
        if notice:
            message += f"\n\n{notice}"
        await ctx.respond(message, ephemeral=False)

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
            if bot.cancel_voice_recovery(guild_id):
                await ctx.respond(
                    "Recording stopped. (It was restarting after an error; "
                    "that restart is cancelled.)"
                )
                return
            await ctx.respond(
                "Not recording right now. Nothing to kill.", ephemeral=True
            )
            return

        # Teardown (voice-thread join, sink close) can take >3s. Defer
        # first so the interaction token stays valid (~15 min) instead
        # of expiring mid-teardown and 404ing as "The application did
        # not respond". Guard the teardown so we always confirm.
        await ctx.defer()
        # Grab the capture before teardown pops the sink from the registry;
        # the object outlives that, and its counters are only final once the
        # in-flight transcriptions have committed below.
        _sink = bot.guild_whisper_sinks.get(guild_id)
        capture = getattr(_sink, "capture", None)
        teardown_ok = True
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
        # Tell the operator what the capture produced and where. Without this
        # nothing surfaces that a capture happened at all, let alone the path
        # needed to go correct the draft manifest.
        capture_line = f"\nCaptured audio: {capture.summary()}" if capture else ""
        if teardown_ok:
            await ctx.followup.send(
                "Recording stopped. Data saved. Standing by for the next run.\n"
                "`/recap` summarises the session, `/transcript` sends the full "
                "text and `/generate_pdf` a PDF."
                f"{capture_line}"
            )
        else:
            await ctx.followup.send(
                "Recording stopped, but cleanup hit an error — check the logs. "
                "If the bot seems stuck, try `/disconnect`."
                f"{capture_line}"
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
        bot.cancel_voice_recovery(guild_id)
        # Guarded so a teardown failure can't skip the disconnect.
        teardown_ok = True
        if bot.guild_is_recording.get(guild_id, False):
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
        # Built from the session's transcript files, not from the sink, so it
        # works mid-session, after /stop or /disconnect, and after a restart,
        # and covers every recording of the session (fwv).
        path = bot.session_file_for_export(ctx.guild_id)
        if not path:
            await ctx.respond(
                "No transcript data yet. Start a recording first.", ephemeral=True
            )
            return
        text, _meta, _files = sessions.session_transcript(path)
        entries = parse_transcript_text(text)
        if not any(e.get("time") for e in entries):
            await ctx.respond(
                "No transcript data yet. Start a recording first.", ephemeral=True
            )
            return
        # A long session takes reportlab more than Discord's 3s to lay out.
        await ctx.defer()
        try:
            pdf_file_path = await pdf_generator(entries)
        except Exception as e:
            logger.error(f"/generate_pdf failed: {e}")
            await ctx.followup.send("PDF generation failed. Check the logs.")
            return
        try:
            with open(pdf_file_path, "rb") as f:
                discord_file = discord.File(
                    f, filename=f"{os.path.splitext(os.path.basename(path))[0]}.pdf"
                )
                await ctx.followup.send(
                    "Dossier compiled. Here's your transcript.", file=discord_file
                )
        finally:
            os.remove(pdf_file_path)

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
            logger.exception("/update_player_map failed")
            await ctx.respond(
                f"Roster sync failed. {command_errors.user_message(e)}", ephemeral=True
            )

    @bot.slash_command(
        name="add_player",
        description="Name someone on the call so the transcript shows their name, not their ID.",
        contexts={discord.InteractionContextType.guild},
    )
    async def add_player(
        ctx: discord.context.ApplicationContext,
        user: discord.Option(
            discord.Member,
            description="Pick the person on the call to name",
        ),
        player: discord.Option(
            str,
            description="Their real name (defaults to their server nickname)",
            required=False,
            default=None,
        ),
        character: discord.Option(
            str,
            description="Their character name (defaults to their server nickname)",
            required=False,
            default=None,
        ),
    ):
        # Pick-from-dropdown UX: no IDs to copy. Sensible defaults so
        # just picking the person already fixes the attribution; names
        # can be refined by re-running. Works DURING a recording — the
        # map is mutated in place so the live session picks it up.
        existed = int(user.id) in bot.player_map
        player_name = player or user.display_name
        character_name = character or user.display_name
        try:
            persisted = bot.upsert_player_entry(user.id, player_name, character_name)
        except Exception as e:
            # The in-memory update already applied (live this session);
            # only the on-disk save failed.
            await ctx.respond(
                f"Applied **{player_name}** ({character_name}) for "
                f"{user.mention} for the current session, but saving to "
                f"the roster file failed — it'll be lost on restart:\n`{e}`",
                ephemeral=True,
            )
            return
        verb = "Updated" if existed else "Added"
        if persisted:
            note = "Saved to the roster and live in the current recording."
        else:
            note = (
                "Live in the current recording, but no roster file is "
                "configured so it won't survive a restart."
            )
        await ctx.respond(
            f"{verb} {user.mention} → **{player_name}** ({character_name}). {note}",
            ephemeral=True,
        )

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
            # Every guild the bot is in, not just ones recorded in this
            # process: after a restart the session is found on disk
            # (roborev #6515).
            for gid in {g.id for g in bot.guilds}:
                # A just-started session nobody has spoken in has a file
                # but no text: not a candidate (roborev #6523).
                path = bot.session_file_for_export(gid)
                if not path or not sessions.session_transcript(path)[0].strip():
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

        # The whole session, not just this recording: after a bot restart the
        # earlier recordings of the same campaign night are joined in front
        # (src/sessions.py). Plus the session's header and note folders.
        # session_file_for_export, not the live sink, so /ask and /recap
        # still work after /stop, /disconnect or a restart (csk, as fwv did
        # for /generate_pdf).
        session_file = bot.session_file_for_export(guild_id)
        transcript_text, header, note_dirs = None, "", []
        if session_file:
            transcript_text, header, note_dirs, _files = sessions.ask_context(
                session_file
            )

        source = "discord-public" if public else "discord"

        def _log(status, started=None, **kw):
            question_log.record(
                source=source,
                question=question,
                status=status,
                asker=getattr(ctx.author, "display_name", None) or str(ctx.author),
                asker_id=getattr(ctx.author, "id", None),
                guild_id=guild_id,
                transcript_path=session_file,
                latency_s=(time.perf_counter() - started) if started else None,
                **kw,
            )

        if not transcript_text or not transcript_text.strip():
            no_data = (
                "No transcript data in the current session yet. Start recording first."
            )
            _log(question_log.REJECTED, error=no_data)
            await ctx.respond(no_data, ephemeral=True)
            return

        # ephemeral=not public: /ask is private to the asker,
        # /ask-public posts the answer to the whole channel. Only the first
        # follow-up inherits the defer's visibility, so every send below
        # passes it explicitly (send_followups; roborev #6486).
        private = not public
        await ctx.defer(ephemeral=private)

        started = time.perf_counter()
        try:
            # llm_ask is blocking and can take tens of seconds, so it must
            # not run on the event loop — that would stall voice receive and
            # the gateway heartbeat, not just this command.
            result = await asyncio.to_thread(
                llm_ask,
                transcript_text,
                question,
                header=header,
                note_dirs=note_dirs,
                max_chars=answer_budget(question),
            )
            # No truncation: anything past Discord's limit is split below.
            answer = clean_answer(result.answer)
            logger.info(
                f"/ask answered by {result.model} (input={result.input_tokens}, "
                f"cached={result.cached_tokens}, output={result.output_tokens}, "
                f"rounds={result.rounds}, notes={result.notes_used or '-'})"
            )

            if not answer:
                empty = "The model returned no answer. Try rephrasing the question."
                _log(question_log.FAILED, started, error=empty, result=result)
                await send_followups(ctx, [empty], ephemeral=private)
                return
            # The model was given a budget (answer_budget) so this is
            # normally one message; if it ran long, post the rest as
            # follow-ups instead of cutting it off.
            messages = discord_messages(question, answer, DISCORD_MESSAGE_LIMIT)
            if len(messages) > 1:
                logger.warning(
                    f"/ask answer ran {len(answer)} chars (budget "
                    f"{answer_budget(question)}); sent as {len(messages)} messages"
                )
            # Record exactly once, after delivery is known (roborev #6459):
            # the full answer goes to the history either way, and a failed
            # send is noted, not logged a second time as a FAILED question
            # by the handler below.
            try:
                await send_followups(ctx, messages, ephemeral=private)
            except Exception as send_err:
                _log(
                    question_log.ANSWERED,
                    started,
                    answer=result.answer,
                    result=result,
                    error=f"answer not delivered to Discord: {send_err!r}",
                )
                logger.error(f"/ask answer could not be delivered: {send_err!r}")
                return
            _log(question_log.ANSWERED, started, answer=result.answer, result=result)
        except AskBlocked as e:
            # The GM check refused it (strategy, min-maxing, secrets). Polite
            # refusal; it went through the same deferred response, so
            # /ask-public refusals are visible like answers would be.
            _log(question_log.BLOCKED, started, error=e.user_message, verdict=e.verdict)
            await send_followups(
                ctx,
                discord_messages(question, e.user_message, DISCORD_MESSAGE_LIMIT),
                ephemeral=private,
            )
        except AskFailed as e:
            # The backend has already chosen the message that fits the reason
            # (bad key, rate limit, out of credit, unreachable, refusal), so
            # pass it straight through rather than guessing here.
            logger.error(f"/ask failed: {e.user_message} ({e.__cause__!r})")
            _log(question_log.FAILED, started, error=e.user_message)
            await send_followups(ctx, [e.user_message], ephemeral=private)
        except Exception as e:
            logger.exception("/ask failed unexpectedly")
            _log(question_log.FAILED, started, error=f"unexpected: {e!r}")
            await send_followups(
                ctx, [f"Failed to query the model.\n`{e}`"], ephemeral=private
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

    @bot.slash_command(
        name="recap",
        description="Summarise the current or most recent session.",
        contexts={
            discord.InteractionContextType.guild,
            discord.InteractionContextType.bot_dm,
        },
    )
    async def recap(
        ctx: discord.context.ApplicationContext,
        public: discord.Option(
            bool,
            description="Post the recap to the whole channel (default: only you)",
            required=False,
            default=False,
        ),
        server: discord.Option(
            str,
            description="Server name (only needed in DMs if you share multiple servers)",
            required=False,
            default=None,
        ),
    ):
        # Same path as /ask (GM check, history, splitting), fixed question.
        await _run_ask(ctx, RECAP_QUESTION, server, public=public)

    @bot.slash_command(
        name="transcript",
        description="Download the current or most recent session as a text file.",
    )
    async def transcript_file(ctx: discord.context.ApplicationContext):
        path = bot.session_file_for_export(ctx.guild_id)
        text = sessions.session_transcript(path)[0] if path else ""
        if not text.strip():
            await ctx.respond(
                "No transcript data yet. Start a recording first.", ephemeral=True
            )
            return
        data = timefmt.for_model(text).encode("utf-8")
        if len(data) > TRANSCRIPT_ATTACHMENT_LIMIT:
            await ctx.respond(
                "This session's transcript is too large for a Discord attachment. "
                "Download it from the dashboard's Transcripts page.",
                ephemeral=True,
            )
            return
        name = os.path.splitext(os.path.basename(path))[0]
        await ctx.respond(
            "Here's the full session transcript.",
            file=discord.File(io.BytesIO(data), filename=f"{name}.txt"),
            ephemeral=True,
        )

    @bot.slash_command(
        name="session",
        description="Show which group, game, campaign and session this recording is tagged with.",
    )
    async def session_info(ctx: discord.context.ApplicationContext):
        sink = bot.guild_whisper_sinks.get(ctx.guild_id)
        path = getattr(sink, "session_file", None)
        if not path:
            await ctx.respond(
                "Not recording right now. Use /scribe first.", ephemeral=True
            )
            return
        lib = library.load()
        meta = sessions.read_meta(path)
        files = [f for f in sessions.session_files(path) if f.exists()]
        d = sessions.describe(lib, meta, max(1, len(files)))
        if not d["tagged"] or not d.get("campaign"):
            await ctx.respond(
                "This recording isn't tied to a campaign. Add this server to a group "
                "in `library/library.yml` (or tag it later with "
                "`scripts/tag_session.py`).",
                ephemeral=True,
            )
            return
        lines = [
            f"**Group:** {d['group']}" + (f" (GM {d['gm']})" if d["gm"] else ""),
            f"**Game:** {d['game']}",
            f"**Campaign:** {d['campaign']}",
            f"**Session:** {d['session_day']}"
            + (f" · {d['recordings']} recordings" if d["recordings"] > 1 else ""),
        ]
        await ctx.respond("\n".join(lines), ephemeral=True)

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
                name="/session",
                value="Show the group, game, campaign and session being recorded.",
                inline=True,
            ),
            discord.EmbedField(
                name="/recap", value="Summarise the session.", inline=True
            ),
            discord.EmbedField(
                name="/transcript", value="Download the session as text.", inline=True
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
                name="/add_player",
                value="Name one person on the call (works mid-session).",
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
        # Reuse the bot's existing HealthCheck instance to avoid re-loading
        # the whisper model on every /health invocation.
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
