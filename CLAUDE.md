# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Discord bot that transcribes voice channel audio to text in real-time using Faster Whisper (local) or OpenAI Whisper API. Built for D&D game sessions. Includes a Flask web dashboard for managing the bot and viewing transcripts.

## Commands

```bash
make start          # Run the bot (python main.py)
make web            # Run the web dashboard (Flask on port 5001)
python main.py -v   # Run with verbose/debug logging
```

Setup: `python -m venv venv && source venv/bin/activate && pip install -r requirements.txt`

## Architecture

**Two separate processes:**
- **Bot process** (`main.py`): Discord bot using Pycord. Slash commands are registered inline in `main.py`, not in cogs. Runs health checks with auto-fix at startup.
- **Web dashboard** (`web/app.py`): Flask app that manages the bot subprocess and displays transcripts. Runs independently.

**Key flow — voice transcription:**
1. `/connect` joins voice channel, creates a `BotHelper` per guild
2. `/scribe` creates a `WhisperSink` (extends Pycord's `Sink`) that receives raw audio per user
3. `WhisperSink.write()` receives audio bytes from Discord, queues them per-speaker
4. A background thread (`insert_voice`) monitors speakers, waits for 1.5s silence gap, then submits audio to a `ThreadPoolExecutor` for transcription
5. Transcription results are written to per-session files (`transcripts/`) and a logging-based transcript log (`.logs/transcripts/`)

**State is tracked per-guild on the `VoloBot` instance:**
- `guild_to_helper` — `BotHelper` (voice client wrapper)
- `guild_is_recording` — recording state flag
- `guild_whisper_sinks` — active `WhisperSink` per guild

**Health system** (`src/bot/health.py`): `HealthCheck` runs startup checks (ffmpeg, opus, whisper model, ollama, env vars, directories) with optional auto-fix. Writes status to `.logs/health_status.json` for the web UI to read.

## Key Dependencies

- **py-cord** (not discord.py) — the `discord` import is Pycord
- **faster_whisper** — local transcription model (large-v3), loaded as module-level singleton in `whisper_sink.py`
- **ollama** — powers the `/ask` command (model: `ai/mistral:latest`)
- **torch** — CPU-only by default (see requirements.txt for CUDA option)
- **ffmpeg** and **libopus** — required system dependencies for voice

## Environment Variables

Required in `.env`:
- `DISCORD_BOT_TOKEN` — bot token
- `TRANSCRIPTION_METHOD` — `local` (default) or `openai`
- `OPENAI_API_KEY` — only if using openai method
- `PLAYER_MAP_FILE_PATH` — optional path to `player_map.yml` mapping Discord user IDs to player/character names

## Code Style

- Formatter: `black`
- Linter: `pylint`
- No test suite currently exists

## Workflow

- **Commit after every significant change.** After a logically-complete unit of work (a feature, a bug fix, a dependency upgrade, a refactor that touches multiple files), stage the relevant files and create a commit before moving on. Do not batch unrelated changes into one commit. Skip commits only for trivial in-progress edits or when the user explicitly asks you to hold off.
- Never include a `Co-Authored-By` line in commit messages.
- Do not stage `.DS_Store`, `__pycache__/`, `.env`, `transcripts/`, or `.logs/` — these are gitignored.

## Known Constraints

- **DAVE protocol blocker (as of 2026-05-12):** Discord enforces E2EE on voice; this project depends on Pycord PR #3159 (`fix/voice-rec-2` branch) for voice-receive DAVE support. `requirements.txt` pins the branch directly. Switch to a stable release once Pycord 2.8.0 ships with the receive PR merged. See `DAVE_MIGRATION.md` for context.
