# Discord Transcript Bot

A Discord bot that sits in a voice channel and writes down what everyone says, as they say it. Built for tabletop sessions over Discord: the transcript is kept per session, speakers are labelled with player and character names, and you can ask questions about the game afterwards.

Transcription runs on your own machine with Whisper (MLX-Whisper on Apple Silicon, faster-whisper elsewhere), or through OpenAI's Whisper API if you prefer.

> This project began in March 2025 as a fork of another Discord transcription bot. The code from that project has since been replaced in full; see [Attribution](#attribution).

## Features

- Live transcription of a voice channel, one speaker at a time, with names from a roster file.
- Per-session transcript files, a JSON log, a text download and a PDF export.
- Web dashboard for managing the bot and viewing transcripts.
- Startup health checks with auto-fix (verifies ffmpeg/opus/the Whisper model, creates directories).
- `/ask`, `/recap` and a dashboard ask box that answer questions about the session. They run on **a local model on your own machine by default** (Ollama or any OpenAI-compatible server: free, and transcripts never leave the machine), or on Claude through the Anthropic API if you add a key.

## Setup

### Prerequisites

- Python 3.10 or newer.
- A bot token from the [Discord Developer Portal](https://discord.com/developers/applications), with the voice and message-content intents enabled.
- `ffmpeg` and `libopus` on the `PATH` (the startup health check tells you if either is missing).
- For `/ask` (optional; transcription works without it), one of:
  - **Recommended:** [Ollama](https://ollama.com) (or another OpenAI-compatible server) and a tool-capable model. See [Questions about the session](#questions-about-the-session-ask).
  - An Anthropic API key.

### Installation

1. **Get the code and a virtual environment.** The Makefile targets expect the venv at `./venv`.

   ```bash
   git clone https://github.com/jimbottle/discord-transcript-bot.git
   cd discord-transcript-bot
   python -m venv venv && source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Warm the model.** Whisper weights are downloaded on first use (about 3 GB); do it now rather than when the first person speaks.

   ```bash
   make prewarm
   ```

3. **Configure.** Copy `.env.sample` to `.env` and fill it in. The essentials:

   ```
   DISCORD_BOT_TOKEN=your_discord_bot_token
   PLAYER_MAP_FILE_PATH=path_to_player_map.yml   # optional
   TRANSCRIPTION_METHOD=local                      # or "openai"
   OPENAI_API_KEY=your_key                         # only if using openai method
   ASK_PROVIDER=local                              # /ask on a local model (see below)
   ```

### Questions about the session (`/ask`)

`/ask`, `/ask-public`, `/recap`, the dashboard's ask box and its Lore page all go through one setting, `ASK_PROVIDER`.

**Option A (recommended): a local model.** Nothing leaves your machine and nothing costs per question.

1. Install [Ollama](https://ollama.com) and pull a model that supports tool calls:

   ```bash
   ollama pull gemma4:26b
   ```

   `gemma4:26b` is the model this was tested with. It needs about 18 GB of free memory (a 32 GB Apple Silicon Mac, or a 24 GB GPU). Smaller tool-capable models such as `qwen3:14b` or `qwen3:8b` should work on smaller machines with weaker answers; they haven't been tested here.

2. In `.env`:

   ```
   ASK_PROVIDER=local
   LOCAL_LLM_MODEL=gemma4:26b
   # LOCAL_LLM_URL=http://localhost:11434/v1   # the default (Ollama)
   ```

   LM Studio, llama.cpp's `llama-server` and vLLM serve the same OpenAI-compatible API: point `LOCAL_LLM_URL` at them (and set `LOCAL_LLM_API_KEY` if yours needs one).

3. Check it: the bot's startup health check (and `/health`) reports whether the server is up and has the model.

What to expect, measured on an Apple Silicon Mac with `gemma4:26b` against a real 3-hour session:

- The first question about a session is the slow one: the model reads the session first. A recap took about 3½ minutes.
- Follow-up questions reuse what it read and take seconds to half a minute. During a live game only the new lines are read.
- The GM check takes 4–10 seconds per question and got the same 12/12 as the Anthropic API on the test questions (`python scripts/check_guard.py`).
- The local model reads **the most recent part of the session**, 100,000 characters by default (about the last hour and a half of play), and says so under the answer when it left the start out. Reading all of a long session was too much for this machine: it ran past the 14-minute limit and crashed the GPU. A faster machine can raise `LOCAL_LLM_MAX_TRANSCRIPT_CHARS` (`0` reads everything).

**Context length.** The server's context window must hold what is sent (100,000 characters is about 35k tokens). Recent Ollama versions size it from the model and your memory. If yours is older or smaller, set `OLLAMA_CONTEXT_LENGTH` (for example `OLLAMA_CONTEXT_LENGTH=131072 ollama serve`). If the window is too small, the server drops the start of the session without any error. The bot notices, from how many tokens the server reports reading, and adds a note to the answer.

**Option B: the Anthropic API.** Faster, reads the whole session, stronger answers; paid per question.

```
ASK_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-...      # better: load it from a secrets manager, see op.env.example
# ASK_MODEL=claude-opus-5-5
```

If `ASK_PROVIDER` is not set, the bot uses Anthropic when `ANTHROPIC_API_KEY` is set, and a local model otherwise. Each question goes to exactly one provider: if it fails, you get the reason, not a silent switch to the other one.

**The GM check.** Before answering, every question is classified on the same provider (strategy, min-maxing, secrets and plot predictions are refused). On a local model it uses the same model, so nothing extra is loaded. `TABLE_GUARD=off` turns it off.

### The roster

`player_map.yml` (the path in `PLAYER_MAP_FILE_PATH`) maps Discord user ids to a player name and a character name; transcript lines are labelled from it. `/update_player_map` fills it from the server's members, `/add_player` names one person mid-session, and the dashboard has an editor.

## Usage

1. **Start the Bot:**

   ```bash
   make start
   # or: python main.py        (add -v for debug logging)
   ```

2. **Start the Web Dashboard:**

   ```bash
   make web
   ```

   The dashboard runs at `http://localhost:5001` and lets you start/stop the bot, view transcripts, and monitor health check status in real time.

3. **Bot Commands:**

   - `/connect`: Connect to your voice channel.
   - `/scribe`: Start transcribing in the current voice channel.
   - `/stop`: Stop transcribing.
   - `/disconnect`: Disconnect from the voice channel.
   - `/ask <question>`: Ask a question about the current or most recent session (works in DMs). Answered by the provider set up under [Questions about the session](#questions-about-the-session-ask).
   - `/recap`: A summary of the current or most recent session (private; `public: True` posts it). Same path as `/ask`.
   - `/transcript`: The full session as a text file, 12-hour times.
   - `/generate_pdf`: Export the current or most recent session (every recording of it) as a PDF. Works after `/stop`, `/disconnect` and a bot restart.
   - `/update_player_map`: Sync player names with the server roster.
   - `/health`: Show system health status.
   - `/help`: Show available commands.

## Docker

`docker compose up --build` builds one image with the bot and the dashboard and serves the dashboard on http://127.0.0.1:5001 (host loopback only; it has no login). Start the bot from the dashboard, or run it on its own with `docker compose run --rm volo python main.py`.

- **Secrets:** run it under 1Password so the host resolves them: `op run --env-file=op.env -- docker compose up`. A local `.env` is read too, if present.
- **Data stays on the host:** `transcripts/`, `captures/`, `history/`, `library/`, `.logs/` and `player_map.yml` are mounted, and `.dockerignore` keeps all of them (and every secret) out of the image. Create them before the first run: `mkdir -p transcripts captures history .logs && touch player_map.yml`. The container runs as uid/gid 1000 (set `DOCKER_UID`/`DOCKER_GID` to match `id -u`/`id -g`), and on Linux Docker creates a missing folder as root, which that user can't write; a missing `player_map.yml` becomes a directory. Folders left root-owned by an earlier run need `sudo chown -R "$(id -u):$(id -g)"`.
- **Local model for `/ask`:** run Ollama on the host as usual; compose points the bot at `http://host.docker.internal:11434/v1`. On Linux, Ollama listens only on 127.0.0.1 by default, which a container can't reach: start it with `OLLAMA_HOST=0.0.0.0` (and keep port 11434 firewalled from your network), or set `LOCAL_LLM_URL`.
- **Model weights:** about 3 GB, downloaded into the `models` volume on first start.
- **Speed:** the container runs faster-whisper on the CPU. MLX-Whisper needs Apple's Metal GPU, so on a Mac the host install (`make start`) is much faster. Use Docker for Linux hosts.

## Contributing

Pull requests are welcome. Run `make preflight` (the health checks and the test suite) before opening one, and add a test for anything you change; `black` and `pylint` are the formatter and linter.

## Attribution

This repository started in March 2025 as a fork of [joshinryz/volo_discord_bot](https://github.com/joshinryz/volo_discord_bot) by Josh Robinett, with contributions from Noah Feinberg. That project publishes no license, and its authors were asked and could not grant one, so in October 2026 every remaining line and asset inherited from it was rewritten or replaced: the voice-connection record, the command-line parser, the PDF export and its background image, and the inherited parts of `main.py`, the bot class and the transcription sink. `git blame` on the current tree attributes nothing to the upstream authors.

The commit history before that point still contains the upstream code as it was received, and the upstream project remains the origin of this one's shape: a Pycord recording sink feeding Whisper, with slash commands for connect, scribe and stop.

## License

**No license file yet.** The code in the current tree is the repository owner's own work (see [Attribution](#attribution)), and a license will be added once one is chosen. Until then, open an issue if you want to use or redistribute it.

## Architecture

- **Bot process** (`main.py`): Discord bot using Pycord, runs health checks at startup with auto-fix.
- **Web dashboard** (`web/app.py`): Flask app that manages the bot process and displays transcripts.
- **Health bridge**: The bot writes `.logs/health_status.json` as checks progress; the web UI reads it to show real-time initialization status (Starting → Initializing → Ready/Failed).

## Acknowledgments

- [Josh Robinett](https://github.com/joshinryz)'s [volo_discord_bot](https://github.com/joshinryz/volo_discord_bot), with contributions from [Noah Feinberg](https://github.com/NoahFeinberg), is where this project started; see [Attribution](#attribution).
- This project uses [Faster Whisper](https://github.com/SYSTRAN/faster-whisper) for audio transcription.
- Built with [Pycord](https://github.com/Pycord-Development/pycord) for Discord integration.
- Uses Claude (Anthropic API) for transcript queries; the transcript is sent as a cached system block.
