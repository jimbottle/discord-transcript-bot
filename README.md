
# V.O.L.O Discord Transcription Bot

This project is a Discord bot that transcribes voice channel audio into text in real-time. It uses Whisper for audio transcription and is capable of handling multiple users in a voice channel.

> **This is a fork of [joshinryz/volo_discord_bot](https://github.com/joshinryz/volo_discord_bot)** by Josh Robinett, with additional contributions from Noah Feinberg. It diverged in March 2025 and has been substantially extended since — see [Attribution](#attribution) and [License](#license).

## Features

- This project uses Pycord (see [Pycord Github](https://github.com/Pycord-Development/pycord))
- This project uses Faster Whisper (see [Faster Whisper Github](https://github.com/SYSTRAN/faster-whisper))
- Transcribes voice channel audio to text.
- Supports multiple users.
- Thread-safe operations for concurrent transcriptions.
- Web dashboard for managing the bot and viewing transcripts.
- Startup health checks with auto-fix (verifies ffmpeg/opus/the Whisper model, creates directories).
- `/ask`, `/recap` and a dashboard ask box that answer questions about the session. They run on **a local model on your own machine by default** (Ollama or any OpenAI-compatible server: free, and transcripts never leave the machine), or on Claude through the Anthropic API if you add a key.

## Setup

To set up and run this Discord bot, follow these steps:

### Prerequisites

- Python 3.8 or higher.
- Discord bot token (see [Discord Developer Portal](https://discord.com/developers/applications)).
- `ffmpeg` installed and added to your system's PATH.
- For `/ask` (optional; transcription works without it), one of:
  - **Recommended:** [Ollama](https://ollama.com) (or another OpenAI-compatible server) and a tool-capable model. See [Questions about the session](#questions-about-the-session-ask).
  - An Anthropic API key.

### Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/jimbottle/discord-transcript-bot.git
   cd discord-transcript-bot
   ```

2. **Create a Virtual Environment (optional but recommended):**

   ```bash
   python -m venv venv
   # Activate the virtual environment
   # On Windows: venv\Scripts\activate
   # On macOS/Linux: source venv/bin/activate
   ```

3. **Install Dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Environment Variables:**

   Create a `.env` file in the root directory:

   Copy `.env.sample` to `.env` and fill it in. The essentials:

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

### Configuration

- Create a `player_map.yml` to map Discord user IDs to player and character names, or use `/update_player_map` to auto-generate it from the server roster.

## Usage

1. **Start the Bot:**

   ```bash
   make start
   # or: python main.py
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

Contributions to this project are welcome. Please ensure to follow the project's coding style and submit pull requests for any new features or bug fixes.

## Attribution

This project is a fork of [joshinryz/volo_discord_bot](https://github.com/joshinryz/volo_discord_bot) (Josh Robinett, 2023–2025; additional contributions by Noah Feinberg). The fork point is upstream commit `5e610b2` (2025-03-17); the full upstream commit history is preserved in this repository's git log.

Code originating upstream and still present here, in whole or in part:

- `src/bot/helper.py`, `src/utils/pdf_generator.py`, `src/utils/commandline.py`, `src/config/cliargs.py` — substantially unchanged from upstream
- `main.py`, `src/sinks/whisper_sink.py`, `src/bot/volo_bot.py` — derived from upstream and substantially rewritten and extended
- `assets/parchment_background.jpg` — inherited from upstream; its own provenance is not documented there

Everything else — the ASR backend selection layer, reference-audio capture, the A/B transcription harness, the health-check system, the web dashboard, the player-map store, and the test suite — is original to this fork.

## License

**No license is granted yet, because the upstream grant is unresolved.**

The upstream project publishes no LICENSE file and never has — its README states "MIT License" but links to a file that has never existed in that repository, so the upstream grant is unresolved. This fork therefore cannot pass on rights it has not clearly received, and makes no license claim of its own.

If you want to use, copy, or redistribute this code, open an issue. Resolving the upstream grant with the original author is tracked as pending work; this section will be replaced with a real license once that is settled.

## Architecture

- **Bot process** (`main.py`): Discord bot using Pycord, runs health checks at startup with auto-fix.
- **Web dashboard** (`web/app.py`): Flask app that manages the bot process and displays transcripts.
- **Health bridge**: The bot writes `.logs/health_status.json` as checks progress; the web UI reads it to show real-time initialization status (Starting → Initializing → Ready/Failed).

## Acknowledgments

- Originally created by [Josh Robinett](https://github.com/joshinryz) as [volo_discord_bot](https://github.com/joshinryz/volo_discord_bot), with contributions from [Noah Feinberg](https://github.com/NoahFeinberg). This fork builds directly on that work.
- This project uses [Faster Whisper](https://github.com/SYSTRAN/faster-whisper) for audio transcription.
- Built with [Pycord](https://github.com/Pycord-Development/pycord) for Discord integration.
- Uses Claude (Anthropic API) for transcript queries; the transcript is sent as a cached system block.
