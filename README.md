
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
- Startup health checks with auto-fix (starts Ollama, pulls models, creates directories).
- `/ask` command to query the current transcript via Ollama (works in DMs too).

## Setup

To set up and run this Discord bot, follow these steps:

### Prerequisites

- Python 3.8 or higher.
- Discord bot token (see [Discord Developer Portal](https://discord.com/developers/applications)).
- `ffmpeg` installed and added to your system's PATH.
- [Ollama](https://ollama.ai) installed (for the `/ask` command).

### Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/your-github-username/discord-transcription-bot.git
   cd discord-transcription-bot
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

   ```
   DISCORD_BOT_TOKEN=your_discord_bot_token
   PLAYER_MAP_FILE_PATH=path_to_player_map.yml   # optional
   TRANSCRIPTION_METHOD=local                      # or "openai"
   OPENAI_API_KEY=your_key                         # only if using openai method
   ```

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
   - `/ask <question>`: Ask a question about the current transcript (works in DMs).
   - `/generate_pdf`: Export the current transcript as a PDF.
   - `/update_player_map`: Sync player names with the server roster.
   - `/health`: Show system health status.
   - `/help`: Show available commands.

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

**No license is currently granted for this project. All rights reserved.**

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
- Uses [Ollama](https://ollama.ai) for local LLM-powered transcript queries.
