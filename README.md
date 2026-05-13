
# V.O.L.O Discord Transcription Bot

This project is a Discord bot that transcribes voice channel audio into text in real-time. It uses Whisper for audio transcription and is capable of handling multiple users in a voice channel.

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

## License

[MIT License](LICENSE)

## Architecture

- **Bot process** (`main.py`): Discord bot using Pycord, runs health checks at startup with auto-fix.
- **Web dashboard** (`web/app.py`): Flask app that manages the bot process and displays transcripts.
- **Health bridge**: The bot writes `.logs/health_status.json` as checks progress; the web UI reads it to show real-time initialization status (Starting → Initializing → Ready/Failed).

## Acknowledgments

- This project uses [Faster Whisper](https://github.com/SYSTRAN/faster-whisper) for audio transcription.
- Built with [Pycord](https://github.com/Pycord-Development/pycord) for Discord integration.
- Uses [Ollama](https://ollama.ai) for local LLM-powered transcript queries.
