# The bot and the dashboard in one image (discord-transcript-bot-ir0).
#
# Linux only runs the faster-whisper CPU backend: MLX-Whisper needs Metal, so
# on a Mac the host install (`make start`) transcribes much faster than this
# container. See README "Docker".
FROM python:3.10-slim-bookworm

# ffmpeg + libopus for voice, git for the pinned Pycord branch, procps/lsof
# for the dashboard's "is a bot already running" check.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libopus0 git procps lsof \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/models \
    DASHBOARD_HOST=0.0.0.0 \
    ASR_BACKEND=faster-whisper

EXPOSE 5001
# The dashboard; its Start button launches main.py in this container.
# `docker compose run --rm volo python main.py` runs the bot on its own.
CMD ["python", "web/app.py"]
