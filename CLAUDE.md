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
6. `/stop` (or `/disconnect`, or the bot being kicked) stops the loop, which then flushes every still-buffered speaker, and `stop_voice_thread` waits up to `DRAIN_TIMEOUT_S` for the in-flight transcriptions before `close()` finalizes the session file — so the last thing said before `/stop` is not lost. `start_recording` returns `False` when the sink could not start and `/scribe` reports that instead of claiming to record.

**State is tracked per-guild on the `VoloBot` instance:**
- `guild_to_helper` — `BotHelper` (voice client wrapper)
- `guild_is_recording` — recording state flag
- `guild_whisper_sinks` — active `WhisperSink` per guild

**Health system** (`src/bot/health.py`): `HealthCheck` runs startup checks (ffmpeg, opus, whisper model, Anthropic key, env vars, directories) with optional auto-fix. Writes status to `.logs/health_status.json` for the web UI to read.

## Key Dependencies

- **py-cord** (not discord.py) — the `discord` import is Pycord
- **faster_whisper** — local transcription model (large-v3), loaded as module-level singleton in `whisper_sink.py`
- **anthropic** (SDK) — `/ask` (`src/llm/claude.py`); **openai** (SDK) is only the optional cloud transcription method
- **torch** — CPU-only by default (see requirements.txt for CUDA option)
- **ffmpeg** and **libopus** — required system dependencies for voice

## Environment Variables

Required in `.env` (a derived copy; 1Password is the source of truth for the secret ones, see "Secrets" below):
- `DISCORD_BOT_TOKEN` — bot token
- `TRANSCRIPTION_METHOD` — `local` (default) or `openai`
- `OPENAI_API_KEY` — only if using openai method
- `PLAYER_MAP_FILE_PATH` — optional path to `player_map.yml` mapping Discord user IDs to player/character names
- `ANTHROPIC_API_KEY` — for `/ask`; load it from 1Password via `op.env` (see "Secrets"). Optional `ASK_MODEL` (default `claude-opus-5-5`) and `ASK_EFFORT` (default `medium`).

### `/ask` (`src/llm/claude.py`)

`/ask` is one call to Claude through the Anthropic SDK. The transcript goes in a cached system block (`cache_control: ephemeral`) so follow-up questions re-read it at the cache rate; the question is the single user message; `output_config.effort` is `ASK_EFFORT`. The SDK's own retries handle 429/5xx; everything else maps to a Discord-ready message (`AskFailed.user_message`: bad key, rate limit, out of credit, unreachable, refusal, empty answer), covered by `tests/test_llm_claude.py`. There is no local fallback: without wifi there is no Discord voice either, so a local tier only ever covered "Anthropic down while Discord up". This replaced a three-tier OpenRouter → Cerebras → Ollama chain (discord-transcript-bot-gh7); do not reintroduce provider failover here.

`health.py:_check_ask_provider` only reports whether the key is present (a test call would spend money on every start) and is never critical: transcription does not depend on `/ask`.

**`src/llm/claude.ask()` is blocking and must be called via `asyncio.to_thread`.** A call can take tens of seconds, which would stall voice receive and the gateway heartbeat rather than just the command.

### Transcription engine selection (`src/asr/selection.py`)

These tune the local transcription backend (no effect when `TRANSCRIPTION_METHOD=openai`):
- `WHISPER_MODEL` — model id. Default `large-v3` (accuracy-first). `large-v3-turbo` is selectable for speed-constrained hosts.
- `ASR_BACKEND` — `auto` (default), `mlx`, or `faster-whisper`. `auto` picks MLX-Whisper (Metal GPU) on Apple Silicon and falls back to faster-whisper-CPU everywhere else (or if MLX can't load). Selection never raises.
- `MLX_WHISPER_MODEL` — optional; override the MLX HF repo (otherwise mapped from `WHISPER_MODEL`, e.g. `large-v3` → `mlx-community/whisper-large-v3-mlx`).
- `WHISPER_BEAM_SIZE` (default 5), `WHISPER_BEST_OF` (default 5), `WHISPER_BATCH_SIZE` (default 8) — faster-whisper decode params. The MLX backend ignores all three: `mlx_whisper` has no beam-search decoder (greedy + temperature fallback only) and no batched pipeline. So the beam5-vs-beam10 accuracy question applies only to the faster-whisper path; the final value is set by the A/B bake-off (discord-transcript-bot-d6j).

**Warm the model before a session:** `make prewarm`. Weights download lazily on the first transcription — ~3 GB for MLX — so a cold cache means the first person to speak stalls the bot for minutes. `make prewarm` downloads and does a real end-to-end decode; `python scripts/prewarm_models.py --check` just reports cache state (exit 1 when the resolved model is cold). The startup health check runs the same probe (`src/asr/model_cache.py`) and logs a loud warning *before* a cold-cache download starts, so a multi-minute stall is never unexplained; the probe mirrors `selection._build_backend`'s choice without loading anything, so keep the two in step.

### Reference-audio capture (`src/session_capture.py`)

- `CAPTURE_SESSION_AUDIO` — **off by default.** When truthy, the sink persists the exact per-speaker WAV bytes it feeds the engine to `captures/<session>/`, plus `manifest.draft.jsonl` pre-filled with the machine transcription. This turns a normal game into reference data for the A/B harness (discord-transcript-bot-3dn), which is the gating input for every remaining accuracy decision.
- `CAPTURE_MAX_GB` — disk ceiling, default 20. Capture stops past it; transcription is unaffected.

The draft manifest's `reference` fields are **machine output, not ground truth** — a human corrects them, then saves as `manifest.jsonl` for `scripts/ab_transcribe.py --manifest`. Scoring against an uncorrected draft compares Whisper to itself and reports a meaninglessly low WER; the harness warns if handed a `.draft` file. Each capture directory gets a README with the correction workflow.

**Marginal clips are order-dependent** (discord-transcript-bot-adg): on near-unintelligible audio the MLX decode depends on what ran earlier in the same process, so those clips add order-dependent noise to the corpus WER. Bake-off methodology: `ab_transcribe.py --order-trials 3 --stable-manifest <out>` flags clips whose hypothesis changes between shuffled-order runs and writes the manifest without them; delete near-unintelligible clips from the reference rather than guessing their text; then score with `--isolate` (one subprocess per config). Per-clip WER and hypotheses are always in `--json-out`.

Capture is best-effort by construction: `SessionCapture` swallows its own errors and the sink wraps its calls again, so no capture bug can cost a live session its transcription.

## Secrets

**Every credential an agent uses lives in 1Password.** API keys, tokens, DSNs, OAuth client secrets, webhook URLs, signing keys: if Claude Code, a cron job, an MCP server, or CI needs it, 1Password is the source of truth, and anything else (a local `.env`, SSM, GitHub Secrets, a server-side env file) is a derived copy seeded from it. This applies to the credentials this repo already uses, not only to new ones.

- **Reference, don't copy.** Tracked files carry env var *names* and `op://<vault>/<item>/<field>` references, never values. Load at run time with `op run --env-file=op.env -- <cmd>` (an `op.env` of `op://` lines, gitignored) or `op read "op://..."` piped straight into the consumer.
- **Name items `<repo>-<service>`** (for example `mc-epn`), one item per service, fields named after the env vars. Use the 1Password account that owns the work (personal `my.1password.com`, VisionWrights, or Newtack) and pass `--account` when more than one is signed in.
- **A literal `.env` is debt, not a pattern.** It must be gitignored and `chmod 600`, and the inventory below marks it as pending migration. Move the values into 1Password the next time you touch them.
- **Never print a secret.** Prove a key works by calling the API. Don't `cat` env files, echo variables, or paste values into transcripts, commit messages, or this file.
- **Found a literal secret in a tracked file or in history?** Stop and tell the owner. It needs rotating and purging, and that is their call.
- **Adding a credential:** create the 1Password item, add the `op://` line to `op.env.example` (or this repo's equivalent template), add a row below, and confirm the local copy is ignored.

`main.py` calls `load_dotenv()`, which does not override variables already in the process, so `op run --env-file=op.env -- ./venv/bin/python main.py` works unchanged and the literal `.env` can shrink to the non-secret settings. `.env.sample` is the template for the non-secret settings; `op.env.example` is the template for `op.env`, and `make start` runs under `op run` whenever `op.env` exists.

### Credentials this repo uses

| Env var | Used by / purpose | Where it lives today | 1Password item |
|---|---|---|---|
| `DISCORD_BOT_TOKEN` | bot login (`src/bot/`, checked by `health.py`) | gitignored `.env`, literal value (migrate) | `dtb-discord` (to create) |
| `ANTHROPIC_API_KEY` | `/ask` (`src/llm/claude.py`, presence checked by `health.py`) | `op.env` → 1Password (the pattern); a literal `.env` value is a stopgap | `Anthropic - evan.j.ray@gmail.com - Discord Transcript Bot` (id `4dslv3wu23yu45xzgtpjenfoci`, field `key`) |
| `OPENAI_API_KEY` | transcription only when `TRANSCRIPTION_METHOD=openai` | not set today; when set, same `.env` (migrate) | `dtb-openai` (to create) |

Not secrets, so not in the table: `DISCORD_CHANNEL_ID`, `PLAYER_MAP_FILE_PATH`, `TRANSCRIPTION_METHOD`, `ASK_MODEL`, `ASK_EFFORT`, `ASR_BACKEND`, `WHISPER_*`, `MLX_WHISPER_MODEL`, `CAPTURE_*`, `BIAS_PROMPT_WITH_ROSTER`, `HF_HOME`, `HF_HUB_CACHE`, `FLASK_DEBUG`.

## Code Style

- Formatter: `black`
- Linter: `pylint`

## Testing

- **Automated coverage is required for new code.** Whenever you add or change behavior, write or extend the corresponding automated check before considering the change done. Targets:
  - Flask routes / HTTP surface → `tests/test_web.py` (pytest + Flask test client)
  - Pure Python helpers (path handling, parsing, formatting) → unit tests alongside existing ones
  - Side-effecting startup logic (health checks, file lifecycle) → smoke scripts under `scripts/`
- Run `make preflight` before reporting work done. It executes the health checks and the pytest suite. If you cannot add an automated test for a change (e.g., it requires live Discord voice), state that explicitly in the commit/PR description and document the manual verification steps.
- Live voice session verification: see `scripts/post_session_verify.sh` and the test plan kept in conversation history (TODO: promote to `TESTING.md` if it gets reused).

## Workflow

- **Commit after every significant change.** After a logically-complete unit of work (a feature, a bug fix, a dependency upgrade, a refactor that touches multiple files), stage the relevant files and create a commit before moving on. Do not batch unrelated changes into one commit. Skip commits only for trivial in-progress edits or when the user explicitly asks you to hold off.
- **Run `make preflight` before each commit involving non-trivial code changes.** If preflight fails, fix the regression before committing.
- Never include a `Co-Authored-By` line in commit messages.
- Do not stage `.DS_Store`, `__pycache__/`, `.env`, `transcripts/`, or `.logs/` — these are gitignored.

## Known Constraints

- **DAVE protocol blocker (rechecked 2026-05-18):** Discord enforces E2EE on voice; this project depends on Pycord PR #3159 (`fix/voice-rec-2` branch) for voice-**receive** DAVE support. `requirements.txt` pins the branch directly. **Pycord 2.8.0 shipped 2026-05-18 but only includes DAVE voice-*send* (PR #3143) — the receive PR #3159 is still a draft, moved to the 2.9.0 milestone.** Do NOT switch to stable `py-cord==2.8.0`; it would re-break `/scribe`. Switch to a stable pin only once #3159 is merged and released (track the 2.9.0 milestone). See `DAVE_MIGRATION.md` for context.
- **100%-core spin while recording (discord-transcript-bot-309):** observed once (2026-08-11) on a silent room; measured to be neither the engine nor `insert_voice` and to stop on `/stop` while still connected — i.e. a thread Pycord starts only while recording. Offline reproduction cleared `PacketRouter`/`JitterBuffer` (0% idle in every scenario) and found that `UDPKeepAlive.run` retries a failed UDP send with **no delay** (~6.5k iterations/s, logged only at DEBUG), plus its `delay = 5000` is used as seconds. `src/bot/pycord_patches.py` installs a guarded 1s backoff on that error path and logs the failure at WARNING; it refuses to patch if the upstream method changes shape. Not yet confirmed live: if `cpu_pct` in `.logs/bot_state.json` sits near 100 while `queue_depth` is 0, run `kill -USR1 <pid>` — `src/utils/thread_dump.py` writes a ranked stack dump to `.logs/thread_dump-*.txt` that names the thread. Upstream report draft: `docs/upstream/pycord-udp-keepalive-spin.md`.


<!-- BEGIN BEADS INTEGRATION v:1 profile:minimal hash:7510c1e2 -->
## Beads Issue Tracker

This project uses **bd (beads)** for issue tracking. Run `bd prime` to see full workflow context and commands.

### Quick Reference

```bash
bd ready              # Find available work
bd show <id>          # View issue details
bd update <id> --claim  # Claim work
bd close <id>         # Complete work
```

### Rules

- Use `bd` for ALL task tracking — do NOT use TodoWrite, TaskCreate, or markdown TODO lists
- Run `bd prime` for detailed command reference and session close protocol
- Use `bd remember` for persistent knowledge — do NOT use MEMORY.md files

**Architecture in one line:** issues live in a local Dolt DB; sync uses `refs/dolt/data` on your git remote; `.beads/issues.jsonl` is a passive export. See https://github.com/gastownhall/beads/blob/main/docs/SYNC_CONCEPTS.md for details and anti-patterns.

**`.beads/` is gitignored in this repo (decision 2026-10-20).** The export, hooks and config under `.beads/` are never committed; tracker state lives only in the local Dolt DB on this machine. Do not `git add -f` anything under `.beads/`, and never write a `chore(bd): sync export` commit.

## Session Completion

**When ending a work session**, you MUST complete ALL steps below. Work is NOT complete until `git push` succeeds.

**MANDATORY WORKFLOW:**

1. **File issues for remaining work** - Create issues for anything that needs follow-up
2. **Run quality gates** (if code changed) - Tests, linters, builds
3. **Update issue status** - Close finished work, update in-progress items
4. **PUSH TO REMOTE** - This is MANDATORY:
   ```bash
   git pull --rebase
   git push
   git status  # MUST show "up to date with origin"
   ```
5. **Clean up** - Clear stashes, prune remote branches
6. **Verify** - All changes committed AND pushed
7. **Hand off** - Provide context for next session

**CRITICAL RULES:**
- Work is NOT complete until `git push` succeeds
- NEVER stop before pushing - that leaves work stranded locally
- NEVER say "ready to push when you are" - YOU must push
- If push fails, resolve and retry until it succeeds
<!-- END BEADS INTEGRATION -->

<!-- BEGIN WYK CONVENTIONS v:1 -->
## wyk — planning & handoff over bd

This repo uses **wyk**, a view + handoff layer over **bd (beads)**. "Plan
it in wyk" = **file the plan as bd issues** (deps via `bd dep add`), not
markdown/TodoWrite. File with **`wyk create`** (same flags as `bd create`,
forwarded verbatim) — it also stamps the Claude session so the TUI's
Session column traces work back to a conversation. A PreToolUse hook
blocks raw `bd create` and tells you to switch; that's expected — just
re-run as `wyk create`.

**Owner column** — whose move it is, label-driven (NOT bd's owner/assignee):
- `human` → **HUMAN** (a human must act).
- `agent-handoff` → **AGENT-HANDOFF**: another agent owns it; don't touch,
  a human coordinates. Excluded from `wyk inbox`.
- agent task blocked by a `human`-flagged dep → **HUMAN-BLOCK** (skip it).
- else → **AGENT** (the default; a null owner is never blank — so a task
  that needs a human MUST be handed off, or the human never sees it).

**Hand off to a human**: `wyk handoff <id>` (or `wyk handoff -create "<title>"`)
sets `human` + writes the runbook. Never hand-roll labels; `-a`/`--claim`
are bd's status, not the badge.

**Pick up work**: `wyk inbox` FIRST (items bounced back to you — WORK them),
then `wyk` / `bd ready`. `wyk conventions` prints the full contract.

**Something wrong? Act — don't shrug.** If a wyk/bd command errors, a
convention looks broken, or the workflow rubs wrong, file a bd issue (with
an owner) and fix or hand it off — don't route around it silently.
Friction with wyk is product data; surfacing it is the job.
<!-- END WYK CONVENTIONS -->
