# DAVE Protocol Migration Plan

## Problem

As of March 2, 2026, Discord enforces the DAVE (Discord Audio & Video End-to-End Encryption) protocol for all non-stage voice calls. Clients without DAVE support are rejected with voice gateway close code **4017**.

This project uses **pycord 2.7.1**, which does not support DAVE. The bot cannot connect to voice channels.

## What Needs DAVE Support

This bot depends on **voice receive** — recording audio from users via pycord's `Sink` / `start_recording()` API, then piping it through Whisper for transcription. This is the harder half of DAVE support because it requires decrypting incoming audio frames using per-sender E2EE keys.

## Upstream Tracking

*Last updated: 2026-04-02*

### Pycord PR #2873 — Voice Internals Rewrite & DAVE Support (original)
- URL: https://github.com/Pycord-Development/pycord/pull/2873
- **Status: CLOSED (not merged) on 2026-03-08** — closed due to unrelated changes bundled in
- Superseded by PR #3143 (send) and PR #3159 (receive)

### Pycord PR #3143 — Voice Internals Rewrite & DAVE Support (send only)
- URL: https://github.com/Pycord-Development/pycord/pull/3143
- **Status: MERGED on 2026-03-14**
- Covers: Voice connection + DAVE handshake + sending audio
- Included in **v2.8.0rc1** (released 2026-03-21 on PyPI)

### Pycord PR #3159 — Voice Receive with DAVE Support
- URL: https://github.com/Pycord-Development/pycord/pull/3159
- **Status as of 2026-04-02: OPEN, labeled "hold: testing", last updated 2026-03-27**
- Supersedes PR #3144 (earlier receive experiment, marked "NOT in a working state")
- Includes DAVE decryption fixes from community contributor vito1317 (cherry-picked 2026-03-19)
- Community testing on 2026-03-22/23 shows receive **mostly working** on `fix/voice-rec-2` branch
- Remaining edge cases: epoch transitions and OpusError handling during rekey
- **This is the critical blocker for our bot**
- Notifications: Subscribed

### Pycord PR #3179 — DAVE uid lookup fallback
- URL: https://github.com/Pycord-Development/pycord/pull/3179
- **Status: MERGED on 2026-03-24**
- Falls back to OPUS_SILENCE when DAVE brute-force uid lookup fails
- Indicates active iteration on receive stability

### Pycord Issue #3139 — Voice Receive Bugs & DAVE Rework
- URL: https://github.com/Pycord-Development/pycord/issues/3139
- Status as of 2026-04-02: Open, community testing ongoing on `fix/voice-rec-2` branch
- Covers: Reworking voice reception to work with DAVE, decryption failures, sink rework
- This is the critical tracking issue for our use case
- Notifications: Subscribed

### Pycord Issue #3135 — Voice connection closed with error 4017
- URL: https://github.com/Pycord-Development/pycord/issues/3135
- Status as of 2026-03-16: Open — send-side fix landed via #3143 but issue remains open
- Confirms the 4017 error on pycord 2.7.1

### Pycord Release: v2.8.0rc1 (2026-03-21)
- Includes DAVE send support (PR #3143)
- Does **NOT** include receive/Sink support yet (PR #3159 still open)
- Install: `pip install py-cord==2.8.0rc1`

### Alternatives Evaluated (as of 2026-04-02)

| Library | Language | DAVE Send | DAVE Receive | Per-User Audio / Sink API | Viable? |
|---|---|---|---|---|---|
| **Pycord** | Python | Merged | PR open (working in tests) | Yes (built-in Sinks) | **Best option** |
| discord.py | Python | Merged (PR #10300) | Never supported | No | No |
| nextcord | Python | No | No | No | No |
| disnake | Python | Merged (PR #1492) | No (no plans, issue #178 open since 2021) | No | No |
| hikari | Python | No | No | No | No |
| interactions.py | Python | No | No | Partial | No |
| **discord.js** | JS/TS | Merged | **Merged & working** (PR #11449) | Yes (VoiceReceiver) | Full rewrite required |
| **JDA** | Java | Via jdave | Via jdave | Yes (AudioReceiveHandler) | Full rewrite required |

**Conclusion:** Pycord is the only Python library with both DAVE support and a voice receive Sink API. The only libraries where DAVE receive fully works today are discord.js and JDA, both of which would require a complete rewrite. Sticking with Pycord and tracking PR #3159 is the best path forward.

#### Notable details
- **discord.py**: Has `discord-ext-listening` (by Sheppsu) for voice receive, but it was last updated Feb 2024 with no DAVE support — likely broken since March 2026 enforcement.
- **disnake**: Uses its own `dave.py` package (nanobind-based C++ libdave binding) rather than `davey`. Maintainers have explicitly said no one is working on voice receive.
- **discord.js**: PR #11449 (merged March 2026) fixed receive-side DAVE decryption. Currently the most mature DAVE voice receive implementation across any library.

### Community Resources
- **`davey` Python package** (v0.1.5 on PyPI, updated 2026-03-29): Python bindings for Discord's DAVE/MLS encryption, used internally by Pycord
- **vito1317's fork** (`github.com/vito1317/pycord`, branch `fix/dave-decryption`): Community fork with working DAVE receive decryption fixes, cherry-picked into official PR #3159

## Migration Steps (when upstream is ready)

### Step 1: Upgrade pycord to v2.8+
```bash
pip install --upgrade py-cord[voice]
```
Update `requirements.txt` to pin the new version.

### Step 2: Verify voice connection
Test that the bot can `/connect` to a voice channel without 4017 errors. DAVE send is available in v2.8.0rc1: `pip install py-cord==2.8.0rc1`.

### Step 3: Verify voice receive / recording
Test that `/scribe` produces transcriptions. This depends on PR #3159 being merged and released. To test early, install from the `fix/voice-rec-2` branch: `pip install git+https://github.com/Pycord-Development/pycord@fix/voice-rec-2`. Key areas to check:
- `WhisperSink.write()` receives decrypted audio bytes (src/sinks/whisper_sink.py)
- Audio data decodes properly through opus (no `OpusError: corrupted stream`)
- Transcription output matches pre-DAVE quality

### Step 4: Check for breaking changes to Sink API
The voice rewrite (PR #3143) notes that sink-related breaking changes may land alongside or after the receive work. Review:
- `WhisperSink` extends `discord.sinks.core.Sink` — confirm base class API unchanged
- `Filters` import and usage in `__init__`
- `vc.start_recording()` call signature
- `vc.stop_recording()` call signature

### Step 5: Check new dependencies
discord.py's DAVE implementation required a `davey` Python package. Pycord may require similar dependencies (possibly `libdave` bindings). Add any new requirements to `requirements.txt`.

### Step 6: Health check updates
`src/bot/health.py` may need a new check to verify DAVE handshake capability, or existing checks may need updates if the voice connection flow changes.

## Files Most Likely to Need Changes

| File | Why |
|------|-----|
| `requirements.txt` | pycord version bump, possible new DAVE dependencies |
| `src/sinks/whisper_sink.py` | Extends `Sink` — API may change in v2.8 |
| `src/bot/volo_bot.py` | `start_recording()` / `stop_recording()` calls, sink creation |
| `src/bot/health.py` | May need DAVE-aware health checks |
| `main.py` | `vc.start_recording()` usage in commands |

## Current Environment

- pycord: 2.7.1
- Python: 3.10 (venv)
- Whisper model: faster_whisper large-v3
- Transcription: local (faster_whisper) or OpenAI API
- OS: macOS (Darwin)
