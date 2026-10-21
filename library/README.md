# Library: games, groups, campaigns, sessions

Everything `/ask` knows beyond the transcript, organised the way a table thinks about it:

- a **game** (Cyberpunk RED, D&D 5e) is a rule system and setting;
- a **group** is a table of people with a GM, playing in a Discord server or voice channel;
- a **campaign** belongs to one group and one game;
- a **session** is one game night of a campaign, however many recordings it took.

**Only this README, `library.example.yml` and `example-note.md` are in git.** The GitHub
repo is public and the notes are the GM's private material, spoilers included, so the rest
of `library/` is gitignored and lives on the bot host. Back it up with the GM's files.

## Layout

```
library/
  library.yml                          # games, groups, campaigns (copy library.example.yml)
  games/<game>/notes/*.md              # rules or setting shared by every campaign of that game
  groups/<group>/notes/*.md            # the table: players and characters, house rules, GM tables
  campaigns/<campaign>/notes/*.md      # the story: NPCs, places, factions, arcs, secrets
  campaigns/<campaign>/raw/            # the GM's source documents, to re-derive notes from
```

Each game can carry a `voice:` line, a light flavour for `/ask` answers. Cyberpunk RED
and D&D 5e have built-in defaults: a word or two of Night City slang, or a phrase of
tavern-chronicler colour, with facts kept plain. Set `voice: ""` to turn it off for a
game, or write your own.

`library.yml` binds each group to Discord servers (`discord_guilds`) or, more specifically,
voice channels (`discord_channels`, which win, so two groups can share a server), and names
the group's `active_campaign`. Switching campaigns is one line.

## Sessions

When `/scribe` starts, the bot writes `transcripts/<ts>.meta.json` next to the transcript:
group, game, campaign, Discord server and channel, start time, and the **session day**.
That is the local date with a 6am cutoff (`SESSION_DAY_CUTOFF_HOUR`), so a restart after
midnight still belongs to the night it started. Recordings with the same campaign and
session day are one session: restarting the bot splits the transcript into files but not
the session. `/ask`, in Discord and on the dashboard, reads every recording of the session,
oldest first with a `--- recording restarted at HH:MM:SS ---` line between them, and is told
the game, campaign, group and date.

- `/session` in Discord shows what the current recording is tagged with.
- The dashboard's Transcripts page lists session, campaign, game and group per recording.
- `scripts/tag_session.py --list` shows every tag. `scripts/tag_session.py <files> --campaign <id>`
  tags recordings made before tagging existed, or fixes a wrong tag.

## Notes and retrieval (why it is cheap)

The model never sees note bodies by default. Each question carries only an **index**: one
line per note in the session's game, group and campaign folders (`id — title: summary
[keywords]`, about 100 tokens each) in a cached system block, plus one tool,
`read_campaign_notes(ids)`. The model calls it only when the question needs background the
transcript lacks and asks only for the notes whose index line matches. A campaign note with
the same id as a game or group note replaces it. See `src/library.py`, `src/sessions.py`,
`src/llm/campaign.py` and `src/llm/claude.py`.

Every note starts with YAML front matter (`example-note.md` is a template):

```yaml
---
id: npcs                    # must match the file name; what the model asks for
title: Named NPCs
summary: One line; this is all the model sees until it asks for the note.
keywords: [NPC, Lilith Kane]
audience: table | gm        # gm notes are hidden unless CAMPAIGN_GM_NOTES=1
---
```

Keep notes short and concrete, one topic each, names spelled the way the transcript spells
them. Put spoilers in `audience: gm` notes: players use `/ask` too. Published rulebooks stay
out (copyrighted, large, and the model knows the rules). `make test` parses every local note.
Edits to notes and `library.yml` apply on the next question; no restart.

## Lore page

The dashboard's **Lore** page answers questions about a campaign from these notes alone, with no
transcript, for prep between sessions. It is dashboard-only; there is no Discord command. GM-only
notes are used only when you tick "Include GM-only notes" for that question.
