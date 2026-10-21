# Campaign context for `/ask`

Background the transcript alone cannot supply: who Lilith Kane is, what happened in Seattle,
which gang the crew wiped out. `/ask` (Discord and the dashboard) reads this folder so a
question like "who is Nakamura again?" can be answered.

**Only this README and `example-note.md` are in git.** The GitHub repo is public and
the notes are the GM's private material, spoilers included, so `notes/` and `raw/` are
gitignored and live only on the bot host. Back them up with the rest of the GM's files.

## Layout

- `notes/*.md` — **model-ready notes**, one topic each, hand-distilled from the GM's files.
  Every note starts with YAML front matter:

  ```yaml
  ---
  id: npcs                    # what the model asks for
  title: Named NPCs
  summary: One line; this is all the model sees until it asks for the note.
  keywords: [NPC, Lilith Kane, ...]   # shown in the index to help the model decide
  audience: table | gm        # gm notes are hidden unless CAMPAIGN_GM_NOTES=1
  ---
  ```
- `raw/` — the GM's own source documents (small .docx files) so a note can be re-derived.
  Published rulebooks, DLC PDFs, audio and token packs stay **out of the repo**: they are
  copyrighted R. Talsorian Games products, over a gigabyte, and the model already knows the
  Cyberpunk RED rules. The full set lives in `~/Downloads/Cyberpunk Red` on the operator's Mac.

## How retrieval works (and why it is cheap)

The model never sees the notes by default. Each `/ask` request carries only the **index**:
one line per note (`id — title: summary [keywords]`, roughly 40 tokens each) in a cached
system block, plus a single tool, `read_campaign_notes(ids)`. The model calls the tool only
when the question needs background the transcript lacks, and asks for just the notes whose
index line matches. A question the transcript answers costs nothing extra. See
`src/llm/campaign.py` (loading, index, audience filter) and `src/llm/claude.py` (the tool loop).

## Adding or updating a note

1. Drop the GM's source file in `raw/` if there is one.
2. Write or edit `notes/<id>.md`: front matter as above, then the facts — short, concrete,
   names spelled the way the transcript spells them. Prefer one note per topic over one big
   note; the model pays for every note it pulls.
3. Keep spoilers in `audience: gm` notes. Players use `/ask` too.
4. Run `make test` — `tests/test_campaign.py` parses every note and fails on a bad header.

Nothing needs restarting for a note edit to take effect; notes are read on each question.
