#!/usr/bin/env python3
"""Write the cached recap of each finished session (discord-transcript-bot-fkj).

`/ask` reads an earlier session's recap before its whole transcript, when
one exists (src/recaps.py). This makes them: one model call per session, on
the configured /ask provider, written to transcripts/<stamp>.recap.json.

    ./venv/bin/python scripts/recap_sessions.py                 # show what exists
    ./venv/bin/python scripts/recap_sessions.py --generate      # write the missing ones
    ./venv/bin/python scripts/recap_sessions.py --generate --campaign return-to-night-city
    ./venv/bin/python scripts/recap_sessions.py --generate --force --day 2026-10-20

Without --generate nothing is sent to a model. Tonight's session is skipped
unless --today is given (it may still be recording). `make recaps` runs
--generate under `op run`, so the Anthropic key is available.
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import campaigns, library, recaps, sessions  # noqa: E402


def _eligible(model, campaign=None, day=None, today=False):
    """``(session, files)`` for every night `/ask` can offer as an earlier
    session: tagged with a campaign, or with a game and a group
    (``sessions._same_story``). ``files`` is the night as /ask groups it
    (``sessions.story_night_files``), which is what the recap is checked
    against; it can hold more recordings than the session's own key does
    when a night is tagged unevenly (roborev #7146). Sessions that name a
    campaign go first, so when two of them share a night the recap is
    written once, from the campaign's point of view."""
    now_day = sessions.session_day(datetime.now())
    taken = set()
    found = []
    for s in sorted(model.sessions(campaign), key=lambda s: not s.campaign_id):
        if not (s.campaign_id or (s.game_id and s.group_id)):
            continue
        if day and s.day != day:
            continue
        if s.day == now_day and not today:
            continue
        files = sessions.story_night_files(s.files[0])
        where = recaps.recap_path(files)
        if where in taken:
            continue
        taken.add(where)
        found.append((s, files))
    return sorted(found, key=lambda pair: (pair[0].day or "", pair[0].key or ""))


def main(argv=None, ask=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--generate", action="store_true", help="write missing recaps")
    ap.add_argument("--force", action="store_true", help="rewrite existing recaps too")
    ap.add_argument("--campaign", help="only this campaign id")
    ap.add_argument("--day", help="only this session day (YYYY-MM-DD)")
    ap.add_argument("--today", action="store_true", help="include tonight's session")
    ap.add_argument(
        "--dir",
        default=str(Path(library.PROJECT_ROOT) / "transcripts"),
        help="transcripts folder",
    )
    args = ap.parse_args(argv)

    model = campaigns.Model(args.dir)
    if args.campaign and args.campaign not in model.lib.campaigns:
        ap.error(f"unknown campaign '{args.campaign}'")
    if args.generate and ask is None:
        from dotenv import load_dotenv

        load_dotenv()

    failed = 0
    found = _eligible(model, args.campaign, args.day, args.today)
    if not found:
        print("no finished, tagged sessions")
        return 0
    for s, files in found:
        have = recaps.load(files)
        label = f"{s.key:44} {len(files)} recording(s)"
        if not args.generate:
            state = f"recap ({have.model}, {have.generated_at})" if have else "no recap"
            print(f"{label}  {state}")
            continue
        if have and not args.force:
            print(f"{label}  kept")
            continue
        try:
            recap = recaps.generate(
                files,
                header=sessions.header_text(model.lib, s.meta, len(files)),
                ask=ask,
            )
        except Exception as e:  # AskFailed, AskBlocked, RecapError, OSError
            failed += 1
            reason = getattr(e, "user_message", None) or str(e) or type(e).__name__
            print(f"{label}  FAILED: {reason}", file=sys.stderr)
            continue
        print(f"{label}  written ({len(recap.text)} chars, {recap.model})")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
