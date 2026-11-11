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
    """Sessions `/ask` can offer as an earlier session: tagged with a
    campaign, or with a game and a group (``sessions._same_story``)."""
    now_day = sessions.session_day(datetime.now())
    for s in model.sessions(campaign):
        if not (s.campaign_id or (s.game_id and s.group_id)):
            continue
        if day and s.day != day:
            continue
        if s.day == now_day and not today:
            continue
        yield s


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
    found = list(_eligible(model, args.campaign, args.day, args.today))
    if not found:
        print("no finished, tagged sessions")
        return 0
    for s in found:
        have = recaps.load(s.files)
        label = f"{s.key:44} {s.recordings} recording(s)"
        if not args.generate:
            state = f"recap ({have.model}, {have.generated_at})" if have else "no recap"
            print(f"{label}  {state}")
            continue
        if have and not args.force:
            print(f"{label}  kept")
            continue
        try:
            recap = recaps.generate(
                s.files,
                header=sessions.header_text(model.lib, s.meta, s.recordings),
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
