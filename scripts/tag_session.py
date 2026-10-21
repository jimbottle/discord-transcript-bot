#!/usr/bin/env python3
"""Tag or retag existing transcripts with group / game / campaign / session.

New recordings are tagged automatically at /scribe from library/library.yml.
Use this for recordings made before tagging existed, or ones tagged wrong.

    ./venv/bin/python scripts/tag_session.py --list
    ./venv/bin/python scripts/tag_session.py transcripts/2026-10-20_*.txt \\
        --campaign return-to-night-city
    ./venv/bin/python scripts/tag_session.py transcripts/X.txt --guild 1467567081514864673

The session day comes from the file's timestamp (with the 6am cutoff).
An existing tag's Discord fields are kept unless overridden.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import library, sessions  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("files", nargs="*", help="transcripts/<ts>.txt files")
    ap.add_argument("--campaign", help="campaign id from library.yml")
    ap.add_argument("--group", help="group id (defaults to the campaign's group)")
    ap.add_argument("--guild", type=int, help="resolve from this Discord guild id")
    ap.add_argument("--list", action="store_true", help="show every transcript's tag")
    args = ap.parse_args(argv)
    lib = library.load()

    if args.list:
        root = Path("transcripts")
        for txt in sorted(root.glob("*.txt")):
            m = sessions.read_meta(txt)
            print(f"{txt.name:28} {m.key if m else '(untagged)'}")
        return 0
    if not args.files:
        ap.error("give transcript files, or --list")
    if args.campaign and args.campaign not in lib.campaigns:
        ap.error(
            f"unknown campaign '{args.campaign}' (have: {', '.join(lib.campaigns)})"
        )

    for name in args.files:
        txt = Path(name)
        if not txt.is_file():
            print(f"skip {txt}: not a file", file=sys.stderr)
            continue
        started = sessions.started_at_from_name(txt)
        if started is None:
            print(
                f"skip {txt}: name is not a <YYYY-MM-DD_HH-MM-SS>.txt timestamp",
                file=sys.stderr,
            )
            continue
        old = sessions.read_meta(txt)
        meta = sessions.build_meta(
            lib,
            guild_id=(
                args.guild
                if args.guild is not None
                else (old.guild_id if old else None)
            ),
            guild=old.guild if old else None,
            channel_id=old.channel_id if old else None,
            channel=old.channel if old else None,
            started_at=started,
            group=args.group,
            campaign=args.campaign,
            source="manual",
        )
        sessions.write_meta(txt, meta)
        print(f"{txt.name:28} {meta.key}  (group={meta.group}, game={meta.game})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
