#!/usr/bin/env python3
"""Remove roster-name prompt-echo and known Whisper-artifact lines from
already-written transcripts and JSON logs.

The live pipeline now filters these at transcription time
(WhisperSink._accept_segment / _is_prompt_echo / _is_hallucination_phrase),
but transcripts captured before that fix are still polluted. This is a
one-shot, backup-safe cleanup for those files. It only removes the
text-detectable garbage (roster echoes + the YouTube-ghost denylist) — the
per-segment metric filter can't be applied retroactively (the logs don't
carry segment metrics), so genuine-looking short noise is left alone.

The text heuristics intentionally mirror whisper_sink so the same lines are
removed here as would be dropped live.

USAGE
-----
    # report only (default):
    python scripts/scrub_hallucinations.py transcripts/SESSION.txt .logs/transcripts/DAY.log
    # actually rewrite (originals saved to <file>.bak):
    python scripts/scrub_hallucinations.py --apply transcripts/SESSION.txt ...

Player map defaults to $PLAYER_MAP_FILE_PATH or ./player_map.yml.
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

import yaml

# Mirror of whisper_sink._HALLUCINATION_PHRASES (kept in sync by hand; this is
# a maintenance tool, not imported by the bot, so it avoids loading the model).
_HALLUCINATION_PHRASES = {
    "thank you for watching",
    "thanks for watching",
    "thank you for watching this video",
    "subtitles by the amara org community",
    "transcription by the amara org community",
    "please subscribe",
    "like and subscribe",
    "see you in the next video",
}
_ECHO_FILLERS = {"says", "said", "say", "speaking", "speaks", "and", "the", "a"}

# Per-line transcript format: [HH:MM:SS] Player (Character) [user_id]: text
_LINE_RE = re.compile(
    r"^\[(\d{2}:\d{2}:\d{2})\]\s+(.*?)\s+\(([^)]*)\)\s+\[([^\]]+)\]:\s?(.*)$"
)


def _normalize(text):
    return " ".join(re.sub(r"[^\w\s]", " ", text.lower()).split())


def _roster_names(player_map):
    names = set()
    for entry in (player_map or {}).values():
        if isinstance(entry, dict):
            for key in ("player", "character"):
                v = (entry.get(key) or "").strip()
                if v:
                    names.add(_normalize(v))
    names.discard("")
    return sorted(names, key=len, reverse=True)


def is_garbage(text, names):
    """True if `text` is a roster-name echo or a known Whisper artifact."""
    norm = _normalize(text)
    if not norm:
        return False
    if norm in _HALLUCINATION_PHRASES:
        return True
    stripped, matched = norm, False
    for n in names:
        new = re.sub(rf"\b{re.escape(n)}\b", " ", stripped)
        if new != stripped:
            matched = True
            stripped = new
    leftover = [w for w in stripped.split() if w not in _ECHO_FILLERS]
    return matched and not leftover


def scrub_txt(path, names, apply):
    lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    kept, removed = [], []
    for line in lines:
        m = _LINE_RE.match(line)
        text = m.group(5) if m else line
        if m and is_garbage(text, names):
            removed.append(line)
        else:
            kept.append(line)
    _commit(path, kept, removed, apply, trailing_newline=bool(lines))
    return removed


def scrub_log(path, names, apply):
    lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    kept, removed = [], []
    for line in lines:
        s = line.strip()
        if not s:
            kept.append(line)
            continue
        try:
            row = json.loads(s)
        except json.JSONDecodeError:
            kept.append(line)
            continue
        if is_garbage(row.get("data") or "", names):
            removed.append(line)
        else:
            kept.append(line)
    _commit(path, kept, removed, apply, trailing_newline=bool(lines))
    return removed


def _commit(path, kept, removed, apply, trailing_newline):
    if apply and removed:
        backup = str(path) + ".bak"
        os.replace(path, backup)  # preserve the original verbatim
        body = "\n".join(kept) + ("\n" if trailing_newline else "")
        Path(path).write_text(body, encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("files", nargs="+", help=".txt transcripts and/or .log JSON")
    parser.add_argument(
        "--apply", action="store_true", help="rewrite files (originals -> .bak)"
    )
    parser.add_argument("--player-map", help="path to player_map.yml")
    parser.add_argument(
        "--show", type=int, default=8, help="sample N removed lines per file"
    )
    args = parser.parse_args(argv)

    pm_path = args.player_map or os.getenv("PLAYER_MAP_FILE_PATH") or "player_map.yml"
    player_map = yaml.safe_load(Path(pm_path).read_text()) or {}
    names = _roster_names(player_map)

    total = 0
    for f in args.files:
        scrubber = scrub_log if f.endswith(".log") else scrub_txt
        removed = scrubber(f, names, args.apply)
        total += len(removed)
        verb = "Removed" if args.apply else "Would remove"
        print(f"\n{f}: {verb} {len(removed)} line(s)")
        for line in removed[: args.show]:
            print(f"    - {line[:110]}")
        if len(removed) > args.show:
            print(f"    ... and {len(removed) - args.show} more")
    mode = "applied (originals -> .bak)" if args.apply else "dry-run (use --apply)"
    print(f"\nTotal: {total} line(s) — {mode}")
    return total


if __name__ == "__main__":
    sys.exit(0 if main() is not None else 1)
