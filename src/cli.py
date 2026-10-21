"""Command-line options for ``main.py``.

One flag so far. ``-v``/``--verbose`` turns on debug logging; the long form
also accepts an explicit value (``--verbose true``, ``--verbose=false``) so
launchers that always pass a value keep working.
"""

from __future__ import annotations

import argparse

_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}


def parse_bool(text: str) -> bool:
    """``"true"``/``"false"`` and friends, case-insensitively."""
    lowered = text.strip().lower()
    if lowered in _TRUE:
        return True
    if lowered in _FALSE:
        return False
    raise argparse.ArgumentTypeError(f"expected true or false, got {text!r}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Discord voice-channel transcription bot.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-v",
        "--verbose",
        nargs="?",
        const=True,
        default=False,
        type=parse_bool,
        metavar="true|false",
        help="log at DEBUG level",
    )
    return parser


def parse_args(argv=None) -> argparse.Namespace:
    """Parse ``argv`` (``sys.argv[1:]`` when None) into a namespace."""
    return build_parser().parse_args(argv)
