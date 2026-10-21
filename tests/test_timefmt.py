"""12-hour clock times for people (src/utils/timefmt.py)."""

import pytest

from src.utils import timefmt


@pytest.mark.parametrize(
    "raw, out",
    [
        ("20:26:09", "8:26 PM"),
        ("00:05", "12:05 AM"),
        ("12:00:00", "12:00 PM"),
        ("09:41:07", "9:41 AM"),
        ("19:38:48.181", "7:38 PM"),
        ("not a time", "not a time"),
        ("25:00", "25:00"),
        (None, ""),
    ],
)
def test_clock(raw, out):
    assert timefmt.clock(raw) == out


def test_when():
    assert timefmt.when("2026-10-20T21:23:02") == "Oct 20, 9:23 PM"
    assert timefmt.when("2026-01-05T00:15:00") == "Jan 5, 12:15 AM"
    assert timefmt.when("garbage") == "garbage"
    assert timefmt.when(None) == ""


def test_for_model_rewrites_line_stamps_and_restart_markers_only():
    t = (
        "[19:40:00] Noah (GM) [1]: we start at 20:00 sharp\n"
        "--- recording restarted at 20:15:26 ---\n"
        "[20:16:03] A (B) [2]: ok\n"
    )
    assert timefmt.for_model(t) == (
        "[7:40 PM] Noah (GM) [1]: we start at 20:00 sharp\n"
        "--- recording restarted at 8:15 PM ---\n"
        "[8:16 PM] A (B) [2]: ok\n"
    )
