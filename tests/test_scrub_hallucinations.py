"""Tests for the one-shot transcript scrubber (scripts/scrub_hallucinations.py).

Covers the text classifier (roster echo / multi-name / artifact / real speech
kept) and a round-trip scrub that backs up the original to .bak.
"""

import importlib.util
import json
from pathlib import Path


def _load():
    path = Path(__file__).resolve().parents[1] / "scripts" / "scrub_hallucinations.py"
    spec = importlib.util.spec_from_file_location("scrub_hallucinations", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


scrub = _load()

_PM = {
    1: {"player": "Sovereign Lord GM", "character": "Noah"},
    2: {"player": "Rahul Patch Sarker", "character": "William"},
}
_NAMES = scrub._roster_names(_PM)


def test_is_garbage_flags_echoes_and_artifacts():
    assert scrub.is_garbage("Sovereign Lord GM.", _NAMES)
    assert scrub.is_garbage("Sovereign Lord GM, Rahul Patch Sarker.", _NAMES)
    assert scrub.is_garbage("Noah says,", _NAMES)
    assert scrub.is_garbage("Subtitles by the Amara.org community", _NAMES)
    assert scrub.is_garbage("Thanks for watching!", _NAMES)


def test_is_garbage_keeps_real_speech():
    assert not scrub.is_garbage("Noah hands William the gun", _NAMES)
    assert not scrub.is_garbage("I cast fireball", _NAMES)
    assert not scrub.is_garbage("thank you for the potion", _NAMES)
    assert not scrub.is_garbage("", _NAMES)


def test_scrub_txt_removes_only_garbage_and_backs_up(tmp_path):
    f = tmp_path / "session.txt"
    f.write_text(
        "\n".join(
            [
                "[22:18:42] Sovereign Lord GM (Noah) [42]: Sovereign Lord GM.",
                "[22:18:50] Reiko Tanaka (Gus) [99]: I loot the body.",
                "[22:20:15] Steve Calderon (Johan) [77]: Thanks for watching!",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    removed = scrub.scrub_txt(str(f), _NAMES, apply=True)
    assert len(removed) == 2
    kept = f.read_text(encoding="utf-8").splitlines()
    assert kept == ["[22:18:50] Reiko Tanaka (Gus) [99]: I loot the body."]
    # original preserved verbatim
    assert (tmp_path / "session.txt.bak").exists()
    assert len(Path(str(f) + ".bak").read_text().splitlines()) == 3


def test_scrub_txt_dry_run_reports_without_writing(tmp_path):
    f = tmp_path / "session.txt"
    original = (
        "[22:18:42] Sovereign Lord GM (Noah) [42]: Sovereign Lord GM.\n"
        "[22:18:50] Reiko Tanaka (Gus) [99]: I loot the body.\n"
    )
    f.write_text(original, encoding="utf-8")
    removed = scrub.scrub_txt(str(f), _NAMES, apply=False)
    assert len(removed) == 1  # still reports what it WOULD remove
    assert f.read_text(encoding="utf-8") == original  # file untouched
    assert not (tmp_path / "session.txt.bak").exists()  # no backup in dry-run


def test_scrub_log_filters_json_rows(tmp_path):
    f = tmp_path / "day.log"
    f.write_text(
        "\n".join(
            [
                json.dumps({"player": "Noah", "data": " Sovereign Lord GM."}),
                json.dumps({"player": "Gus", "data": " I loot the body."}),
                "{ not json",  # malformed lines are preserved untouched
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    removed = scrub.scrub_log(str(f), _NAMES, apply=True)
    assert len(removed) == 1
    kept = [ln for ln in f.read_text().splitlines() if ln.strip()]
    assert any("loot the body" in ln for ln in kept)
    assert any("not json" in ln for ln in kept)  # malformed kept
