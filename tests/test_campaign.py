"""Campaign notes (src/llm/campaign.py): front-matter parsing, the index,
the audience filter, and a check that every shipped note is well-formed."""

import pytest

from src.llm import campaign

GOOD = """---
id: places
title: Places
summary: Where things are.
keywords: [map, Little Europe]
---
# Places
Little Europe is a district.
"""


def test_parse_note_reads_front_matter_and_body():
    n = campaign.parse_note(GOOD)
    assert (n.id, n.title, n.summary) == ("places", "Places", "Where things are.")
    assert n.keywords == ["map", "Little Europe"]
    assert n.audience == "table"
    assert n.body.startswith("# Places")
    assert n.index_line() == "- places — Places: Where things are. [map, Little Europe]"


@pytest.mark.parametrize(
    "text, fragment",
    [
        ("no front matter", "missing front matter"),
        ("---\nid: x\n", "unterminated"),
        ("---\ntitle: T\nsummary: S\n---\nbody", "'id'"),
        ("---\nid: x\ntitle: T\nsummary: S\naudience: dm\n---\nbody", "audience"),
        ("---\n- a list\n---\nbody", "not a mapping"),
    ],
)
def test_parse_note_rejects_bad_headers(text, fragment):
    with pytest.raises(campaign.NoteError) as e:
        campaign.parse_note(text)
    assert fragment in str(e.value)


def test_note_id_must_match_file_name(tmp_path):
    p = tmp_path / "other.md"
    p.write_text(GOOD, encoding="utf-8")
    with pytest.raises(campaign.NoteError):
        campaign.parse_note(GOOD, p)


def test_load_notes_skips_a_broken_note_and_keeps_the_rest(
    tmp_path, monkeypatch, caplog
):
    (tmp_path / "places.md").write_text(GOOD, encoding="utf-8")
    (tmp_path / "broken.md").write_text("not a note", encoding="utf-8")
    monkeypatch.setenv("CAMPAIGN_NOTES_DIR", str(tmp_path))
    notes = campaign.load_notes()
    assert [n.id for n in notes] == ["places"]
    assert any("broken.md" in r.message for r in caplog.records)


def test_load_notes_with_missing_dir_is_empty(tmp_path, monkeypatch):
    monkeypatch.setenv("CAMPAIGN_NOTES_DIR", str(tmp_path / "nope"))
    assert campaign.load_notes() == []
    assert campaign.index_text([]) == ""


def test_read_notes_returns_found_bodies_and_names_missing_ids():
    n = campaign.parse_note(GOOD)
    text, is_error = campaign.read_notes(["places", "ghost"], [n])
    assert not is_error  # something was found
    assert "## Places (id: places)" in text
    assert "Little Europe is a district." in text
    assert "Unknown note id(s): ghost" in text
    text, is_error = campaign.read_notes(["ghost"], [n])
    assert is_error


def test_tool_schema_is_strict():
    assert campaign.TOOL["strict"] is True
    assert campaign.TOOL["input_schema"]["additionalProperties"] is False
    assert campaign.TOOL["input_schema"]["required"] == ["ids"]


# ── the local notes in campaign/notes (gitignored; absent on a fresh clone) ──

_LOCAL = sorted(campaign.DEFAULT_NOTES_DIR.glob("*.md"))
_needs_local_notes = pytest.mark.skipif(
    not _LOCAL, reason="campaign/notes is local-only and empty here"
)


def test_example_note_parses():
    p = campaign.PROJECT_ROOT / "campaign" / "example-note.md"
    campaign.parse_note(p.read_text(encoding="utf-8"), p)


@_needs_local_notes
def test_every_shipped_note_parses(monkeypatch):
    monkeypatch.delenv("CAMPAIGN_NOTES_DIR", raising=False)
    monkeypatch.setenv("CAMPAIGN_GM_NOTES", "1")
    paths = _LOCAL
    for p in paths:
        campaign.parse_note(p.read_text(encoding="utf-8"), p)  # raises on a bad note
    notes = campaign.load_notes()
    assert len(notes) == len(paths)


@pytest.mark.skipif(
    not (campaign.DEFAULT_NOTES_DIR / "gm-secrets.md").exists(),
    reason="no local gm-secrets note",
)
def test_shipped_gm_notes_are_hidden_by_default(monkeypatch):
    monkeypatch.delenv("CAMPAIGN_NOTES_DIR", raising=False)
    monkeypatch.delenv("CAMPAIGN_GM_NOTES", raising=False)
    ids = [n.id for n in campaign.load_notes()]
    assert "gm-secrets" not in ids
    assert "npcs" in ids


@_needs_local_notes
def test_shipped_index_is_small(monkeypatch):
    """The index rides on every question; keep it cheap (~40 tokens/note)."""
    monkeypatch.delenv("CAMPAIGN_NOTES_DIR", raising=False)
    monkeypatch.setenv("CAMPAIGN_GM_NOTES", "1")
    index = campaign.index_text(campaign.load_notes())
    # 8 notes measured at ~800 tokens; this guards against a note whose
    # summary or keyword list grows into a second body.
    assert len(index) < 400 * len(campaign.load_notes()), len(index)
