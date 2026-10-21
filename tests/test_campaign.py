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
    notes = campaign.load_notes([tmp_path])
    assert [n.id for n in notes] == ["places"]
    assert any("broken.md" in r.message for r in caplog.records)


def test_load_notes_with_missing_dir_is_empty(tmp_path, monkeypatch):
    assert campaign.load_notes([tmp_path / "nope"]) == []
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


def test_later_folder_overrides_an_earlier_note_with_the_same_id(tmp_path):
    game, camp = tmp_path / "game", tmp_path / "camp"
    game.mkdir()
    camp.mkdir()
    (game / "places.md").write_text(GOOD, encoding="utf-8")
    (game / "rules.md").write_text(
        GOOD.replace("id: places", "id: rules"), encoding="utf-8"
    )
    (camp / "places.md").write_text(
        GOOD.replace("Where things are.", "Campaign places."), encoding="utf-8"
    )
    notes = {n.id: n for n in campaign.load_notes([game, camp])}
    assert set(notes) == {"places", "rules"}
    assert notes["places"].summary == "Campaign places."


# ── the local library (gitignored; absent on a fresh clone) ──

from src import library  # noqa: E402

_LOCAL = sorted(library.DEFAULT_LIBRARY_DIR.glob("*/*/notes/*.md"))
_needs_local_notes = pytest.mark.skipif(
    not _LOCAL, reason="library notes are local-only and absent here"
)


def test_example_note_parses():
    p = library.PROJECT_ROOT / "library" / "example-note.md"
    campaign.parse_note(p.read_text(encoding="utf-8"), p)


@_needs_local_notes
def test_every_local_note_parses():
    for p in _LOCAL:
        campaign.parse_note(p.read_text(encoding="utf-8"), p)  # raises on a bad note


@_needs_local_notes
def test_local_gm_notes_are_hidden_by_default(monkeypatch):
    monkeypatch.delenv("CAMPAIGN_GM_NOTES", raising=False)
    dirs = sorted({p.parent for p in _LOCAL})
    all_ids = {p.stem for p in _LOCAL}
    shown = {n.id for n in campaign.load_notes(dirs)}
    hidden = {n.id for n in campaign.load_notes(dirs, include_gm=True)} - shown
    assert shown | hidden == all_ids
    for note_id in hidden:
        assert (
            campaign.parse_note(
                next(p for p in _LOCAL if p.stem == note_id).read_text(encoding="utf-8")
            ).audience
            == "gm"
        )


@_needs_local_notes
def test_local_index_is_small():
    """The index rides on every question; keep it cheap (~40 tokens/note)."""
    dirs = sorted({p.parent for p in _LOCAL})
    notes = campaign.load_notes(dirs, include_gm=True)
    # 8 notes measured at ~800 tokens; this guards against a note whose
    # summary or keyword list grows into a second body.
    assert len(campaign.index_text(notes)) < 400 * len(notes)
