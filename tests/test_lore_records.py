"""Session records for the Lore search and the summary generator
(discord-transcript-bot-3bx): recaps and transcripts are offered on demand,
never sent wholesale."""

from src import campaigns, recaps, sessions
from src.llm import campaign, claude
from src.llm.common import AskResult
from tests import test_library_sessions as _base
from tests.test_library_sessions import _earlier_fixture

lib = _base.lib  # the same library fixture, under its usual name

FULL = (
    "TOPICS: Fixie, Militech\n\nSUMMARY:\nThe deal went bad, choom.\n\n"
    "RECAP:\nAt 7:01 PM the deal with Militech went bad."
)


def _night_13(t):
    return [t / "2026-10-13_19-00-00.txt", t / "2026-10-13_21-00-00.txt"]


def _recap_13(t):
    def ask(transcript, question, *, header="", **context):
        return AskResult(answer=FULL, model="m")

    return recaps.generate(_night_13(t), ask=ask)


# ── every session of a campaign, as on-demand notes ───────────────────


def test_campaign_session_notes_list_every_night_of_that_campaign(lib, tmp_path):
    _earlier_fixture(lib, tmp_path)
    _recap_13(tmp_path)
    notes = sessions.campaign_session_notes(tmp_path, "return-to-night-city")
    # all three nights (not only the ones before some recording), the
    # recap ahead of its transcript, and nothing from the other campaign
    assert [n.id for n in notes] == [
        "session-2026-10-06",
        "recap-2026-10-13",
        "session-2026-10-13",
        "session-2026-10-20",
    ]
    by_id = {n.id: n for n in notes}
    assert by_id["session-2026-10-20"].title == "Session, Tuesday, October 20, 2026"
    assert "topics: Fixie, Militech" in by_id["recap-2026-10-13"].index_line()
    assert "choom" not in by_id["recap-2026-10-13"].body  # the plain recap
    assert "we ran" in by_id["session-2026-10-13"].body  # both recordings
    assert "goblins" not in "".join(n.body for n in notes)
    assert sessions.campaign_session_notes(tmp_path, "no-such-campaign") == []
    assert sessions.campaign_session_notes(tmp_path, None) == []
    assert sessions.campaign_session_notes(tmp_path / "missing", "x") == []


def test_only_index_lines_ride_with_a_question(lib, tmp_path):
    """The efficiency contract: what is sent up front is one short line per
    record, whatever the transcripts weigh."""
    _earlier_fixture(lib, tmp_path)
    big = tmp_path / "2026-10-06_19-00-00.txt"
    big.write_text("[19:01:00] Jim (Jill) [371]: " + "blah " * 40_000 + "\n")
    _recap_13(tmp_path)
    notes = sessions.campaign_session_notes(tmp_path, "return-to-night-city")
    index = campaign.index_text(notes)
    assert sum(len(n.body) for n in notes) > 200_000
    assert len(index) < 1_500
    assert "blah blah" not in index


def test_the_model_offers_a_campaigns_session_notes(lib, tmp_path):
    _earlier_fixture(lib, tmp_path)
    model = campaigns.Model(tmp_path, lib)
    ids = [n.id for n in model.campaign("return-to-night-city").session_notes()]
    assert ids == ["session-2026-10-06", "session-2026-10-13", "session-2026-10-20"]
    assert (
        campaigns.Model(None, lib).campaign("return-to-night-city").session_notes()
        == []
    )


# ── Lore reads them through the same tool ─────────────────────────────


def test_lore_prompt_explains_the_session_records():
    text = claude.LORE_SYSTEM_PROMPT
    assert "recap-" in text and "session-" in text
    assert text.index("recap-") < text.index("transcript only when")


# ── the summary generator knows the same context ──────────────────────


def test_the_generator_is_offered_notes_and_earlier_sessions_on_demand(lib, tmp_path):
    now = _earlier_fixture(lib, tmp_path)
    _recap_13(tmp_path)
    seen = {}

    def ask(transcript, question, *, header="", note_dirs=(), extra_notes=()):
        seen.update(
            transcript=transcript,
            header=header,
            dirs=list(note_dirs),
            extra=extra_notes,
        )
        return AskResult(answer=FULL, model="m")

    recaps.generate([now], header="SESSION: X", lib=lib, ask=ask)
    # the campaign's note folders (game, group, campaign) ...
    assert [d.parent.name for d in seen["dirs"]] == [
        "cyberpunk-red",
        "noah-gm",
        "return-to-night-city",
    ]
    # ... and only the sessions BEFORE this one, as index entries
    assert [n.id for n in seen["extra"]] == [
        "session-2026-10-06",
        "recap-2026-10-13",
        "session-2026-10-13",
    ]
    assert recaps.CONTEXT_RULES in seen["header"]
    # the prompt itself carries tonight's transcript and no earlier one
    assert "tonight" in seen["transcript"]
    assert "the deal went bad" not in seen["transcript"] + seen["header"]


def test_the_generator_has_no_context_without_a_library_or_a_tag(lib, tmp_path):
    _earlier_fixture(lib, tmp_path)
    seen = []

    def ask(transcript, question, *, header="", note_dirs=(), extra_notes=()):
        seen.append((header, list(note_dirs), list(extra_notes)))
        return AskResult(answer=FULL, model="m")

    recaps.generate(_night_13(tmp_path), header="SESSION: X", ask=ask)  # no lib
    untagged = tmp_path / "2026-10-01_19-00-00.txt"
    recaps.generate([untagged], lib=lib, ask=ask)
    for header, dirs, extra in seen:
        assert dirs == [] and extra == [] and "CONTEXT:" not in header
    assert recaps.context_for([], lib) == ([], [])


def test_context_rules_keep_the_summary_about_this_session():
    rules = recaps.CONTEXT_RULES.lower()
    for phrase in (
        "only to get a name or its spelling right",
        "do not read an earlier session's full transcript",
        "about this session's events alone",
    ):
        assert phrase in rules, phrase
