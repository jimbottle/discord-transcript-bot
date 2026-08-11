"""Tests for reference-audio capture (discord-transcript-bot-h7j).

Capture exists to make a live session yield A/B reference data, so the two
properties that matter are: the clips + manifest it writes are actually
consumable by scripts/ab_transcribe.py, and a capture failure can never
disturb transcription during a live game.
"""

import io
import json
import wave
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.session_capture import (
    MANIFEST_NAME,
    README_NAME,
    SESSION_INFO_NAME,
    SessionCapture,
)


def _wav_bytes(seconds=0.5):
    """Discord-format (48kHz/16-bit/stereo) silence, as the sink produces."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(48000)
        w.writeframes(b"\x00\x00\x00\x00" * int(48000 * seconds))
    return buf.getvalue()


def _speaker(user=42, player="Gus", character="Thorin"):
    return SimpleNamespace(user=user, player=player, character=character)


def test_writes_clip_and_manifest_line(tmp_path):
    cap = SessionCapture(str(tmp_path / "sess"))
    name = cap.save_clip(_wav_bytes(), 7, _speaker())
    cap.record(name, _speaker(), "I cast fireball", begin="19:30:45")
    cap.close()

    assert name == "0007_Gus.wav", "clip name carries submission order + speaker"
    assert (tmp_path / "sess" / name).exists()

    lines = (tmp_path / "sess" / MANIFEST_NAME).read_text().splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["audio"] == name
    assert entry["reference"] == "I cast fireball"
    assert entry["player"] == "Gus"
    assert "needs_reference" not in entry


def test_manifest_is_consumable_by_the_ab_harness(tmp_path):
    """The whole point: what capture writes must load in ab_transcribe.

    Guards the contract between the two — a schema drift here would only
    surface when someone tries to run the bake-off on real captured data.
    """
    from scripts.ab_transcribe import load_manifest

    cap = SessionCapture(str(tmp_path / "sess"))
    for seq, text in enumerate(["hello there", "roll for initiative"]):
        name = cap.save_clip(_wav_bytes(), seq, _speaker())
        cap.record(name, _speaker(), text)
    cap.close()

    # A corrected manifest is the draft renamed, so load the draft directly.
    manifest = tmp_path / "sess" / MANIFEST_NAME
    clips = load_manifest(SimpleNamespace(manifest=str(manifest)))

    assert len(clips) == 2
    for audio_path, reference in clips:
        assert Path(audio_path).exists(), "manifest paths must resolve to real clips"
        assert reference


def test_empty_transcription_is_flagged_for_review(tmp_path):
    """A dropped/silent segment still gets a clip (it may be real speech the
    hallucination filter ate) but must be marked, so nobody leaves an empty
    reference in place to skew the corpus WER."""
    cap = SessionCapture(str(tmp_path / "sess"))
    name = cap.save_clip(_wav_bytes(), 1, _speaker())
    cap.record(name, _speaker(), "")
    cap.close()

    entry = json.loads((tmp_path / "sess" / MANIFEST_NAME).read_text().splitlines()[0])
    assert entry["reference"] == ""
    assert entry["needs_reference"] is True


def test_readme_is_written_so_the_draft_is_not_mistaken_for_truth(tmp_path):
    cap = SessionCapture(str(tmp_path / "sess"))
    cap.save_clip(_wav_bytes(), 0, _speaker())
    cap.close()

    readme = (tmp_path / "sess" / README_NAME).read_text()
    assert "NOT ground truth" in readme
    assert "ab_transcribe.py" in readme


def test_nothing_written_until_a_clip_arrives(tmp_path):
    """An enabled-but-silent session must not litter an empty directory."""
    cap = SessionCapture(str(tmp_path / "sess"))
    cap.close()
    assert not (tmp_path / "sess").exists()
    assert cap.summary() == "no audio captured"


def test_unsafe_speaker_names_are_sanitized(tmp_path):
    """Player/character names are arbitrary user input and must never escape
    the capture directory or produce an unopenable filename."""
    cap = SessionCapture(str(tmp_path / "sess"))
    name = cap.save_clip(_wav_bytes(), 3, _speaker(player="../../etc/passwd"))
    cap.close()

    assert "/" not in name
    assert (tmp_path / "sess" / name).exists()


def test_missing_player_falls_back_to_user_id(tmp_path):
    cap = SessionCapture(str(tmp_path / "sess"))
    name = cap.save_clip(_wav_bytes(), 2, _speaker(player=None, character=None))
    cap.close()
    assert name == "0002_42.wav"


def test_disk_cap_stops_capture_without_raising(tmp_path):
    """Past the cap, capture stops silently rather than filling the disk
    mid-session; transcription is unaffected either way."""
    cap = SessionCapture(str(tmp_path / "sess"), max_bytes=len(_wav_bytes()) + 10)
    first = cap.save_clip(_wav_bytes(), 0, _speaker())
    second = cap.save_clip(_wav_bytes(), 1, _speaker())
    cap.close()

    assert first is not None
    assert second is None, "second clip exceeds the cap and must be skipped"
    assert cap.clips == 1


def test_write_failure_disables_capture_but_never_raises(tmp_path, monkeypatch):
    """The live-session guarantee: an unwritable capture directory must not
    propagate an exception into the transcription path."""
    cap = SessionCapture(str(tmp_path / "sess"))

    def boom(*a, **k):
        raise OSError("read-only filesystem")

    monkeypatch.setattr("os.makedirs", boom)

    assert cap.save_clip(_wav_bytes(), 0, _speaker()) is None
    cap.record("x.wav", _speaker(), "text")  # must also stay quiet
    cap.close()


def test_record_is_a_noop_without_a_clip(tmp_path):
    """If the clip write was skipped, there's nothing to reference — the
    manifest must not gain an orphan line pointing at a missing file."""
    cap = SessionCapture(str(tmp_path / "sess"))
    cap.record(None, _speaker(), "text")
    cap.close()
    assert not (tmp_path / "sess").exists()


def test_summary_reports_what_was_captured(tmp_path):
    cap = SessionCapture(str(tmp_path / "sess"))
    cap.save_clip(_wav_bytes(seconds=1.0), 0, _speaker())
    summary = cap.summary()
    cap.close()
    assert "1 clips" in summary
    assert "sess" in summary


@pytest.mark.parametrize("clips", [1, 5])
def test_clip_order_matches_manifest_order(tmp_path, clips):
    """Sorted clip filenames and manifest line order must agree, so a human
    correcting the draft can follow along in the directory listing."""
    cap = SessionCapture(str(tmp_path / "sess"))
    for seq in range(clips):
        name = cap.save_clip(_wav_bytes(), seq, _speaker())
        cap.record(name, _speaker(), f"line {seq}")
    cap.close()

    on_disk = sorted(p.name for p in (tmp_path / "sess").glob("*.wav"))
    in_manifest = [
        json.loads(line)["audio"]
        for line in (tmp_path / "sess" / MANIFEST_NAME).read_text().splitlines()
    ]
    assert on_disk == in_manifest


def test_missing_sequence_number_does_not_collide(tmp_path):
    """A caller with no submission sequence must still get unique filenames —
    otherwise two segments from one speaker silently overwrite each other and
    the manifest points at the wrong audio."""
    cap = SessionCapture(str(tmp_path / "sess"))
    first = cap.save_clip(_wav_bytes(), None, _speaker())
    second = cap.save_clip(_wav_bytes(), None, _speaker())
    cap.close()

    assert first != second
    assert len(list((tmp_path / "sess").glob("*.wav"))) == 2


def test_session_info_records_the_biased_prompt(tmp_path):
    """The bake-off must be able to reproduce the conditions that produced
    the draft — above all the roster-biased initial_prompt, which is rebuilt
    from a roster that may change before anyone runs it."""
    info = {"initial_prompt": "...Names: Thorin, Gus.", "model": "large-v3"}
    cap = SessionCapture(str(tmp_path / "sess"), session_info=info)
    cap.save_clip(_wav_bytes(), 0, _speaker())
    cap.close()

    written = json.loads((tmp_path / "sess" / SESSION_INFO_NAME).read_text())
    assert written["initial_prompt"] == "...Names: Thorin, Gus."
    assert written["model"] == "large-v3"


# --- roborev #3510: clips must never be orphaned without a manifest line ---


def test_manifest_line_still_written_after_close(tmp_path):
    """The session-tail case. A transcription submitted just before /stop
    finishes AFTER close() (the executor is never shut down), so its clip is
    already on disk. Refusing the manifest append would orphan exactly the
    final utterances of every captured session."""
    cap = SessionCapture(str(tmp_path / "sess"))
    name = cap.save_clip(_wav_bytes(), 0, _speaker())
    cap.close()

    cap.record(name, _speaker(), "the last thing anyone said")

    entries = [
        json.loads(line)
        for line in (tmp_path / "sess" / MANIFEST_NAME).read_text().splitlines()
    ]
    assert [e["audio"] for e in entries] == [name]
    assert entries[0]["reference"] == "the last thing anyone said"


def test_manifest_line_still_written_after_disable(tmp_path):
    """The disk-cap case. When capture trips its cap, clips already saved and
    still in flight must keep their manifest lines — they are valid reference
    data and the human has no other way to learn they exist."""
    cap = SessionCapture(str(tmp_path / "sess"), max_bytes=len(_wav_bytes()) + 10)
    name = cap.save_clip(_wav_bytes(), 0, _speaker())
    assert cap.save_clip(_wav_bytes(), 1, _speaker()) is None  # trips the cap

    cap.record(name, _speaker(), "still valid reference data")
    cap.close()

    entries = [
        json.loads(line)
        for line in (tmp_path / "sess" / MANIFEST_NAME).read_text().splitlines()
    ]
    assert [e["audio"] for e in entries] == [name]


def test_every_saved_clip_has_a_manifest_line(tmp_path):
    """The invariant behind both cases above, stated directly: no .wav on
    disk without a line describing it."""
    cap = SessionCapture(str(tmp_path / "sess"))
    names = [cap.save_clip(_wav_bytes(), i, _speaker()) for i in range(3)]
    cap.close()  # close BEFORE the commits land, as a real teardown does
    for i, name in enumerate(names):
        cap.record(name, _speaker(), f"line {i}")

    on_disk = sorted(p.name for p in (tmp_path / "sess").glob("*.wav"))
    in_manifest = sorted(
        json.loads(line)["audio"]
        for line in (tmp_path / "sess" / MANIFEST_NAME).read_text().splitlines()
    )
    assert on_disk == in_manifest


def test_record_before_any_clip_creates_nothing(tmp_path):
    """Dropping the closed/disabled guard must not let a stray record() call
    conjure a capture directory for a session that never captured."""
    cap = SessionCapture(str(tmp_path / "sess"))
    cap.record("phantom.wav", _speaker(), "text")
    cap.close()
    assert not (tmp_path / "sess").exists()


# --- roborev #3510: participants must be told audio is kept ---------------


def test_scribe_notice_only_when_capturing(tmp_path):
    from types import SimpleNamespace as NS

    from src.session_capture import PARTICIPANT_NOTICE, scribe_notice

    assert scribe_notice(NS(capture=None)) == ""
    assert scribe_notice(None) == ""
    notice = scribe_notice(NS(capture=SessionCapture(str(tmp_path / "s"))))
    assert notice == PARTICIPANT_NOTICE
    # The whole point is disclosing that AUDIO is kept, not just transcribed.
    assert "Audio recording is ON" in notice
    assert "saved to disk" in notice
