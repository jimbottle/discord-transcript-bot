"""/generate_pdf builds from the session's transcript files (fwv)."""

import asyncio
import os
from types import SimpleNamespace

from src.bot.volo_bot import VoloBot
from src.transcript_lines import parse_transcript_text
from src.utils import pdf_generator as pg


def test_pdf_builds_from_parsed_lines_and_escapes_markup(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    # The parchment background lives in the repo's assets/; draw nothing here.
    monkeypatch.setattr(pg, "add_parchment_background", lambda c, doc: None)
    entries = parse_transcript_text(
        "[19:40:00] Jim (Rook) [1]: I roll <3 & pray\n"
        "--- recording restarted at 20:15:26 ---\n"
        "[20:16:00] Noah (Noah) [2]: <b>bold</b>?\n"
    )
    path = asyncio.run(pg.pdf_generator(entries))
    assert os.path.getsize(path) > 0
    with open(path, "rb") as f:
        assert f.read(5) == b"%PDF-"


def test_speaker_label():
    assert pg.speaker_label({"player": "Jim", "character": "Rook"}) == "Jim (Rook)"
    assert pg.speaker_label({"player": "Noah", "character": "Noah"}) == "Noah"
    assert pg.speaker_label({"player": None, "character": None}) == ""


def test_export_uses_this_process_recording_then_falls_back_to_disk(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    live = tmp_path / "live.txt"
    live.write_text("x")
    fake = SimpleNamespace(guild_last_session_file={1: str(live)})
    assert VoloBot.session_file_for_export(fake, 1) == str(live)

    # After a restart nothing is remembered: the newest file tagged with the
    # guild on disk is used.
    found = tmp_path / "transcripts" / "2026-10-20_20-53-37.txt"
    monkeypatch.setattr(
        "src.sessions.latest_for_guild", lambda gid, d: found if gid == 2 else None
    )
    fake = SimpleNamespace(guild_last_session_file={})
    assert VoloBot.session_file_for_export(fake, 2) == str(found)
    assert VoloBot.session_file_for_export(fake, 3) is None

    # A remembered file that was deleted from the dashboard falls through too.
    fake = SimpleNamespace(guild_last_session_file={2: str(tmp_path / "gone.txt")})
    assert VoloBot.session_file_for_export(fake, 2) == str(found)


def test_teardown_no_longer_drains_anything():
    """/stop and /disconnect used to drain the transcript queue and throw the
    result away, which left /generate_pdf empty."""
    with open(os.path.join(os.path.dirname(__file__), "..", "main.py")) as f:
        src = f.read()
    assert "get_transcription" not in src


def test_a_recording_nobody_has_spoken_in_keeps_its_session(tmp_path, monkeypatch):
    """roborev #6511/#6521: the .txt appears on the first line. Until then
    the fresh recording is still the one to export: an earlier recording of
    the same session (a restart mid-game) supplies the text, and a brand-new
    session reads as empty, never as the previous session."""
    from src import sessions

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "src.sessions.latest_for_guild", lambda gid, d: tmp_path / "older.txt"
    )
    meta = sessions.SessionMeta(session_day="2026-10-27", campaign="c", guild_id=1)

    earlier = tmp_path / "2026-10-27_19-00-00.txt"
    earlier.write_text("[19:01:00] A (B) [1]: before the restart\n")
    sessions.write_meta(earlier, meta)
    fresh = tmp_path / "2026-10-27_20-00-00.txt"
    sessions.write_meta(fresh, meta)

    fake = SimpleNamespace(guild_last_session_file={1: str(fresh)})
    path = VoloBot.session_file_for_export(fake, 1)
    assert path == str(fresh)
    assert "before the restart" in sessions.session_transcript(path)[0]

    # A new session nobody has spoken in: empty, not the older session.
    lonely = tmp_path / "2026-11-03_19-00-00.txt"
    sessions.write_meta(
        lonely, sessions.SessionMeta(session_day="2026-11-03", campaign="c")
    )
    fake = SimpleNamespace(guild_last_session_file={1: str(lonely)})
    path = VoloBot.session_file_for_export(fake, 1)
    assert path == str(lonely)
    assert sessions.session_transcript(path)[0] == ""
