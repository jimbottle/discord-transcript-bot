import logging

from src.utils import bot_log


def test_bot_log_keeps_bot_messages_and_skips_transcript_lines(tmp_path):
    handler = bot_log.install(directory=str(tmp_path), fmt="%(name)s %(message)s")
    try:
        logging.getLogger("src.bot.volo_bot").warning("Process CPU at 99%")
        logging.getLogger("transcription").warning('{"data": "hi"}')
        handler.flush()
        text = (tmp_path / "bot.log").read_text()
        assert "Process CPU at 99%" in text
        assert '"data"' not in text
    finally:
        logging.getLogger().removeHandler(handler)
        handler.close()


def test_unwritable_directory_disables_it_without_raising(tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("x")
    assert bot_log.install(directory=str(blocker / "sub")) is None
