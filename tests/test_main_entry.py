"""main.py's module-level helpers. main.py builds the bot under
``if __name__ == "__main__"`` so importing it is side-effect free apart from
load_dotenv()."""

import asyncio
import logging
from types import SimpleNamespace

import main


def test_command_mention_uses_registered_id_or_backticks():
    ctx = SimpleNamespace(
        bot=SimpleNamespace(
            application_commands=[
                SimpleNamespace(name="connect", id=123),
                SimpleNamespace(name="draft", id=None),
            ]
        )
    )
    assert main.command_mention(ctx, "connect") == "</connect:123>"
    assert main.command_mention(ctx, "draft") == "`/draft`"
    assert main.command_mention(ctx, "missing") == "`/missing`"


def test_help_table_matches_the_registered_slash_commands():
    """Every command main.py registers is listed in /help, and nothing else."""
    import re

    with open(main.__file__, encoding="utf-8") as f:
        src = f.read()
    registered = set(re.findall(r'slash_command\(\s*name="([^"]+)"', src))
    listed = {name for name, _ in main.COMMAND_SUMMARY}
    assert listed == registered


def test_configure_logging_routes_transcript_records_to_the_day_file(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(main.bot_log, "install", lambda **kw: None)
    main.configure_logging(verbose=True)
    assert logging.getLogger().level == logging.DEBUG
    assert logging.getLogger("discord").level == logging.WARNING
    logging.getLogger("transcription").info('{"data": "hello"}')
    files = list((tmp_path / ".logs" / "transcripts").glob("*-transcription.log"))
    assert len(files) == 1
    assert files[0].read_text().strip() == '{"data": "hello"}'
    # detach so later tests don't write into tmp_path
    for h in list(logging.getLogger("transcription").handlers):
        logging.getLogger("transcription").removeHandler(h)
        h.close()


def test_run_until_interrupted_shuts_down_cleanly():
    calls = []

    class Bot:
        async def start(self, token):
            calls.append(("start", token))
            raise KeyboardInterrupt

        async def stop_and_cleanup(self):
            calls.append("stop_and_cleanup")

        async def close(self):
            calls.append("close")

    loop = asyncio.new_event_loop()
    main.run_until_interrupted(Bot(), loop, "tok")
    assert calls == [("start", "tok"), "stop_and_cleanup", "close"]
    assert loop.is_closed()
