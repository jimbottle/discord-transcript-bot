"""The command-line parser (src/cli.py) and the per-guild voice record
(src/bot/guild_voice.py)."""

import argparse

import pytest

from src import cli
from src.bot.guild_voice import GuildVoice


@pytest.mark.parametrize(
    "argv, expected",
    [
        ([], False),
        (["-v"], True),
        (["--verbose"], True),
        (["--verbose", "true"], True),
        (["--verbose=false"], False),
        (["--verbose", "No"], False),
    ],
)
def test_verbose_flag_forms(argv, expected):
    assert cli.parse_args(argv).verbose is expected


def test_verbose_rejects_garbage():
    with pytest.raises(SystemExit):
        cli.parse_args(["--verbose", "sometimes"])


def test_parse_bool_values():
    assert cli.parse_bool(" TRUE ") is True
    assert cli.parse_bool("off") is False
    with pytest.raises(argparse.ArgumentTypeError):
        cli.parse_bool("maybe")


def test_guild_voice_tracks_client_and_clears():
    bot = object()
    vc = object()
    voice = GuildVoice(bot, 42, vc)
    assert voice.connected and voice.vc is vc and voice.guild_id == 42
    voice.set_vc(None)
    assert not voice.connected
    voice.set_vc(vc)
    voice.clear()
    assert voice.vc is None and voice.guild_id is None and voice.bot is bot
