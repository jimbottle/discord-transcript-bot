"""Unit tests for src.utils.voice.disconnect_targets.

Regression guard for roborev #786 (LOW): /disconnect's voice-client
selection/dedup is pure logic and must be covered without live Discord.
The stale-helper.vc case is the live bug ("said disconnected but stayed
in the call").
"""
from types import SimpleNamespace

from src.utils.voice import disconnect_targets


def _vc(guild_id=None):
    g = SimpleNamespace(id=guild_id) if guild_id is not None else None
    return SimpleNamespace(guild=g)


def test_live_and_helper_same_object_deduped():
    vc = _vc(1)
    assert disconnect_targets(vc, vc, [vc], 1) == [vc]


def test_stale_helper_and_distinct_live_both_included_live_first():
    live = _vc(1)
    stale_helper = _vc(1)
    targets = disconnect_targets(live, stale_helper, [], 1)
    assert targets == [live, stale_helper]


def test_extra_guild_clients_added_and_deduped():
    live = _vc(7)
    other = _vc(7)
    targets = disconnect_targets(live, None, [live, other], 7)
    assert targets == [live, other]  # live not duplicated, other appended


def test_clients_for_other_guilds_excluded():
    live = _vc(7)
    foreign = _vc(99)
    no_guild = _vc(None)
    targets = disconnect_targets(live, None, [foreign, no_guild], 7)
    assert targets == [live]


def test_none_live_and_helper_uses_voice_clients():
    only = _vc(3)
    assert disconnect_targets(None, None, [only], 3) == [only]


def test_all_empty_returns_empty():
    assert disconnect_targets(None, None, [], 5) == []
    assert disconnect_targets(None, None, None, 5) == []
