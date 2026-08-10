"""Unit tests for src.player_map_store — the shared atomic roster reader/
writer used by both VoloBot.upsert_player_entry and the web roster editor.

Mirrors the guarantees the bot's /add_player tests assert (merge-preserve,
str-id coercion, non-dict refusal, atomic no-orphan) plus delete, so either
caller relies on one tested implementation.
"""

import pytest
import yaml

from src import player_map_store as store


def test_load_missing_file_returns_empty(tmp_path):
    assert store.load(str(tmp_path / "nope.yml")) == {}


def test_load_falsy_path_returns_empty():
    assert store.load(None) == {}
    assert store.load("") == {}


def test_load_empty_file_returns_empty(tmp_path):
    pm = tmp_path / "player_map.yml"
    pm.write_text("", encoding="utf-8")
    assert store.load(str(pm)) == {}


def test_load_returns_mapping(tmp_path):
    pm = tmp_path / "player_map.yml"
    pm.write_text(
        yaml.dump({7: {"player": "Ed", "character": "Volo"}}), encoding="utf-8"
    )
    assert store.load(str(pm)) == {7: {"player": "Ed", "character": "Volo"}}


def test_load_non_mapping_raises(tmp_path):
    pm = tmp_path / "player_map.yml"
    pm.write_text(yaml.dump(["not", "a", "mapping"]), encoding="utf-8")
    with pytest.raises(ValueError):
        store.load(str(pm))


def test_upsert_merge_preserves_others(tmp_path):
    pm = tmp_path / "player_map.yml"
    pm.write_text(
        yaml.dump({999: {"player": "Existing", "character": "Keep"}}), encoding="utf-8"
    )
    store.upsert(str(pm), 7, "Cody", "Jim")
    on_disk = yaml.safe_load(pm.read_text())
    assert on_disk[7] == {"player": "Cody", "character": "Jim"}
    assert on_disk[999] == {"player": "Existing", "character": "Keep"}


def test_upsert_coerces_str_id_and_updates_existing(tmp_path):
    pm = tmp_path / "player_map.yml"
    pm.write_text(
        yaml.dump({7: {"player": "old", "character": "old"}}), encoding="utf-8"
    )
    store.upsert(str(pm), "7", "New", "NewChar")  # str id
    assert yaml.safe_load(pm.read_text())[7] == {
        "player": "New",
        "character": "NewChar",
    }


def test_upsert_refuses_non_mapping_and_leaves_file(tmp_path):
    pm = tmp_path / "player_map.yml"
    pm.write_text(yaml.dump(["not", "a", "mapping"]), encoding="utf-8")
    with pytest.raises(ValueError):
        store.upsert(str(pm), 7, "A", "B")
    assert yaml.safe_load(pm.read_text()) == ["not", "a", "mapping"]


def test_upsert_atomic_no_orphan_on_write_failure(tmp_path, monkeypatch):
    pm = tmp_path / "player_map.yml"
    pm.write_text(
        yaml.dump({1: {"player": "Keep", "character": "Keep"}}), encoding="utf-8"
    )
    monkeypatch.setattr(
        store.yaml, "dump", lambda *a, **k: (_ for _ in ()).throw(OSError("disk full"))
    )
    with pytest.raises(OSError):
        store.upsert(str(pm), 7, "X", "Y")
    assert not (tmp_path / "player_map.yml.tmp").exists()  # no orphan
    assert yaml.safe_load(pm.read_text()) == {
        1: {"player": "Keep", "character": "Keep"}
    }


def test_delete_removes_and_preserves_others(tmp_path):
    pm = tmp_path / "player_map.yml"
    pm.write_text(
        yaml.dump(
            {
                7: {"player": "Go", "character": "Go"},
                9: {"player": "Stay", "character": "Stay"},
            }
        ),
        encoding="utf-8",
    )
    assert store.delete(str(pm), "7") is True  # str id coerced
    on_disk = yaml.safe_load(pm.read_text())
    assert 7 not in on_disk
    assert on_disk[9] == {"player": "Stay", "character": "Stay"}


def test_delete_absent_returns_false_and_no_write(tmp_path):
    pm = tmp_path / "player_map.yml"
    original = {9: {"player": "Stay", "character": "Stay"}}
    pm.write_text(yaml.dump(original), encoding="utf-8")
    mtime_before = pm.stat().st_mtime_ns
    assert store.delete(str(pm), 7) is False
    assert yaml.safe_load(pm.read_text()) == original
    assert pm.stat().st_mtime_ns == mtime_before  # never rewritten


def test_delete_non_mapping_raises(tmp_path):
    pm = tmp_path / "player_map.yml"
    pm.write_text(yaml.dump(["not", "a", "mapping"]), encoding="utf-8")
    with pytest.raises(ValueError):
        store.delete(str(pm), 7)
