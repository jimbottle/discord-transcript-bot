"""Tests for the pure WER scoring helpers (src/wer.py).

No model import — these cover the math/normalization the A/B harness relies
on. discord-transcript-bot-61z.
"""

from src.wer import aggregate_wer, normalize_text, word_error_rate


def test_normalize_lowercases_and_strips_punctuation():
    assert normalize_text("Hello, WORLD!") == ["hello", "world"]


def test_normalize_keeps_intra_word_apostrophe():
    assert normalize_text("don't") == ["don't"]


def test_normalize_empty():
    assert normalize_text("") == []
    assert normalize_text(None) == []


def test_wer_exact_match_is_zero():
    s = word_error_rate("the quick brown fox", "The quick brown fox.")
    assert s["wer"] == 0.0
    assert s["edits"] == 0
    assert s["ref_words"] == 4


def test_wer_single_substitution():
    s = word_error_rate("the quick brown fox", "the quick brown dog")
    assert s["edits"] == 1
    assert s["wer"] == 0.25


def test_wer_insertion_and_deletion():
    # one insertion
    assert word_error_rate("a b c", "a b c d")["edits"] == 1
    # one deletion
    assert word_error_rate("a b c", "a c")["edits"] == 1


def test_wer_empty_reference():
    # nothing expected, nothing said -> perfect
    assert word_error_rate("", "")["wer"] == 0.0
    # nothing expected but words produced -> all insertions, capped at 1.0
    assert word_error_rate("", "hello there")["wer"] == 1.0


def test_aggregate_is_corpus_level_not_mean():
    # Clip A: 1 edit / 100 ref words (0.01). Clip B: 1 edit / 1 ref word (1.0).
    # Mean would be ~0.505; corpus WER is 2/101 ≈ 0.0198.
    a = {"wer": 0.01, "edits": 1, "ref_words": 100, "hyp_words": 100}
    b = {"wer": 1.0, "edits": 1, "ref_words": 1, "hyp_words": 2}
    agg = aggregate_wer([a, b])
    assert agg["edits"] == 2
    assert agg["ref_words"] == 101
    assert abs(agg["wer"] - (2 / 101)) < 1e-9


def test_aggregate_empty():
    assert aggregate_wer([])["wer"] == 0.0
