"""Unit tests for src.utils.answer.clean_ollama_answer."""
from src.utils.answer import (
    DISCORD_LIMIT,
    TRUNCATION_SUFFIX,
    clean_ollama_answer,
)


def test_plain_answer_unchanged():
    assert clean_ollama_answer("Building 20 is the HR building.") == \
        "Building 20 is the HR building."


def test_surrounding_whitespace_trimmed():
    assert clean_ollama_answer("  The next session is in March.\n") == \
        "The next session is in March."


def test_empty_and_whitespace_only():
    assert clean_ollama_answer("") == ""
    assert clean_ollama_answer("   \n\t ") == ""


def test_single_line_think_block_removed():
    assert clean_ollama_answer("<think>pondering</think>The answer.") == \
        "The answer."


def test_multiline_think_block_removed():
    raw = "<think>\nstep 1\nstep 2\n</think>\n\nCody makes the death save."
    assert clean_ollama_answer(raw) == "Cody makes the death save."


def test_think_tag_case_insensitive():
    assert clean_ollama_answer("<THINK>x</Think> Done.") == "Done."


def test_multiple_think_blocks_removed():
    raw = "<think>a</think>First. <think>b</think>Second."
    assert clean_ollama_answer(raw) == "First. Second."


def test_dangling_unterminated_think_dropped():
    # num_predict can cut a reasoning trace off mid-block; everything from
    # the dangling <think> onward must go.
    raw = "Partial answer. <think>reasoning that never closes because it"
    assert clean_ollama_answer(raw) == "Partial answer."


def test_only_reasoning_returns_empty():
    assert clean_ollama_answer("<think>all reasoning, no answer</think>") == ""


def test_realistic_deepseek_shape():
    raw = (
        "<think>\nOkay, the user asks about Building 20. Let me scan the "
        "transcript...\nit says HR building, three floors.\n</think>\n\n"
        "Building 20 is the HR building and has three floors."
    )
    assert clean_ollama_answer(raw) == \
        "Building 20 is the HR building and has three floors."


def test_under_and_at_limit_not_truncated():
    at_limit = "a" * DISCORD_LIMIT
    assert clean_ollama_answer(at_limit) == at_limit
    assert TRUNCATION_SUFFIX not in clean_ollama_answer(at_limit)


def test_over_limit_truncated_with_suffix():
    long = "b" * (DISCORD_LIMIT + 500)
    out = clean_ollama_answer(long)
    assert out == "b" * DISCORD_LIMIT + TRUNCATION_SUFFIX


def test_limit_applied_after_think_removal():
    # The size check must be on the final text, not the raw response.
    raw = "<think>" + "z" * 5000 + "</think>" + "short answer"
    assert clean_ollama_answer(raw) == "short answer"
