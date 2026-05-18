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


# ── roborev #770 (MEDIUM): opening-tag-less / nested reasoning ─────────
# QwQ / deepseek-r1 distills whose chat template primes the assistant
# turn inside <think> emit content that starts mid-reasoning with only a
# closing </think>. The old logic (regex needs an opener; find("<think>")
# returns -1) leaked the entire monologue plus a stray tag to Discord.

def test_closing_tag_only_strips_leading_reasoning():
    raw = (
        "Okay, the user wants Building 20. Scanning the transcript... "
        "it says HR, three floors.\n</think>\n\n"
        "Building 20 is the HR building with three floors."
    )
    assert clean_ollama_answer(raw) == \
        "Building 20 is the HR building with three floors."


def test_closing_tag_only_no_answer_returns_empty():
    assert clean_ollama_answer("all reasoning, never got to an answer</think>") == ""


def test_closing_tag_case_insensitive():
    assert clean_ollama_answer("internal monologue</THINK>The answer.") == \
        "The answer."


def test_nested_think_blocks_removed():
    raw = "<think>outer <think>inner</think> still outer</think>The answer."
    assert clean_ollama_answer(raw) == "The answer."


def test_well_formed_block_then_closing_only_tail():
    # A clean block, then the model re-enters reasoning and only closes it.
    raw = "<think>first pass</think>more reasoning</think>Final answer."
    assert clean_ollama_answer(raw) == "Final answer."
