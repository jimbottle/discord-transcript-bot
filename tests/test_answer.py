"""Unit tests for src.utils.answer."""

from src.utils.answer import (
    DISCORD_LIMIT,
    TRUNCATION_SUFFIX,
    clamp_message,
    clean_answer,
)


def test_plain_answer_unchanged():
    assert (
        clean_answer("Building 20 is the HR building.")
        == "Building 20 is the HR building."
    )


def test_surrounding_whitespace_trimmed():
    assert (
        clean_answer("  The next session is in March.\n")
        == "The next session is in March."
    )


def test_empty_and_whitespace_only():
    assert clean_answer("") == ""
    assert clean_answer("   \n\t ") == ""


def test_single_line_think_block_removed():
    assert clean_answer("<think>pondering</think>The answer.") == "The answer."


def test_multiline_think_block_removed():
    raw = "<think>\nstep 1\nstep 2\n</think>\n\nCody makes the death save."
    assert clean_answer(raw) == "Cody makes the death save."


def test_think_tag_case_insensitive():
    assert clean_answer("<THINK>x</Think> Done.") == "Done."


def test_multiple_think_blocks_removed():
    raw = "<think>a</think>First. <think>b</think>Second."
    assert clean_answer(raw) == "First. Second."


def test_dangling_unterminated_think_dropped():
    # num_predict can cut a reasoning trace off mid-block; everything from
    # the dangling <think> onward must go.
    raw = "Partial answer. <think>reasoning that never closes because it"
    assert clean_answer(raw) == "Partial answer."


def test_only_reasoning_returns_empty():
    assert clean_answer("<think>all reasoning, no answer</think>") == ""


def test_realistic_deepseek_shape():
    raw = (
        "<think>\nOkay, the user asks about Building 20. Let me scan the "
        "transcript...\nit says HR building, three floors.\n</think>\n\n"
        "Building 20 is the HR building and has three floors."
    )
    assert clean_answer(raw) == "Building 20 is the HR building and has three floors."


def test_under_and_at_limit_not_truncated():
    at_limit = "a" * DISCORD_LIMIT
    assert clean_answer(at_limit) == at_limit
    assert TRUNCATION_SUFFIX not in clean_answer(at_limit)


def test_over_limit_truncated_only_when_a_limit_is_given():
    long = "b" * (DISCORD_LIMIT + 500)
    assert clean_answer(long) == long  # /ask splits instead of truncating
    assert clean_answer(long, DISCORD_LIMIT) == "b" * DISCORD_LIMIT + TRUNCATION_SUFFIX


def test_limit_applied_after_think_removal():
    # The size check must be on the final text, not the raw response.
    raw = "<think>" + "z" * 5000 + "</think>" + "short answer"
    assert clean_answer(raw) == "short answer"


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
    assert clean_answer(raw) == "Building 20 is the HR building with three floors."


def test_closing_tag_only_no_answer_returns_empty():
    assert clean_answer("all reasoning, never got to an answer</think>") == ""


def test_closing_tag_case_insensitive():
    assert clean_answer("internal monologue</THINK>The answer.") == "The answer."


def test_nested_think_blocks_removed():
    raw = "<think>outer <think>inner</think> still outer</think>The answer."
    assert clean_answer(raw) == "The answer."


def test_well_formed_block_then_closing_only_tail():
    # A clean block, then the model re-enters reasoning and only closes it.
    raw = "<think>first pass</think>more reasoning</think>Final answer."
    assert clean_answer(raw) == "Final answer."


def test_literal_close_tag_in_answer_is_truncated():
    """roborev #776 (LOW): documents the accepted tradeoff — a genuine
    answer that contains the literal ``</think>`` loses everything up to
    and including the last occurrence. This asserts the (destructive)
    behavior so the limitation is explicit, not invisible."""
    raw = "To strip reasoning, drop everything before </think> in the text."
    assert clean_answer(raw) == "in the text."


# ── clamp_message (roborev #776 LOW: composed-message clamp had no test) ──


def test_clamp_message_under_limit_unchanged():
    assert clamp_message("short", 2000) == "short"


def test_clamp_message_at_limit_unchanged():
    exact = "a" * 2000
    assert clamp_message(exact, 2000) == exact


def test_clamp_message_over_limit_truncated_to_exactly_limit():
    out = clamp_message("b" * 5000, 2000)
    assert len(out) == 2000
    assert out.endswith(TRUNCATION_SUFFIX)
    assert out == "b" * (2000 - len(TRUNCATION_SUFFIX)) + TRUNCATION_SUFFIX


def test_clamp_message_tiny_limit_no_negative_slice():
    # limit smaller than the suffix must not raise / negative-slice.
    out = clamp_message("x" * 100, 5)
    assert out == TRUNCATION_SUFFIX


# ── Discord delivery: budget + split, never truncate ───────────────────

from src.utils.answer import (  # noqa: E402
    DISCORD_HARD_LIMIT,
    answer_budget,
    discord_messages,
    split_message,
)


def _para(n, ch="x"):
    return (ch * 9 + " ") * n


def test_short_answer_is_one_message():
    assert discord_messages("Who?", "Kane.") == ["**Q:** Who?\n\nKane."]


def test_long_answer_splits_on_paragraphs_and_keeps_everything():
    paras = [_para(60, c).strip() for c in "abcde"]  # ~600 chars each
    answer = "\n\n".join(paras)
    msgs = discord_messages("What happened?", answer)
    assert len(msgs) >= 2
    assert all(len(m) <= DISCORD_HARD_LIMIT for m in msgs)
    assert msgs[0].startswith("**Q:** What happened?")
    joined = "\n\n".join(msgs)
    for p in paras:
        assert p in joined  # no paragraph cut in half, nothing dropped
    assert "truncated" not in joined


def test_one_huge_paragraph_splits_on_words():
    answer = _para(500).strip()  # ~5000 chars, no newlines
    msgs = split_message(answer)
    assert all(len(m) <= DISCORD_HARD_LIMIT for m in msgs)
    assert " ".join(msgs).split() == answer.split()


def test_unbroken_token_is_hard_split():
    msgs = split_message("z" * 4500)
    assert all(len(m) <= DISCORD_HARD_LIMIT for m in msgs)
    assert "".join(msgs) == "z" * 4500


def test_code_block_split_is_closed_and_reopened():
    code = "```\n" + "\n".join(f"line {i} " + "y" * 40 for i in range(80)) + "\n```"
    msgs = split_message(code)
    assert len(msgs) >= 2
    for m in msgs:
        assert m.count("```") % 2 == 0, m[:80]
        assert len(m) <= DISCORD_HARD_LIMIT


def test_budget_shrinks_with_long_questions_but_stays_useful():
    assert answer_budget("Who?") == 1700  # capped
    # 2000 - len("**Q:** ") - 900 - len("\n\n") - 200 headroom
    assert answer_budget("q" * 900) == 891
    assert answer_budget("q" * 1900) == 600  # floor
