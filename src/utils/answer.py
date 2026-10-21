"""Post-processing for raw `/ask` answers before they go to Discord.

Kept as a pure, side-effect-free helper so it can be unit-tested without
a live Discord or model API (the `/ask` command itself needs both).
"""

import re

# Claude returns reasoning in separate thinking blocks that src/llm/claude
# never includes in the answer, so inline <think>...</think> text should
# not occur. The stripping below predates the move to the Anthropic API
# (open-weight reasoning models put their monologue inline) and stays as
# defense in depth: posting a reasoning trace to the channel is the worse
# failure.
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)

DISCORD_LIMIT = 1900
TRUNCATION_SUFFIX = "\n\n...(truncated)"


def clean_answer(text: str, limit=None) -> str:
    """Strip reasoning blocks and enforce Discord's message-size limit.

    - Removes well-formed ``<think>...</think>`` blocks (incl. nested).
    - Drops a leading reasoning trace that has only a closing
      ``</think>`` and no opener (QwQ / deepseek-r1 distills whose chat
      template primes the assistant turn inside ``<think>``).
    - Drops a dangling unterminated ``<think>`` (a reasoning trace cut
      off by ``num_predict``) and everything after it.
    - Trims surrounding whitespace.
    - Truncates to ``limit`` with a suffix **only when a limit is given**.
      /ask no longer passes one: a long answer is split across Discord
      messages (:func:`discord_messages`) instead of being cut off.

    A plain answer with no reasoning blocks and length <= limit is
    returned unchanged apart from whitespace trimming.

    Known limitation: the stray-closing-tag strip keys off the literal
    text ``</think>``. If a *genuine* answer itself contains that
    literal (e.g. someone asks ``/ask`` to explain think-tag
    stripping), everything up to and including the last occurrence is
    discarded. This is an accepted tradeoff — preventing a reasoning
    leak matters more for this bot than answering questions about the
    tag itself — and is asserted explicitly in
    ``test_answer.py::test_literal_close_tag_in_answer_is_truncated``.
    """
    if not text:
        return ""

    cleaned = _THINK_BLOCK_RE.sub("", text)

    # A stray closing tag with no matching opener means the model began
    # its turn already inside a reasoning trace, or it's the leftover
    # from a nested block the non-greedy regex couldn't fully consume
    # (e.g. ``<think>a<think>b</think>c</think>ans`` → ``c</think>ans``).
    # Everything up to and including the last such ``</think>`` is
    # reasoning. ``</think>`` is 8 chars regardless of case.
    lowered = cleaned.lower()
    close_idx = lowered.rfind("</think>")
    if close_idx != -1:
        cleaned = cleaned[close_idx + len("</think>") :]

    open_idx = cleaned.lower().find("<think>")
    if open_idx != -1:
        cleaned = cleaned[:open_idx]

    cleaned = cleaned.strip()

    if limit is not None and len(cleaned) > limit:
        cleaned = cleaned[:limit] + TRUNCATION_SUFFIX

    return cleaned


def clamp_message(message: str, limit: int) -> str:
    """Clamp a fully composed message to a hard character ``limit``.

    ``clean_answer`` only bounds the *answer*; the rendered
    ``**Q:** <question>\\n\\n<answer>`` can still exceed Discord's hard
    2000-char limit when the question is long. Returns ``message``
    unchanged if it already fits, otherwise truncates and appends
    :data:`TRUNCATION_SUFFIX`. For any realistic ``limit`` (>= the
    suffix length) the result is exactly ``limit`` chars; for a
    degenerate tiny ``limit`` it is just the suffix — never a negative
    slice / crash.
    """
    if len(message) <= limit:
        return message
    keep = max(0, limit - len(TRUNCATION_SUFFIX))
    return message[:keep] + TRUNCATION_SUFFIX


# ── Discord delivery: budget the model, split instead of truncating ──────
# Discord rejects a message over 2000 characters. /ask tells the model a
# character budget (answer_budget) so answers normally fit in one message;
# when one still runs long it is split at paragraph, then line, then word
# boundaries into several messages, so nothing is ever cut off.

DISCORD_HARD_LIMIT = 2000
_FENCE = "```"


def answer_budget(question: str, limit: int = DISCORD_HARD_LIMIT) -> int:
    """Characters the model should aim for so ``**Q:** <question>`` plus the
    answer fits in one Discord message, with headroom because models count
    characters loosely. Never below 600 (a long question still gets a real
    answer; the split below catches any overflow)."""
    overhead = len("**Q:** ") + len(question) + len("\n\n")
    return max(600, min(1700, limit - overhead - 200))


def _pieces(text: str, limit: int) -> list:
    """Break ``text`` into pieces no longer than ``limit``, preferring
    paragraph, then line, then word boundaries; hard-cut only a single
    token longer than ``limit``."""
    if len(text) <= limit:
        return [text]
    for sep in ("\n\n", "\n", " "):
        parts = text.split(sep)
        if len(parts) > 1:
            out, cur = [], ""
            for part in parts:
                cand = part if not cur else cur + sep + part
                if len(cand) <= limit:
                    cur = cand
                    continue
                if cur:
                    out.append(cur)
                if len(part) <= limit:
                    cur = part
                else:
                    sub = _pieces(part, limit)
                    out.extend(sub[:-1])
                    cur = sub[-1]
            if cur:
                out.append(cur)
            return out
    return [text[i : i + limit] for i in range(0, len(text), limit)]


_FENCE_LINE = re.compile(r"^```[^\n`]*", re.M)


def split_message(text: str, limit: int = DISCORD_HARD_LIMIT) -> list:
    """Split ``text`` into Discord-sized messages without losing anything.
    A code block split across messages is closed at the end of one and
    reopened at the start of the next with its original opening line
    (```python keeps its highlighting; roborev #6486)."""
    text = text.strip()
    if not text:
        return []
    longest_opener = max((len(m) for m in _FENCE_LINE.findall(text)), default=3)
    reserve = len("\n" + _FENCE) + longest_opener + 1
    chunks = [c.strip("\n") for c in _pieces(text, limit - reserve) if c.strip()]
    out, opener = [], None
    for chunk in chunks:
        if opener:
            chunk = opener + "\n" + chunk
        # Parity over every ``` (Discord treats a mid-line ``` as a fence
        # too); the opener text comes from the last line-start fence.
        if chunk.count(_FENCE) % 2 == 1:
            starts = _FENCE_LINE.findall(chunk)
            opener = starts[-1] if starts else _FENCE
            chunk += "\n" + _FENCE
        else:
            opener = None
        out.append(chunk)
    return out


def discord_messages(
    question: str, answer: str, limit: int = DISCORD_HARD_LIMIT
) -> list:
    """The messages to post for one /ask answer, in order: the first opens
    with the question; every message fits Discord's limit; the joined text
    is the whole answer."""
    return split_message(f"**Q:** {question}\n\n{answer}", limit)
