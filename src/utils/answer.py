"""Post-processing for raw Ollama `/ask` answers before they go to Discord.

Kept as a pure, side-effect-free helper so it can be unit-tested without
a live Discord or Ollama (the `/ask` command itself needs both).
"""
import re

# Reasoning models (e.g. deepseek-r1) stream their chain-of-thought
# inline in message.content as <think>...</think> rather than in
# Ollama's native `thinking` field. Without stripping it the bot would
# post the entire reasoning monologue to the channel. Gemma 4 defaults
# thinking ON in Ollama, so the call site also passes think=False; this
# is the defense-in-depth for any model that ignores that.
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)

DISCORD_LIMIT = 1900
TRUNCATION_SUFFIX = "\n\n...(truncated)"


def clean_ollama_answer(text: str, limit: int = DISCORD_LIMIT) -> str:
    """Strip reasoning blocks and enforce Discord's message-size limit.

    - Removes well-formed ``<think>...</think>`` blocks (incl. nested).
    - Drops a leading reasoning trace that has only a closing
      ``</think>`` and no opener (QwQ / deepseek-r1 distills whose chat
      template primes the assistant turn inside ``<think>``).
    - Drops a dangling unterminated ``<think>`` (a reasoning trace cut
      off by ``num_predict``) and everything after it.
    - Trims surrounding whitespace.
    - Truncates to ``limit`` with a suffix (behaviour preserved verbatim
      from the previous inline logic in main.py).

    A plain answer with no reasoning blocks and length <= limit is
    returned unchanged apart from whitespace trimming.
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
        cleaned = cleaned[close_idx + len("</think>"):]

    open_idx = cleaned.lower().find("<think>")
    if open_idx != -1:
        cleaned = cleaned[:open_idx]

    cleaned = cleaned.strip()

    if len(cleaned) > limit:
        cleaned = cleaned[:limit] + TRUNCATION_SUFFIX

    return cleaned
