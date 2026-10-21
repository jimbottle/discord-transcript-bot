"""What every /ask backend shares: the prompts, the result and error types,
and the output-length instruction.

The backends are ``src/llm/local.py`` (a model on your own machine through
any OpenAI-compatible server, Ollama by default) and ``src/llm/claude.py``
(the Anthropic API). ``src/llm/provider.py`` picks one. Prompts live here so
both answer under the same table rules.
"""

import os
from dataclasses import dataclass, field
from typing import Optional

# Tool rounds per question. One is the norm (the model asks for the notes it
# needs in a single call); the cap keeps a confused model from looping.
MAX_TOOL_ROUNDS = 3

TIME_STYLE = (
    " Write clock times the way the transcript shows them, 12-hour with "
    "AM/PM (for example 8:26 PM), never 24-hour."
)
TABLE_RULES = (
    " Table rules from the GM: do not give tactical or strategic advice on "
    "beating a scenario, do not help optimise (min-max) characters, and do "
    "not reveal or speculate about anything the characters have not learned "
    "or where the plot is going. If asked, decline in one sentence and offer "
    "to recall what happened instead."
)

SYSTEM_PROMPT = (
    "You are an assistant that answers questions about a voice chat "
    "transcript of a tabletop RPG session (the SESSION line, when present, names the game and campaign; follow its STYLE line if there is one). Be concise and direct. Answer from "
    "the transcript; if the answer isn't there, say so. When a CAMPAIGN NOTES "
    "INDEX is provided you may read campaign notes with the read_campaign_notes "
    "tool, but only when the question needs background the transcript lacks, "
    "and say when an answer comes from the notes rather than the transcript."
    + TIME_STYLE
    + TABLE_RULES
)

LORE_SYSTEM_PROMPT = (
    "You answer questions about a tabletop RPG campaign for its players and "
    "GM, using the campaign's reference notes. The CAMPAIGN line names the "
    "game, campaign and group; follow its STYLE line if there is one. Read "
    "the notes you need with the read_campaign_notes tool (request every "
    "relevant note in one call) and answer only from them. If the notes "
    "don't cover the question, say so plainly rather than inventing "
    "campaign facts; you may add general knowledge of the game's published "
    "rules or setting if you label it as such. Be concise and direct."
    + TIME_STYLE
    + TABLE_RULES
)

EMPTY_MSG = "The model returned no answer. Try rephrasing the question."
TRUNCATION_NOTE = "\n\n_(answer cut off at the token limit)_"
ONLY_GM_NOTES_MSG = (
    "This campaign only has GM-only notes. Tick 'Include GM-only notes' to "
    "ask about them."
)
NO_NOTES_MSG = (
    "This campaign has no reference notes yet, so there is nothing to answer from."
)


def env(name: str) -> str:
    """Environment lookup that treats whitespace-only as unset (``.env.sample``
    uses the ``FOO=`` convention, so a blank value is a realistic footgun)."""
    return os.getenv(name, "").strip()


class AskFailed(Exception):
    """`/ask` could not produce an answer. ``user_message`` is safe to post
    to Discord as-is; the original error is chained for the log."""

    def __init__(self, user_message: str, cause: Optional[BaseException] = None):
        super().__init__(user_message)
        self.user_message = user_message
        self.__cause__ = cause


class AskBlocked(Exception):
    """The GM check refused the question before any answer was generated.
    Not an AskFailed, so callers record it as ``blocked``, not a failure.
    ``user_message`` is safe to post as-is."""

    def __init__(self, user_message: str, verdict):
        super().__init__(user_message)
        self.user_message = user_message
        self.verdict = verdict


@dataclass
class AskResult:
    answer: str
    model: str
    input_tokens: int = 0
    cached_tokens: int = 0
    output_tokens: int = 0
    # Campaign notes the model pulled in (ids), in order; [] when it answered
    # from the transcript alone. Token counts above are summed over rounds.
    notes_used: list = field(default_factory=list)
    rounds: int = 1
    # The GM check that let the question through (None when disabled).
    guard: Optional[object] = None


def length_instruction(max_chars: int) -> str:
    """Tell the model its hard output budget (Discord messages are capped).
    Models overshoot a stated character count by up to about a third
    (measured: 1700 asked, 2241 written), so the target is ~70% of the
    hard limit and the limit itself is stated as a ceiling."""
    target = int(max_chars * 0.7)
    return (
        f"OUTPUT LIMIT: this answer is posted as one Discord message with a "
        f"hard cap. Aim for about {target} characters (roughly {target // 6} "
        f"words) and never exceed {max_chars}. For broad requests like a full "
        "recap, summarise the highlights instead of covering everything. Lead "
        "with what was asked and finish cleanly within the limit."
    )


def dedupe(ids) -> list:
    seen = set()
    return [i for i in ids if not (i in seen or seen.add(i))]
