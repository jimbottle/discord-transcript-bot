"""Word Error Rate (WER) scoring for the transcription A/B harness.

Pure, dependency-free helpers (no whisper / faster_whisper import) so they
can be unit-tested without loading any model. Used by
scripts/ab_transcribe.py to compare transcription configs on real session
audio. discord-transcript-bot-61z.
"""

import re

# Strip everything that isn't a word char, whitespace, or an intra-word
# apostrophe (so "don't" stays one token, not "don" + "t").
_NON_WORD = re.compile(r"[^\w\s']")


def normalize_text(text):
    """Lowercase, drop punctuation, and split into comparable word tokens.

    This is the canonical normalization used on BOTH reference and
    hypothesis before scoring, so casing/punctuation differences don't
    count as errors — only real word substitutions/insertions/deletions do.
    """
    if not text:
        return []
    text = text.lower()
    text = _NON_WORD.sub(" ", text)
    return text.split()


def _word_levenshtein(ref_words, hyp_words):
    """Edit distance (substitutions + insertions + deletions) between two
    token lists, computed with the standard O(n*m) DP over two rows."""
    n, m = len(ref_words), len(hyp_words)
    if n == 0:
        return m
    if m == 0:
        return n
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        ref_w = ref_words[i - 1]
        for j in range(1, m + 1):
            cost = 0 if ref_w == hyp_words[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[m]


def word_error_rate(reference, hypothesis):
    """Score one hypothesis against one reference.

    Returns a dict: ``{"wer", "edits", "ref_words", "hyp_words"}``. ``wer``
    is edits / ref_words (the standard WER), or 0.0 when the reference is
    empty and the hypothesis is too, else 1.0 (all insertions). Aggregate
    across many clips by summing ``edits`` and ``ref_words`` separately and
    dividing — NOT by averaging per-clip WER (which over-weights short
    clips). See ``aggregate_wer``.
    """
    ref_words = normalize_text(reference)
    hyp_words = normalize_text(hypothesis)
    edits = _word_levenshtein(ref_words, hyp_words)
    ref_len = len(ref_words)
    if ref_len == 0:
        wer = 0.0 if len(hyp_words) == 0 else 1.0
    else:
        wer = edits / ref_len
    return {
        "wer": wer,
        "edits": edits,
        "ref_words": ref_len,
        "hyp_words": len(hyp_words),
    }


def aggregate_wer(scores):
    """Corpus-level WER over many per-clip score dicts (from
    ``word_error_rate``): total edits / total reference words. Returns the
    same dict shape with summed counts. Empty input -> wer 0.0."""
    total_edits = sum(s["edits"] for s in scores)
    total_ref = sum(s["ref_words"] for s in scores)
    total_hyp = sum(s["hyp_words"] for s in scores)
    wer = (total_edits / total_ref) if total_ref else 0.0
    return {
        "wer": wer,
        "edits": total_edits,
        "ref_words": total_ref,
        "hyp_words": total_hyp,
    }
