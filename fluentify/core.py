import re
from difflib import SequenceMatcher
from fluentify.config import MAX_CONTEXT_CHARS


def _trim_context(context_text: str) -> str:
    """Hard-cap context length to protect latency/cost."""
    if not context_text:
        return ""
    context_text = context_text.strip()
    if len(context_text) <= MAX_CONTEXT_CHARS:
        return context_text
    # Keep the most recent tail (usually most relevant for tense/pronouns)
    return context_text[-MAX_CONTEXT_CHARS:].lstrip()


def _sanitize_llm_output(raw_text: str) -> str:
    """
    Make the output robust across models:
    - remove any accidental <think>...</think>
    - take the last non-empty line (in case the model adds prefaces)
    - trim whitespace
    """
    if not raw_text:
        return ""
    text = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL).strip()
    if not text:
        return ""
    # Use the last non-empty line to avoid "Sure:" / explanations leaking
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return lines[-1] if lines else ""


def _is_trivial_change(original: str, corrected: str) -> bool:
    """
    Return True if the correction is too minor to be worth showing.

    Compares normalized word sequences using SequenceMatcher.ratio(),
    which accounts for word order and duplicates unlike a plain set overlap.
    A ratio >= 0.90 with a word-count difference of at most 1 is considered trivial
    (e.g. synonym swaps, added intensifiers, punctuation-only changes).
    """
    orig_words = normalize_for_compare(original).split()
    corr_words = normalize_for_compare(corrected).split()

    # Sequence similarity: respects word order and duplicate words
    ratio = SequenceMatcher(None, orig_words, corr_words).ratio()

    # Word count difference
    len_diff = abs(len(orig_words) - len(corr_words))

    # Treat as trivial if sequences are ≥90% similar and differ by at most 1 word
    return ratio >= 0.90 and len_diff <= 1


def normalize_for_compare(text: str) -> str:
    """
    Normalize text for fuzzy equality comparison.
    Strips case differences, punctuation, emoji, and extra whitespace
    so that trivial cosmetic edits don't count as a real correction.

    Examples:
        "I'm good!"  →  "im good"
        "im good"    →  "im good"   ← same → Perfect!
    """
    text = text.lower()
    # Remove everything except alphanumeric and spaces (covers punctuation + emoji)
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()
