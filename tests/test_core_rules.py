import pytest
from fluentify.core import (
    normalize_for_compare,
    _sanitize_llm_output,
    _trim_context,
    _is_trivial_change,
)
from fluentify.config import MAX_CONTEXT_CHARS


# ── normalize_for_compare ────────────────────────────────────────────────────

def test_normalize_strips_punctuation():
    assert normalize_for_compare("I'm good!") == "im good"


def test_normalize_strips_emoji():
    assert normalize_for_compare("Good job 🎉") == "good job"


def test_normalize_lowercases():
    assert normalize_for_compare("Hello World") == "hello world"


def test_normalize_collapses_whitespace():
    assert normalize_for_compare("  hello   world  ") == "hello world"


def test_normalize_makes_punctuation_variants_equal():
    assert normalize_for_compare("I'm good!") == normalize_for_compare("im good")


def test_normalize_empty_string():
    assert normalize_for_compare("") == ""


# ── _sanitize_llm_output ─────────────────────────────────────────────────────

def test_sanitize_removes_think_tags():
    raw = "<think>step by step reasoning</think>I went to the gym."
    assert _sanitize_llm_output(raw) == "I went to the gym."


def test_sanitize_returns_last_non_empty_line():
    raw = "Sure, here's the correction:\nI went to the gym."
    assert _sanitize_llm_output(raw) == "I went to the gym."


def test_sanitize_returns_empty_for_blank_input():
    assert _sanitize_llm_output("") == ""
    assert _sanitize_llm_output("   ") == ""


def test_sanitize_handles_think_tag_only():
    assert _sanitize_llm_output("<think>nothing useful</think>") == ""


def test_sanitize_strips_surrounding_whitespace():
    assert _sanitize_llm_output("  hello  ") == "hello"


# ── _trim_context ────────────────────────────────────────────────────────────

def test_trim_context_leaves_short_text_unchanged():
    text = "short context"
    assert _trim_context(text) == text


def test_trim_context_truncates_long_text_to_max():
    long_text = "x" * (MAX_CONTEXT_CHARS + 100)
    result = _trim_context(long_text)
    assert len(result) <= MAX_CONTEXT_CHARS


def test_trim_context_keeps_the_tail():
    tail = "most recent message"
    long_text = "a" * MAX_CONTEXT_CHARS + tail
    assert _trim_context(long_text).endswith(tail)


def test_trim_context_handles_empty_and_none():
    assert _trim_context("") == ""
    assert _trim_context(None) == ""


# ── _is_trivial_change ───────────────────────────────────────────────────────

def test_is_trivial_identical_sentence():
    assert _is_trivial_change("I woke up today", "I woke up today") is True


def test_is_trivial_single_intensifier_insertion():
    # "way" inserted — 7→8 words, high sequence similarity
    assert _is_trivial_change(
        "The rent is too expensive",
        "The rent is way too expensive",
    ) is True


def test_is_trivial_adverb_insertion_before_verb():
    # "really" inserted — should be trivial
    assert _is_trivial_change(
        "I need to get my life together",
        "I really need to get my life together",
    ) is True


def test_is_not_trivial_tense_correction():
    # "go" → "went" is a real grammar fix
    assert _is_trivial_change(
        "Yesterday I go to the gym",
        "Yesterday I went to the gym",
    ) is False


def test_is_not_trivial_idiom_replacement():
    # "going so fast" → "flying by" is a meaningful change
    assert _is_trivial_change(
        "This year is going so fast",
        "This year is flying by",
    ) is False


def test_is_not_trivial_double_negative_fix():
    # Word count drops by 1, but sequence similarity is too low
    assert _is_trivial_change(
        "I don't have no idea what to do",
        "I have no idea what to do",
    ) is False
