"""
Regression tests derived from the 2026-03-07 live test session.
Each test pins a specific over-correction bug so it cannot reappear silently.
"""
import pytest
from unittest.mock import AsyncMock, patch

from fluentify.pipeline import generate_correction
from fluentify.core import _is_trivial_change, normalize_for_compare


# ── Intensifier injection (code-level guard) ─────────────────────────────────

@pytest.mark.asyncio
async def test_regression_adding_way_gives_perfect():
    """'way' injected into 'too expensive' must not show as a correction."""
    with patch("fluentify.pipeline._call_llm", new=AsyncMock(return_value="The rent is way too expensive")):
        assert await generate_correction("The rent is too expensive", "") == "PERFECT"


@pytest.mark.asyncio
async def test_regression_adding_really_gives_perfect():
    """'really' injected before verb must not show as a correction."""
    with patch("fluentify.pipeline._call_llm", new=AsyncMock(return_value="I really need to get my life together")):
        assert await generate_correction("I need to get my life together", "") == "PERFECT"


# ── Synonym / time-word swap (relies on LLM PERFECT sentinel) ────────────────

@pytest.mark.asyncio
async def test_regression_today_vs_this_morning_gives_perfect():
    """'today' → 'this morning' is not a real correction; LLM should say PERFECT."""
    with patch("fluentify.pipeline._call_llm", new=AsyncMock(return_value="PERFECT")):
        assert await generate_correction("I woke up way too early today.", "") == "PERFECT"


@pytest.mark.asyncio
async def test_regression_thinking_about_vs_of_gives_perfect():
    """'thinking about' vs 'thinking of' are equally correct."""
    with patch("fluentify.pipeline._call_llm", new=AsyncMock(return_value="PERFECT")):
        assert await generate_correction("I've been thinking about moving to another city.", "") == "PERFECT"


# ── Colloquial filler preservation ───────────────────────────────────────────

@pytest.mark.asyncio
async def test_regression_like_filler_preserved_gives_perfect():
    """'Like,' at the start is an intentional filler; LLM must not strip it."""
    with patch("fluentify.pipeline._call_llm", new=AsyncMock(return_value="PERFECT")):
        assert await generate_correction("Like, where did the time even go?", "") == "PERFECT"


# ── Real corrections must NOT be swallowed ───────────────────────────────────

@pytest.mark.asyncio
async def test_regression_tense_correction_is_not_swallowed():
    """'I go' → 'I went' is a real fix; must not be suppressed as trivial."""
    with patch("fluentify.pipeline._call_llm", new=AsyncMock(return_value="Yesterday I went to the gym.")):
        result = await generate_correction("Yesterday I go to the gym.", "")
    assert result == "Yesterday I went to the gym."


@pytest.mark.asyncio
async def test_regression_double_negative_fix_is_not_swallowed():
    """'don't have no idea' → 'have no idea' must remain a real correction."""
    with patch("fluentify.pipeline._call_llm", new=AsyncMock(return_value="I have no idea what to do with my life right now.")):
        result = await generate_correction("I don't have no idea what to do with my life right now.", "")
    assert result == "I have no idea what to do with my life right now."


@pytest.mark.asyncio
async def test_regression_phrasal_verb_paid_off_is_not_swallowed():
    """'paid' → 'paid off' must not be suppressed."""
    with patch("fluentify.pipeline._call_llm", new=AsyncMock(return_value="All the hard work finally paid off.")):
        result = await generate_correction("All the hard effort finally paid.", "")
    assert result == "All the hard work finally paid off."


# ── _is_trivial_change direct assertions ─────────────────────────────────────

def test_regression_trivial_single_word_insert():
    assert _is_trivial_change("I need to go", "I really need to go") is True


def test_regression_non_trivial_idiom_swap():
    assert _is_trivial_change("This year is going so fast", "This year is flying by") is False
