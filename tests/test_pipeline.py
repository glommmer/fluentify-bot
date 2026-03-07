import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

import discord
from fluentify.pipeline import (
    generate_correction,
    _build_bot_corrections,
    _has_bot_approval,
    _reviewed_text_for_message,
    _truncate_history_text,
    build_context,
    process_message,
)
from fluentify.config import MAX_HISTORY_MSG_CHARS
from tests.conftest import make_message, make_reaction, async_iter


# ── generate_correction ──────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_generate_correction_returns_perfect_when_llm_says_perfect():
    with patch("fluentify.pipeline._call_llm", new=AsyncMock(return_value="PERFECT")):
        assert await generate_correction("I woke up today.", "") == "PERFECT"


@pytest.mark.asyncio
async def test_generate_correction_returns_perfect_on_normalize_match():
    # Punctuation-only difference → normalize makes them equal
    with patch("fluentify.pipeline._call_llm", new=AsyncMock(return_value="I woke up today")):
        assert await generate_correction("I woke up today!", "") == "PERFECT"


@pytest.mark.asyncio
async def test_generate_correction_returns_perfect_for_trivial_intensifier():
    with patch("fluentify.pipeline._call_llm", new=AsyncMock(return_value="The rent is way too expensive")):
        assert await generate_correction("The rent is too expensive", "") == "PERFECT"


@pytest.mark.asyncio
async def test_generate_correction_returns_corrected_sentence():
    with patch("fluentify.pipeline._call_llm", new=AsyncMock(return_value="Yesterday I went to the gym.")):
        result = await generate_correction("Yesterday I go to the gym.", "")
    assert result == "Yesterday I went to the gym."


@pytest.mark.asyncio
async def test_generate_correction_falls_back_on_timeout():
    call_count = 0

    async def flaky_llm(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise asyncio.TimeoutError()
        return "Yesterday I went to the gym."

    with patch("fluentify.pipeline._call_llm", side_effect=flaky_llm):
        result = await generate_correction("Yesterday I go to the gym.", "")

    assert result == "Yesterday I went to the gym."
    assert call_count == 2


@pytest.mark.asyncio
async def test_generate_correction_falls_back_on_rate_limit():
    call_count = 0

    async def rate_limited(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise Exception("429 rate limit exceeded")
        return "I worked on that report."

    with patch("fluentify.pipeline._call_llm", side_effect=rate_limited):
        result = await generate_correction("I work on that report.", "")

    assert result == "I worked on that report."
    assert call_count == 2


@pytest.mark.asyncio
async def test_generate_correction_returns_error_when_all_models_fail():
    with patch("fluentify.pipeline._call_llm", side_effect=Exception("server error")):
        result = await generate_correction("I go to gym.", "")
    assert result == "Error"


# ── _truncate_history_text ───────────────────────────────────────────────────

def test_truncate_history_text_short_unchanged():
    assert _truncate_history_text("hello world") == "hello world"


def test_truncate_history_text_long_gets_ellipsis():
    long_text = "w" * (MAX_HISTORY_MSG_CHARS + 50)
    result = _truncate_history_text(long_text)
    assert result.endswith("…")
    assert len(result) <= MAX_HISTORY_MSG_CHARS + 1


def test_truncate_history_text_empty():
    assert _truncate_history_text("") == ""
    assert _truncate_history_text(None) == ""


# ── _has_bot_approval ────────────────────────────────────────────────────────

def test_has_bot_approval_true_when_bot_reacted():
    msg = make_message(reactions=[make_reaction("✅", me=True)])
    assert _has_bot_approval(msg) is True


def test_has_bot_approval_false_when_other_user_reacted():
    msg = make_message(reactions=[make_reaction("✅", me=False)])
    assert _has_bot_approval(msg) is False


def test_has_bot_approval_false_when_no_reactions():
    msg = make_message(reactions=[])
    assert _has_bot_approval(msg) is False


def test_has_bot_approval_false_for_different_emoji():
    msg = make_message(reactions=[make_reaction("👍", me=True)])
    assert _has_bot_approval(msg) is False


# ── _build_bot_corrections ───────────────────────────────────────────────────

def test_build_bot_corrections_extracts_correction(bot_id):
    bot_reply = make_message(
        id=200, content="I went to the gym.",
        author_id=bot_id, is_bot=True, reference_message_id=1,
    )
    result = _build_bot_corrections([bot_reply], bot_user_id=bot_id)
    assert result == {1: "I went to the gym."}


def test_build_bot_corrections_ignores_human_messages(bot_id):
    human = make_message(id=100, content="hello", author_id=50, reference_message_id=1)
    assert _build_bot_corrections([human], bot_user_id=bot_id) == {}


def test_build_bot_corrections_returns_empty_when_bot_id_none():
    bot_reply = make_message(id=200, content="correction", author_id=999, is_bot=True, reference_message_id=1)
    assert _build_bot_corrections([bot_reply], bot_user_id=None) == {}


def test_build_bot_corrections_keeps_newest_correction(bot_id):
    first = make_message(id=201, content="newest correction", author_id=bot_id, is_bot=True, reference_message_id=1)
    second = make_message(id=200, content="older correction", author_id=bot_id, is_bot=True, reference_message_id=1)
    # history is newest → oldest
    result = _build_bot_corrections([first, second], bot_user_id=bot_id)
    assert result[1] == "newest correction"


# ── _reviewed_text_for_message ───────────────────────────────────────────────

def test_reviewed_text_uses_bot_correction():
    msg = make_message(id=1, content="original")
    text, source = _reviewed_text_for_message(msg, {1: "corrected"})
    assert text == "corrected"
    assert source == "corrected"


def test_reviewed_text_uses_approved_original():
    msg = make_message(id=1, content="approved text", reactions=[make_reaction("✅", me=True)])
    text, source = _reviewed_text_for_message(msg, {})
    assert text == "approved text"
    assert source == "approved"


def test_reviewed_text_returns_none_for_unreviewed():
    msg = make_message(id=1, content="unreviewed")
    text, source = _reviewed_text_for_message(msg, {})
    assert text is None
    assert source == "none"


def test_reviewed_text_returns_raw_when_fallback_enabled():
    msg = make_message(id=1, content="raw text")
    text, source = _reviewed_text_for_message(msg, {}, allow_unreviewed_fallback=True)
    assert text == "raw text"
    assert source == "raw"


def test_reviewed_text_correction_takes_priority_over_approval():
    msg = make_message(id=1, content="original", reactions=[make_reaction("✅", me=True)])
    text, source = _reviewed_text_for_message(msg, {1: "bot corrected"})
    assert text == "bot corrected"
    assert source == "corrected"


# ── build_context ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_build_context_includes_author_id(bot_id):
    message = make_message(content="hello", author_id=123)
    message.channel.history = MagicMock(return_value=async_iter([]))
    message.guild.me.id = bot_id

    result = await build_context(message)
    assert "[current_user_id]" in result
    assert "123" in result


@pytest.mark.asyncio
async def test_build_context_shows_none_when_no_reviewed_history(bot_id):
    message = make_message(content="hello", author_id=123)
    # History has a message but it's unreviewed (no reaction, no bot reply)
    unreviewed = make_message(id=10, content="some msg", author_id=50)
    message.channel.history = MagicMock(return_value=async_iter([unreviewed]))
    message.guild.me.id = bot_id

    result = await build_context(message)
    assert "- (none)" in result


@pytest.mark.asyncio
async def test_build_context_includes_approved_message(bot_id):
    message = make_message(content="new msg", author_id=123)
    approved = make_message(
        id=10, content="previously approved",
        author_id=50, reactions=[make_reaction("✅", me=True)],
    )
    message.channel.history = MagicMock(return_value=async_iter([approved]))
    message.guild.me.id = bot_id

    result = await build_context(message)
    assert "previously approved" in result


# ── process_message ──────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_process_message_returns_correction(bot_id):
    message = make_message(content="Yesterday I go to the gym.", author_id=123)
    message.channel.history = MagicMock(return_value=async_iter([]))
    message.guild.me.id = bot_id

    with patch("fluentify.pipeline._call_llm", new=AsyncMock(return_value="Yesterday I went to the gym.")):
        result = await process_message(message)

    assert result == "Yesterday I went to the gym."


@pytest.mark.asyncio
async def test_process_message_returns_perfect_for_natural_sentence(bot_id):
    message = make_message(content="I can't believe it's already March.", author_id=123)
    message.channel.history = MagicMock(return_value=async_iter([]))
    message.guild.me.id = bot_id

    with patch("fluentify.pipeline._call_llm", new=AsyncMock(return_value="PERFECT")):
        result = await process_message(message)

    assert result == "PERFECT"
