import pytest
from unittest.mock import MagicMock, AsyncMock
import discord


def make_reaction(emoji: str, me: bool) -> MagicMock:
    r = MagicMock()
    r.emoji = emoji
    r.me = me
    return r


def make_message(
    *,
    id: int = 1,
    content: str = "hello",
    author_id: int = 100,
    is_bot: bool = False,
    reactions: list | None = None,
    reference_message_id: int | None = None,
) -> MagicMock:
    msg = MagicMock(spec=discord.Message)
    msg.id = id
    msg.content = content
    msg.author = MagicMock()
    msg.author.id = author_id
    msg.author.bot = is_bot
    msg.reactions = reactions or []
    if reference_message_id is not None:
        msg.reference = MagicMock()
        msg.reference.message_id = reference_message_id
    else:
        msg.reference = None
    return msg


async def async_iter(items: list):
    """Helper: list → async generator for channel.history() mocking."""
    for item in items:
        yield item


@pytest.fixture
def bot_id() -> int:
    return 9999


@pytest.fixture
def correction_reaction():
    return make_reaction("✅", me=True)
