import discord
import asyncio
import textwrap
from fluentify.config import (
    CLIENT_LLM,
    LLM_SEMAPHORE,
    LLM_TIMEOUT_SECONDS,
    LLM_GENERATE_TEMPERATURE,
    FALLBACK_MODELS,
    MAX_HISTORY_MSG_CHARS,
    MAX_AUTHOR_HISTORY,
)
from fluentify.core import (
    _sanitize_llm_output,
    _trim_context,
    _is_trivial_change,
    normalize_for_compare,
)

# prompt template
GENERATOR_PROMPT = textwrap.dedent("""\
You are an expert in natural, conversational American English text messaging.
Your task is to fix the [target_sentence] so it sounds natural, like something a real person would type in everyday conversation.

Context usage:
- [conversation_history] shows the recent conversation in chronological order, identified by user IDs.
- [current_user_id] tells you WHO is speaking the Target Sentence.
- Use context ONLY to understand the situation, tense, and pronouns.
- Do NOT drag words from the previous context into the target sentence.

Correction Rules:
1. MINIMAL INTERVENTION: Fix ONLY what is unnatural or grammatically wrong.
   Do NOT rephrase sentences that are already understandable.
   Do NOT use slang, heavy colloquialisms, or overly casual expressions unless they appear in the original.
2. PRESERVE CORE MEANING & PERSON: The author of the Target Sentence is identified by [ID of the current message author].
   Do NOT change the original intent, emotion, or nuance.
   Do NOT add information that is not in the original sentence.
   Do NOT change who is speaking or who is being addressed.
3. PRESERVE SENTENCE TYPE: Questions stay questions (?). Statements stay statements.

EXAMPLES OF THE PERFECT BALANCE:
- Target: "It's better what I call there."
  Good: "I should probably call them." or "I'd better give them a call."
  Bad (Too passive): "It's better if I call them."

- Target: "The restaurant has not gotten my phone call."
  Good: "The restaurant isn't answering." or "They're not picking up."
  Bad (Too creative): "No wonder they never got my call."

- Target: "God, why did you trash me??"
  Good: "God, why are you doing this to me??" or "God, why did you ditch me??"
  Bad (Too passive): "Why did you trash me?"

- Target: "Should I go the restaurant again?"
  Good: "Should I just go to that restaurant again?"
  Bad (Changing person): "Want to grab food from that restaurant again?"

- Target: "I was going to leave early, but time passed so fast."
  Good: "I was gonna leave early, but time flew by."
  Bad (Adding new info): "I totally lost track of time and now it's too late to leave early."

- Target: "Maybe I should skip breakfast because I don't have no time."
  Good: "Maybe I should skip breakfast since I don't have any time."
  Bad (Changing intent): "I'll just have to skip breakfast, I'm running super behind."

- Target: "The weather is nicer than I expected, though."
  Good: "The weather's nicer than I thought it'd be, though."
  Bad (Flipping nuance): "At least the weather's not as bad as I thought it'd be."

- Target: "I wanna have some food."
  Good: "I wanna get something to eat."
  Bad (Too slangy): "I'm so down for some grub."

OUTPUT RULE:
Output the single word "PERFECT" — and nothing else — if ANY of the following apply:
- The sentence is grammatically correct AND already sounds like natural native speech.
- The ONLY difference would be swapping a preposition or article where both are equally
  correct (e.g. "thinking about" vs "thinking of", "on the weekend" vs "at the weekend").
- The ONLY difference would be adding an intensifier not present in the original
  (e.g. inserting "way", "really", "so", "just" where none existed).
- The sentence contains an intentional colloquial filler
  (e.g. "Like, ...", "I mean, ...", "You know, ...") — never remove fillers.

Do NOT output "Perfect!" just because a sentence is understandable.
If a more natural idiomatic expression exists (e.g. "flying by" for "going so fast"),
apply it — that is a valid correction.

Otherwise, output EXACTLY ONE corrected sentence and nothing else. No explanation.
Do NOT add words, intensifiers, or details that are not in the original sentence.

{context_text}
""")


async def _call_llm(
    model_name: str, system_prompt: str, user_prompt: str, temperature: float
) -> str:
    """
    Send a chat completion request to the specified LLM and return the response text.

    Acquires LLM_SEMAPHORE before dispatching the request to limit concurrency,
    and enforces a timeout of LLM_TIMEOUT_SECONDS on the underlying API call.
    The raw response is passed through _sanitize_llm_output before being returned.

    Args:
        model_name: Identifier of the model to use (e.g., "gpt-4o").
        system_prompt: System-role instruction that sets model behavior and context.
        user_prompt: User-role input or query to be answered by the model.
        temperature: Sampling temperature in [0, 2]; lower values yield more
                     deterministic outputs, higher values more diverse ones.

    Returns:
        Sanitized response text from the model, with control characters and
        extraneous whitespace removed.

    Raises:
        asyncio.TimeoutError: If the API call does not complete within
            LLM_TIMEOUT_SECONDS.
        openai.APIError: If the upstream API returns an error response.
    """
    async with LLM_SEMAPHORE:
        response = await asyncio.wait_for(
            CLIENT_LLM.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
            ),
            timeout=LLM_TIMEOUT_SECONDS,
        )
    raw_text = response.choices[0].message.content
    return _sanitize_llm_output(raw_text)


async def _run_llm_with_fallback(
    system_prompt: str, user_prompt: str, temperature: float
) -> str:
    """
    Attempt a chat completion request across multiple models in fallback order.

    Iterates through FALLBACK_MODELS sequentially, falling through to the next
    model on any of the following conditions: asyncio.TimeoutError, a rate-limit
    response (HTTP 429), or any other exception from _call_llm. Returns the
    first successful response, or the sentinel string "Error" if all models
    in FALLBACK_MODELS are exhausted.

    Args:
        system_prompt: System-role instruction that sets model behavior and context.
        user_prompt: User-role input or query to be answered by the model.
        temperature: Sampling temperature in [0, 2]; lower values yield more
                     deterministic outputs, higher values more diverse ones.

    Returns:
        Sanitized response text from the first successful model,
        or "Error" if every model in FALLBACK_MODELS fails.
    """
    for model_name in FALLBACK_MODELS:
        try:
            return await _call_llm(model_name, system_prompt, user_prompt, temperature)
        except asyncio.TimeoutError:
            print(
                f"⏳ [{model_name}] Timeout after {LLM_TIMEOUT_SECONDS}s. Switching to the next model..."
            )
            continue
        except Exception as e:
            error_msg = str(e).lower()
            if "429" in error_msg or "rate limit" in error_msg:
                print(
                    f"⚠️ [{model_name}] API rate limit exceeded. Switching to the next model..."
                )
                continue
            print(f"❌ [{model_name}] API Error: {str(e)}")
            continue
    return "Error"


async def generate_correction(target_text: str, context_text: str) -> str:
    """
    Generate a grammatically corrected version of the target text using an LLM,
    with fallback across multiple models.

    Trims context_text via _trim_context before dispatching, then iterates through
    FALLBACK_MODELS sequentially, returning the first successful response.
    Falls through to the next model on asyncio.TimeoutError, a rate-limit response
    (HTTP 429), or any other exception from _call_llm. Returns "Error" if all
    models in FALLBACK_MODELS are exhausted.

    Args:
        target_text: The original sentence to be corrected.
        context_text: Surrounding conversation context provided to the LLM
                      to improve correction accuracy; trimmed internally before use.

    Returns:
        The corrected sentence from the first successful model,
        or "Error" if every model in FALLBACK_MODELS fails.
    """
    context_text = _trim_context(context_text)
    for model_name in FALLBACK_MODELS:
        try:
            candidate = await _call_llm(
                model_name=model_name,
                system_prompt=GENERATOR_PROMPT.format(context_text=context_text),
                user_prompt=f"[target_sentence]\n{target_text}",
                temperature=LLM_GENERATE_TEMPERATURE,
            )

            if candidate.strip().upper() == "PERFECT":
                return "PERFECT"

            if normalize_for_compare(candidate) == normalize_for_compare(target_text):
                return "PERFECT"

            # TO-DO: whitelist-based trivial check is context-unaware — move this logic to the prompt.
            # if _is_trivial_change(target_text, candidate):
            #     return "PERFECT"

            return candidate
        except asyncio.TimeoutError:
            print(
                f"⏳ [{model_name}] Timeout after {LLM_TIMEOUT_SECONDS}s. Switching to the next model..."
            )
            continue
        except Exception as e:
            error_msg = str(e).lower()
            if "429" in error_msg or "rate limit" in error_msg:
                print(
                    f"⚠️ [{model_name}] API rate limit exceeded. Switching to the next model..."
                )
                continue
            print(f"❌ [{model_name}] API Error: {str(e)}")
            continue
    return "Error"


def _truncate_history_text(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return ""
    if len(text) > MAX_HISTORY_MSG_CHARS:
        return text[:MAX_HISTORY_MSG_CHARS].rstrip() + "…"
    return text


def _get_current_bot_user_id(message: discord.Message) -> int | None:
    """
    Best-effort way to identify the currently running bot user.
    Works in normal guild text channels.
    """
    try:
        if message.guild and message.guild.me:
            return message.guild.me.id
    except Exception:
        pass
    return None


def _has_bot_approval(msg: discord.Message) -> bool:
    """
    Returns True only if the current bot itself added a ✅ reaction.
    Uses Reaction.me, which indicates whether the current client reacted.
    """
    return any(str(r.emoji) == "✅" and getattr(r, "me", False) for r in msg.reactions)


def _build_bot_corrections(
    history_msgs: list[discord.Message],
    bot_user_id: int | None,
) -> dict[int, str]:
    """
    Build {original_message_id: corrected_text} from recent bot reply messages.

    Assumes history_msgs are ordered newest -> oldest.
    Keeps the first seen correction so the newest correction wins.
    """
    corrections: dict[int, str] = {}

    if bot_user_id is None:
        return corrections

    for msg in history_msgs:
        if not msg.author.bot:
            continue
        if msg.author.id != bot_user_id:
            continue
        if not msg.reference or not msg.reference.message_id:
            continue

        correction = (msg.content or "").strip()
        if not correction:
            continue

        original_message_id = msg.reference.message_id
        if original_message_id not in corrections:
            corrections[original_message_id] = correction

    return corrections


def _reviewed_text_for_message(
    msg: discord.Message,
    bot_corrections: dict[int, str],
    *,
    allow_unreviewed_fallback: bool = False,
) -> tuple[str | None, str]:
    """
    Return the best text to use for context for a message.

    Priority:
    1) bot-corrected text
    2) original text if bot approved with ✅
    3) original text only if allow_unreviewed_fallback=True

    Returns:
        (text_or_none, source)
        source is one of: corrected / approved / raw / none
    """
    original = (msg.content or "").strip()
    if not original:
        return None, "none"

    if msg.id in bot_corrections:
        return _truncate_history_text(bot_corrections[msg.id]), "corrected"

    if _has_bot_approval(msg):
        return _truncate_history_text(original), "approved"

    if allow_unreviewed_fallback:
        return _truncate_history_text(original), "raw"

    return None, "none"


async def build_context(message: discord.Message) -> str:
    """
    Build a structured conversational context for the LLM.

    author_context:
      - only recent messages from the same author that the bot has already reviewed
      - corrected messages use the bot's corrected text
      - ✅-approved messages use the user's original text

    partner_context:
      - if the current message is a reply, use the replied-to message
      - otherwise, for very short reactive messages (<= 6 words), use the most recent
        adjacent message from another user
      - partner text prefers reviewed text, but can fall back to raw text

    dialogue_state:
      - target speaker
      - speech act
      - polarity
      - relation metadata
    """
    author_lines: list[str] = []

    # Recent history: newest -> oldest
    history_msgs: list[discord.Message] = []
    async for msg in message.channel.history(limit=25, before=message):
        history_msgs.append(msg)

    bot_user_id = _get_current_bot_user_id(message)
    bot_corrections = _build_bot_corrections(history_msgs, bot_user_id)

    for msg in history_msgs:
        if msg.author.bot:
            continue

        reviewed_text, source = _reviewed_text_for_message(
            msg,
            bot_corrections,
            allow_unreviewed_fallback=False,
        )
        if not reviewed_text:
            continue

        author_lines.append(f"- {msg.author.id}: {reviewed_text}")

        if len(author_lines) >= MAX_AUTHOR_HISTORY:
            break

    author_lines.reverse()

    context_text = textwrap.dedent(f"""\
[conversation_history]
{chr(10).join(author_lines) if author_lines else "- (none)"}

[current_user_id]
{message.author.id}
    """).strip()

    return _trim_context(context_text)


async def process_message(message: discord.Message) -> str:
    """
    Build a structured conversational context string for the LLM.

    Fetches up to 25 messages preceding the current message from the channel
    history, then collects up to MAX_AUTHOR_HISTORY non-bot messages that have
    a reviewed text record (via _reviewed_text_for_message with no unreviewed
    fallback). Messages are assembled in chronological order and formatted as
    two labeled sections: [conversation_history] containing each message as
    "- {author_id}: {reviewed_text}", and [current_user_id] containing the
    author ID of the triggering message. The final string is passed through
    _trim_context before being returned.

    Args:
        message: The incoming Discord message that triggered context assembly;
                 its channel history is queried and its author ID is embedded
                 in the [current_user_id] section.

    Returns:
        A trimmed, LLM-ready context string with [conversation_history] and
        [current_user_id] sections, suitable for direct injection into a prompt.
    """
    context_text = await build_context(message)
    corrected = await generate_correction(message.content, context_text)

    return corrected
