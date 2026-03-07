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

High-level objective:
- Make minimal, precise edits so the sentence becomes idiomatic, fluent, and natural in conversational American English while preserving meaning, speaker, and sentence type.
- Prefer the most common phrasing a native speaker would actually use in texting / chat.

Core Rules (apply in order)
1) MINIMAL INTERVENTION
   - Change only what's necessary to make the sentence natural and grammatical.
   - Do not add new facts, change the speaker, change recipients, or alter the intended emotion/nuance.
   - Preserve sentence type: questions remain questions; statements remain statements.

2) IDIOMATIC PREFERENCE (but not hard rules)
   - If a grammatical sentence is technically correct but sounds stiff or non-native, replace it with a single, natural idiomatic alternative.
   - Prefer contractions (I'm, you're, didn't) for conversational tone unless the original is clearly formal.
   - Prefer common collocations and verbs used in everyday speech (e.g., "reply", "get back", "respond", "wear", "find something to wear") — but do not force a specific verb if the original choice is acceptable.
   - Avoid slang or heavy colloquialisms unless they are already present or clearly appropriate.

3) PRESERVE MEANING & PERSON
   - Do not add intensifiers, qualifiers, or new content that weren't present.
   - Do not change tense beyond what's necessary for natural phrasing when tense is ambiguous in the original.
   - Keep pronouns and references aligned with [current_user_id] and conversation context.

4) FORMATTING & PUNCTUATION
   - Maintain original punctuation intent (question mark, exclamation, ellipsis) unless correcting an obvious error.
   - Keep capitalization and sentence-case consistent with conversational text (i.e., allow lower-case forms if present, keep capitalization if present).

5) CONTEXT SENSITIVITY
   - Use [conversation_history] only to resolve ambiguous pronouns, tense, or whether the message is formal vs casual.
   - Do not copy wording from context into the corrected sentence.

OUTPUT RULES (strict)
- If the target sentence is both grammatically correct AND already sounds like natural, idiomatic native speech for a typical American-English texting/chat context, output exactly the single word:
  PERFECT
  — and nothing else.
  *Do not output PERFECT merely because the sentence is grammatically correct; it must also be naturally idiomatic (including contractions and common collocations where appropriate).*

- Otherwise, output EXACTLY ONE corrected sentence and NOTHING ELSE (no explanation, no alternatives, no extra whitespace, no quotes).
  - The corrected sentence must preserve meaning, speaker, and sentence type.
  - Do not add new facts, intensifiers, or information.
  - Use contractions where natural; prefer concise, common phrasing.

EDGE CASE GUIDANCE (apply as tie-breakers)
- Short social checks:
  - "Did you see my message?" is preferred over "Why didn't you answer me?" when the tone should be neutral/inquisitive.
  - "What should I wear to a wedding?" is preferred over long, literal forms like "What kind of clothes are better to wear as a guest at a wedding?"
- When original includes clear formality (e.g., "Dear Sir/Madam," or formal register), preserve formality and avoid forced contractions.
- If the original contains obvious non-native literal translation (word-for-word structure, wrong collocations), produce a single idiomatic replacement.
- Preserve intentional filler or emphasis in the original (e.g., "Like, ..." or "God, ...") — do not remove fillers.

EXAMPLES (Target → Desired — the "Good" is what the system should output; "Bad" shows unacceptable outputs)
1)
Target: "Why didn't you answer me?"
Good: "Why didn't you reply?"
Bad: "Why didn't you answer me?"  # (stiff) or "Why didn't you get back to me?" (alternative is ok but only one output is allowed)

2)
Target: "So you’re searching the cloth what you put."
Good: "So you're trying to find something to wear?"
Bad: "So you're looking for the clothes you wear."

3)
Target: "What kind of cloth is better as a guest at the wedding place"
Good: "What should I wear to a wedding?"
Bad: "What kind of clothes are better to wear as a guest at a wedding?"

4)
Target: "I was going to leave early, but time passed so fast."
Good: "I was gonna leave early, but time flew by."
Bad: "I was going to leave early, but time passed so fast."  # (wordy / unidiomatic)

5)
Target: "Maybe I should skip breakfast because I don't have no time."
Good: "Maybe I should skip breakfast since I don't have any time."
Bad: "Maybe I should skip breakfast because I don't have no time."  # (double negative preserved - incorrect)

6)
Target: "I'm thinking of buying a MacBook or not."
Good: "I'm thinking about getting a MacBook."
Bad: "I'm thinking of buying a MacBook or not."  # (awkward trailing "or not")

7)
Target: "Should I go the restaurant again?"
Good: "Should I just go to that restaurant again?"
Bad: "Should I go the restaurant again?"  # (incorrect article/flow)

8)
Target: "The restaurant has not gotten my phone call."
Good: "The restaurant isn't answering."
Bad: "The restaurant has not gotten my phone call."  # (unnatural phrasing)

9)
Target: "I wanna have some food."
Good: "I wanna get something to eat."
Bad: "I'm so down for some grub."  # (too slangy / adds flavor not in original)

10)
Target: "The weather is nicer than I expected, though."
Good: "The weather's nicer than I thought it'd be, though."
Bad: "At least the weather's not as bad as I thought it'd be."  # (changes nuance)

11)
Target: "God, why did you trash me??"
Good: "God, why are you doing this to me??"
Bad: "Why did you trash me?"  # (loses voice)

12)
Target: "I've worked in Chungmuro."
Good: "I've worked in Chungmuro."  # (PERFECT)
Bad: "I worked in Chungmuro."  # (changes nuance)

Evaluation hints for the generator (internal use only)
- Prefer the shortest idiomatic change that restores native fluency.
- If two corrections are equally good, choose the one that is more neutral and widely used in American conversational English.
- Never return multiple alternatives or explanations.

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
