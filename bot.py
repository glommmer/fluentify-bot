import discord
import os
import re
import textwrap
from dotenv import load_dotenv
from groq import AsyncGroq
from keep_alive import keep_alive

# Load environment variables
load_dotenv()
TOKEN = os.getenv("DISCORD_BOT_TOKEN")
LLM_API_KEY = os.getenv("LLM_API_KEY")

# Configure the client
client_ai = AsyncGroq(api_key=LLM_API_KEY)
intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)

# The specific channel where the bot operates
TARGET_CHANNEL_NAME = "english-chat"

# Retry using preconfigured fallback models
FALLBACK_MODELS = [
    "qwen/qwen3-32b",
    "openai/gpt-oss-20b",
    "llama-3.1-8b-instant",
    "llama-3.3-70b-versatile",
    "mixtral-8x7b-32768",
    "gemma2-9b-it",
]

# System prompt template
SYSTEM_PROMPT_TEMPLATE = textwrap.dedent("""\
You are a native American English conversational editor whose sole job is to turn a single non-native English utterance into a natural, idiomatic American-English sentence suitable for casual conversation in a chat (e.g., Discord).  
Read the [Conversation Context] to resolve tense, pronouns, or references, but use that context ONLY as described in rule 3.

[Conversation Context]
{context_text}

CRITICAL RULES:
1) PRIMARY TASK: Evaluate and CORRECT the [Target Sentence] to a single natural-sounding American-English sentence. Preserve the original meaning, intent, tone, and any proper nouns. You may restructure freely to achieve a natural conversational phrasing, but do NOT add new factual content or change the user's meaning.

2) IGNORE TRIVIAL ERRORS: If the target sentence only has trivial issues (minor capitalization, a missing period at the end, an obvious single-character typo where meaning is clear), respond exactly with:
Perfect!
Do NOT produce any other text, punctuation, or explanation.

3) CONTEXT USE (very important): Use the provided conversation context ONLY to:
   - choose correct tense/aspect (past/present/future),
   - select appropriate pronouns (he/she/they/it),
   - and keep references coherent (who or what the sentence refers to).
Do NOT rewrite the sentence to reflect extra information from the context, and do NOT invent or infer facts beyond what's necessary for tense/pronoun clarity.

4) CONVERSATIONAL & NATIVE: Always produce how a native American friend would say it in everyday speech. Avoid textbook/formal phrasing unless the user's original tone is clearly formal. Prefer natural contractions (I'm, don't, it's) and idiomatic collocations.

5) BREVITY & CLARITY: Prefer concise, clear phrasings. Avoid adding extra clauses unless required to preserve meaning. If the original is short, keep the corrected sentence short.

6) TENSE & ASPECT: Correct tense mismatches. If context indicates a different time frame than the target sentence implies, adjust the tense to match context.

7) PRESERVE REGISTER & EMOTION: Keep the user's tone (polite, casual, enthusiastic, angry, etc.). If tone is ambiguous, default to casual-friendly.

8) NON-ENGLISH DETECTION: If the target sentence is mostly non-English (e.g., primarily Korean or other non-English text) and not meaningfully correctable into English, output exactly:
Not English
Do NOT add anything else.

9) MULTI-SENTENCE INPUT: If the target contains multiple sentences, correct each and return them as a short sequence separated by a single space. Still follow all other rules (brevity, no added facts).

10) SLANG, NAMES, EMOJI: Preserve slang, nicknames, usernames, emojis, and proper nouns as-is unless they clearly prevent understanding. If a non-standard word makes the sentence awkward, replace it with a natural equivalent only if that preserves intent.

11) OUTPUT CLEANLINESS: Output ONLY the final corrected sentence (or the exact token Perfect! or Not English). Do NOT wrap it in quotes. Do NOT add explanations, alternatives, suggestions, markup, or extra whitespace. Trim leading/trailing spaces and newlines.

12) UNHANDLED OR UNCERTAIN: If the sentence is too fragmentary to correct without inventing meaning, make the minimal natural repair that preserves intent. If that's impossible, reply with Not English.

BEHAVIOR EXAMPLES (follow these patterns):
- Target: I have a lot of thoughts in my brain.
  -> I have a lot on my mind.
- Target: It's hard to me today.
  -> I'm having a hard time today.
- Target: Are you a bot?
  -> Perfect!
- Target: What do you want for present usually?
  -> What kind of gifts do you usually like?
- Target: I usually have beers everyday. So do today.
  -> I usually have a beer every day, and today is no exception.
- Target: (Korean) 저는 학교에 갑니다.
  -> Not English

IMPLEMENTATION NOTES (for the model's internal use):
- Favor a low-verbosity, high-precision rewrite style (short, idiomatic).
- Do not output multiple alternatives — produce exactly one corrected sentence.
- If context_text is empty, still correct the sentence using the rules above but do not guess unseen facts.

Now, given the [Conversation Context] above and a single [Target Sentence], output the corrected sentence following these rules and nothing else."""
)


async def correct_english(target_text: str, context_text: str) -> str:
    """
    Attempts to correct an English sentence using multiple AI models with a fallback mechanism.

    This function sends the target sentence to an AI model for correction while providing
    additional context through a system prompt. If the request fails (e.g., due to API
    rate limits or other errors), the function automatically retries using the next
    model listed in FALLBACK_MODELS.

    For each model:
    - A chat completion request is sent with a system prompt and the target sentence.
    - The response text is cleaned by removing any <think>...</think> tags.
    - The cleaned corrected sentence is returned immediately if successful.

    Error handling:
    - If a rate limit error (HTTP 429) occurs, the function switches to the next model.
    - If other errors occur (such as network issues), it also proceeds to the next model.

    If all models fail, the function returns "Error".

    Args:
        target_text (str): The sentence that needs to be corrected.
        context_text (str): Additional context used to guide the correction.

    Returns:
        str: The corrected sentence, or "Error" if all models fail.
    """
    for model_name in FALLBACK_MODELS:
        try:
            response = await client_ai.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT_TEMPLATE.format(
                            context_text=context_text
                        ),
                    },
                    {"role": "user", "content": f"[Target Sentence]\n{target_text}"},
                ],
                temperature=0.7,
            )

            raw_text = response.choices[0].message.content
            clean_text = re.sub(
                r"<think>.*?</think>", "", raw_text, flags=re.DOTALL
            ).strip()
            return clean_text

        except Exception as e:
            error_msg = str(e).lower()
            # When a rate limit (429) error occurs
            if "429" in error_msg or "rate limit" in error_msg:
                print(f"⚠️ [{model_name}] API rate limit exceeded. Switching to the next model...")
                continue  # Move on to the next model
            else:
                # If other errors occur (e.g., network issues), try the next model
                print(f"❌ [{model_name}] API Error: {str(e)}")
                continue

    return "Error"


@client.event
async def on_ready():
    """
    Event handler triggered when the Discord bot has successfully connected
    and finished logging in.

    This function runs once the client is ready to start interacting with
    Discord. It prints a confirmation message indicating that the bot has
    logged in and that Fluentify is now running.
    """
    print(f"✅ Logged in as: {client.user}")
    print("Fluentify is now running!")


@client.event
async def on_message(message):
    """
    Handles incoming messages and performs English correction within a specific channel.

    This event handler is triggered whenever a new message is posted. The function
    filters messages to ensure the bot does not respond to itself and only operates
    within the configured target channel.

    Workflow:
    1. Ignore messages sent by the bot itself.
    2. Ignore messages outside the configured TARGET_CHANNEL_NAME.
    3. Retrieve the previous 5 user messages from the channel to build conversation
       context (bot messages are excluded).
    4. Send the current message along with the collected context to the
       correct_english() function.
    5. Based on the AI response:
       - Do nothing if the message is not English.
       - Add a checkmark reaction if the sentence is already perfect.
       - Reply with the corrected sentence if a correction is provided.

    The bot replies directly in the main chat instead of creating a thread in order
    to keep the interaction simple and unobtrusive.
    """
    if message.author == client.user:
        return

    # Operate only in the specific channel
    if message.channel.name != TARGET_CHANNEL_NAME:
        return

    # Reaction indicating the message is being processed (optional)
    # await message.add_reaction('👀')

    # Retrieve the previous 5 (or more) messages to understand the context
    history_messages = []
    limit = 5
    async for msg in message.channel.history(limit=limit, before=message):
        # Exclude correction messages sent by the bot and collect only user conversations
        if not msg.author.bot:
            history_messages.append(f"{msg.author.display_name}: {msg.content}")

    # Sort in chronological order (oldest first)
    history_messages.reverse()
    context_text = "\n".join(history_messages)

    # Send both the context and the current text to the AI
    corrected_text = await correct_english(message.content, context_text)

    # await message.remove_reaction('👀', client.user)

    if "Not English" in corrected_text or corrected_text == "":
        return

    elif "Perfect!" in corrected_text:
        await message.add_reaction("✅")

    elif "Error" in corrected_text:
        pass

    else:
        # Reaction indicating correction completed
        # await message.add_reaction('📝')

        # Reply directly in the main chat with only the corrected sentence
        await message.reply(corrected_text)


keep_alive()
client.run(TOKEN)
