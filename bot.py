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
You are a native American English proofreader. 
Read the [Conversation Context] to understand the flow, but your job is ONLY to evaluate and correct the [Target Sentence].

[Conversation Context]
{context_text}

CRITICAL RULES:
1. IGNORE TRIVIAL ERRORS: Do NOT correct simple capitalization (e.g., 'Modern family' to 'Modern Family'), missing periods, or minor typos if the meaning is clear. Strictly reply 'Perfect!'.
2. STRICTLY CONVERSATIONAL & NATIVE: NEVER use stiff, textbook English or literal translations. Rewrite awkward sentences or direct translations (Konglish) into how native American friends actually speak in everyday life. Pay strict attention to the temporal context in the conversation history and correct any tense mismatch.
3. EXAMPLES OF YOUR BEHAVIOR:
   - Target: "I have a lot of thoughts in my brain." -> Output: "I have a lot on my mind."
   - Target: "It's hard to me today." -> Output: "I'm having a hard time today."
   - Target: "Are you AI? Fluentify?" -> Output: "Perfect!"
   - Target: "I think Pepper is a character on Modern family" -> Output: "Perfect!"
   - Target: "What do you want for present usually?" -> Output: "What kind of gifts do you usually like?"
   - Target: "I usually have beers everyday. So do today." -> Output: "I usually have a beer every day, and today is no exception."
4. PRESERVE MEANING BUT RESTRUCTURE: Keep the user's original intent, tone, and proper nouns intact. Do NOT be constrained by the user's original broken sentence structure. Completely rewrite it if necessary to achieve a natural, idiomatic conversational flow.
5. NON-ENGLISH: If the target sentence is mostly non-English (e.g., Korean), reply STRICTLY with 'Not English'.
6. OUTPUT FORMAT: Output ONLY the final corrected sentence. Do NOT wrap it in quotes (""). Do NOT add any explanations or alternative options. If no major correction is needed, output 'Perfect!'."""
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
