import os
import asyncio
import spacy
from dotenv import load_dotenv
from groq import AsyncGroq

# Load environment variables
load_dotenv()
TOKEN = os.getenv("DISCORD_BOT_TOKEN")
LLM_API_KEY = os.getenv("LLM_API_KEY")

# Configure the client
CLIENT_LLM = AsyncGroq(api_key=LLM_API_KEY)

# The specific channel where the bot operates
TARGET_CHANNEL_NAME = "english-chat"

# LLM tuning
LLM_GENERATE_TEMPERATURE = 0.2
LLM_TIMEOUT_SECONDS = 12

# Cap context to prevent token/latency blowups in Discord chats.
MAX_CONTEXT_CHARS = 1200
MAX_HISTORY_MSG_CHARS = 220
MAX_AUTHOR_HISTORY = 5

# Limit concurrent LLM calls to reduce rate-limit cascades & latency spikes.
LLM_CONCURRENCY = 3
LLM_SEMAPHORE = asyncio.Semaphore(LLM_CONCURRENCY)

# Retry using preconfigured fallback models
FALLBACK_MODELS = [
    "llama-3.3-70b-versatile",
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "qwen/qwen3-32b",
    "llama-3.1-8b-instant",
    "openai/gpt-oss-20b",
]
