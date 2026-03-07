import discord
from fluentify.config import TARGET_CHANNEL_NAME
from fluentify.pipeline import process_message


def create_client() -> discord.Client:
    # Configure the client
    intents = discord.Intents.default()
    intents.message_content = True
    client = discord.Client(intents=intents)

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
        Handle incoming Discord messages and respond with grammar correction results.

        Ignores messages from the bot itself and any channel other than TARGET_CHANNEL_NAME.
        Delegates processing to process_message and responds based on the result:
        - "NOT_ENGLISH" or empty: silently ignored.
        - "PERFECT": reacts with a ✅ emoji.
        - "ERROR": silently ignored.
        - Any other string: replied to the original message as a correction.
        """
        if message.author == client.user:
            return

        if message.channel.name != TARGET_CHANNEL_NAME:
            return

        result = (await process_message(message)).strip()

        if result.upper() in {"NOT_ENGLISH", ""}:
            return

        if result.upper() == "PERFECT":
            await message.add_reaction("✅")
            return

        if result.upper() == "ERROR":
            return

        await message.reply(result)

    return client
