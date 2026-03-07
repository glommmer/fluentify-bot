from keep_alive import keep_alive
from fluentify.config import TOKEN
from fluentify.discord_app import create_client


def main():
    keep_alive()
    client = create_client()
    client.run(TOKEN)


if __name__ == "__main__":
    main()
