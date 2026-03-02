from flask import Flask
from threading import Thread

app = Flask(__name__)

@app.route('/')
def home():
    # Koyeb uses this message to determine that the bot is running successfully
    return "Fluentify Bot is Alive!"

def run():
    # Start a web server on port 8000, the default port used by Koyeb
    app.run(host='0.0.0.0', port=8000)

def keep_alive():
    # Run the web server in a separate thread so it does not interfere with the bot's main operation
    t = Thread(target=run)
    t.start()