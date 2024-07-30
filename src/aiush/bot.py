# This example requires the 'message_content' intent.
import os
import discord
from aiush.run import ChatModel 
from dotenv import load_dotenv
load_dotenv()


prompt = """
    You are a human. Provide realistic and coherent responses. Keep answers concise. No emojis or hahaha"
    """
chatbot = ChatModel("meta-llama/Meta-Llama-3.1-8B-Instruct",
                  "./models/meta-llama/Meta-Llama-3.1-8B-Instruct/",
                  prompt)

intents = discord.Intents.default()
intents.message_content = True

DISCORD_KEY = os.environ["DISCORD_API_KEY"]

client = discord.Client(intents=intents)

@client.event
async def on_ready():
    print(f'We have logged in as {client.user}')

@client.event
async def on_message(message):
    if message.author == client.user:
        return
    if message != "":
        user_msg = {
            "role": "user",
            "content": message.content
        }
        chatbot.messages.append(user_msg)

        response = chatbot.run()
        assistant_msg = {
            "role": "assistant",
            "content": response[-1]["content"]
        }
        chatbot.messages.append(assistant_msg)
        await message.channel.send(response[-1]["content"])

client.run(DISCORD_KEY)
