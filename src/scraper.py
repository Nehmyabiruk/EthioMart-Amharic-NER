from telethon.sync import TelegramClient
import pandas as pd
import os

api_id = 37709643 
api_hash = "b81d94ed45968ac90c09b66788191a6f"

channels = [
    "@ZemenExpress",
    "@Shewabrand",
    "@ethio_brand_collection",
    "@gebeyaadama",
    "@nevacomputer"
]
all_messages = []

with TelegramClient("session_name", api_id, api_hash) as client:
    for channel in channels:
        print(f"Fetching messages from {channel}...")

        for message in client.iter_messages(channel, limit=500):

            if message.text:
                all_messages.append({
                    "channel": channel,
                    "message_id": message.id,
                    "text": message.text,
                    "date": message.date,
                    "views": message.views,
                    "image_path": None
                })

    df = pd.DataFrame(all_messages)

    df.drop_duplicates(subset=["text"], inplace=True)
    df.dropna(subset=["text"], inplace=True)

    os.makedirs("data/raw", exist_ok=True)
    df.to_csv("data/raw/telegram_raw.csv", index=False)

    print("data collection saved")