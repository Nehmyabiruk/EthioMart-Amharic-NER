from telethon import TelegramClient
import csv
import os
from dotenv import load_dotenv

load_dotenv('.env')
api_id = os.getenv('TG_API_ID')
api_hash = os.getenv('TG_API_HASH')
phone = os.getenv('phone')

async def scrape_channel(client, channel_username, writer, media_dir):
    try:
        entity = await client.get_entity(channel_username)
        channel_title = entity.title
        async for message in client.iter_messages(entity, limit=500):
            media_path = None
            if message.media and hasattr(message.media, 'photo'):
                filename = f"{channel_username}_{message.id}.jpg"
                media_path = os.path.join(media_dir, filename)
                await client.download_media(message.media, media_path)
            writer.writerow([channel_title, channel_username, message.id, message.message, message.date, media_path])
    except Exception as e:
        print(f"Error scraping {channel_username}: {e}")

client = TelegramClient('scraping_session', api_id, api_hash)

async def main():
    await client.start()
    media_dir = 'data/photos'
    os.makedirs(media_dir, exist_ok=True)

    with open('data/telegram_data.csv', 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Channel Title', 'Channel Username', 'ID', 'Message', 'Date', 'Media Path'])
        channels = ['@Shageronlinestore', '@EthioShop', '@AddisMarket', '@TedaShop', '@HabeshaStore']
        for channel in channels:
            await scrape_channel(client, channel, writer, media_dir)
            print(f"Scraped data from {channel}")

with client:
    client.loop.run_until_complete(main())