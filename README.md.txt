EthioMart Amharic NER Project
This project is part of the 10 Academy Week 4 challenge to build a Named Entity Recognition (NER) system that extracts products, prices, and locations from Amharic-language Telegram messages.
🔍 Objectives

Scrape Amharic e-commerce Telegram messages.
Label entities using CoNLL format.
Fine-tune NER models like XLM-Roberta or AfroXLMR.
Enable EthioMart to become a centralized e-commerce hub.

⚙️ Setup
Install dependencies:
pip install -r requirements.txt

Configure .env:
TG_API_ID=your_api_id
TG_API_HASH=your_api_hash
phone=+2519XXXXXXX

Run the scraper:
python scraper.py

📁 Output

data/telegram_data.csv – scraped messages from channels like @Shageronlinestore, @EthioShop, @AddisMarket, @TedaShop, @HabeshaStore
labeling/labeled_data.conll – labeled entities

📄 License
MIT License