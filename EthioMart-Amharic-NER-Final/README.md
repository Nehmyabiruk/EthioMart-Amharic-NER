# EthioMart Amharic NER Project

This repository contains the Week 4 submission for the EthioMart Amharic Named Entity Recognition (NER) pipeline, part of the 10 Academy KAIM Challenge.

## 🧠 Objectives
- Scrape Amharic-language product messages from Telegram vendors.
- Annotate named entities (products, prices, locations) using CoNLL format.
- Fine-tune a transformer-based NER model (AfroXLMR, XLM-Roberta).
- Deliver business insight to EthioMart for vendor analysis and micro-lending.

## 📦 Project Structure
```
EthioMart-Amharic-NER/
├── README.md
├── scraper.py
├── requirements.txt
├── .env.example
├── data/
│   ├── telegram_data.csv
│   └── photos/
└── labeling/
    └── labeled_data.conll
```

## ⚙️ Installation

```bash
pip install -r requirements.txt
```

Create a `.env` file:
```
TG_API_ID=your_api_id
TG_API_HASH=your_api_hash
phone=+2519XXXXXXX
```

Then run the scraper:
```bash
python scraper.py
```

## 📁 Output

- `telegram_data.csv`: Extracted messages from Telegram.
- `labeled_data.conll`: Labeled entities for fine-tuning.
