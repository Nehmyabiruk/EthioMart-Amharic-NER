# EthioMart Amharic NER  
**Named Entity Recognition for Amharic E-commerce Telegram Messages**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Transformers](https://img.shields.io/badge/🤗-Transformers-orange?logo=huggingface)](https://huggingface.co/docs/transformers)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Fine-tuned **XLM-RoBERTa** model to extract key entities (**Product**, **PRICE**, **LOC**) from Amharic Telegram e-commerce posts — powering vendor analytics and micro-lending decisions for **EthioMart**.

## 🎯 Project Goal

Transform messy Telegram vendor messages into structured data to:
- Centralize e-commerce information from multiple channels
- Identify promising vendors for micro-lending
- Support EthioMart's vision of becoming Ethiopia's unified Telegram e-commerce hub

**Target entities**:
- **Product** (B-Product / I-Product)
- **PRICE** (B-PRICE / I-PRICE)
- **LOCATION** (B-LOC / I-LOC)

## ✨ Features

- Telegram message scraping from 5+ Ethiopian e-commerce channels
- Text cleaning & basic rule-based pre-annotation
- Manual labeling in CoNLL format
- Fine-tuning XLM-RoBERTa for Amharic NER
- Per-class & overall evaluation metrics (seqeval)
- Automatic performance bar chart popup after training

## 📂 Project Structure

ethiomart-amharic-ner/
├── src/
│   ├── scraper.py              # Collects messages from Telegram channels
│   ├── preprocessing.py        # Cleans text & generates pre-annotated CSV
│   └── train_Ner.py            # Full NER training & evaluation pipeline
├── data/
│   ├── raw/
│   │   └── telegram_raw.csv    # Raw scraped messages (channel, text, views, date...)
│   ├── processed/
│   │   └── annotated_ner.csv   # Pre-annotated or manually corrected data
│   └── labels/
│       ├── train.conll         # Training set in CoNLL format
│       └── valid.conll         # Validation set
├── models/
│   └── xlm-roberta-amharic-ner-final/   # Saved fine-tuned model (not in git)
├── requirements.txt
├── .gitignore
└── README.md


## 🚀 Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/Nehmyabiruk/EthioMart-Amharic-NER.git
cd EthioMart-Amharic-NER

2. Install dependencies
Bashpip install -r requirements.txt
requirements.txt contents (recommended):
texttorch
transformers>=4.30
datasets>=2.14
evaluate
seqeval
pandas
numpy
matplotlib
seaborn
telethon
3. Prepare data

Run scraper to collect fresh messages:

Bashpython src/scraper.py

Pre-annotate (optional) or manually label more data:

Bashpython src/preprocessing.py

Make sure data/labels/train.conll and valid.conll exist with correct CoNLL format

4. Train the model
Bashpython src/train_Ner.py
After training finishes:

A bar chart window will pop up showing Precision / Recall / F1 / Accuracy
Model is saved to models/xlm-roberta-amharic-ner-final/

📊 Current Performance
(As of last run – very small labeled dataset)

Precision: ~0.00
Recall: ~0.00
F1-score: ~0.00–0.01
Accuracy: ~0.06–0.21

Note: These low numbers are expected with only ~50–60 labeled sentences.
With 300–1000+ properly labeled examples, F1 should reach 0.5–0.8 (especially for PRICE and LOC).
🔧 Next Steps / Roadmap

 Label 300–1000+ Telegram messages (use Label Studio / Doccano)
 Balance classes (more PRICE & LOC examples)
 Experiment with smaller/faster models (afro-xlmr-mini, distilbert-multilingual)
 Add confusion matrix & entity-level error analysis
 Integrate extracted entities with vendor metadata (views, frequency) → Lending Scorecard
 Add SHAP/LIME interpretability (Task 5)
 Deploy inference API or Gradio demo

📄 License
MIT License – feel free to use, modify, and share.
🙏 Acknowledgments

10 Academy KAIM 5 Week 4 Challenge
Hugging Face Transformers & Datasets teams
Telethon library for Telegram scraping

Built with ❤️ in Ethiopia for EthioMart 🚀