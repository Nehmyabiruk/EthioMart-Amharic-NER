# src/preprocessing.py  (updated)1

import os
import pandas as pd
import re
from typing import List, Dict

os.makedirs("data/processed", exist_ok=True)

def clean_amharic_text(text):
    if pd.isna(text) or not text:
        return ""
    text = re.sub(r'[^\w\s፡።.,!?]', '', str(text))
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def get_preprocessed_data(annotated_csv_path="data/processed/annotated_ner.csv") -> List[Dict]:
    """
    Load or create annotated NER data in HF format: list of {"tokens": [...], "ner_tags": [...]}
    """
    if os.path.exists(annotated_csv_path):
        df_anno = pd.read_csv(annotated_csv_path)
        print(f"Loaded {len(df_anno)} annotated examples")
    else:
        print("No annotated file found → falling back to minimal example or raise error")
        # For now: return your dummy or raise
        return [
            {"tokens": ["አዲስ", "አበባ", "500", "ብር"], "ner_tags": [5, 6, 3, 3]},
            {"tokens": ["ስልክ", "ከፍተኛ", "1000", "ብር"], "ner_tags": [1, 2, 3, 3]}
        ]

    data = []
    for _, row in df_anno.iterrows():
        tokens = row["tokens"].split()   # assume you saved as space-separated string
        tags   = list(map(int, row["ner_tags"].split()))
        if len(tokens) == len(tags):
            data.append({"tokens": tokens, "ner_tags": tags})

    return data

# Optional: small helper to start annotation (run once)
def start_annotation(input_csv="data/raw/telegram_raw.csv", output_csv="data/processed/annotated_ner.csv", limit=50):
    df = pd.read_csv(input_csv)
    df["clean_text"] = df["text"].apply(clean_amharic_text)
    df = df[df["clean_text"].str.len() > 10].head(limit)  # skip very short/empty

    # Here you would manually annotate → for demo, we can simulate rule-based pre-annotation
    # Real annotation needs tool like Doccano, Label Studio, or even Excel + regex

    # Example very basic rule-based pre-fill (improve this!)
    annotated = []
    for text in df["clean_text"]:
        tokens = text.split()
        tags = [0] * len(tokens)  # default O

        for i, tok in enumerate(tokens):
            if re.match(r'^\d+(,\d+)?$', tok):  # number like 500 or 1,200
                tags[i] = 3  # B-PRICE
                if i+1 < len(tokens) and tokens[i+1] in ["ብር", "ብር።"]:
                    tags[i+1] = 4  # I-PRICE
            elif "አዲስ አበባ" in text or tok in ["አዲስ", "አዳማ", "ባህር", "ዳር"]:
                # simplistic location detection
                tags[i] = 5 if tags[i] == 0 else tags[i]

        annotated.append({"tokens": " ".join(tokens), "ner_tags": " ".join(map(str, tags))})

    pd.DataFrame(annotated).to_csv(output_csv, index=False)
    print(f"Saved {len(annotated)} pre-annotated examples to {output_csv}")

if __name__ == "__main__":
    # Run once to create starter annotation file
    start_annotation(limit=100)  # change limit as you annotate more
    # Then manually correct in Excel/Label Studio