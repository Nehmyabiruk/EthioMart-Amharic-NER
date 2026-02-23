# src/train_Ner.py
# Optimized – skips network checks + pre-loads model + chart pops up & stays open

import os
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
)
import evaluate
from collections import Counter

# Force TkAgg backend – chart window pops up and stays open
matplotlib.use('TkAgg')

# ────────────────────────────────────────────────────────────────
# Pre-load model once (makes every run after much faster)
# ────────────────────────────────────────────────────────────────
print("Pre-loading cached XLM-RoBERTa (only slow first time)...")
_ = AutoModelForTokenClassification.from_pretrained(
    "xlm-roberta-base",
    local_files_only=True,
    trust_remote_code=False
)
print("Model pre-loaded.")

# ────────────────────────────────────────────────────────────────
# Labels
# ────────────────────────────────────────────────────────────────
labels = ["O", "B-Product", "I-Product", "B-PRICE", "I-PRICE", "B-LOC", "I-LOC"]
label2id = {lbl: idx for idx, lbl in enumerate(labels)}
id2label = {idx: lbl for lbl, idx in label2id.items()}

NUM_LABELS = len(labels)

print("Labels:", labels)
print("Mapping:", label2id)

# ────────────────────────────────────────────────────────────────
# CoNLL reader
# ────────────────────────────────────────────────────────────────
def read_conll(filepath):
    sentence_tokens = []
    sentence_tags = []
    current_tokens = []
    current_tags = []
    label_counter = Counter()

    with open(filepath, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                if current_tokens:
                    sentence_tokens.append(current_tokens)
                    sentence_tags.append(current_tags)
                    current_tokens = []
                    current_tags = []
                continue

            parts = line.split()
            if len(parts) < 2:
                continue

            token = parts[0]
            tag_str = parts[-1].strip()

            if tag_str not in label2id:
                print(f"Line {line_num}: Unknown tag '{tag_str}' → forced to 'O'")
                tag_id = 0
            else:
                tag_id = label2id[tag_str]

            label_counter[tag_str] += 1
            current_tokens.append(token)
            current_tags.append(tag_id)

    if current_tokens:
        sentence_tokens.append(current_tokens)
        sentence_tags.append(current_tags)

    print(f"\nLoaded {len(sentence_tokens)} sentences from {os.path.basename(filepath)}")
    print("Label distribution (strings):", dict(label_counter.most_common()))
    print("ID distribution:", Counter(tag for tags in sentence_tags for tag in tags))

    return {"tokens": sentence_tokens, "ner_tags": sentence_tags}

# ────────────────────────────────────────────────────────────────
# Load data
# ────────────────────────────────────────────────────────────────
train_data = read_conll("data/labels/train.conll")
valid_data = read_conll("data/labels/valid.conll")

train_dataset = Dataset.from_dict(train_data)
valid_dataset = Dataset.from_dict(valid_data)

print(f"\nTrain: {len(train_dataset)} | Valid: {len(valid_dataset)}")

# ────────────────────────────────────────────────────────────────
# Tokenization
# ────────────────────────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base", local_files_only=True)

def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"],
        is_split_into_words=True,
        truncation=True,
        max_length=512,
    )

    aligned_labels = []
    for i, tags in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []

        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(tags[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx

        aligned_labels.append(label_ids)

    tokenized_inputs["labels"] = aligned_labels
    return tokenized_inputs

tokenized_train = train_dataset.map(tokenize_and_align_labels, batched=True, remove_columns=["tokens"])
tokenized_valid = valid_dataset.map(tokenize_and_align_labels, batched=True, remove_columns=["tokens"])

# ────────────────────────────────────────────────────────────────
# Model – now fast because pre-loaded
# ────────────────────────────────────────────────────────────────
model = AutoModelForTokenClassification.from_pretrained(
    "xlm-roberta-base",
    num_labels=NUM_LABELS,
    id2label=id2label,
    label2id=label2id,
    local_files_only=True,
    trust_remote_code=False
)

# ────────────────────────────────────────────────────────────────
# Training args
# ────────────────────────────────────────────────────────────────
training_args = TrainingArguments(
    output_dir="../models/xlm-roberta-amharic-ner",
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=3,
    learning_rate=1e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,
    num_train_epochs=10,
    weight_decay=0.01,
    warmup_steps=100,
    logging_steps=20,
    load_best_model_at_end=True,
    metric_for_best_model="eval_f1",
    greater_is_better=True,
    fp16=False,
    report_to="none",
)

# ────────────────────────────────────────────────────────────────
# Metrics
# ────────────────────────────────────────────────────────────────
seqeval = evaluate.load("seqeval")

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=-1)

    true_preds = [[id2label[p] for p, l in zip(pred, lab) if l != -100] for pred, lab in zip(predictions, labels)]
    true_labs  = [[id2label[l] for l in lab if l != -100] for lab in labels]

    results = seqeval.compute(predictions=true_preds, references=true_labs, zero_division=0)
    
    print("\nPer-class metrics:")
    for entity in ["Product", "PRICE", "LOC"]:
        print(f"{entity}: P={results.get(f'{entity}', {}).get('precision', 0):.4f} | "
              f"R={results.get(f'{entity}', {}).get('recall', 0):.4f} | "
              f"F1={results.get(f'{entity}', {}).get('f1', 0):.4f}")

    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

# ────────────────────────────────────────────────────────────────
# Trainer
# ────────────────────────────────────────────────────────────────
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_valid,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# ────────────────────────────────────────────────────────────────
# Train & Evaluate + Chart
# ────────────────────────────────────────────────────────────────
trainer.train()
metrics = trainer.evaluate()
print("\nFinal Metrics:", metrics)

# Chart – pops up and stays open
df = pd.DataFrame({
    "Metric": ["Precision", "Recall", "F1", "Accuracy"],
    "Score": [
        metrics.get("eval_precision", 0.0),
        metrics.get("eval_recall", 0.0),
        metrics.get("eval_f1", 0.0),
        metrics.get("eval_accuracy", 0.0)
    ]
})

plt.figure(figsize=(8, 5))
sns.barplot(x="Metric", y="Score", data=df, palette="Blues_d")
plt.title("Amharic NER – Final Performance (Validation Set)")
plt.ylim(0, 1)
plt.ylabel("Score")
plt.xlabel("Metric")
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.show(block=True)  # waits until you close the window

print("Chart window closed – script finished.")
trainer.save_model("../models/xlm-roberta-amharic-ner-final")
tokenizer.save_pretrained("../models/xlm-roberta-amharic-ner-final")
print("Model saved.")