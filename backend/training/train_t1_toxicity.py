"""
Train Agent T1 — RoBERTa Toxicity Classifier
Datasets: Jigsaw Toxic Comments + Kaggle Cyberbullying
Run: python training/train_t1_toxicity.py
"""

import os
import torch
import numpy as np
from datasets import load_dataset, concatenate_datasets, Dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, DataCollatorWithPadding,
)
from sklearn.metrics import f1_score, accuracy_score
import pandas as pd

MODEL_NAME = "roberta-base"
OUTPUT_DIR = "./models/text_toxicity"
MAX_LENGTH = 128
BATCH_SIZE = 16
EPOCHS = 3
LABEL_MAP = {0: "safe", 1: "harassment", 2: "threat", 3: "hate_speech", 4: "identity_hate"}
NUM_LABELS = 5


def load_jigsaw():
    """Load Jigsaw dataset from Kaggle (download manually first)"""
    try:
        df = pd.read_csv("./data/raw/jigsaw_train.csv")
        labels = []
        for _, row in df.iterrows():
            if row.get("threat", 0):
                labels.append(2)
            elif row.get("identity_hate", 0):
                labels.append(4)
            elif row.get("obscene", 0) or row.get("insult", 0):
                labels.append(1)
            elif row.get("toxic", 0):
                labels.append(1)
            else:
                labels.append(0)
        return Dataset.from_dict({"text": df["comment_text"].tolist(), "label": labels})
    except FileNotFoundError:
        print("Jigsaw CSV not found. Download from: kaggle competitions download jigsaw-toxic-comment-classification-challenge")
        return None


def load_cyberbullying_kaggle():
    """Load Kaggle cyberbullying classification dataset"""
    try:
        df = pd.read_csv("./data/raw/cyberbullying_tweets.csv")
        label_mapping = {
            "not_cyberbullying": 0,
            "gender": 1,
            "religion": 3,
            "age": 1,
            "ethnicity": 3,
            "other_cyberbullying": 1,
        }
        labels = [label_mapping.get(l, 0) for l in df["cyberbullying_type"].tolist()]
        return Dataset.from_dict({"text": df["tweet_text"].tolist(), "label": labels})
    except FileNotFoundError:
        print("Cyberbullying CSV not found. Download from Kaggle.")
        return None


def load_hatexplain():
    try:
        ds = load_dataset("hatexplain", split="train")
        texts, labels = [], []
        for ex in ds:
            text = " ".join(ex["post_tokens"])
            label_list = ex["annotators"]["label"]
            majority = max(set(label_list), key=label_list.count)
            label_map_h = {0: 0, 1: 3, 2: 0}  # normal:0, hatespeech:3, offensive:1
            texts.append(text)
            labels.append(label_map_h.get(majority, 0))
        return Dataset.from_dict({"text": texts, "label": labels})
    except Exception as e:
        print(f"HateXplain load error: {e}")
        return None


def tokenize(examples, tokenizer):
    return tokenizer(examples["text"], truncation=True, max_length=MAX_LENGTH)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "f1_macro": f1_score(labels, predictions, average="macro"),
        "accuracy": accuracy_score(labels, predictions),
    }


def main():
    print("Loading datasets...")
    datasets_list = []
    for loader in [load_jigsaw, load_cyberbullying_kaggle, load_hatexplain]:
        ds = loader()
        if ds is not None:
            datasets_list.append(ds)

    if not datasets_list:
        print("No datasets found. Please download at least one dataset.")
        return

    combined = concatenate_datasets(datasets_list)
    combined = combined.shuffle(seed=42)
    split = combined.train_test_split(test_size=0.1, seed=42)

    print(f"Train: {len(split['train'])}, Eval: {len(split['test'])}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenized_train = split["train"].map(lambda x: tokenize(x, tokenizer), batched=True)
    tokenized_eval = split["test"].map(lambda x: tokenize(x, tokenizer), batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=NUM_LABELS
    )
    model.config.id2label = LABEL_MAP
    model.config.label2id = {v: k for k, v in LABEL_MAP.items()}

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        logging_steps=100,
        fp16=torch.cuda.is_available(),
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )

    print("Training T1 toxicity model...")
    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Model saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
