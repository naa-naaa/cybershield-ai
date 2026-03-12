"""
Train Agent T3 — Emotion Classifier
Datasets: GoEmotions + TRAC (Aggression Identification)
Run: python training/train_t3_emotion.py
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

MODEL_NAME = "distilroberta-base"
OUTPUT_DIR = "./models/text_emotion"
MAX_LENGTH = 128
BATCH_SIZE = 16
EPOCHS = 3

# We use 7 emotions: anger, disgust, fear, joy, neutral, sadness, surprise
EMOTION_MAP = {0: "anger", 1: "disgust", 2: "fear", 3: "joy", 4: "neutral", 5: "sadness", 6: "surprise"}
NUM_LABELS = 7

# GoEmotions has 28 labels — map to our 7
GO_EMOTION_MAP = {
    "anger": 0, "annoyance": 0, "disapproval": 0,
    "disgust": 1,
    "fear": 2, "nervousness": 2,
    "admiration": 3, "amusement": 3, "approval": 3, "caring": 3,
    "excitement": 3, "gratitude": 3, "joy": 3, "love": 3,
    "optimism": 3, "pride": 3, "relief": 3,
    "neutral": 4, "realization": 4,
    "disappointment": 5, "embarrassment": 5, "grief": 5,
    "remorse": 5, "sadness": 5,
    "confusion": 6, "curiosity": 6, "desire": 6, "surprise": 6,
}


def load_go_emotions():
    try:
        ds = load_dataset("google-research-datasets/go_emotions", "simplified", split="train")
        texts, labels = [], []
        all_labels = ds.features["labels"].feature.names
        for ex in ds:
            if len(ex["labels"]) == 0:
                continue
            raw_label = all_labels[ex["labels"][0]]
            mapped = GO_EMOTION_MAP.get(raw_label, 4)
            texts.append(ex["text"])
            labels.append(mapped)
        return Dataset.from_dict({"text": texts, "label": labels})
    except Exception as e:
        print(f"GoEmotions load error: {e}")
        return None


def load_trac():
    """TRAC aggression identification dataset — maps to anger(0)"""
    try:
        ds = load_dataset("trac-2", split="train")
        texts, labels = [], []
        trac_map = {"NAG": 4, "CAG": 0, "OAG": 0}  # Non/Covert/Overt aggression
        for ex in ds:
            texts.append(ex["text"])
            labels.append(trac_map.get(ex["label"], 4))
        return Dataset.from_dict({"text": texts, "label": labels})
    except Exception as e:
        print(f"TRAC load error: {e}")
        return None


def tokenize(examples, tokenizer):
    return tokenizer(examples["text"], truncation=True, max_length=MAX_LENGTH)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "f1_macro": f1_score(labels, preds, average="macro"),
        "accuracy": accuracy_score(labels, preds),
    }


def main():
    print("Loading emotion datasets...")
    datasets_list = []
    for loader in [load_go_emotions, load_trac]:
        ds = loader()
        if ds is not None:
            datasets_list.append(ds)

    if not datasets_list:
        print("No datasets loaded.")
        return

    combined = concatenate_datasets(datasets_list).shuffle(seed=42)
    split = combined.train_test_split(test_size=0.1, seed=42)
    print(f"Train: {len(split['train'])}, Eval: {len(split['test'])}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenized_train = split["train"].map(lambda x: tokenize(x, tokenizer), batched=True)
    tokenized_eval = split["test"].map(lambda x: tokenize(x, tokenizer), batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS)
    model.config.id2label = EMOTION_MAP
    model.config.label2id = {v: k for k, v in EMOTION_MAP.items()}

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
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

    print("Training T3 emotion model...")
    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Model saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
