"""Train and use a binary sentiment classifier. See README.md for data format."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


LABEL_NAMES = {0: "negative/complaint", 1: "positive/non-complaint"}


def load_data(path: Path):
    import pandas as pd

    data = pd.read_csv(path)
    if not {"text", "label"}.issubset(data.columns):
        raise ValueError("CSV must have 'text' and 'label' columns")
    if data[["text", "label"]].isna().any().any():
        raise ValueError("'text' and 'label' cannot contain missing values")
    if not data["label"].isin([0, 1]).all():
        raise ValueError("labels must be 0 or 1")
    if data["label"].value_counts().reindex([0, 1], fill_value=0).min() < 2:
        raise ValueError("each class needs at least two rows")
    texts = data["text"].astype(str)
    if texts.str.strip().eq("").any():
        raise ValueError("text cannot be empty")
    return texts.tolist(), data["label"].astype(int).tolist()


def split_data(texts, labels, test_size: float, seed: int):
    from sklearn.model_selection import train_test_split

    try:
        return train_test_split(
            texts, labels, test_size=test_size, random_state=seed, stratify=labels
        )
    except ValueError as exc:
        raise ValueError(
            "cannot make a stratified split; add more rows or adjust --validation-size"
        ) from exc


def report_metrics(labels, probabilities):
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

    predictions = (probabilities >= 0.5).astype(int)
    metrics = {
        "accuracy": float(accuracy_score(labels, predictions)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
    }
    if len(set(labels)) == 2:
        metrics["roc_auc"] = float(roc_auc_score(labels, probabilities))
    print(json.dumps(metrics, indent=2))


def train_baseline(args):
    import joblib
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.pipeline import make_pipeline

    texts, labels = load_data(args.data)
    x_train, x_val, y_train, y_val = split_data(
        texts, labels, args.validation_size, args.seed
    )
    model = make_pipeline(TfidfVectorizer(ngram_range=(1, 2)), MultinomialNB())
    model.fit(x_train, y_train)
    import numpy as np

    probabilities = np.asarray(model.predict_proba(x_val))[:, 1]
    report_metrics(y_val, probabilities)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.output)
    print(f"Saved baseline to {args.output}")


def train_bert(args):
    import numpy as np
    import torch
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    texts, labels = load_data(args.data)
    x_train, x_val, y_train, y_val = split_data(
        texts, labels, args.validation_size, args.seed
    )
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.checkpoint, num_labels=2,
        id2label={0: LABEL_NAMES[0], 1: LABEL_NAMES[1]},
        label2id={LABEL_NAMES[0]: 0, LABEL_NAMES[1]: 1},
    )

    class TextDataset(Dataset):
        def __init__(self, items, targets):
            self.items = items
            self.targets = targets

        def __len__(self):
            return len(self.items)

        def __getitem__(self, index):
            encoded = tokenizer(
                self.items[index], truncation=True, max_length=args.max_length
            )
            encoded["labels"] = self.targets[index]
            return encoded

    def collate(batch):
        targets = [item.pop("labels") for item in batch]
        padded = tokenizer.pad(batch, padding=True, return_tensors="pt")
        padded["labels"] = torch.tensor(targets, dtype=torch.long)
        return padded

    train_loader = DataLoader(
        TextDataset(x_train, y_train), batch_size=args.batch_size,
        shuffle=True, collate_fn=collate,
    )
    val_loader = DataLoader(
        TextDataset(x_val, y_val), batch_size=args.batch_size,
        collate_fn=collate,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad()
            outputs = model(**batch)
            outputs.loss.backward()
            optimizer.step()
            total_loss += outputs.loss.item()
        print(f"Epoch {epoch + 1}/{args.epochs}: training loss {total_loss / len(train_loader):.4f}")

    model.eval()
    probabilities = []
    with torch.no_grad():
        for batch in val_loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            logits = model(**batch).logits
            probabilities.extend(torch.softmax(logits, dim=1)[:, 1].cpu().tolist())
    report_metrics(y_val, np.asarray(probabilities))
    args.output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    print(f"Saved BERT model to {args.output}")


def predict(args):
    if args.model.is_file():
        import joblib

        model = joblib.load(args.model)
        probability = float(model.predict_proba([args.text])[0][1])
    elif args.model.is_dir():
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForSequenceClassification.from_pretrained(args.model)
        model.eval()
        inputs = tokenizer(args.text, return_tensors="pt", truncation=True)
        with torch.no_grad():
            probability = float(torch.softmax(model(**inputs).logits, dim=1)[0, 1])
    else:
        raise FileNotFoundError(args.model)
    result = {"label": LABEL_NAMES[int(probability >= 0.5)], "positive_probability": probability}
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("train-baseline", "train-bert"):
        command = commands.add_parser(name)
        command.add_argument("--data", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--validation-size", type=float, default=0.2)
        command.add_argument("--seed", type=int, default=42)
        if name == "train-bert":
            command.add_argument("--checkpoint", default="google-bert/bert-base-uncased")
            command.add_argument("--epochs", type=int, default=2)
            command.add_argument("--batch-size", type=int, default=8)
            command.add_argument("--learning-rate", type=float, default=2e-5)
            command.add_argument("--max-length", type=int, default=128)
    command = commands.add_parser("predict")
    command.add_argument("--model", type=Path, required=True)
    command.add_argument("--text", required=True)
    args = parser.parse_args()
    if args.command.startswith("train"):
        if not 0 < args.validation_size < 1:
            parser.error("--validation-size must be between 0 and 1")
        if args.command == "train-bert" and (
            args.epochs < 1 or args.batch_size < 1 or args.max_length < 2
            or args.learning_rate <= 0
        ):
            parser.error("BERT training options must be positive")
    {"train-baseline": train_baseline, "train-bert": train_bert, "predict": predict}[
        args.command
    ](args)


if __name__ == "__main__":
    main()
