# Bert Model

An in-progress project for binary sentiment classification with a TF-IDF + Naive Bayes baseline and a fine-tuned BERT model. The structure is inspired by [chriskhanhtran/bert-for-sentiment-analysis](https://github.com/chriskhanhtran/bert-for-sentiment-analysis); this repository contains an original implementation and does not copy its notebook, results, or images.

> **Work in progress:** I am still working on this code. The training and prediction commands are available, but I have not published a dataset, trained weights, or verified benchmark results. Evaluation, documentation, and tuning are ongoing.

## What the code does

- Reads a labeled CSV with `text` and `label` columns. Labels must be `0` (negative/complaint) or `1` (positive/non-complaint).
- Uses a stratified train/validation split shared by both approaches.
- Trains a TF-IDF + Multinomial Naive Bayes baseline.
- Fine-tunes `google-bert/bert-base-uncased` with PyTorch and Hugging Face Transformers.
- Prints validation accuracy, F1, and ROC AUC (when defined), then saves the fitted model.
- Predicts one text string from a saved baseline or BERT model.

## Setup

Use Python 3.10–3.12. Create a virtual environment and install dependencies:

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
```

The first BERT run downloads model files from Hugging Face. Training can be slow on a CPU; a CUDA GPU is used automatically when available.

## Data format

Provide your own licensed CSV. Do not commit private text or model outputs. Example rows:

```csv
text,label
"The flight was delayed again",0
"The staff handled my request well",1
```

Both classes need at least two rows for a stratified split. The example above only illustrates the format and is too small for training or meaningful evaluation.

## Run

```bash
python bert_model.py train-baseline --data path/to/data.csv --output models/baseline.joblib
python bert_model.py train-bert --data path/to/data.csv --output models/bert --epochs 2
python bert_model.py predict --model models/baseline.joblib --text "The service was great"
python bert_model.py predict --model models/bert --text "My flight was cancelled"
```

Use `python bert_model.py --help` for all options. `--seed` controls the split and random seeds. The default split reserves 20% for validation. The script does not claim a benchmark score; scores are produced from the supplied dataset.

## Current status and next steps

- [x] Reproducible split and baseline training
- [x] BERT fine-tuning and saved-model prediction code
- [ ] Run and document a full experiment on a shareable dataset
- [ ] Tune hyperparameters and compare results across multiple seeds
- [ ] Add dataset-specific analysis and error review

The reference project studies airline complaint tweets and reports its own results. Those numbers are **not** results from this repository.
