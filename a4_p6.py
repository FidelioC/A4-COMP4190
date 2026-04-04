# distilbert_assignment_starter.py

# !pip install -q transformers datasets

import torch
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    AutoModelForSequenceClassification,
)
from datasets import load_dataset
from transformers import Trainer, TrainingArguments


# -------------------------
# Device
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)


# -------------------------
# MLM Task
# -------------------------
def run_mlm_example():
    sentence = "Machine learning is very [MASK]"

    # TODO: tokenize sentence and move to device
    inputs = tokenizer(sentence, return_tensors="pt")
    inputs = inputs.to(device)

    # TODO: load masked LM model and move to device
    model = None

    # TODO: run model and get logits
    logits = None

    # TODO: find index of [MASK]
    mask_index = None

    # TODO: compute probabilities and get top predictions
    # TODO: print reconstructed sentences (replace [MASK])


# -------------------------
# Load dataset
# -------------------------
def load_data():
    dataset = load_dataset("imdb")

    # TODO: select subset (500 train, 200 validation)
    return None


# -------------------------
# Tokenization
# -------------------------
def tokenize_function(example):
    # TODO: tokenize with truncation, padding, max_length=64
    return None


# -------------------------
# Training
# -------------------------
def train_model(dataset):

    # TODO: tokenize datasets
    tokenized_train = None
    tokenized_val = None

    # TODO: set format for torch
    # tokenized_train.set_format(...)
    # tokenized_val.set_format(...)

    # TODO: load classification model
    model = None

    # IMPORTANT: keep these hyperparameters fixed
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=1,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        logging_steps=20,
    )

    # TODO: define Trainer and train
    trainer = None

    return trainer


# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    run_mlm_example()

    dataset = load_data()
    trainer = train_model(dataset)

    # TODO: evaluate model and print accuracy
