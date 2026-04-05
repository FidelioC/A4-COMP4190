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


# https://huggingface.co/docs/transformers/en/model_doc/distilbert
# -------------------------
# MLM Task
# -------------------------
def run_mlm_example():
    sentence = "Machine learning is very [MASK]"

    # tokenize sentence and move to device
    inputs = tokenizer(sentence, return_tensors="pt")
    inputs = inputs.to(device)

    # load masked LM model and move to device
    model = AutoModelForMaskedLM.from_pretrained(model_name)
    model = model.to(
        device
    )  # move model to GPU (cuda) or CPU depending on availability

    # run model and get logits
    with torch.no_grad():
        logits = model(**inputs).logits

    # find index of [MASK]
    mask_index = (
        (inputs.input_ids == tokenizer.mask_token_id)[0]
        .nonzero(as_tuple=True)[0]
        .item()
    )

    # compute probabilities and get top predictions
    probability = torch.softmax(
        logits[0, mask_index], dim=0
    )  # compute probabilities from logits convert using softmax
    top = torch.topk(probability, k=10)  # display top predicted words, e.g., 10

    # print reconstructed sentences (replace [MASK])
    for token_id, prob in zip(top.indices, top.values):
        word = tokenizer.decode([token_id.item()])
        reconstructed = sentence.replace("[MASK]", word)
        print(f"{reconstructed}  (prob: {prob*100:.2f}%)")


# -------------------------
# Load dataset
# -------------------------
def load_data():
    dataset = load_dataset("imdb")

    # select subset (500 train, 200 validation)
    train = dataset["train"].shuffle(seed=42).select(range(500))
    validation = dataset["test"].shuffle(seed=42).select(range(200))
    return train, validation


# -------------------------
# Tokenization
# -------------------------
def tokenize_function(example):
    # tokenize with truncation, padding, max_length=64
    inputs = tokenizer(
        example["text"],
        padding="max_length",
        truncation=True,
        max_length=64,
    )

    return inputs


# -------------------------
# Training
# -------------------------
def train_model(dataset):
    train, validation = dataset

    # tokenize datasets
    tokenized_train = train.map(tokenize_function)
    tokenized_val = validation.map(tokenize_function)

    # set format for torch
    # tokenized_train.set_format(...)
    # tokenized_val.set_format(...)
    # set format with torch, and specify columns to include input_ids, attention_mask, and label
    # input_ids: tokenized input representation of the text
    # attention_mask: indicates which tokens are padding (0) vs real tokens (1)
    # label: the sentiment label (0 or 1)
    tokenized_train.set_format(
        type="torch", columns=["input_ids", "attention_mask", "label"]
    )
    tokenized_val.set_format(
        type="torch", columns=["input_ids", "attention_mask", "label"]
    )

    # load classification model
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model = model.to(device)

    # IMPORTANT: keep these hyperparameters fixed
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=1,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        logging_steps=20,
    )

    # define Trainer and train
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
    )
    trainer.train()

    return trainer


# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    run_mlm_example()

    dataset = load_data()
    trainer = train_model(dataset)

    # evaluate model and print accuracy
    _, val = dataset
    tokenized_val = val.map(tokenize_function)
    tokenized_val.set_format(
        type="torch", columns=["input_ids", "attention_mask", "label"]
    )
    predictions = trainer.predict(tokenized_val)
    correct = 0
    for pred, label in zip(predictions.predictions, predictions.label_ids):
        if pred[0] < pred[1]:  # pred[1] > pred[0] means class 1 is predicted
            predicted_class = 1
        else:
            predicted_class = 0
        if predicted_class == label:
            correct += 1

    accuracy = correct / len(predictions.label_ids)
    print(f"Validation Accuracy: {accuracy:.4f}")
