# hyperparameter_tuning_starter.py

import numpy as np
import matplotlib.pyplot as plt


# -------------------------
# Data generation
# -------------------------
def generate_data(n=200):
    # TODO: generate X ~ N(0, I)
    X = None

    # TODO: assign labels using circle rule
    y = None

    return X, y


# -------------------------
# Feature mapping
# -------------------------
def feature_map(X):
    # TODO: return [x1, x2, x1^2, x2^2]
    return None


# -------------------------
# Split data
# -------------------------
def split_data(X, y):
    # TODO: split into train/val/test (60/20/20)
    return None


# -------------------------
# Loss
# -------------------------
def compute_loss(X, y, w, b):
    return None


# -------------------------
# Gradients
# -------------------------
def compute_gradients(X, y, w, b):
    return None, None


# -------------------------
# Training
# -------------------------
def train(X, y, lr, epochs=200):
    # TODO: initialize parameters
    w = None
    b = None

    for _ in range(epochs):
        # TODO: update parameters
        pass

    return w, b


# -------------------------
# Prediction
# -------------------------
def predict(X, w, b):
    # TODO: return -1 or +1
    return None


# -------------------------
# Error
# -------------------------
def compute_error(y_true, y_pred):
    return np.mean(y_true != y_pred)


# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    X, y = generate_data()
    X = feature_map(X)

    data = split_data(X, y)
    X_train, y_train, X_val, y_val, X_test, y_test = data

    learning_rates = [0.001, 0.01, 0.1, 0.5]
    val_errors = []

    for lr in learning_rates:
        w, b = train(X_train, y_train, lr)

        preds_val = predict(X_val, w, b)
        err = compute_error(y_val, preds_val)

        val_errors.append(err)

    # TODO: choose best lr
    best_lr = None

    # TODO: retrain on training set
    # TODO: evaluate on test set

    # TODO: plot validation error vs lr