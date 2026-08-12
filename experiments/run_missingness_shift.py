from src.data.loaders import load_amazon_google
from src.matcher.save_model import load_matcher
from src.matcher.predict import create_feature_matrix, predict_scores

from src.drift.shifts.missingness_shift import (
    apply_missingness_shift
)

import pandas as pd


A,B,train,valid,test = load_amazon_google(
    "data/raw/amazon_google"
)


model = load_matcher(
    "models/random_forest_amazon_google.pkl"
)


# baseline

X_base = create_feature_matrix(
    A,
    B,
    test
)

base_scores = predict_scores(
    model,
    X_base
)


# missingness drift

A_shift = apply_missingness_shift(
    A,
    [
        "manufacturer"
    ],
    probability=0.5
)


B_shift = apply_missingness_shift(
    B,
    [
        "manufacturer"
    ],
    probability=0.5
)


X_shift = create_feature_matrix(
    A_shift,
    B_shift,
    test
)


shift_scores = predict_scores(
    model,
    X_shift
)


result = test.copy()

result["base_score"] = base_scores
result["shift_score"] = shift_scores


result.to_csv(
    "results/missingness_shift_scores.csv",
    index=False
)


print(result.head())


print("\nMean scores")
print(
    "Before:",
    base_scores.mean()
)

print(
    "After:",
    shift_scores.mean()
)


print("\nSaved:")
print(
    "results/missingness_shift_scores.csv"
)
