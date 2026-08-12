from src.data.loaders import load_amazon_google
from src.matcher.save_model import load_matcher
from src.matcher.predict import (
    create_feature_matrix,
    predict_scores
)

from src.drift.shifts.title_degradation import (
    apply_title_degradation
)

import pandas as pd


A,B,train,valid,test = load_amazon_google(
    "data/raw/amazon_google"
)


model = load_matcher(
    "models/random_forest_amazon_google.pkl"
)


# Baseline

X_base = create_feature_matrix(
    A,
    B,
    test
)

base_scores = predict_scores(
    model,
    X_base
)


# Title degradation

A_shift = apply_title_degradation(
    A,
    keep_tokens=2
)


B_shift = apply_title_degradation(
    B,
    keep_tokens=2
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
    "results/title_shift_scores.csv",
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
    "results/title_shift_scores.csv"
)
