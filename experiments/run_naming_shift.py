import pandas as pd

from src.data.loaders import load_amazon_google
from src.matcher.save_model import load_matcher
from src.matcher.predict import create_feature_matrix, predict_scores

from src.drift.shifts.naming_shift import apply_naming_shift


A, B, train, valid, test = load_amazon_google(
    "data/raw/amazon_google"
)


model = load_matcher(
    "models/random_forest_amazon_google.pkl"
)


# -------------------------
# Baseline stream
# -------------------------

X_base = create_feature_matrix(
    A,
    B,
    test
)

base_scores = predict_scores(
    model,
    X_base
)


# -------------------------
# Apply naming shift
# -------------------------

A_shifted = apply_naming_shift(
    A,
    [
        "title",
        "manufacturer"
    ]
)

B_shifted = apply_naming_shift(
    B,
    [
        "title",
        "manufacturer"
    ]
)


X_shift = create_feature_matrix(
    A_shifted,
    B_shifted,
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
    "results/naming_shift_scores.csv",
    index=False
)


print(result.head())

print("\nMean score:")
print("Before:", base_scores.mean())
print("After :", shift_scores.mean())

print("\nSaved:")
print("results/naming_shift_scores.csv")
