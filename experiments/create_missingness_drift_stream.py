import pandas as pd

from src.data.loaders import load_amazon_google
from src.matcher.save_model import load_matcher
from src.matcher.predict import (
    create_feature_matrix,
    predict_scores,
)

from src.drift.shifts.missingness_shift import (
    apply_missingness_shift,
)


# --------------------------------------------------
# Load data and trained matcher
# --------------------------------------------------

A, B, train, valid, test = load_amazon_google(
    "data/raw/amazon_google"
)

model = load_matcher(
    "models/random_forest_amazon_google.pkl"
)


# --------------------------------------------------
# Stable phase
# --------------------------------------------------

stable_size = int(len(test) * 0.6)

stable_pairs = test.iloc[:stable_size]

stable_X = create_feature_matrix(
    A,
    B,
    stable_pairs,
)

stable_scores = predict_scores(
    model,
    stable_X,
)


# --------------------------------------------------
# Drift phase
# --------------------------------------------------

drift_pairs = test.iloc[stable_size:]


A_shift = apply_missingness_shift(
    A,
    ["manufacturer"],
    probability=0.5,
)

B_shift = apply_missingness_shift(
    B,
    ["manufacturer"],
    probability=0.5,
)

drift_X = create_feature_matrix(
    A_shift,
    B_shift,
    drift_pairs,
)

drift_scores = predict_scores(
    model,
    drift_X,
)


# --------------------------------------------------
# Combine stream
# --------------------------------------------------

stable_df = stable_pairs.copy()
stable_df["score"] = stable_scores
stable_df["phase"] = "stable"

drift_df = drift_pairs.copy()
drift_df["score"] = drift_scores
drift_df["phase"] = "drift"


stream = pd.concat(
    [stable_df, drift_df],
    ignore_index=True,
)

stream["index"] = range(len(stream))


# --------------------------------------------------
# Save
# --------------------------------------------------

output_path = (
    "results/missingness_drift_stream.csv"
)

stream.to_csv(
    output_path,
    index=False,
)


print(stream.head())

print("\nPhase counts:")
print(
    stream["phase"].value_counts()
)

print("\nTrue drift point:")
print(stable_size)

print("\nSaved:")
print(output_path)
