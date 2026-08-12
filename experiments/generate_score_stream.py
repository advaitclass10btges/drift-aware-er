import pandas as pd

from src.data.loaders import load_amazon_google
from src.matcher.predict import create_feature_matrix, predict_scores
from src.matcher.save_model import load_matcher


A, B, train, valid, test = load_amazon_google(
    "data/raw/amazon_google"
)


model = load_matcher(
    "models/random_forest_amazon_google.pkl"
)


X_test = create_feature_matrix(
    A,
    B,
    test
)


scores = predict_scores(
    model,
    X_test
)


stream = test.copy()

stream["score"] = scores


stream.to_csv(
    "results/amazon_google_score_stream.csv",
    index=False
)


print(stream.head())

print("\nSaved:")
print("results/amazon_google_score_stream.csv")
