import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

from src.data.loaders import load_amazon_google
from src.matcher.dataset import create_feature_dataset
from src.matcher.predict import create_feature_matrix, predict_scores
from src.matcher.threshold import find_best_threshold


A,B,train,valid,test = load_amazon_google(
    "data/raw/amazon_google"
)


# Train
X_train,y_train = create_feature_dataset(
    A,B,train
)

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)

model.fit(
    X_train,
    y_train
)


# Validation threshold

X_valid = create_feature_matrix(
    A,B,valid
)

valid_scores = predict_scores(
    model,
    X_valid
)

threshold, f1 = find_best_threshold(
    valid["label"],
    valid_scores
)

print("Best threshold:", threshold)
print("Validation F1:", f1)


# Test evaluation

X_test = create_feature_matrix(
    A,B,test
)

test_scores = predict_scores(
    model,
    X_test
)

test_predictions = (
    test_scores >= threshold
).astype(int)


print("\nTest results")
print(
    classification_report(
        test["label"],
        test_predictions
    )
)
