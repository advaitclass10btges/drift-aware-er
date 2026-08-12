import joblib

from sklearn.ensemble import RandomForestClassifier


def train_matcher(X_train, y_train):

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

    return model


def save_model(model, path):

    joblib.dump(
        model,
        path
    )
