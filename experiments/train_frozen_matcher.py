from sklearn.ensemble import RandomForestClassifier

from src.data.loaders import load_amazon_google
from src.matcher.dataset import create_feature_dataset
from src.matcher.save_model import save_matcher


A,B,train,valid,test = load_amazon_google(
    "data/raw/amazon_google"
)


X_train,y_train = create_feature_dataset(
    A,
    B,
    train
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


save_matcher(
    model,
    "models/random_forest_amazon_google.pkl"
)


print("Saved frozen matcher:")
print("models/random_forest_amazon_google.pkl")
