from src.data.loaders import load_amazon_google
from src.matcher.dataset import create_feature_dataset
from src.matcher.train import train_matcher


A,B,train,valid,test = load_amazon_google(
    "data/raw/amazon_google"
)


X_train,y_train = create_feature_dataset(
    A,
    B,
    train
)


print("Feature matrix:")
print(X_train.shape)

print("\nLabels:")
print(y_train.value_counts())


model = train_matcher(
    X_train,
    y_train
)


print("\nTraining complete")
