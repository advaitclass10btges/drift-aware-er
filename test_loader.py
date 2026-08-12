from src.data.loaders import load_amazon_google


path = "data/raw/amazon_google"

A, B, train, valid, test = load_amazon_google(path)

print("TABLE A")
print(A.head())

print("\nTABLE B")
print(B.head())

print("\nTRAIN")
print(train.head())

print("\nSizes:")
print(A.shape)
print(B.shape)
print(train.shape)
print(valid.shape)
print(test.shape)
