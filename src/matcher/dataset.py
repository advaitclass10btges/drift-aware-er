import pandas as pd
from src.matcher.features import build_features


def create_feature_dataset(
    table_a,
    table_b,
    pairs
):
    """
    Convert labeled ER pairs into ML features.

    Returns:
        X -> dataframe of features
        y -> labels
    """

    feature_rows = []
    labels = []

    table_a_index = table_a.set_index("id")
    table_b_index = table_b.set_index("id")


    for _, row in pairs.iterrows():

        left = table_a_index.loc[row["ltable_id"]]
        right = table_b_index.loc[row["rtable_id"]]


        features = build_features(
            left,
            right
        )

        feature_rows.append(features)
        labels.append(row["label"])


    X = pd.DataFrame(feature_rows)
    y = pd.Series(labels)

    return X, y
