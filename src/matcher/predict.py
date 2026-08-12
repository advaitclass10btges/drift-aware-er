import pandas as pd

from src.matcher.features import build_features


def create_feature_matrix(table_a, table_b, pairs):

    rows = []

    table_a_index = table_a.set_index("id")
    table_b_index = table_b.set_index("id")

    for _, row in pairs.iterrows():

        left = table_a_index.loc[row["ltable_id"]]
        right = table_b_index.loc[row["rtable_id"]]

        rows.append(
            build_features(left, right)
        )

    return pd.DataFrame(rows)


def predict_scores(model, X):

    return model.predict_proba(X)[:, 1]
