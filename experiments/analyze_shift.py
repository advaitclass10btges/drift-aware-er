import sys
import pandas as pd

from src.evaluation.drift_metrics import (
    score_statistics,
    boundary_mass,
    score_entropy
)

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score
)


path = sys.argv[1]


df = pd.read_csv(path)


for name, scores in [
    ("Before shift", df["base_score"]),
    ("After shift", df["shift_score"])
]:

    print("\n====================")
    print(name)

    print(
        score_statistics(scores)
    )

    print(
        "Boundary mass:",
        boundary_mass(
            scores,
            threshold=0.36
        )
    )

    print(
        "Entropy:",
        score_entropy(scores)
    )


print("\nClassification")


for name, scores in [
    ("Before shift", df["base_score"]),
    ("After shift", df["shift_score"])
]:

    predictions = (
        scores >= 0.36
    ).astype(int)


    print("\n", name)

    print(
        "Precision:",
        precision_score(
            df["label"],
            predictions
        )
    )

    print(
        "Recall:",
        recall_score(
            df["label"],
            predictions
        )
    )

    print(
        "F1:",
        f1_score(
            df["label"],
            predictions
        )
    )
