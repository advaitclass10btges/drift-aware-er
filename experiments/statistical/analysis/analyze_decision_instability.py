import pandas as pd
import os

os.makedirs(
    "results/statistical/analysis",
    exist_ok=True
)

from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
)

STREAMS = {
    "title": "results/title_drift_stream.csv",
    "naming": "results/naming_drift_stream.csv",
    "missingness": "results/missingness_drift_stream.csv",
}

THRESHOLD = 0.36


def evaluate(df):

    y_true = df["label"]

    y_pred = (
        df["score"] >= THRESHOLD
    ).astype(int)

    return {
        "f1":
            f1_score(
                y_true,
                y_pred,
                zero_division=0,
            ),

        "precision":
            precision_score(
                y_true,
                y_pred,
                zero_division=0,
            ),

        "recall":
            recall_score(
                y_true,
                y_pred,
                zero_division=0,
            ),
    }


rows=[]


for name,path in STREAMS.items():

    df=pd.read_csv(path)

    stable=df[
        df.phase=="stable"
    ]

    drift=df[
        df.phase=="drift"
    ]


    s=evaluate(stable)
    d=evaluate(drift)


    rows.append({

        "shift":name,

        "stable_f1":s["f1"],
        "drift_f1":d["f1"],

        "f1_drop":
            s["f1"]-d["f1"],

        "stable_precision":
            s["precision"],

        "drift_precision":
            d["precision"],

        "stable_recall":
            s["recall"],

        "drift_recall":
            d["recall"],
    })


out=pd.DataFrame(rows)

out.to_csv(
    "results/statistical/analysis/decision_instability.csv",
    index=False,
)

print(out)
