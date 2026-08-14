import os
import numpy as np
import pandas as pd

from src.evaluation.drift_metrics import (
    boundary_mass,
    score_entropy,
)


STREAMS = {
    "title": "results/title_drift_stream.csv",
    "naming": "results/naming_drift_stream.csv",
    "missingness": "results/missingness_drift_stream.csv",
}


OUTPUT = (
    "results/statistical/analysis/"
    "boundary_vs_statistics.csv"
)


TRUE_DRIFT = 1375


def feature_values(scores):

    return {
        "mean":
            np.mean(scores),

        "std":
            np.std(scores),

        "entropy":
            score_entropy(scores),

        "boundary_mass":
            boundary_mass(
                scores,
                threshold=0.36,
                epsilon=0.05
            )
    }



def compare_stream(path, name):

    df = pd.read_csv(path)

    pre = (
        df[df.phase=="stable"]
        .score
        .values
    )

    post = (
        df[df.phase=="drift"]
        .score
        .values
    )


    pre_features = feature_values(pre)
    post_features = feature_values(post)


    rows=[]


    for feature in pre_features:

        rows.append({

            "shift": name,

            "feature": feature,

            "stable_value":
                pre_features[feature],

            "drift_value":
                post_features[feature],

            "absolute_change":
                abs(
                    post_features[feature]
                    -
                    pre_features[feature]
                )
        })


    return rows



def main():

    results=[]


    for name,path in STREAMS.items():

        results.extend(
            compare_stream(
                path,
                name
            )
        )


    df=pd.DataFrame(results)


    os.makedirs(
        os.path.dirname(OUTPUT),
        exist_ok=True
    )


    df.to_csv(
        OUTPUT,
        index=False
    )


    print(df)



if __name__=="__main__":
    main()