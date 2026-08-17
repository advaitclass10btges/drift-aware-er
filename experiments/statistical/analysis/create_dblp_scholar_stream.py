import os
import numpy as np
import pandas as pd

from rapidfuzz.fuzz import ratio
import jellyfish


BASE = (
    "datasets/real/dblp_scholar/processed"
)

OUTPUT = (
    "results/"
    "dblp_scholar_score_stream.csv"
)


np.random.seed(42)


def clean(x):

    if pd.isna(x):
        return ""

    return str(x).lower().strip()



def similarity(a,b):

    return ratio(
        clean(a),
        clean(b)
    ) / 100.0



def jaro(a,b):

    return jellyfish.jaro_winkler_similarity(
        clean(a),
        clean(b)
    )



def compute_score(left,right):

    title = similarity(
        left["title"],
        right["title"]
    )


    authors = jaro(
        left["authors"],
        right["authors"]
    )


    venue = similarity(
        left["venue"],
        right["venue"]
    )


    if pd.isna(left["year"]) or pd.isna(right["year"]):

        year = 0

    else:

        year = 1 / (
            1 + abs(
                left["year"]
                -
                right["year"]
            )
        )


    score = (
        0.55*title
        +
        0.25*authors
        +
        0.15*venue
        +
        0.05*year
    )


    return score



def main():

    tableA = pd.read_csv(
        f"{BASE}/tableA.csv"
    )

    tableB = pd.read_csv(
        f"{BASE}/tableB.csv"
    )

    pairs = pd.read_csv(
        f"{BASE}/test.csv"
    )


    A = tableA.set_index("id")
    B = tableB.set_index("id")


    results=[]


    for _,row in pairs.iterrows():

        left = A.loc[
            row["ltable_id"]
        ]

        right = B.loc[
            row["rtable_id"]
        ]


        score = compute_score(
            left,
            right
        )


        results.append({

            "ltable_id":
                row["ltable_id"],

            "rtable_id":
                row["rtable_id"],

            "label":
                row["label"],

            "score":
                score

        })


    df = pd.DataFrame(results)


    # deterministic stream ordering
    df=df.sample(
        frac=1,
        random_state=42
    ).reset_index(drop=True)


    os.makedirs(
        "results",
        exist_ok=True
    )


    df.to_csv(
        OUTPUT,
        index=False
    )


    print(df.head())
    print()
    print(df.label.value_counts())
    print(df.score.describe())



if __name__=="__main__":
    main()