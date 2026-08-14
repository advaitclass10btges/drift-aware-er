import os
import numpy as np
import pandas as pd


from src.evaluation.drift_metrics import (
    boundary_mass,
    score_entropy,
)



DATASET = (
    "results/"
    "amazon_google_score_stream.csv"
)


OUTPUT = (
    "results/statistical/analysis/"
    "amazon_google_boundary.csv"
)



def analyze(scores, labels):


    return {

        "samples":
            len(scores),

        "positive_rate":
            labels.mean(),

        "mean_score":
            scores.mean(),

        "std_score":
            scores.std(),

        "boundary_mass":
            boundary_mass(
                scores,
                threshold=0.36,
                epsilon=0.05
            ),

        "entropy":
            score_entropy(
                scores
            )
    }



def main():

    df = pd.read_csv(DATASET)


    scores = (
        df.score
        .values
    )

    labels = (
        df.label
        .values
    )


    results=[]


    # confidence ordered degradation

    order = np.argsort(scores)


    chunks = np.array_split(
        order,
        5
    )


    for i,idx in enumerate(chunks):

        result = analyze(
            scores[idx],
            labels[idx]
        )

        result["confidence_bin"]=i

        results.append(result)



    out=pd.DataFrame(results)


    os.makedirs(
        os.path.dirname(OUTPUT),
        exist_ok=True
    )


    out.to_csv(
        OUTPUT,
        index=False
    )


    print(out)



if __name__=="__main__":
    main()