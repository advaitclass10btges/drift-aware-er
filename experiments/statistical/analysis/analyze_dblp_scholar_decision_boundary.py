import os
import numpy as np
import pandas as pd


DATASET = (
    "results/"
    "dblp_scholar_score_stream.csv"
)


OUTPUT = (
    "results/statistical/analysis/"
    "dblp_scholar_decision_boundary.csv"
)


DRIFT_POINT_RATIO = 0.7


def compress(scores, point, alpha):

    scores = scores.copy()

    scores[point:] *= alpha

    return scores



def main():

    df = pd.read_csv(DATASET)

    scores = df.score.values


    threshold = np.quantile(
        scores[:int(len(scores)*DRIFT_POINT_RATIO)],
        0.5
    )


    drift_point = int(
        len(scores)
        *
        DRIFT_POINT_RATIO
    )


    results=[]


    stable_scores = scores[:drift_point]


    stable_prediction = (
        stable_scores >= threshold
    )


    for alpha in [
        0.95,
        0.90,
        0.85,
        0.80,
        0.75
    ]:

        drift_scores = compress(
            scores,
            drift_point,
            alpha
        )


        drift_region = drift_scores[drift_point:]


        stable_region = scores[drift_point:]


        stable_pred = (
            stable_region >= threshold
        )


        drift_pred = (
            drift_region >= threshold
        )


        flip_rate = np.mean(
            stable_pred != drift_pred
        )


        results.append({

            "dataset":
                "DBLP-Scholar",

            "severity":
                alpha,

            "threshold":
                threshold,

            "decision_flip_rate":
                flip_rate,

            "stable_positive_rate":
                stable_pred.mean(),

            "drift_positive_rate":
                drift_pred.mean()

        })


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