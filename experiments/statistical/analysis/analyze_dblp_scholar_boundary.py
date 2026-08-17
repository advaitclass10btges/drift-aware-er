import os
import numpy as np
import pandas as pd


DATASET = (
    "results/"
    "dblp_scholar_score_stream.csv"
)

OUTPUT = (
    "results/statistical/analysis/"
    "dblp_scholar_boundary.csv"
)


DRIFT_POINT_RATIO = 0.7


def calibrate_boundary_threshold(scores):
    """
    Calibrate decision boundary from stable score distribution.

    Uses median score as operational decision threshold.
    """

    return np.quantile(
        scores,
        0.50
    )



def boundary_mass(scores, threshold):
    """
    Measure uncertainty mass around the calibrated boundary.

    Boundary region:
        threshold +/- 10%
    """

    scores = np.asarray(
        scores
    )

    margin = 0.10


    lower = threshold - margin
    upper = threshold + margin


    return np.mean(
        (scores >= lower)
        &
        (scores <= upper)
    )



def compress(scores, point, alpha):

    scores = scores.copy()

    scores[point:] *= alpha

    return scores



def main():

    df = pd.read_csv(
        DATASET
    )


    scores = (
        df["score"]
        .values
    )


    drift_point = int(
        len(scores)
        *
        DRIFT_POINT_RATIO
    )


    stable_original = (
        scores[:drift_point]
    )


    threshold = calibrate_boundary_threshold(
        stable_original
    )


    print(
        "Calibrated threshold:",
        threshold
    )


    results = []


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


        stable = (
            drift_scores[:drift_point]
        )

        drift = (
            drift_scores[drift_point:]
        )


        stable_mass = boundary_mass(
            stable,
            threshold
        )

        drift_mass = boundary_mass(
            drift,
            threshold
        )


        results.append({

            "dataset":
                "DBLP-Scholar",

            "severity":
                alpha,

            "threshold":
                threshold,

            "stable_boundary_mass":
                stable_mass,

            "drift_boundary_mass":
                drift_mass,

            "increase":
                drift_mass
                -
                stable_mass

        })


    output = pd.DataFrame(
        results
    )


    os.makedirs(
        os.path.dirname(OUTPUT),
        exist_ok=True
    )


    output.to_csv(
        OUTPUT,
        index=False
    )


    print(output)



if __name__ == "__main__":
    main()