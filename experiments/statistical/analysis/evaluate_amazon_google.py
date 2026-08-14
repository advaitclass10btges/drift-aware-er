import os
import numpy as np
import pandas as pd

"""
Controlled robustness evaluation on the Amazon-Google
entity resolution benchmark.

The experiment applies score-distribution degradation
(score compression) after a fixed drift point and evaluates
whether the boundary-aware detector identifies the resulting
decision-boundary instability.

This experiment is separate from natural boundary analysis.
"""


from src.drift.boundary_detector import (
    calibrate_detector,
    detect,
)


DATASET = (
    "results/"
    "amazon_google_score_stream.csv"
)


OUTPUT = (
    "results/statistical/analysis/"
    "amazon_google_results.csv"
)


TRUE_DRIFT = 1375

REFERENCE_SIZE = 500
WINDOW_SIZE = 100
CALIBRATION_STEP = 50
DETECTION_STEP = 25

THRESHOLD = 0.36
EPSILON = 0.05
QUANTILE = 0.99


def inject_score_compression(
    scores,
    drift_point,
    alpha,
):

    scores = np.asarray(
        scores,
        dtype=float
    ).copy()

    scores[drift_point:] *= alpha

    return scores



def evaluate_boundary(scores):

    stable = scores[:TRUE_DRIFT]


    calibration = calibrate_detector(
        stable,
        reference_size=REFERENCE_SIZE,
        window_size=WINDOW_SIZE,
        step=CALIBRATION_STEP,
        threshold=THRESHOLD,
        epsilon=EPSILON,
        quantile=QUANTILE,
    )


    detections = detect(
        scores,
        calibration,
        window_size=WINDOW_SIZE,
        step=DETECTION_STEP,
        threshold=THRESHOLD,
        epsilon=EPSILON,
    )


    post = [
        d for d in detections
        if d["start"] >= TRUE_DRIFT
    ]


    pre = [
        d for d in detections
        if d["start"] < TRUE_DRIFT
    ]


    if post:

        delay = (
            post[0]["start"]
            -
            TRUE_DRIFT
        )

        detected = True

    else:

        delay = np.nan
        detected = False


    stable_windows = (
        (TRUE_DRIFT-WINDOW_SIZE)
        //
        DETECTION_STEP
        + 1
    )


    return {
        "detected": detected,
        "delay": delay,
        "false_alarm_rate":
            len(pre)/stable_windows,
        "detections":
            len(detections),
    }



def main():

    df = pd.read_csv(DATASET)

    scores = (
        df["score"]
        .values
    )


    results = []


    for alpha in [
        0.95,
        0.90,
        0.85,
        0.80,
        0.75,
    ]:


        drift_scores = (
            inject_score_compression(
                scores,
                TRUE_DRIFT,
                alpha,
            )
        )


        result = evaluate_boundary(
            drift_scores
        )


        results.append({

            "dataset":
                "Amazon-Google",

            "drift_type":
                "score_compression",

            "severity":
                alpha,

            **result

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