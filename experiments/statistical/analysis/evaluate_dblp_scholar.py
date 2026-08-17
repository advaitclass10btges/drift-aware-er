import os
import numpy as np
import pandas as pd

from src.drift.boundary_detector import (
    calibrate_detector,
    detect,
)


DATASET = (
    "results/"
    "dblp_scholar_score_stream.csv"
)


OUTPUT = (
    "results/statistical/analysis/"
    "dblp_scholar_results.csv"
)


REFERENCE_SIZE = 500
WINDOW_SIZE = 100
CALIBRATION_STEP = 50
DETECTION_STEP = 25

THRESHOLD = 0.36
EPSILON = 0.05
QUANTILE = 0.99


def compress(scores, point, alpha):

    scores = np.asarray(
        scores,
        dtype=float
    ).copy()

    scores[point:] *= alpha

    return scores



def evaluate(scores, drift_point):

    calibration = calibrate_detector(
        scores[:drift_point],
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
        if d["start"] >= drift_point
    ]


    pre = [
        d for d in detections
        if d["start"] < drift_point
    ]


    if post:

        delay = (
            post[0]["start"]
            -
            drift_point
        )

        detected=True

    else:

        delay=np.nan
        detected=False


    stable_windows = (
        (drift_point-WINDOW_SIZE)
        //
        DETECTION_STEP
        +1
    )


    return {
        "detected": detected,
        "delay": delay,
        "false_alarm_rate":
            len(pre)/stable_windows,
        "detections":
            len(detections)
    }



def main():

    df=pd.read_csv(DATASET)

    scores=df.score.values


    drift_point=int(
        len(scores)*0.7
    )


    results=[]


    for alpha in [
        0.95,
        0.90,
        0.85,
        0.80,
        0.75
    ]:

        drift_scores=compress(
            scores,
            drift_point,
            alpha
        )


        result=evaluate(
            drift_scores,
            drift_point
        )


        results.append({

            "dataset":
                "DBLP-Scholar",

            "drift_type":
                "score_compression",

            "severity":
                alpha,

            **result
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