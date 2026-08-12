import pandas as pd
import numpy as np

from river.drift import ADWIN, PageHinkley

from src.drift.boundary_detector import (
    calibrate_detector,
    detect,
)


STREAMS = {
    "title": "results/title_drift_stream.csv",
    "naming": "results/naming_drift_stream.csv",
    "missingness": "results/missingness_drift_stream.csv",
}

TRUE_DRIFT = 1375

REFERENCE_SIZE = 500
WINDOW_SIZE = 100
CALIBRATION_STEP = 50
DETECTION_STEP = 25

THRESHOLD = 0.36
EPSILON = 0.05
QUANTILE = 0.99


def evaluate_baseline(detector, scores):

    detections = []

    for i, score in enumerate(scores):

        detector.update(float(score))

        if detector.drift_detected:
            detections.append(i)

    pre = [
        x for x in detections
        if x < TRUE_DRIFT
    ]

    post = [
        x for x in detections
        if x >= TRUE_DRIFT
    ]

    if post:
        first = post[0]
        delay = first - TRUE_DRIFT
    else:
        first = np.nan
        delay = np.nan

    return {
        "total_detections": len(detections),
        "false_alarms": len(pre),
        "post_drift_detections": len(post),
        "first_post_drift_detection": first,
        "detection_delay": delay,
        "detections": detections,
    }


def evaluate_boundary(scores):

    stable_scores = scores[:TRUE_DRIFT]

    calibration = calibrate_detector(
        stable_scores,
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

    pre = [
        d for d in detections
        if d["start"] < TRUE_DRIFT
    ]

    post = [
        d for d in detections
        if d["start"] >= TRUE_DRIFT
    ]

    if post:
        first = post[0]["start"]
        delay = first - TRUE_DRIFT
    else:
        first = np.nan
        delay = np.nan

    return {
        "total_detections": len(detections),
        "false_alarms": len(pre),
        "post_drift_detections": len(post),
        "first_post_drift_detection": first,
        "detection_delay": delay,
        "detections": [
            d["start"] for d in detections
        ],
    }


results = []

for shift, path in STREAMS.items():

    print("\n" + "=" * 70)
    print(f"{shift.upper()} DRIFT")
    print("=" * 70)

    df = pd.read_csv(path)

    scores = df["score"].to_numpy(dtype=float)

    detectors = {
        "ADWIN": ADWIN(),
        "Page-Hinkley": PageHinkley(),
    }

    for name, detector in detectors.items():

        result = evaluate_baseline(
            detector,
            scores,
        )

        print(f"\n{name}")
        print(result)

        results.append({
            "shift": shift,
            "detector": name,
            **{
                k: v
                for k, v in result.items()
                if k != "detections"
            },
        })

    result = evaluate_boundary(scores)

    print("\nBoundary-Aware")
    print(result)

    results.append({
        "shift": shift,
        "detector": "Boundary-Aware",
        **{
            k: v
            for k, v in result.items()
            if k != "detections"
        },
    })


summary = pd.DataFrame(results)

print("\n\n" + "=" * 70)
print("ALL DETECTORS × ALL DRIFTS")
print("=" * 70)

print(
    summary[
        [
            "shift",
            "detector",
            "total_detections",
            "false_alarms",
            "post_drift_detections",
            "first_post_drift_detection",
            "detection_delay",
        ]
    ].to_string(index=False)
)

output_path = (
    "results/all_detector_comparison.csv"
)

summary.to_csv(
    output_path,
    index=False,
)

print("\nSaved:")
print(output_path)
