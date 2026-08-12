import pandas as pd
import numpy as np

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


def evaluate_stream(name, path):

    print("\n" + "=" * 70)
    print(f"BOUNDARY-AWARE DETECTOR: {name.upper()}")
    print("=" * 70)

    df = pd.read_csv(path)

    scores = df["score"].to_numpy(dtype=float)

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

    false_alarms = [
        d for d in detections
        if d["start"] < TRUE_DRIFT
    ]

    post_drift = [
        d for d in detections
        if d["start"] >= TRUE_DRIFT
    ]

    if post_drift:
        first_detection = post_drift[0]["start"]
        detection_delay = first_detection - TRUE_DRIFT
    else:
        first_detection = np.nan
        detection_delay = np.nan

    total_detections = len(detections)
    false_alarm_count = len(false_alarms)
    post_drift_count = len(post_drift)

    false_alarm_rate = (
        false_alarm_count / total_detections
        if total_detections > 0
        else 0.0
    )

    print("\nStream size:")
    print(len(scores))

    print("\nTrue drift point:")
    print(TRUE_DRIFT)

    print("\nCalibration threshold:")
    print(calibration["threshold"])

    print("\nTotal detections:")
    print(total_detections)

    print("\nFalse alarms:")
    print(false_alarm_count)

    print("\nPost-drift detections:")
    print(post_drift_count)

    print("\nFalse alarm rate:")
    print(false_alarm_rate)

    print("\nFirst post-drift detection:")
    print(first_detection)

    print("\nDetection delay:")
    print(detection_delay)

    return {
        "shift": name,
        "stream_size": len(scores),
        "true_drift": TRUE_DRIFT,
        "calibration_threshold": calibration["threshold"],
        "total_detections": total_detections,
        "false_alarms": false_alarm_count,
        "post_drift_detections": post_drift_count,
        "false_alarm_rate": false_alarm_rate,
        "first_post_drift_detection": first_detection,
        "detection_delay": detection_delay,
    }


results = []

for name, path in STREAMS.items():
    results.append(
        evaluate_stream(name, path)
    )

summary = pd.DataFrame(results)

print("\n\n" + "=" * 70)
print("BOUNDARY-AWARE DETECTOR SUMMARY")
print("=" * 70)

print(summary.to_string(index=False))

output_path = (
    "results/boundary_detector_all_shifts.csv"
)

summary.to_csv(
    output_path,
    index=False,
)

print("\nSaved:")
print(output_path)
