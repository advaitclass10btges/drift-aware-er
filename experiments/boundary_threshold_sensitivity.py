import pandas as pd
import numpy as np

from src.drift.boundary_detector import (
    calibrate_detector,
    detect,
)


STREAM_PATH = "results/title_drift_stream.csv"

TRUE_DRIFT = 1375

WINDOW_SIZE = 100
DETECTION_STEP = 25

CALIBRATION_REFERENCE = 500
CALIBRATION_WINDOW = 100
CALIBRATION_STEP = 50

QUANTILES = [
    0.95,
    0.975,
    0.99,
    0.995,
    0.999,
]


# --------------------------------------------------
# Load stream
# --------------------------------------------------

df = pd.read_csv(STREAM_PATH)

scores = df["score"].to_numpy(dtype=float)

stable_scores = scores[:TRUE_DRIFT]


print("=" * 60)
print("BOUNDARY-AWARE THRESHOLD SENSITIVITY")
print("=" * 60)

print("\nStream size:", len(scores))
print("Stable samples:", len(stable_scores))
print("True drift point:", TRUE_DRIFT)


results = []


# --------------------------------------------------
# Evaluate each calibration quantile
# --------------------------------------------------

for quantile in QUANTILES:

    calibration = calibrate_detector(
        stable_scores,
        reference_size=CALIBRATION_REFERENCE,
        window_size=CALIBRATION_WINDOW,
        step=CALIBRATION_STEP,
        quantile=quantile,
    )

    detections = detect(
        scores,
        calibration,
        window_size=WINDOW_SIZE,
        step=DETECTION_STEP,
    )

    # ----------------------------------------------
    # Separate false alarms and post-drift detections
    # ----------------------------------------------

    false_alarms = [
        d for d in detections
        if d["start"] < TRUE_DRIFT
    ]

    post_drift = [
        d for d in detections
        if d["start"] >= TRUE_DRIFT
    ]

    if post_drift:

        first_detection = min(
            d["start"] for d in post_drift
        )

        detection_delay = (
            first_detection - TRUE_DRIFT
        )

    else:

        first_detection = None
        detection_delay = None


    # Number of possible detection windows whose
    # starting point is before the true drift.
    possible_pre_drift_windows = len(
        range(
            0,
            TRUE_DRIFT,
            DETECTION_STEP,
        )
    )

    false_alarm_rate = (
        len(false_alarms) /
        possible_pre_drift_windows
    )


    results.append({

        "quantile": quantile,

        "calibration_threshold":
            calibration["threshold"],

        "total_detections":
            len(detections),

        "false_alarms":
            len(false_alarms),

        "post_drift_detections":
            len(post_drift),

        "false_alarm_rate":
            false_alarm_rate,

        "first_post_drift_detection":
            first_detection,

        "detection_delay":
            detection_delay,
    })


# --------------------------------------------------
# Results
# --------------------------------------------------

results_df = pd.DataFrame(results)

print("\n")
print("=" * 60)
print("SENSITIVITY RESULTS")
print("=" * 60)

print(
    results_df.to_string(
        index=False
    )
)


# --------------------------------------------------
# Save
# --------------------------------------------------

output_path = (
    "results/"
    "boundary_threshold_sensitivity.csv"
)

results_df.to_csv(
    output_path,
    index=False
)

print("\nSaved:")
print(output_path)
