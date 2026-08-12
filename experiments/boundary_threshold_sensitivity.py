import pandas as pd
import numpy as np

from src.drift.boundary_detector import (
    calibrate_detector,
    detect,
)

# --------------------------------------------------
# Configuration
# --------------------------------------------------

STREAM_PATH = "results/title_drift_stream.csv"

TRUE_DRIFT = 1375

WINDOW_SIZE = 100
DETECTION_STEP = 25

CALIBRATION_REFERENCE = 500
CALIBRATION_WINDOW = 100
CALIBRATION_STEP = 50

# Keep calibration quantile fixed.
CALIBRATION_QUANTILE = 0.99

# Actual detector thresholds to evaluate.
THRESHOLDS = [
    0.20,
    0.25,
    0.30,
    0.35,
    0.36,
    0.40,
    0.45,
    0.50,
]

EPSILON = 0.05


# --------------------------------------------------
# Load stream
# --------------------------------------------------

df = pd.read_csv(STREAM_PATH)

scores = df["score"].to_numpy(dtype=float)

stable_scores = scores[:TRUE_DRIFT]


print("=" * 70)
print("BOUNDARY-AWARE DETECTOR THRESHOLD SENSITIVITY")
print("=" * 70)

print("\nStream size:", len(scores))
print("Stable samples:", len(stable_scores))
print("True drift point:", TRUE_DRIFT)

print("\nFixed calibration quantile:", CALIBRATION_QUANTILE)
print("Thresholds:", THRESHOLDS)


# --------------------------------------------------
# Calibrate once
# --------------------------------------------------

calibration = calibrate_detector(
    stable_scores,
    reference_size=CALIBRATION_REFERENCE,
    window_size=CALIBRATION_WINDOW,
    step=CALIBRATION_STEP,
    quantile=CALIBRATION_QUANTILE,
)


print("\nCalibration threshold:")
print(calibration["threshold"])


# --------------------------------------------------
# Evaluate each detector threshold
# --------------------------------------------------

results = []


for threshold in THRESHOLDS:

    print("\n" + "-" * 70)
    print("Testing detector threshold:", threshold)

    detections = detect(
        scores,
        calibration,
        window_size=WINDOW_SIZE,
        step=DETECTION_STEP,
        threshold=threshold,
        epsilon=EPSILON,
    )

    # ----------------------------------------------
    # False alarms
    # ----------------------------------------------

    false_alarms = [
        d
        for d in detections
        if d["start"] < TRUE_DRIFT
    ]

    # ----------------------------------------------
    # Post-drift detections
    # ----------------------------------------------

    post_drift = [
        d
        for d in detections
        if d["start"] >= TRUE_DRIFT
    ]

    # ----------------------------------------------
    # First post-drift detection
    # ----------------------------------------------

    if post_drift:

        first_detection = min(
            d["start"]
            for d in post_drift
        )

        detection_delay = (
            first_detection - TRUE_DRIFT
        )

    else:

        first_detection = np.nan
        detection_delay = np.nan

    # ----------------------------------------------
    # Number of possible pre-drift windows
    # ----------------------------------------------

    possible_pre_drift_windows = len(
        range(
            0,
            TRUE_DRIFT,
            DETECTION_STEP,
        )
    )

    false_alarm_rate = (
        len(false_alarms)
        / possible_pre_drift_windows
    )

    # ----------------------------------------------
    # Detection coverage
    # ----------------------------------------------

    detected = len(post_drift) > 0

    results.append(
        {
            "threshold": threshold,
            "calibration_quantile":
                CALIBRATION_QUANTILE,
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
            "detected":
                detected,
        }
    )

    print(
        "Total detections:",
        len(detections),
    )

    print(
        "False alarms:",
        len(false_alarms),
    )

    print(
        "Post-drift detections:",
        len(post_drift),
    )

    print(
        "False alarm rate:",
        round(false_alarm_rate, 6),
    )

    print(
        "First post-drift detection:",
        first_detection,
    )

    print(
        "Detection delay:",
        detection_delay,
    )


# --------------------------------------------------
# Results
# --------------------------------------------------

results_df = pd.DataFrame(results)


print("\n")
print("=" * 70)
print("THRESHOLD SENSITIVITY RESULTS")
print("=" * 70)

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
    "boundary_detector_threshold_sensitivity.csv"
)

results_df.to_csv(
    output_path,
    index=False,
)


print("\nSaved:")
print(output_path)
