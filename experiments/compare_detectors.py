import pandas as pd
import numpy as np

from river.drift import ADWIN, PageHinkley

from src.drift.boundary_detector import (
    calibrate_detector,
    detect,
)


STREAM_PATH = "results/title_drift_stream.csv"

TRUE_DRIFT = 1375

# Boundary-aware parameters
REFERENCE_SIZE = 500
CALIBRATION_WINDOW = 100
CALIBRATION_STEP = 50

DETECTION_WINDOW = 100
DETECTION_STEP = 25

QUANTILE = 0.99


# ============================================================
# Load stream
# ============================================================

df = pd.read_csv(STREAM_PATH)

scores = df["score"].to_numpy(dtype=float)

print("=" * 65)
print("DETECTOR COMPARISON")
print("=" * 65)

print("\nStream size:", len(scores))
print("True drift point:", TRUE_DRIFT)


# ============================================================
# ADWIN
# ============================================================

adwin = ADWIN()

adwin_detections = []

for i, score in enumerate(scores):

    adwin.update(score)

    if adwin.drift_detected:
        adwin_detections.append(i)


# ============================================================
# Page-Hinkley
# ============================================================

page_hinkley = PageHinkley()

ph_detections = []

for i, score in enumerate(scores):

    page_hinkley.update(score)

    if page_hinkley.drift_detected:
        ph_detections.append(i)


# ============================================================
# Boundary-Aware
# ============================================================

stable_scores = scores[:TRUE_DRIFT]

calibration = calibrate_detector(
    stable_scores,
    reference_size=REFERENCE_SIZE,
    window_size=CALIBRATION_WINDOW,
    step=CALIBRATION_STEP,
    quantile=QUANTILE,
)

boundary_detections = detect(
    scores,
    calibration,
    window_size=DETECTION_WINDOW,
    step=DETECTION_STEP,
)


# Convert window detections to starting indices
boundary_indices = [
    d["start"]
    for d in boundary_detections
]


# ============================================================
# Evaluation helper
# ============================================================

def evaluate_detection_indices(
    detections,
    true_drift,
):

    false_alarms = [
        d for d in detections
        if d < true_drift
    ]

    post_drift = [
        d for d in detections
        if d >= true_drift
    ]

    if post_drift:

        first_detection = min(post_drift)

        delay = (
            first_detection -
            true_drift
        )

    else:

        first_detection = None
        delay = None

    possible_pre_drift = len(
        range(
            0,
            true_drift,
        )
    )

    false_alarm_rate = (
        len(false_alarms) /
        possible_pre_drift
    )

    return {
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
            delay,
    }


# ============================================================
# Evaluate
# ============================================================

adwin_results = evaluate_detection_indices(
    adwin_detections,
    TRUE_DRIFT,
)

ph_results = evaluate_detection_indices(
    ph_detections,
    TRUE_DRIFT,
)

boundary_results = evaluate_detection_indices(
    boundary_indices,
    TRUE_DRIFT,
)


# ============================================================
# Comparison table
# ============================================================

results = pd.DataFrame([

    {
        "detector": "ADWIN",
        **adwin_results,
    },

    {
        "detector": "Page-Hinkley",
        **ph_results,
    },

    {
        "detector": "Boundary-Aware",
        **boundary_results,
    },

])


print("\n")
print("=" * 65)
print("COMPARISON RESULTS")
print("=" * 65)

print(
    results.to_string(
        index=False
    )
)


# ============================================================
# Detection locations
# ============================================================

print("\n")
print("=" * 65)
print("DETECTION LOCATIONS")
print("=" * 65)

print("\nADWIN:")
print(adwin_detections)

print("\nPage-Hinkley:")
print(ph_detections)

print("\nBoundary-Aware:")
print(boundary_indices)


# ============================================================
# Boundary calibration information
# ============================================================

print("\n")
print("=" * 65)
print("BOUNDARY-AWARE CALIBRATION")
print("=" * 65)

print(
    "Calibration threshold:",
    calibration["threshold"]
)


# ============================================================
# Save
# ============================================================

output_path = (
    "results/detector_comparison_title_drift.csv"
)

results.to_csv(
    output_path,
    index=False
)

print("\nSaved:")
print(output_path)
