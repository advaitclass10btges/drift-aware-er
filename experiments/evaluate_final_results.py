import pandas as pd
import numpy as np

PATH = "results/all_detector_comparison.csv"

TRUE_DRIFT = 1375
WINDOW_SIZE = 100
STEP = 25

df = pd.read_csv(PATH)

# Number of detector windows whose start lies
# completely before the true drift point.
stable_windows = len(
    range(
        0,
        TRUE_DRIFT - WINDOW_SIZE + 1,
        STEP
    )
)

df["stable_windows"] = stable_windows

df["false_alarm_rate"] = (
    df["false_alarms"] /
    stable_windows
)

# Whether the detector successfully detected
# the injected drift.
df["detected"] = (
    df["post_drift_detections"] > 0
)

print("\n" + "=" * 90)
print("FINAL DRIFT DETECTION EVALUATION")
print("=" * 90)

print("\nStable evaluation windows:")
print(stable_windows)

print("\nResults:")
print(
    df[
        [
            "shift",
            "detector",
            "total_detections",
            "false_alarms",
            "stable_windows",
            "false_alarm_rate",
            "post_drift_detections",
            "first_post_drift_detection",
            "detection_delay",
            "detected",
        ]
    ].to_string(index=False)
)

print("\n" + "=" * 90)
print("DETECTOR-LEVEL SUMMARY")
print("=" * 90)

summary = (
    df
    .groupby("detector")
    .agg(
        shifts_tested=("shift", "count"),
        shifts_detected=("detected", "sum"),
        total_false_alarms=("false_alarms", "sum"),
        mean_false_alarm_rate=("false_alarm_rate", "mean"),
        mean_detection_delay=("detection_delay", "mean"),
    )
    .reset_index()
)

print(summary.to_string(index=False))

output = "results/final_detector_evaluation.csv"

df.to_csv(output, index=False)

print("\nSaved:")
print(output)
