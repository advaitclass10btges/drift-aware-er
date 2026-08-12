import os

import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# CONFIG
# ============================================================

EVALUATION_PATH = "results/final_detector_evaluation.csv"
OUTPUT_DIR = "results"

TRUE_DRIFT = 1375


# ============================================================
# LOAD FINAL EVALUATION
# ============================================================

df = pd.read_csv(EVALUATION_PATH)

print("\nLoaded final evaluation:")
print(df)


# ============================================================
# PLOT 1 — DETECTION DELAY BY SHIFT
# ============================================================

def plot_detection_delay(df):
    boundary = df[
        (df["detector"] == "Boundary-Aware")
        & (df["detected"] == True)
    ].copy()

    if boundary.empty:
        print("No Boundary-Aware detections available.")
        return

    plt.figure(figsize=(8, 5))

    plt.bar(
        boundary["shift"],
        boundary["detection_delay"]
    )

    plt.xlabel("Drift type")
    plt.ylabel("Detection delay")
    plt.title("Boundary-Aware Detection Delay")

    plt.tight_layout()

    path = os.path.join(
        OUTPUT_DIR,
        "final_detection_delay.png"
    )

    plt.savefig(path, dpi=300)
    plt.close()

    print(f"Saved: {path}")


# ============================================================
# PLOT 2 — FALSE ALARM RATE
# ============================================================

def plot_false_alarm_rate(df):

    summary = (
        df.groupby("detector")["false_alarm_rate"]
        .mean()
        .reset_index()
    )

    plt.figure(figsize=(8, 5))

    plt.bar(
        summary["detector"],
        summary["false_alarm_rate"]
    )

    plt.xlabel("Detector")
    plt.ylabel("Mean false alarm rate")
    plt.title("Detector False Alarm Rate")

    plt.tight_layout()

    path = os.path.join(
        OUTPUT_DIR,
        "final_false_alarm_rate.png"
    )

    plt.savefig(path, dpi=300)
    plt.close()

    print(f"Saved: {path}")


# ============================================================
# PLOT 3 — SHIFTS DETECTED
# ============================================================

def plot_shifts_detected(df):

    summary = (
        df.groupby("detector")["detected"]
        .sum()
        .reset_index()
    )

    plt.figure(figsize=(8, 5))

    plt.bar(
        summary["detector"],
        summary["detected"]
    )

    plt.xlabel("Detector")
    plt.ylabel("Number of shifts detected")
    plt.title("Drift Detection Coverage")

    plt.tight_layout()

    path = os.path.join(
        OUTPUT_DIR,
        "final_detection_coverage.png"
    )

    plt.savefig(path, dpi=300)
    plt.close()

    print(f"Saved: {path}")


# ============================================================
# PLOT 4 — DETECTOR COMPARISON
# ============================================================

def plot_detector_comparison(df):

    summary = (
        df.groupby("detector")
        .agg(
            shifts_detected=("detected", "sum"),
            total_false_alarms=("false_alarms", "sum"),
        )
        .reset_index()
    )

    plt.figure(figsize=(9, 5))

    x = range(len(summary))

    plt.bar(
        x,
        summary["shifts_detected"],
        label="Shifts detected"
    )

    plt.xticks(
        list(x),
        summary["detector"]
    )

    plt.xlabel("Detector")
    plt.ylabel("Count")
    plt.title("Detector Comparison")

    plt.legend()

    plt.tight_layout()

    path = os.path.join(
        OUTPUT_DIR,
        "final_detector_comparison.png"
    )

    plt.savefig(path, dpi=300)
    plt.close()

    print(f"Saved: {path}")


# ============================================================
# PLOT 5 — STREAM WITH TRUE DRIFT POINT
# ============================================================

def plot_stream(
    stream_path,
    title,
    output_name
):

    stream = pd.read_csv(stream_path)

    scores = stream["score"].to_numpy()

    plt.figure(figsize=(12, 5))

    plt.plot(
        scores,
        linewidth=1
    )

    plt.axvline(
        TRUE_DRIFT,
        linestyle="--",
        linewidth=2,
        label="True drift point"
    )

    plt.xlabel("Stream index")
    plt.ylabel("Matcher score")

    plt.title(title)

    plt.legend()

    plt.tight_layout()

    path = os.path.join(
        OUTPUT_DIR,
        output_name
    )

    plt.savefig(
        path,
        dpi=300
    )

    plt.close()

    print(f"Saved: {path}")


# ============================================================
# PLOT 6 — ALL STREAMS
# ============================================================

def plot_all_streams():

    streams = [
        (
            "results/title_drift_stream.csv",
            "Title Drift Stream",
            "final_title_drift_stream.png",
        ),
        (
            "results/naming_drift_stream.csv",
            "Naming Drift Stream",
            "final_naming_drift_stream.png",
        ),
        (
            "results/missingness_drift_stream.csv",
            "Missingness Drift Stream",
            "final_missingness_drift_stream.png",
        ),
    ]

    for path, title, output_name in streams:

        if not os.path.exists(path):
            print(f"Missing stream: {path}")
            continue

        plot_stream(
            path,
            title,
            output_name
        )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("\n======================================")
    print("FINAL RESULTS VISUALIZATION")
    print("======================================")

    plot_detection_delay(df)

    plot_false_alarm_rate(df)

    plot_shifts_detected(df)

    plot_detector_comparison(df)

    plot_all_streams()

    print("\n======================================")
    print("ALL PLOTS GENERATED")
    print("======================================")