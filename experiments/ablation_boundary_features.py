import pandas as pd
import numpy as np

from src.evaluation.drift_metrics import (
    score_statistics,
    boundary_mass,
    score_entropy,
)


STREAMS = {
    "title": "results/title_drift_stream.csv",
    "naming": "results/naming_drift_stream.csv",
    "missingness": "results/missingness_drift_stream.csv",
}

TRUE_DRIFT = 1375

REFERENCE_SIZE = 500
CALIBRATION_WINDOW = 100
CALIBRATION_STEP = 50

DETECTION_WINDOW = 100
DETECTION_STEP = 25

BOUNDARY_THRESHOLD = 0.36
EPSILON = 0.05
QUANTILE = 0.99


FEATURE_NAMES = [
    "mean",
    "std",
    "boundary_mass",
    "entropy",
]


ABLATIONS = {
    "Full": [
        "mean",
        "std",
        "boundary_mass",
        "entropy",
    ],

    "No_Mean": [
        "std",
        "boundary_mass",
        "entropy",
    ],

    "No_Std": [
        "mean",
        "boundary_mass",
        "entropy",
    ],

    "No_Boundary": [
        "mean",
        "std",
        "entropy",
    ],

    "No_Entropy": [
        "mean",
        "std",
        "boundary_mass",
    ],

    "Mean_Only": [
        "mean",
    ],

    "Std_Only": [
        "std",
    ],

    "Boundary_Only": [
        "boundary_mass",
    ],

    "Entropy_Only": [
        "entropy",
    ],
}


def extract_features(
    scores,
    threshold=BOUNDARY_THRESHOLD,
    epsilon=EPSILON,
):
    """
    Extract all four distributional features.
    """

    scores = np.asarray(scores, dtype=float)

    stats = score_statistics(scores)

    return {
        "mean": stats["mean"],
        "std": stats["std"],
        "boundary_mass": boundary_mass(
            scores,
            threshold=threshold,
            epsilon=epsilon,
        ),
        "entropy": score_entropy(scores),
    }


def feature_vector(
    scores,
    selected_features,
):
    """
    Convert selected feature names into a numerical vector.
    """

    features = extract_features(scores)

    return np.array(
        [
            features[name]
            for name in selected_features
        ],
        dtype=float,
    )


def feature_distance(
    reference,
    current,
    scale,
):
    """
    Standardized Euclidean distance.
    """

    reference = np.asarray(
        reference,
        dtype=float,
    )

    current = np.asarray(
        current,
        dtype=float,
    )

    scale = np.asarray(
        scale,
        dtype=float,
    )

    scale = np.maximum(
        scale,
        1e-8,
    )

    return float(
        np.sqrt(
            np.mean(
                ((current - reference) / scale) ** 2
            )
        )
    )


def calibrate(
    stable_scores,
    selected_features,
):
    """
    Calibrate one feature configuration using
    stable data only.
    """

    reference_scores = stable_scores[
        :REFERENCE_SIZE
    ]

    reference_features = feature_vector(
        reference_scores,
        selected_features,
    )

    calibration_features = []

    start = REFERENCE_SIZE

    while (
        start + CALIBRATION_WINDOW
        <= len(stable_scores)
    ):

        current_scores = stable_scores[
            start:
            start + CALIBRATION_WINDOW
        ]

        features = feature_vector(
            current_scores,
            selected_features,
        )

        calibration_features.append(
            features
        )

        start += CALIBRATION_STEP

    calibration_features = np.asarray(
        calibration_features
    )

    scale = np.std(
        calibration_features,
        axis=0,
    )

    scale = np.maximum(
        scale,
        1e-6,
    )

    distances = np.array(
        [
            feature_distance(
                reference_features,
                features,
                scale,
            )
            for features in calibration_features
        ]
    )

    threshold = float(
        np.quantile(
            distances,
            QUANTILE,
        )
    )

    return {
        "reference": reference_features,
        "scale": scale,
        "threshold": threshold,
    }


def detect(
    scores,
    calibration,
    selected_features,
):
    """
    Run the detector over the complete stream.
    """

    detections = []

    start = 0

    while (
        start + DETECTION_WINDOW
        <= len(scores)
    ):

        current_scores = scores[
            start:
            start + DETECTION_WINDOW
        ]

        current_features = feature_vector(
            current_scores,
            selected_features,
        )

        distance = feature_distance(
            calibration["reference"],
            current_features,
            calibration["scale"],
        )

        if distance > calibration["threshold"]:

            detections.append(
                {
                    "start": start,
                    "distance": distance,
                }
            )

        start += DETECTION_STEP

    return detections


def evaluate_variant(
    scores,
    shift_name,
    variant_name,
    selected_features,
):
    """
    Evaluate one feature configuration.
    """

    stable_scores = scores[
        :TRUE_DRIFT
    ]

    calibration = calibrate(
        stable_scores,
        selected_features,
    )

    detections = detect(
        scores,
        calibration,
        selected_features,
    )

    false_alarms = [
        d
        for d in detections
        if d["start"] < TRUE_DRIFT
    ]

    post_drift = [
        d
        for d in detections
        if d["start"] >= TRUE_DRIFT
    ]

    if post_drift:

        first_detection = min(
            d["start"]
            for d in post_drift
        )

        detection_delay = (
            first_detection
            - TRUE_DRIFT
        )

    else:

        first_detection = np.nan
        detection_delay = np.nan

    stable_windows = len(
        range(
            0,
            TRUE_DRIFT - DETECTION_WINDOW + 1,
            DETECTION_STEP,
        )
    )

    false_alarm_rate = (
        len(false_alarms)
        / stable_windows
    )

    return {
        "shift": shift_name,
        "variant": variant_name,
        "features": "+".join(
            selected_features
        ),
        "calibration_threshold":
            calibration["threshold"],
        "total_detections":
            len(detections),
        "false_alarms":
            len(false_alarms),
        "post_drift_detections":
            len(post_drift),
        "stable_windows":
            stable_windows,
        "false_alarm_rate":
            false_alarm_rate,
        "first_post_drift_detection":
            first_detection,
        "detection_delay":
            detection_delay,
        "detected":
            len(post_drift) > 0,
    }


# ==========================================================
# MAIN
# ==========================================================

print("=" * 80)
print("BOUNDARY-AWARE FEATURE ABLATION")
print("=" * 80)

print("\nConfiguration:")
print("Reference size:", REFERENCE_SIZE)
print("Calibration window:", CALIBRATION_WINDOW)
print("Calibration step:", CALIBRATION_STEP)
print("Detection window:", DETECTION_WINDOW)
print("Detection step:", DETECTION_STEP)
print("Quantile:", QUANTILE)
print("Boundary threshold:", BOUNDARY_THRESHOLD)
print("True drift:", TRUE_DRIFT)

results = []


for shift_name, path in STREAMS.items():

    print("\n")
    print("=" * 80)
    print(f"{shift_name.upper()} DRIFT")
    print("=" * 80)

    df = pd.read_csv(path)

    scores = df["score"].to_numpy(
        dtype=float
    )

    print("Stream size:", len(scores))

    for variant_name, selected_features in ABLATIONS.items():

        result = evaluate_variant(
            scores,
            shift_name,
            variant_name,
            selected_features,
        )

        results.append(result)

        print(
            f"{variant_name:15s} | "
            f"features={'+'.join(selected_features):35s} | "
            f"FA={result['false_alarm_rate']:.4f} | "
            f"delay={result['detection_delay']} | "
            f"detected={result['detected']}"
        )


results_df = pd.DataFrame(results)


print("\n\n")
print("=" * 100)
print("FEATURE ABLATION RESULTS")
print("=" * 100)

print(
    results_df[
        [
            "shift",
            "variant",
            "false_alarms",
            "post_drift_detections",
            "false_alarm_rate",
            "first_post_drift_detection",
            "detection_delay",
            "detected",
        ]
    ].to_string(index=False)
)


# ==========================================================
# Detector-level summary
# ==========================================================

summary = (
    results_df
    .groupby("variant")
    .agg(
        shifts_tested=("shift", "count"),
        shifts_detected=("detected", "sum"),
        total_false_alarms=("false_alarms", "sum"),
        mean_false_alarm_rate=(
            "false_alarm_rate",
            "mean",
        ),
        mean_detection_delay=(
            "detection_delay",
            "mean",
        ),
    )
    .reset_index()
)


print("\n\n")
print("=" * 100)
print("ABLATION SUMMARY")
print("=" * 100)

print(
    summary.to_string(
        index=False
    )
)


# ==========================================================
# Save
# ==========================================================

output_path = (
    "results/"
    "boundary_feature_ablation.csv"
)

results_df.to_csv(
    output_path,
    index=False,
)

summary_path = (
    "results/"
    "boundary_feature_ablation_summary.csv"
)

summary.to_csv(
    summary_path,
    index=False,
)

print("\nSaved:")
print(output_path)
print(summary_path)
