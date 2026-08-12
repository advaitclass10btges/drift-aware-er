import pandas as pd
import numpy as np

from src.drift.boundary_detector import (
    calibrate_detector,
    detect,
)


STREAM_PATH = "results/title_drift_stream.csv"

TRUE_DRIFT = 1375

REFERENCE_SIZE = 500
CALIBRATION_END = TRUE_DRIFT

WINDOW_SIZE = 100
STEP = 25

THRESHOLD = 0.36
EPSILON = 0.05


df = pd.read_csv(STREAM_PATH)

scores = df["score"].to_numpy(dtype=float)


# --------------------------------------------------
# Stable phase used for calibration
# --------------------------------------------------

stable_scores = scores[:CALIBRATION_END]


calibration = calibrate_detector(
    stable_scores,
    reference_size=REFERENCE_SIZE,
    window_size=WINDOW_SIZE,
    step=50,
    threshold=THRESHOLD,
    epsilon=EPSILON,
    quantile=0.99,
)


print("\n==============================")
print("BOUNDARY-AWARE DETECTOR")
print("==============================")


print("\nReference features:")

feature_names = [
    "mean",
    "std",
    "boundary_mass",
    "entropy",
]

for name, value in zip(
    feature_names,
    calibration["reference_features"],
):
    print(
        f"{name}: {value:.6f}"
    )


print("\nFeature scales:")

for name, value in zip(
    feature_names,
    calibration["scale"],
):
    print(
        f"{name}: {value:.6f}"
    )


print("\nCalibration threshold:")
print(
    calibration["threshold"]
)


print("\nStable calibration distances:")
print(
    {
        "mean":
            float(
                np.mean(
                    calibration[
                        "calibration_distances"
                    ]
                )
            ),
        "max":
            float(
                np.max(
                    calibration[
                        "calibration_distances"
                    ]
                )
            ),
        "99th_percentile":
            float(
                np.quantile(
                    calibration[
                        "calibration_distances"
                    ],
                    0.99,
                )
            ),
    }
)


# --------------------------------------------------
# Detect over complete stream
# --------------------------------------------------

detections = detect(
    scores,
    calibration,
    window_size=WINDOW_SIZE,
    step=STEP,
    threshold=THRESHOLD,
    epsilon=EPSILON,
)


# --------------------------------------------------
# Evaluate detections
# --------------------------------------------------

print("\nTrue drift point:")
print(TRUE_DRIFT)


print("\nNumber of detections:")
print(len(detections))


if detections:

    print("\nFirst detections:")

    for detection in detections[:10]:

        print(
            {
                "start":
                    detection["start"],
                "end":
                    detection["end"],
                "distance":
                    round(
                        detection["distance"],
                        6,
                    ),
            }
        )


    post_drift = [
        d
        for d in detections
        if d["start"] >= TRUE_DRIFT
    ]


    if post_drift:

        first = post_drift[0]

        detection_point = first["start"]

        delay = (
            detection_point -
            TRUE_DRIFT
        )

        print("\nFirst post-drift detection:")
        print(detection_point)

        print("\nDetection delay:")
        print(delay)

    else:

        print(
            "\nDetections occurred, "
            "but none started after the true drift point."
        )

else:

    print("\nNo detection.")


# --------------------------------------------------
# Save detections
# --------------------------------------------------

rows = []

for detection in detections:

    rows.append({
        "start":
            detection["start"],

        "end":
            detection["end"],

        "distance":
            detection["distance"],
    })


result = pd.DataFrame(rows)

result.to_csv(
    "results/boundary_detector_detections.csv",
    index=False,
)


print("\nSaved:")
print(
    "results/boundary_detector_detections.csv"
)
