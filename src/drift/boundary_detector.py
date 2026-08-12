import numpy as np

from src.evaluation.drift_metrics import (
    score_statistics,
    boundary_mass,
    score_entropy,
)


def window_features(scores, threshold=0.36, epsilon=0.05):
    """
    Extract the four distributional signals used by the
    boundary-aware drift detector.
    """

    scores = np.asarray(scores, dtype=float)

    stats = score_statistics(scores)

    return np.array([
        stats["mean"],
        stats["std"],
        boundary_mass(
            scores,
            threshold=threshold,
            epsilon=epsilon,
        ),
        score_entropy(scores),
    ], dtype=float)


def feature_distance(reference, current, scale):
    """
    Standardized Euclidean distance between two feature vectors.
    """

    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)
    scale = np.asarray(scale, dtype=float)

    scale = np.maximum(scale, 1e-8)

    return float(
        np.sqrt(
            np.mean(
                ((current - reference) / scale) ** 2
            )
        )
    )


def calibrate_detector(
    scores,
    reference_size=500,
    window_size=100,
    step=50,
    threshold=0.36,
    epsilon=0.05,
    quantile=0.99,
):
    """
    Calibrate the detector using stable data only.

    The first reference_size samples form the reference
    distribution.

    Remaining stable windows estimate normal variation.
    """

    scores = np.asarray(scores, dtype=float)

    if len(scores) < reference_size + window_size:
        raise ValueError(
            "Not enough stable samples for calibration."
        )

    reference_scores = scores[:reference_size]

    reference_features = window_features(
        reference_scores,
        threshold=threshold,
        epsilon=epsilon,
    )

    calibration_features = []

    start = reference_size

    while start + window_size <= len(scores):

        current_scores = scores[
            start:start + window_size
        ]

        features = window_features(
            current_scores,
            threshold=threshold,
            epsilon=epsilon,
        )

        calibration_features.append(features)

        start += step

    calibration_features = np.asarray(
        calibration_features
    )

    # Robust scale estimated from stable windows.
    scale = np.std(
        calibration_features,
        axis=0
    )

    scale = np.maximum(scale, 1e-6)

    calibration_distances = np.array([
        feature_distance(
            reference_features,
            features,
            scale,
        )
        for features in calibration_features
    ])

    detection_threshold = float(
        np.quantile(
            calibration_distances,
            quantile,
        )
    )

    return {
        "reference_features": reference_features,
        "scale": scale,
        "threshold": detection_threshold,
        "calibration_distances": calibration_distances,
    }


def detect(
    scores,
    calibration,
    window_size=100,
    step=25,
    threshold=0.36,
    epsilon=0.05,
):
    """
    Apply the calibrated detector to a score stream.

    Returns every window whose multivariate distance
    exceeds the calibrated threshold.
    """

    scores = np.asarray(scores, dtype=float)

    reference_features = calibration[
        "reference_features"
    ]

    scale = calibration["scale"]

    detection_threshold = calibration[
        "threshold"
    ]

    detections = []

    start = 0

    while start + window_size <= len(scores):

        current_scores = scores[
            start:start + window_size
        ]

        current_features = window_features(
            current_scores,
            threshold=threshold,
            epsilon=epsilon,
        )

        distance = feature_distance(
            reference_features,
            current_features,
            scale,
        )

        if distance > detection_threshold:

            detections.append({
                "start": start,
                "end": start + window_size,
                "distance": distance,
                "features": current_features,
            })

        start += step

    return detections
