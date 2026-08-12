import numpy as np
import pandas as pd

from river.drift import ADWIN, PageHinkley

from src.drift.boundary_detector import (
    calibrate_detector,
    detect,
)


# ============================================================
# CONFIGURATION
# ============================================================

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

N_REPS = 300
BLOCK_SIZE = 50

RANDOM_SEED = 20260812

CI_CONFIDENCE = 0.95
BOOTSTRAP_CI_REPS = 5000
PERMUTATIONS = 10000


# ============================================================
# DATA LOADING
# ============================================================

def load_stream(path):
    """
    Load a score stream from CSV.
    """
    df = pd.read_csv(path)

    if "score" not in df.columns:
        raise ValueError(
            f"'score' column not found in {path}"
        )

    return df["score"].to_numpy(dtype=float)


# ============================================================
# MOVING BLOCK BOOTSTRAP
# ============================================================

def moving_block_bootstrap_segment(
    segment,
    target_size,
    block_size,
    rng,
):
    """
    Moving block bootstrap.

    Samples contiguous blocks with replacement so that
    local temporal dependence is partially preserved.
    """

    segment = np.asarray(segment, dtype=float)

    if len(segment) < block_size:
        raise ValueError(
            "Segment shorter than block size."
        )

    blocks = []
    current_size = 0

    while current_size < target_size:

        start = rng.integers(
            0,
            len(segment) - block_size + 1,
        )

        block = segment[
            start:start + block_size
        ]

        blocks.append(block)
        current_size += len(block)

    result = np.concatenate(blocks)

    return result[:target_size]


def bootstrap_stream(scores, rng):
    """
    Resample stable and drift regions separately.

    This preserves the known drift location while allowing
    repeated stochastic realizations of both regimes.
    """

    stable = scores[:TRUE_DRIFT]
    drift = scores[TRUE_DRIFT:]

    stable_boot = moving_block_bootstrap_segment(
        stable,
        len(stable),
        BLOCK_SIZE,
        rng,
    )

    drift_boot = moving_block_bootstrap_segment(
        drift,
        len(drift),
        BLOCK_SIZE,
        rng,
    )

    return np.concatenate(
        [stable_boot, drift_boot]
    )


# ============================================================
# BASELINE DETECTORS
# ============================================================

def evaluate_baseline(detector, scores):
    """
    Evaluate ADWIN or Page-Hinkley.
    """

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
        detected = True

    else:

        first = np.nan
        delay = np.nan
        detected = False

    stable_windows = max(
        1,
        (TRUE_DRIFT - WINDOW_SIZE)
        // DETECTION_STEP
        + 1,
    )

    false_alarm_rate = (
        len(pre) / stable_windows
    )

    return {
        "total_detections": len(detections),
        "false_alarms": len(pre),
        "post_drift_detections": len(post),
        "false_alarm_rate": false_alarm_rate,
        "first_post_drift_detection": first,
        "detection_delay": delay,
        "detected": detected,
    }


# ============================================================
# BOUNDARY-AWARE DETECTOR
# ============================================================

def evaluate_boundary(scores):
    """
    Evaluate the Boundary-Aware detector.
    """

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
        detected = True

    else:

        first = np.nan
        delay = np.nan
        detected = False

    stable_windows = max(
        1,
        (TRUE_DRIFT - WINDOW_SIZE)
        // DETECTION_STEP
        + 1,
    )

    false_alarm_rate = (
        len(pre) / stable_windows
    )

    return {
        "total_detections": len(detections),
        "false_alarms": len(pre),
        "post_drift_detections": len(post),
        "false_alarm_rate": false_alarm_rate,
        "first_post_drift_detection": first,
        "detection_delay": delay,
        "detected": detected,
        "calibration_threshold": calibration["threshold"],
    }


# ============================================================
# EVALUATE ALL DETECTORS
# ============================================================

def evaluate_all(scores):
    """
    Evaluate all three detectors on exactly the same
    bootstrap stream.
    """

    results = []

    detectors = {
        "ADWIN": ADWIN(),
        "Page-Hinkley": PageHinkley(),
    }

    for name, detector in detectors.items():

        result = evaluate_baseline(
            detector,
            scores,
        )

        results.append({
            "detector": name,
            **result,
        })

    boundary_result = evaluate_boundary(
        scores
    )

    results.append({
        "detector": "Boundary-Aware",
        **boundary_result,
    })

    return results


# ============================================================
# BOOTSTRAP CONFIDENCE INTERVAL
# ============================================================

def bootstrap_ci(
    values,
    statistic=np.mean,
    confidence=CI_CONFIDENCE,
    seed=RANDOM_SEED + 999,
):
    """
    Non-parametric percentile bootstrap confidence interval.
    """

    values = np.asarray(
        values,
        dtype=float,
    )

    values = values[
        np.isfinite(values)
    ]

    if len(values) == 0:
        return np.nan, np.nan, np.nan

    rng = np.random.default_rng(seed)

    boot_statistics = []

    for _ in range(
        BOOTSTRAP_CI_REPS
    ):

        sample = rng.choice(
            values,
            size=len(values),
            replace=True,
        )

        boot_statistics.append(
            statistic(sample)
        )

    alpha = 1.0 - confidence

    lower = np.quantile(
        boot_statistics,
        alpha / 2,
    )

    upper = np.quantile(
        boot_statistics,
        1 - alpha / 2,
    )

    return (
        statistic(values),
        lower,
        upper,
    )


# ============================================================
# PAIRED PERMUTATION TEST
# ============================================================

def paired_permutation_pvalue(
    x,
    y,
    statistic=np.mean,
    n_permutations=PERMUTATIONS,
    seed=20260813,
):
    """
    Paired randomization test.

    H0:
        The paired difference between x and y
        is exchangeable around zero.
    """

    x = np.asarray(
        x,
        dtype=float,
    )

    y = np.asarray(
        y,
        dtype=float,
    )

    mask = (
        np.isfinite(x)
        & np.isfinite(y)
    )

    x = x[mask]
    y = y[mask]

    if len(x) == 0:
        return np.nan

    differences = x - y

    observed = statistic(
        differences
    )

    rng = np.random.default_rng(
        seed
    )

    extreme = 0

    for _ in range(
        n_permutations
    ):

        signs = rng.choice(
            [-1.0, 1.0],
            size=len(differences),
        )

        permuted = statistic(
            differences * signs
        )

        if abs(permuted) >= abs(observed):
            extreme += 1

    return (
        extreme + 1
    ) / (
        n_permutations + 1
    )


# ============================================================
# CLIFF'S DELTA
# ============================================================

def cliffs_delta(x, y):
    """
    Cliff's delta effect size.

    Positive values indicate x tends to be larger than y.

    delta = P(x > y) - P(x < y)
    """

    x = np.asarray(
        x,
        dtype=float,
    )

    y = np.asarray(
        y,
        dtype=float,
    )

    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]

    if len(x) == 0 or len(y) == 0:
        return np.nan

    greater = 0
    less = 0

    for xi in x:

        greater += np.sum(
            xi > y
        )

        less += np.sum(
            xi < y
        )

    return (
        greater - less
    ) / (
        len(x) * len(y)
    )


# ============================================================
# EFFECT SIZE INTERPRETATION
# ============================================================

def interpret_cliffs_delta(delta):
    """
    Interpret Cliff's delta magnitude.
    """

    if not np.isfinite(delta):
        return "undefined"

    magnitude = abs(delta)

    if magnitude < 0.147:
        return "negligible"

    if magnitude < 0.33:
        return "small"

    if magnitude < 0.474:
        return "medium"

    return "large"


# ============================================================
# RUN BLOCK-BOOTSTRAP EXPERIMENT
# ============================================================

def run_bootstrap_experiment():

    raw_results = []

    master_rng = np.random.default_rng(
        RANDOM_SEED
    )

    for shift, path in STREAMS.items():

        print("\n" + "=" * 70)
        print(f"{shift.upper()} DRIFT")
        print("=" * 70)

        original_scores = load_stream(
            path
        )

        for rep in range(
            N_REPS
        ):

            rng = np.random.default_rng(
                master_rng.integers(
                    0,
                    2**63 - 1,
                )
            )

            boot_scores = bootstrap_stream(
                original_scores,
                rng,
            )

            evaluation = evaluate_all(
                boot_scores
            )

            for result in evaluation:

                raw_results.append({
                    "shift": shift,
                    "rep": rep,
                    **result,
                })

        print(
            f"Completed {N_REPS} repetitions."
        )

    raw_df = pd.DataFrame(
        raw_results
    )

    output_path = (
        "results/statistical/"
        "bootstrap_detector_results.csv"
    )

    raw_df.to_csv(
        output_path,
        index=False,
    )

    print("\nSaved:")
    print(output_path)

    return raw_df


# ============================================================
# SUMMARIZE BOOTSTRAP RESULTS
# ============================================================

def summarize_bootstrap(raw_df):

    summary = []

    for (
        shift,
        detector,
    ), group in raw_df.groupby(
        ["shift", "detector"]
    ):

        detection_rate = (
            group["detected"]
            .astype(float)
            .mean()
        )

        _, detection_low, detection_high = (
            bootstrap_ci(
                group["detected"].astype(float)
            )
        )

        mean_far = (
            group["false_alarm_rate"]
            .mean()
        )

        _, far_low, far_high = (
            bootstrap_ci(
                group["false_alarm_rate"]
            )
        )

        delay_values = group[
            "detection_delay"
        ].to_numpy(
            dtype=float
        )

        if np.isfinite(
            delay_values
        ).any():

            mean_delay, delay_low, delay_high = (
                bootstrap_ci(
                    delay_values
                )
            )

        else:

            mean_delay = np.nan
            delay_low = np.nan
            delay_high = np.nan

        summary.append({
            "shift": shift,
            "detector": detector,
            "repetitions": len(group),

            "detection_rate":
                detection_rate,

            "detection_rate_ci_low":
                detection_low,

            "detection_rate_ci_high":
                detection_high,

            "mean_false_alarm_rate":
                mean_far,

            "false_alarm_rate_ci_low":
                far_low,

            "false_alarm_rate_ci_high":
                far_high,

            "mean_detection_delay":
                mean_delay,

            "detection_delay_ci_low":
                delay_low,

            "detection_delay_ci_high":
                delay_high,
        })

    summary_df = pd.DataFrame(
        summary
    )

    output_path = (
        "results/statistical/"
        "bootstrap_detector_summary.csv"
    )

    summary_df.to_csv(
        output_path,
        index=False,
    )

    return summary_df


# ============================================================
# PAIRED DETECTOR COMPARISONS
# ============================================================

def paired_comparisons(raw_df):

    comparisons = []

    baseline_detectors = [
        "ADWIN",
        "Page-Hinkley",
    ]

    for shift in STREAMS.keys():

        boundary = raw_df[
            (raw_df["shift"] == shift)
            & (
                raw_df["detector"]
                == "Boundary-Aware"
            )
        ].sort_values("rep")

        for baseline_name in (
            baseline_detectors
        ):

            baseline = raw_df[
                (raw_df["shift"] == shift)
                & (
                    raw_df["detector"]
                    == baseline_name
                )
            ].sort_values("rep")

            # ------------------------------------------------
            # Detection rate comparison
            # ------------------------------------------------

            boundary_detection = (
                boundary["detected"]
                .astype(float)
                .to_numpy()
            )

            baseline_detection = (
                baseline["detected"]
                .astype(float)
                .to_numpy()
            )

            detection_difference = (
                boundary_detection
                - baseline_detection
            )

            detection_rate_difference = (
                np.mean(
                    detection_difference
                )
            )

            detection_p = (
                paired_permutation_pvalue(
                    boundary_detection,
                    baseline_detection,
                    statistic=np.mean,
                    seed=20260820,
                )
            )

            detection_cliff = (
                cliffs_delta(
                    boundary_detection,
                    baseline_detection,
                )
            )

            # ------------------------------------------------
            # False alarm comparison
            # ------------------------------------------------

            boundary_far = (
                boundary[
                    "false_alarm_rate"
                ].to_numpy(dtype=float)
            )

            baseline_far = (
                baseline[
                    "false_alarm_rate"
                ].to_numpy(dtype=float)
            )

            far_difference = (
                boundary_far
                - baseline_far
            )

            mean_far_difference = (
                np.nanmean(
                    far_difference
                )
            )

            far_p = (
                paired_permutation_pvalue(
                    boundary_far,
                    baseline_far,
                    statistic=np.mean,
                    seed=20260821,
                )
            )

            far_cliff = (
                cliffs_delta(
                    boundary_far,
                    baseline_far,
                )
            )

            # ------------------------------------------------
            # Detection delay comparison
            # ------------------------------------------------

            boundary_delay = (
                boundary[
                    "detection_delay"
                ].to_numpy(dtype=float)
            )

            baseline_delay = (
                baseline[
                    "detection_delay"
                ].to_numpy(dtype=float)
            )

            paired_delay_mask = (
                np.isfinite(boundary_delay)
                & np.isfinite(baseline_delay)
            )

            boundary_delay_paired = (
                boundary_delay[
                    paired_delay_mask
                ]
            )

            baseline_delay_paired = (
                baseline_delay[
                    paired_delay_mask
                ]
            )

            if len(
                boundary_delay_paired
            ) > 0:

                mean_delay_difference = (
                    np.mean(
                        boundary_delay_paired
                        - baseline_delay_paired
                    )
                )

                delay_p = (
                    paired_permutation_pvalue(
                        boundary_delay_paired,
                        baseline_delay_paired,
                        statistic=np.mean,
                        seed=20260822,
                    )
                )

                delay_cliff = (
                    cliffs_delta(
                        boundary_delay_paired,
                        baseline_delay_paired,
                    )
                )

            else:

                paired_delay_count = 0
                mean_delay_difference = np.nan
                delay_p = np.nan
                delay_cliff = np.nan

            paired_delay_count = len(
                boundary_delay_paired
            )

            comparisons.append({
                "shift": shift,
                "baseline": baseline_name,

                "mean_detection_rate_difference":
                    detection_rate_difference,

                "detection_rate_pvalue":
                    detection_p,

                "detection_rate_cliffs_delta":
                    detection_cliff,

                "detection_rate_cliffs_magnitude":
                    interpret_cliffs_delta(
                        detection_cliff
                    ),

                "mean_far_difference":
                    mean_far_difference,

                "far_pvalue":
                    far_p,

                "far_cliffs_delta":
                    far_cliff,

                "far_cliffs_magnitude":
                    interpret_cliffs_delta(
                        far_cliff
                    ),

                "paired_delay_n":
                    paired_delay_count,

                "mean_delay_difference":
                    mean_delay_difference,

                "delay_pvalue":
                    delay_p,

                "delay_cliffs_delta":
                    delay_cliff,

                "delay_cliffs_magnitude":
                    interpret_cliffs_delta(
                        delay_cliff
                    ),
            })

    comparison_df = pd.DataFrame(
        comparisons
    )

    output_path = (
        "results/statistical/"
        "paired_detector_comparisons.csv"
    )

    comparison_df.to_csv(
        output_path,
        index=False,
    )

    return comparison_df


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print(
        "V0.2 BLOCK-BOOTSTRAP "
        "STATISTICAL VALIDATION"
    )
    print("=" * 70)

    print(f"Repetitions: {N_REPS}")
    print(f"Block size: {BLOCK_SIZE}")
    print(f"Seed: {RANDOM_SEED}")
    print(f"True drift: {TRUE_DRIFT}")

    # --------------------------------------------------------
    # 1. Run repeated block bootstrap
    # --------------------------------------------------------

    raw = run_bootstrap_experiment()

    # --------------------------------------------------------
    # 2. Bootstrap confidence intervals
    # --------------------------------------------------------

    summary = summarize_bootstrap(
        raw
    )

    # --------------------------------------------------------
    # 3. Paired detector comparisons
    # --------------------------------------------------------

    comparisons = paired_comparisons(
        raw
    )

    # --------------------------------------------------------
    # 4. Print summary
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("BOOTSTRAP SUMMARY")
    print("=" * 70)

    print(
        summary.to_string(
            index=False
        )
    )

    print("\n" + "=" * 70)
    print("PAIRED DETECTOR COMPARISONS")
    print("=" * 70)

    print(
        comparisons.to_string(
            index=False
        )
    )

    print("\n" + "=" * 70)
    print("FILES SAVED")
    print("=" * 70)

    print(
        "results/statistical/"
        "bootstrap_detector_results.csv"
    )

    print(
        "results/statistical/"
        "bootstrap_detector_summary.csv"
    )

    print(
        "results/statistical/"
        "paired_detector_comparisons.csv"
    )


if __name__ == "__main__":
    main()

