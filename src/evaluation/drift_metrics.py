import numpy as np
import pandas as pd


def score_statistics(scores):
    """
    Basic score distribution statistics.
    """

    scores = np.asarray(scores)

    return {
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
        "min": float(np.min(scores)),
        "max": float(np.max(scores)),
        "median": float(np.median(scores))
    }



def boundary_mass(scores, threshold=0.36, epsilon=0.05):
    """
    Fraction of scores near decision boundary.

    High boundary mass means more uncertain decisions.
    """

    scores = np.asarray(scores)

    return float(
        np.mean(
            np.abs(scores - threshold) < epsilon
        )
    )



def score_entropy(scores, bins=50):
    """
    Estimate entropy of score distribution.

    Higher entropy:
        less concentrated confidence
    """

    hist, _ = np.histogram(
        scores,
        bins=bins,
        range=(0,1),
        density=True
    )

    hist = hist[hist > 0]

    probabilities = hist / np.sum(hist)

    entropy = -np.sum(
        probabilities *
        np.log(probabilities)
    )

    return float(entropy)



def classification_metrics(
    labels,
    predictions
):
    """
    Classification degradation metrics.
    """

    from sklearn.metrics import (
        precision_score,
        recall_score,
        f1_score
    )


    return {
        "precision":
            float(
                precision_score(
                    labels,
                    predictions
                )
            ),

        "recall":
            float(
                recall_score(
                    labels,
                    predictions
                )
            ),

        "f1":
            float(
                f1_score(
                    labels,
                    predictions
                )
            )
    }



def compare_distributions(
    baseline_scores,
    current_scores
):
    """
    Compare score distributions.

    Simple statistics shift.
    """

    base = score_statistics(
        baseline_scores
    )

    current = score_statistics(
        current_scores
    )


    return {

        "mean_shift":
            current["mean"] -
            base["mean"],

        "std_shift":
            current["std"] -
            base["std"],

        "median_shift":
            current["median"] -
            base["median"]
    }
