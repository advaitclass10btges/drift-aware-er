import pandas as pd

from src.evaluation.drift_metrics import (
    score_statistics,
    boundary_mass,
    score_entropy,
)

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
)


THRESHOLD = 0.36
EPSILON = 0.05


def analyze_shift(name, path):
    df = pd.read_csv(path)

    before = df["base_score"]
    after = df["shift_score"]

    before_pred = (before >= THRESHOLD).astype(int)
    after_pred = (after >= THRESHOLD).astype(int)

    before_stats = score_statistics(before)
    after_stats = score_statistics(after)

    before_boundary = boundary_mass(
        before,
        threshold=THRESHOLD,
        epsilon=EPSILON,
    )

    after_boundary = boundary_mass(
        after,
        threshold=THRESHOLD,
        epsilon=EPSILON,
    )

    before_entropy = score_entropy(before)
    after_entropy = score_entropy(after)

    return {
        "shift": name,

        "mean_before": before_stats["mean"],
        "mean_after": after_stats["mean"],
        "mean_change": (
            after_stats["mean"] -
            before_stats["mean"]
        ),

        "std_before": before_stats["std"],
        "std_after": after_stats["std"],

        "median_before": before_stats["median"],
        "median_after": after_stats["median"],

        "boundary_mass_before": before_boundary,
        "boundary_mass_after": after_boundary,

        "entropy_before": before_entropy,
        "entropy_after": after_entropy,

        "precision_before": precision_score(
            df["label"],
            before_pred,
            zero_division=0,
        ),

        "precision_after": precision_score(
            df["label"],
            after_pred,
            zero_division=0,
        ),

        "recall_before": recall_score(
            df["label"],
            before_pred,
            zero_division=0,
        ),

        "recall_after": recall_score(
            df["label"],
            after_pred,
            zero_division=0,
        ),

        "f1_before": f1_score(
            df["label"],
            before_pred,
            zero_division=0,
        ),

        "f1_after": f1_score(
            df["label"],
            after_pred,
            zero_division=0,
        ),
    }


experiments = [
    (
        "Naming shift",
        "results/naming_shift_scores.csv",
    ),
    (
        "Missingness shift",
        "results/missingness_shift_scores.csv",
    ),
    (
        "Title shift",
        "results/title_shift_scores.csv",
    ),
]


results = []

for name, path in experiments:
    print(f"\nAnalyzing: {name}")
    results.append(
        analyze_shift(name, path)
    )


summary = pd.DataFrame(results)

pd.set_option(
    "display.max_columns",
    None,
)

pd.set_option(
    "display.width",
    200,
)

pd.set_option(
    "display.float_format",
    lambda x: f"{x:.4f}",
)


print("\n")
print("=" * 100)
print("SHIFT EXPERIMENT SUMMARY")
print("=" * 100)

print(summary.to_string(index=False))

summary.to_csv(
    "results/shift_summary.csv",
    index=False,
)

print("\nSaved:")
print("results/shift_summary.csv")
