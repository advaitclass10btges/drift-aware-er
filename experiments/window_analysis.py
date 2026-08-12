import pandas as pd

from src.evaluation.drift_metrics import (
    score_statistics,
    boundary_mass,
    score_entropy
)


df = pd.read_csv(
    "results/amazon_google_score_stream.csv"
)


scores = df["score"]


WINDOW = 500


results = []


for start in range(
    0,
    len(scores),
    WINDOW
):

    end = min(
        start + WINDOW,
        len(scores)
    )

    window_scores = scores[start:end]


    stats = score_statistics(
        window_scores
    )


    results.append({

        "start": start,
        "end": end,

        "mean":
            stats["mean"],

        "std":
            stats["std"],

        "entropy":
            score_entropy(
                window_scores
            ),

        "boundary_mass":
            boundary_mass(
                window_scores,
                threshold=0.36
            )
    })


result_df = pd.DataFrame(results)


print(result_df)


result_df.to_csv(
    "results/window_metrics.csv",
    index=False
)


print("\nSaved:")
print("results/window_metrics.csv")
