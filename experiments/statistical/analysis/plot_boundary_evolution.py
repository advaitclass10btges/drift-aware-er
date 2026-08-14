import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.evaluation.drift_metrics import boundary_mass


STREAMS = {
    "title": "results/title_drift_stream.csv",
    "naming": "results/naming_drift_stream.csv",
    "missingness": "results/missingness_drift_stream.csv",
}


TRUE_DRIFT = 1375

WINDOW_SIZE = 100
STEP = 25

THRESHOLD = 0.36
EPSILON = 0.05


OUT = "results/figures"

os.makedirs(
    OUT,
    exist_ok=True
)


def compute_boundary_evolution(scores):

    positions = []
    masses = []

    start = 0

    while start + WINDOW_SIZE <= len(scores):

        window = scores[
            start:start + WINDOW_SIZE
        ]

        mass = boundary_mass(
            window,
            threshold=THRESHOLD,
            epsilon=EPSILON,
        )

        positions.append(
            start + WINDOW_SIZE // 2
        )

        masses.append(
            mass
        )

        start += STEP


    return (
        np.array(positions),
        np.array(masses)
    )


fig, axes = plt.subplots(
    3,
    1,
    figsize=(10, 9),
    sharex=True
)


for ax, (shift, path) in zip(
    axes,
    STREAMS.items()
):

    df = pd.read_csv(path)

    scores = df["score"].to_numpy(
        dtype=float
    )


    x, y = compute_boundary_evolution(
        scores
    )


    ax.plot(
        x,
        y,
        linewidth=2,
    )


    ax.axvline(
        TRUE_DRIFT,
        linestyle="--",
        linewidth=2,
    )


    ax.set_ylabel(
        "Boundary Mass"
    )

    ax.set_title(
        shift.title()
        + " Drift"
    )


    ax.grid(
        alpha=0.3
    )


axes[-1].set_xlabel(
    "Stream Position"
)


fig.suptitle(
    "Evolution of Decision-Boundary Instability Under Entity Resolution Drift",
    fontsize=14,
)


fig.tight_layout()


fig.savefig(
    f"{OUT}/boundary_evolution.png",
    dpi=300,
    bbox_inches="tight",
)


fig.savefig(
    f"{OUT}/boundary_evolution.svg",
    bbox_inches="tight",
)


print(
    "Saved:",
    f"{OUT}/boundary_evolution.png"
)
