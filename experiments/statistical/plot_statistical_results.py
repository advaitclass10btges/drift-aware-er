import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


SUMMARY = "results/statistical/bootstrap_detector_summary.csv"
RAW = "results/statistical/bootstrap_detector_results.csv"
PAIRED = "results/statistical/paired_detector_comparisons.csv"

OUT = "results/statistical/figures"
os.makedirs(OUT, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

summary = pd.read_csv(SUMMARY)
raw = pd.read_csv(RAW)
paired = pd.read_csv(PAIRED)


# ============================================================
# NORMALIZE NAMES
# ============================================================

summary["shift"] = summary["shift"].str.title()
paired["shift"] = paired["shift"].str.title()

summary["detector"] = summary["detector"].replace({
    "Boundary-Aware": "Boundary-Aware",
    "Page-Hinkley": "Page-Hinkley",
    "ADWIN": "ADWIN",
})

paired["baseline"] = paired["baseline"].replace({
    "Page-Hinkley": "Page-Hinkley",
    "ADWIN": "ADWIN",
})


# Consistent ordering
shift_order = [
    "Title",
    "Naming",
    "Missingness",
]

detector_order = [
    "ADWIN",
    "Page-Hinkley",
    "Boundary-Aware",
]


# ============================================================
# FIGURE 1
# DETECTION PROBABILITY WITH 95% CI
# ============================================================

fig, ax = plt.subplots(figsize=(9, 6))

for detector in detector_order:

    d = summary[
        summary["detector"] == detector
    ].copy()

    d["shift"] = pd.Categorical(
        d["shift"],
        categories=shift_order,
        ordered=True,
    )

    d = d.sort_values("shift")

    x = np.arange(len(d))

    ax.errorbar(
        x,
        d["detection_rate"],
        yerr=[
            d["detection_rate"]
            - d["detection_rate_ci_low"],
            d["detection_rate_ci_high"]
            - d["detection_rate"],
        ],
        marker="o",
        capsize=4,
        linewidth=2,
        label=detector,
    )

ax.set_xticks(range(len(shift_order)))
ax.set_xticklabels(shift_order)

ax.set_ylim(-0.05, 1.05)
ax.set_ylabel("Detection probability")
ax.set_xlabel("Drift type")
ax.set_title(
    "Drift Detection Probability Across 300 Block-Bootstrap Replications"
)

ax.legend()
ax.grid(axis="y", alpha=0.25)

fig.tight_layout()

fig.savefig(
    f"{OUT}/detection_probability_ci.png",
    dpi=300,
    bbox_inches="tight",
)

fig.savefig(
    f"{OUT}/detection_probability_ci.svg",
    bbox_inches="tight",
)

plt.close(fig)


# ============================================================
# FIGURE 2
# FALSE ALARM RATE WITH 95% CI
# ============================================================

fig, ax = plt.subplots(figsize=(9, 6))

for detector in detector_order:

    d = summary[
        summary["detector"] == detector
    ].copy()

    d["shift"] = pd.Categorical(
        d["shift"],
        categories=shift_order,
        ordered=True,
    )

    d = d.sort_values("shift")

    x = np.arange(len(d))

    ax.errorbar(
        x,
        d["mean_false_alarm_rate"],
        yerr=[
            d["mean_false_alarm_rate"]
            - d["false_alarm_rate_ci_low"],
            d["false_alarm_rate_ci_high"]
            - d["mean_false_alarm_rate"],
        ],
        marker="o",
        capsize=4,
        linewidth=2,
        label=detector,
    )

ax.set_xticks(range(len(shift_order)))
ax.set_xticklabels(shift_order)

ax.set_ylabel("False-alarm rate")
ax.set_xlabel("Drift type")
ax.set_title(
    "False-Alarm Rate Across 300 Block-Bootstrap Replications"
)

ax.legend()
ax.grid(axis="y", alpha=0.25)

fig.tight_layout()

fig.savefig(
    f"{OUT}/false_alarm_rate_ci.png",
    dpi=300,
    bbox_inches="tight",
)

fig.savefig(
    f"{OUT}/false_alarm_rate_ci.svg",
    bbox_inches="tight",
)

plt.close(fig)


# ============================================================
# FIGURE 3
# DETECTION DELAY WITH 95% CI
# ============================================================

fig, ax = plt.subplots(figsize=(9, 6))

for detector in detector_order:

    d = summary[
        summary["detector"] == detector
    ].copy()

    d["shift"] = pd.Categorical(
        d["shift"],
        categories=shift_order,
        ordered=True,
    )

    d = d.sort_values("shift")

    # Only plot detectors that actually have delay estimates.
    d = d[np.isfinite(d["mean_detection_delay"])]

    if len(d) == 0:
        continue

    x = [
        shift_order.index(s)
        for s in d["shift"].astype(str)
    ]

    ax.errorbar(
        x,
        d["mean_detection_delay"],
        yerr=[
            d["mean_detection_delay"]
            - d["detection_delay_ci_low"],
            d["detection_delay_ci_high"]
            - d["mean_detection_delay"],
        ],
        marker="o",
        capsize=4,
        linewidth=2,
        label=detector,
    )

ax.axhline(
    0,
    linewidth=1,
)

ax.set_xticks(range(len(shift_order)))
ax.set_xticklabels(shift_order)

ax.set_ylabel("Detection delay (samples)")
ax.set_xlabel("Drift type")

ax.set_title(
    "Detection Delay Conditional on Successful Detection"
)

ax.legend()
ax.grid(axis="y", alpha=0.25)

fig.tight_layout()

fig.savefig(
    f"{OUT}/detection_delay_ci.png",
    dpi=300,
    bbox_inches="tight",
)

fig.savefig(
    f"{OUT}/detection_delay_ci.svg",
    bbox_inches="tight",
)

plt.close(fig)


# ============================================================
# FIGURE 4
# EFFECT SIZE FOREST PLOT
# ============================================================

effect_rows = []

for _, row in paired.iterrows():

    effect_rows.append({
        "label": (
            f"{row['shift']} vs "
            f"{row['baseline']}"
        ),
        "shift": row["shift"],
        "baseline": row["baseline"],
        "delta": row["detection_rate_cliffs_delta"],
        "magnitude": row["detection_rate_cliffs_magnitude"],
        "p": row["detection_rate_pvalue"],
    })

effects = pd.DataFrame(effect_rows)

fig, ax = plt.subplots(figsize=(10, 7))

if len(effects) > 0:

    y = np.arange(len(effects))

    ax.scatter(
        effects["delta"],
        y,
        s=70,
    )

    for i, row in effects.iterrows():

        if row["p"] < 0.001:
            marker = "***"
        elif row["p"] < 0.01:
            marker = "**"
        elif row["p"] < 0.05:
            marker = "*"
        else:
            marker = "ns"

        ax.text(
            row["delta"] + 0.02,
            i,
            marker,
            va="center",
        )

    ax.axvline(
        0,
        linewidth=1,
    )

    ax.set_yticks(y)
    ax.set_yticklabels(effects["label"])

ax.set_xlabel("Cliff's delta")
ax.set_title(
    "Effect Size of Boundary-Aware Detection Advantage"
)

ax.grid(axis="x", alpha=0.25)

fig.tight_layout()

fig.savefig(
    f"{OUT}/detection_effect_sizes.png",
    dpi=300,
    bbox_inches="tight",
)

fig.savefig(
    f"{OUT}/detection_effect_sizes.svg",
    bbox_inches="tight",
)

plt.close(fig)


# ============================================================
# FIGURE 5
# DETECTION RATE HEATMAP
# ============================================================

heat = summary.pivot(
    index="detector",
    columns="shift",
    values="detection_rate",
)

heat = heat.reindex(detector_order)
heat = heat.reindex(columns=shift_order)

fig, ax = plt.subplots(figsize=(8, 5))

im = ax.imshow(
    heat.values,
    aspect="auto",
)

ax.set_xticks(range(len(shift_order)))
ax.set_xticklabels(shift_order)

ax.set_yticks(range(len(detector_order)))
ax.set_yticklabels(detector_order)

for i in range(heat.shape[0]):
    for j in range(heat.shape[1]):

        value = heat.iloc[i, j]

        if pd.notna(value):

            ax.text(
                j,
                i,
                f"{value:.2f}",
                ha="center",
                va="center",
            )

ax.set_xlabel("Drift type")
ax.set_ylabel("Detector")

ax.set_title(
    "Detection Probability Matrix"
)

fig.colorbar(
    im,
    ax=ax,
    label="Detection probability",
)

fig.tight_layout()

fig.savefig(
    f"{OUT}/detection_probability_heatmap.png",
    dpi=300,
    bbox_inches="tight",
)

fig.savefig(
    f"{OUT}/detection_probability_heatmap.svg",
    bbox_inches="tight",
)

plt.close(fig)


# ============================================================
# FIGURE 6
# PERFORMANCE FRONTIER:
# DETECTION RATE VS FALSE-ALARM RATE
# ============================================================

fig, ax = plt.subplots(figsize=(8, 6))

for detector in detector_order:

    d = summary[
        summary["detector"] == detector
    ]

    ax.scatter(
        d["mean_false_alarm_rate"],
        d["detection_rate"],
        s=80,
        label=detector,
    )

    for _, row in d.iterrows():

        ax.annotate(
            row["shift"],
            (
                row["mean_false_alarm_rate"],
                row["detection_rate"],
            ),
            xytext=(5, 5),
            textcoords="offset points",
        )

ax.set_xlabel("Mean false-alarm rate")
ax.set_ylabel("Detection probability")

ax.set_ylim(-0.05, 1.05)

ax.set_title(
    "Detection–False-Alarm Performance Frontier"
)

ax.legend()
ax.grid(alpha=0.25)

fig.tight_layout()

fig.savefig(
    f"{OUT}/performance_frontier.png",
    dpi=300,
    bbox_inches="tight",
)

fig.savefig(
    f"{OUT}/performance_frontier.svg",
    bbox_inches="tight",
)

plt.close(fig)


# ============================================================
# SUMMARY
# ============================================================

print("=" * 70)
print("STATISTICAL FIGURES GENERATED")
print("=" * 70)

for filename in sorted(os.listdir(OUT)):
    print(f"{OUT}/{filename}")

print("=" * 70)
