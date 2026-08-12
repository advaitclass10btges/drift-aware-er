import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv("results/shift_summary.csv")


# --------------------------------------------------
# Figure 1: Mean score before vs after
# --------------------------------------------------

plt.figure(figsize=(10, 6))

x = range(len(df))
width = 0.35

plt.bar(
    [i - width / 2 for i in x],
    df["mean_before"],
    width=width,
    label="Before shift",
)

plt.bar(
    [i + width / 2 for i in x],
    df["mean_after"],
    width=width,
    label="After shift",
)

plt.xticks(
    list(x),
    df["shift"],
)

plt.ylabel("Mean matcher score")
plt.title("Matcher Score Change Under Different Shifts")
plt.legend()

plt.tight_layout()

plt.savefig(
    "results/shift_comparison.png",
    dpi=200,
)

plt.close()


# --------------------------------------------------
# Figure 2: F1 degradation
# --------------------------------------------------

plt.figure(figsize=(10, 6))

plt.bar(
    [i - width / 2 for i in x],
    df["f1_before"],
    width=width,
    label="Before shift",
)

plt.bar(
    [i + width / 2 for i in x],
    df["f1_after"],
    width=width,
    label="After shift",
)

plt.xticks(
    list(x),
    df["shift"],
)

plt.ylabel("F1 score")
plt.title("Matcher Performance Under Distribution Shifts")
plt.legend()

plt.tight_layout()

plt.savefig(
    "results/f1_degradation.png",
    dpi=200,
)

plt.close()


# --------------------------------------------------
# Figure 3: Boundary mass
# --------------------------------------------------

plt.figure(figsize=(10, 6))

plt.bar(
    [i - width / 2 for i in x],
    df["boundary_mass_before"],
    width=width,
    label="Before shift",
)

plt.bar(
    [i + width / 2 for i in x],
    df["boundary_mass_after"],
    width=width,
    label="After shift",
)

plt.xticks(
    list(x),
    df["shift"],
)

plt.ylabel("Boundary mass")
plt.title("Boundary-Aware Uncertainty Under Distribution Shifts")
plt.legend()

plt.tight_layout()

plt.savefig(
    "results/boundary_mass_comparison.png",
    dpi=200,
)

plt.close()


print("Saved:")
print("results/shift_comparison.png")
print("results/f1_degradation.png")
print("results/boundary_mass_comparison.png")
