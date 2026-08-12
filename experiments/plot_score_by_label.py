import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv(
    "results/amazon_google_score_stream.csv"
)


plt.figure(figsize=(8,5))

plt.hist(
    df[df["label"] == 0]["score"],
    bins=50,
    alpha=0.7,
    label="Non-match"
)

plt.hist(
    df[df["label"] == 1]["score"],
    bins=50,
    alpha=0.7,
    label="Match"
)


plt.xlabel("Matcher score")
plt.ylabel("Frequency")
plt.title("Amazon-Google Score Distribution by Ground Truth")

plt.legend()

plt.savefig(
    "results/score_distribution_by_label.png",
    dpi=300,
    bbox_inches="tight"
)

print("Saved")
