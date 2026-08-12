import pandas as pd
import matplotlib.pyplot as plt


scores = pd.read_csv(
    "results/amazon_google_score_stream.csv"
)


plt.figure(figsize=(8,5))

plt.hist(
    scores["score"],
    bins=50
)

plt.xlabel("Matcher score")
plt.ylabel("Frequency")
plt.title("Amazon-Google Matcher Score Distribution")

plt.savefig(
    "results/score_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

print("Saved results/score_distribution.png")
