import os
import pandas as pd
import numpy as np
from pathlib import Path


RAW = Path(
    "datasets/real/dblp_scholar"
)

OUT = RAW / "processed"

OUT.mkdir(
    parents=True,
    exist_ok=True
)


np.random.seed(42)


# -----------------------------
# Load tables
# -----------------------------

dblp = pd.read_csv(
    RAW / "DBLP1.csv",
    encoding="latin1"
)

scholar = pd.read_csv(
    RAW / "Scholar.csv",
    encoding="latin1"
)


mapping = pd.read_csv(
    RAW / "DBLP-Scholar_perfectMapping.csv",
    encoding="latin1"
)

dblp = pd.read_csv(
    RAW / "DBLP1.csv",
    encoding="latin1"
)

scholar = pd.read_csv(
    RAW / "Scholar.csv",
    encoding="latin1"
)

mapping = pd.read_csv(
    RAW / "DBLP-Scholar_perfectMapping.csv",
    encoding="latin1"
)


# Clean column names
for df in [dblp, scholar, mapping]:
    df.columns = (
        df.columns
        .str.strip()
        .str.replace("\ufeff", "", regex=False)
        .str.replace("ï»¿", "", regex=False)
        .str.replace('"', "", regex=False)
    )

assert "id" in dblp.columns
assert "id" in scholar.columns
assert "idDBLP" in mapping.columns
assert "idScholar" in mapping.columns


# -----------------------------
# Save normalized tables
# -----------------------------

dblp.to_csv(
    OUT / "tableA.csv",
    index=False
)

scholar.to_csv(
    OUT / "tableB.csv",
    index=False
)


# -----------------------------
# Positive pairs
# -----------------------------

positive = mapping.rename(
    columns={
        "idDBLP": "ltable_id",
        "idScholar": "rtable_id"
    }
)

positive["label"] = 1

print("DBLP columns:", dblp.columns.tolist())
print("Scholar columns:", scholar.columns.tolist())
print("Mapping columns:", mapping.columns.tolist())

# -----------------------------
# Negative sampling
# -----------------------------

positive_keys = set(
    zip(
        positive.ltable_id,
        positive.rtable_id
    )
)


dblp_ids = dblp["id"].values
scholar_ids = scholar["id"].values


negative = []


target = len(positive) * 4


while len(negative) < target:

    l = np.random.choice(dblp_ids)
    r = np.random.choice(scholar_ids)

    if (l, r) not in positive_keys:
        negative.append(
            {
                "ltable_id": l,
                "rtable_id": r,
                "label": 0
            }
        )


negative = pd.DataFrame(
    negative
)


pairs = pd.concat(
    [
        positive,
        negative
    ],
    ignore_index=True
)


pairs = pairs.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)


# -----------------------------
# Train / validation / test split
# -----------------------------

n = len(pairs)

train_end = int(
    0.7*n
)

valid_end = int(
    0.85*n
)


train = pairs.iloc[
    :train_end
]

valid = pairs.iloc[
    train_end:valid_end
]

test = pairs.iloc[
    valid_end:
]


train.to_csv(
    OUT/"train.csv",
    index=False
)

valid.to_csv(
    OUT/"valid.csv",
    index=False
)

test.to_csv(
    OUT/"test.csv",
    index=False
)


print("DBLP-Scholar prepared")
print("Pairs:", len(pairs))
print("Positive:", pairs.label.sum())
print("Negative:", (pairs.label==0).sum())
print("Train:", len(train))
print("Valid:", len(valid))
print("Test:", len(test))