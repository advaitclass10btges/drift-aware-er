import pandas as pd

import os

os.makedirs(
    "results/statistical/analysis",
    exist_ok=True
)

STREAMS = {
    "title": "results/title_drift_stream.csv",
    "naming": "results/naming_drift_stream.csv",
    "missingness": "results/missingness_drift_stream.csv",
}

THRESHOLD = 0.36
EPSILON = 0.05


def boundary_mass(scores):

    return (
        (
            abs(scores - THRESHOLD)
            <= EPSILON
        )
        .mean()
    )


rows=[]


for name,path in STREAMS.items():

    df=pd.read_csv(path)

    stable=df[
        df.phase=="stable"
    ]

    drift=df[
        df.phase=="drift"
    ]

    stable_mass = boundary_mass(
        stable.score.values
    )

    drift_mass = boundary_mass(
        drift.score.values
    )

    rows.append({

        "shift":name,

        "stable_boundary_mass":
            stable_mass,

        "drift_boundary_mass":
            drift_mass,

        "increase":
            drift_mass-stable_mass
    })


out=pd.DataFrame(rows)

out.to_csv(
    "results/statistical/analysis/boundary_instability.csv",
    index=False
)

print(out)
