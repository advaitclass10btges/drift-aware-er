import pandas as pd


rows = []


# detector comparison

data = pd.read_csv(
    "results/statistical/bootstrap_detector_summary.csv"
)


for shift in data["shift"].unique():

    subset = data[data["shift"] == shift]

    boundary = subset[
        subset.detector=="Boundary-Aware"
    ]

    for baseline in ["ADWIN","Page-Hinkley"]:

        base = subset[
            subset.detector==baseline
        ]

        if len(base)==0:
            continue

        b_rate = float(
            boundary["detection_rate"].iloc[0]
        )

        base_rate=float(
            base["detection_rate"].iloc[0]
        )

        if base_rate > 0:

            relative_gain = (
                (b_rate - base_rate)
                / base_rate
            ) * 100

        else:
            relative_gain = None


        rows.append({

            "shift": shift,

            "baseline": baseline,

            "boundary_detection_rate":
                round(b_rate * 100, 2),

            "baseline_detection_rate":
                round(base_rate * 100, 2),

            "absolute_gain_percentage_points":
                round(
                    (b_rate-base_rate)*100,
                    2
                ),

            "relative_gain_percent":
                relative_gain,

            "baseline_failed":
                base_rate == 0
        })


df=pd.DataFrame(rows)

df.to_csv(
    "results/statistical/analysis/"
    "percentage_summary.csv",
    index=False
)

print(df)