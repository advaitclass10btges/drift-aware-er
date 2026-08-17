import pandas as pd


INPUT = (
    "results/statistical/"
    "bootstrap_detector_results.csv"
)

OUTPUT = (
    "results/statistical/analysis/"
    "delay_summary.csv"
)


df = pd.read_csv(INPUT)


rows = []


for shift in df["shift"].unique():

    subset = df[
        df["shift"] == shift
    ]

    boundary = subset[
        subset["detector"] == "Boundary-Aware"
    ]

    for baseline_name in [
        "ADWIN",
        "Page-Hinkley"
    ]:

        baseline = subset[
            subset["detector"] == baseline_name
        ]


        if len(boundary) == 0:
            continue


        boundary_delay = (
            boundary["detection_delay"]
            .dropna()
            .mean()
        )


        baseline_delay = (
            baseline["detection_delay"]
            .dropna()
            .mean()
        )


        if pd.notna(baseline_delay):

            reduction = (
                (baseline_delay - boundary_delay)
                /
                baseline_delay
            ) * 100

        else:
            reduction = None


        rows.append({

            "shift": shift,

            "baseline": baseline_name,

            "boundary_mean_delay":
                boundary_delay,

            "baseline_mean_delay":
                baseline_delay,

            "delay_reduction_percent":
                reduction,

            "boundary_detection_rate":
                boundary["detected"]
                .mean(),

            "baseline_detection_rate":
                baseline["detected"]
                .mean()
        })


result = pd.DataFrame(rows)


result.to_csv(
    OUTPUT,
    index=False
)


print(result)