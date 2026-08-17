import os
import pandas as pd


OUTPUT = (
    "results/statistical/analysis/"
    "final_results_summary.csv"
)


def main():

    rows = []


    # ============================
    # Bootstrap synthetic evaluation
    # ============================

    bootstrap = pd.read_csv(
        "results/statistical/bootstrap_detector_summary.csv"
    )


    boundary = bootstrap[
        bootstrap.detector == "Boundary-Aware"
    ]


    for _, row in boundary.iterrows():

        rows.append({

            "dataset":
                "Synthetic",

            "shift":
                row["shift"],

            "detector":
                "Boundary-Aware",

            "severity":
                "-",

            "detection_rate":
                round(
                    row["detection_rate"]*100,
                    2
                ),

            "mean_delay":
                round(
                    row["mean_detection_delay"],
                    2
                ),

            "false_alarm_rate":
                round(
                    row["mean_false_alarm_rate"]*100,
                    2
                ),

            "decision_flip_rate":
                "-"

        })


    # ============================
    # Amazon Google
    # ============================

    amazon = pd.read_csv(
        "results/statistical/analysis/"
        "amazon_google_results.csv"
    )


    for _, row in amazon.iterrows():

        rows.append({

            "dataset":
                "Amazon-Google",

            "shift":
                row["drift_type"],

            "detector":
                "Boundary-Aware",

            "severity":
                row["severity"],

            "detection_rate":
                100 if row["detected"]
                else 0,

            "mean_delay":
                row["delay"],

            "false_alarm_rate":
                round(
                    row["false_alarm_rate"]*100,
                    2
                ),

            "decision_flip_rate":
                "-"

        })


    # ============================
    # DBLP Scholar
    # ============================

    dblp = pd.read_csv(
        "results/statistical/analysis/"
        "dblp_scholar_results.csv"
    )


    flips = pd.read_csv(
        "results/statistical/analysis/"
        "dblp_scholar_decision_boundary.csv"
    )


    dblp = dblp.merge(
        flips[
            [
                "severity",
                "decision_flip_rate"
            ]
        ],
        on="severity"
    )


    for _, row in dblp.iterrows():

        rows.append({

            "dataset":
                "DBLP-Scholar",

            "shift":
                row["drift_type"],

            "detector":
                "Boundary-Aware",

            "severity":
                row["severity"],

            "detection_rate":
                100 if row["detected"]
                else 0,

            "mean_delay":
                row["delay"],

            "false_alarm_rate":
                round(
                    row["false_alarm_rate"]*100,
                    2
                ),

            "decision_flip_rate":
                round(
                    row["decision_flip_rate"]*100,
                    2
                )

        })


    output = pd.DataFrame(rows)


    os.makedirs(
        os.path.dirname(OUTPUT),
        exist_ok=True
    )


    output.to_csv(
        OUTPUT,
        index=False
    )


    print(output)



if __name__ == "__main__":
    main()