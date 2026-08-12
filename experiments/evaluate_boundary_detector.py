import pandas as pd


TRUE_DRIFT = 1375


detections = pd.read_csv(
    "results/boundary_detector_detections.csv"
)


false_alarms = detections[
    detections["start"] < TRUE_DRIFT
]


true_detections = detections[
    detections["start"] >= TRUE_DRIFT
]


print("Total detections:")
print(len(detections))


print("\nFalse alarms:")
print(len(false_alarms))


print(false_alarms)


print("\nPost drift detections:")
print(len(true_detections))


if len(true_detections):

    delay = (
        true_detections.iloc[0]["start"]
        -
        TRUE_DRIFT
    )

    print("\nDetection delay:")
    print(delay)
