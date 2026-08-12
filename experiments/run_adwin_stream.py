import pandas as pd

from river.drift import ADWIN


df = pd.read_csv(
    "results/title_drift_stream.csv"
)


detector = ADWIN()


detections = []


for i, score in enumerate(df["score"]):

    detector.update(
        score
    )

    if detector.drift_detected:

        detections.append(i)



print("True drift point:")
print(1375)


print("\nADWIN detections:")
print(detections)


if detections:

    delay = detections[0] - 1375

    print("\nDetection delay:")
    print(delay)

else:

    print("\nNo detection")
