import pandas as pd

from src.drift.page_hinkley_detector import PageHinkleyDetector


df = pd.read_csv(
    "results/amazon_google_score_stream.csv"
)


detector = PageHinkleyDetector()

detections = []

for i, score in enumerate(df["score"]):

    if detector.update(score):
        detections.append(i)


print("Number of detections:", len(detections))
print("Detection points:", detections)
