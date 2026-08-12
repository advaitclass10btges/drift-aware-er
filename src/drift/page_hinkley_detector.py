from river import drift


class PageHinkleyDetector:

    def __init__(self):

        self.detector = drift.PageHinkley()


    def update(self, value):

        self.detector.update(value)

        return self.detector.drift_detected
