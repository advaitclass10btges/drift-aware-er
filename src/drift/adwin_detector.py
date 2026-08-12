from river import drift


class ADWINDetector:

    def __init__(self, delta=0.002):

        self.detector = drift.ADWIN(
            delta=delta
        )


    def update(self, value):

        self.detector.update(value)

        return self.detector.drift_detected
