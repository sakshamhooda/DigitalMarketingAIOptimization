from river import drift

class ConceptDriftDetector:
    def __init__(self):
        self.drift_detector = drift.ADWIN()

    def detect_drift(self, X):
        drift_points = []
        for i, row in X.iterrows():
            self.drift_detector.update(row['ROAS'])
            if self.drift_detector.drift_detected:
                drift_points.append(i)
                print(f"Concept drift detected at index {i}")
                # Reset the detector after drift is detected
                self.drift_detector = drift.ADWIN()
        return drift_points