from sklearn.ensemble import IsolationForest

class AnomalyDetector:
    def __init__(self, contamination=0.1):
        self.iso_forest = IsolationForest(contamination=contamination)

    def detect_outliers(self, X):
        return self.iso_forest.fit_predict(X)