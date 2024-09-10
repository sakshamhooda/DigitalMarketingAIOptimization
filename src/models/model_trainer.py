from sklearn.model_selection import train_test_split
from src.models.ensemble_model import EnsembleModel

class ModelTrainer:
    def __init__(self):
        self.model = EnsembleModel()

    def train(self, X, y, test_size=0.2, random_state=42):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        self.model.fit(X_train, y_train)
        return X_train, X_test, y_train, y_test

    def get_trained_model(self):
        return self.model