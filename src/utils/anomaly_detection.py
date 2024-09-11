from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd

class AnomalyDetector:
    def __init__(self):
        self.preprocessor = None
        self.iso_forest = None

    def preprocess_data(self, X):
        # Identify numeric and categorical columns
        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
        categorical_features = X.select_dtypes(include=['object', 'category']).columns

        # Create preprocessing steps
        numeric_transformer = 'passthrough'
        categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse=False)

        # Combine preprocessing steps
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])

        # Create pipeline
        self.iso_forest = Pipeline([
            ('preprocessor', self.preprocessor),
            ('classifier', IsolationForest(contamination=0.1, random_state=42))
        ])

    def detect_outliers(self, X):
        if self.iso_forest is None:
            self.preprocess_data(X)
        return self.iso_forest.fit_predict(X)