from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit

class FeatureSelector:
    def __init__(self):
        self.selector = None

    def select_features(self, X, y):
        self.selector = RFECV(estimator=RandomForestRegressor(), step=1, cv=TimeSeriesSplit(n_splits=5))
        return self.selector.fit_transform(X, y)

    def get_selected_features(self):
        return self.selector.support_