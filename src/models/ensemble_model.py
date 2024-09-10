import optuna
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
import xgboost as xgb
import numpy as np

class EnsembleModel:
    def __init__(self):
        self.best_xgb = None
        self.stacking_model = None

    def optimize_xgboost(self, X, y):
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_loguniform('learning_rate', 1e-3, 1.0),
                'subsample': trial.suggest_uniform('subsample', 0.6, 1.0)
            }
            model = xgb.XGBRegressor(**params)
            return np.mean(cross_val_score(model, X, y, cv=TimeSeriesSplit(n_splits=5)))

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=100)
        best_params = study.best_params
        self.best_xgb = xgb.XGBRegressor(**best_params)

    def create_stacking_ensemble(self):
        base_models = [
            ('xgb', self.best_xgb),
            ('rf', RandomForestRegressor()),
            ('mlp', MLPRegressor())
        ]
        self.stacking_model = StackingRegressor(estimators=base_models, final_estimator=RandomForestRegressor())

    def fit(self, X, y):
        self.optimize_xgboost(X, y)
        self.create_stacking_ensemble()
        self.stacking_model.fit(X, y)

    def predict(self, X):
        return self.stacking_model.predict(X)