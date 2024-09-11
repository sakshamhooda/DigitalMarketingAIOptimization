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
                'learning_rate': trial.suggest_float('learning_rate', 1e-3, 1.0, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0)
            }
            model = xgb.XGBRegressor(**params)
            return np.mean(cross_val_score(model, X, y, cv=TimeSeriesSplit(n_splits=5)))
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=100)
        best_params = study.best_params
        
        self.best_xgb = xgb.XGBRegressor(**best_params)
        print("Fitting XGBoost model...")
        self.best_xgb.fit(X, y)  # Fit the best model
        print("XGBoost model fitted.")

    def create_stacking_ensemble(self):
        base_models = [
            ('xgb', self.best_xgb),
            ('rf', RandomForestRegressor()),
            ('mlp', MLPRegressor(max_iter=500, learning_rate_init=0.001))
        ]
        self.stacking_model = StackingRegressor(estimators=base_models, final_estimator=RandomForestRegressor())

    def fit(self, X, y):
        print("Starting model fitting process...")
        print("Optimizing XGBoost...")
        self.optimize_xgboost(X, y)
        print("Creating stacking ensemble...")
        self.create_stacking_ensemble()
        print("Fitting stacking model...")
        self.stacking_model.fit(X, y)
        print("Model fitting complete.")


    def predict(self, X):
        return self.stacking_model.predict(X)

    def get_fitted_xgb(self):
        if self.best_xgb is None:
            raise ValueError("XGBoost model has not been initialized. Call fit() first.")
        if not hasattr(self.best_xgb, 'fitted_'):
            raise ValueError("XGBoost model has been initialized but not fitted. There might be an issue in the fit() method.")
        return self.best_xgb