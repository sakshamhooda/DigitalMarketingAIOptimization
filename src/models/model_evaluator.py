from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

class ModelEvaluator:
    @staticmethod
    def evaluate_model(y_true, y_pred):
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        return {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2
        }

    @staticmethod
    def calculate_roas(revenue, spend):
        return revenue / spend if spend != 0 else 0

    @staticmethod
    def evaluate_roas(true_revenue, true_spend, pred_revenue, pred_spend):
        true_roas = ModelEvaluator.calculate_roas(true_revenue, true_spend)
        pred_roas = ModelEvaluator.calculate_roas(pred_revenue, pred_spend)
        return ModelEvaluator.evaluate_model(true_roas, pred_roas)