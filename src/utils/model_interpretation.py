import shap
import matplotlib.pyplot as plt

class ModelInterpreter:
    @staticmethod
    def explain_model(model, X):
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        return shap_values

    @staticmethod
    def plot_shap_summary(shap_values, X):
        shap.summary_plot(shap_values, X)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_feature_importance(model, X):
        feature_importance = model.feature_importances_
        sorted_idx = feature_importance.argsort()
        pos = np.arange(sorted_idx.shape[0]) + .5
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.barh(pos, feature_importance[sorted_idx], align='center')
        ax.set_yticks(pos)
        ax.set_yticklabels(X.columns[sorted_idx])
        ax.set_xlabel('Feature Importance')
        ax.set_title('Feature Importance (MDI)')
        plt.tight_layout()
        plt.show()