import shap

class ModelInterpreter:
    @staticmethod
    def explain_model(model, X):
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        return shap_values

    @staticmethod
    def plot_shap_summary(shap_values, X):
        shap.summary_plot(shap_values, X)