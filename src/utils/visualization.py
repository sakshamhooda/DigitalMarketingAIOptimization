import matplotlib.pyplot as plt
import seaborn as sns

def plot_actual_vs_predicted(y_true, y_pred):
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('Actual vs Predicted')
    plt.show()

def plot_feature_importance(model, feature_names):
    importance = model.feature_importances_
    indices = np.argsort(importance)[::-1]

    plt.figure(figsize=(12, 8))
    plt.title("Feature Importances")
    plt.bar(range(len(importance)), importance[indices])
    plt.xticks(range(len(importance)), feature_names[indices], rotation=90)
    plt.tight_layout()
    plt.show()

def plot_time_series(df, date_column, value_column, title):
    plt.figure(figsize=(12, 6))
    plt.plot(df[date_column], df[value_column])
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel(value_column)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_correlation_heatmap(df):
    plt.figure(figsize=(12, 10))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title('Correlation Heatmap')
    plt.tight_layout()
    plt.show()