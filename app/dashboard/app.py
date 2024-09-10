import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from src.models.model_trainer import ModelTrainer
from src.models.model_evaluator import ModelEvaluator
from src.utils.visualization import plot_actual_vs_predicted, plot_feature_importance

def load_data():
    return pd.read_csv('../../data/processed/preprocessed_data.csv')

def main():
    st.title('Ad Spend Optimization Dashboard')

    data = load_data()

    st.sidebar.header('Navigation')
    page = st.sidebar.radio('Go to', ['Data Overview', 'Model Performance', 'Predictions'])

    if page == 'Data Overview':
        st.header('Data Overview')
        st.write(data.head())
        st.write(data.describe())

        st.subheader('ROAS Over Time')
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(data['Date'], data['ROAS'])
        ax.set_xlabel('Date')
        ax.set_ylabel('ROAS')
        st.pyplot(fig)

    elif page == 'Model Performance':
        st.header('Model Performance')

        X = data.drop(['ROAS', 'Date'], axis=1)
        y = data['ROAS']

        trainer = ModelTrainer()
        X_train, X_test, y_train, y_test = trainer.train(X, y)
        model = trainer.get_trained_model()

        y_pred = model.predict(X_test)

        evaluator = ModelEvaluator()
        metrics = evaluator.evaluate_model(y_test, y_pred)

        for metric, value in metrics.items():
            st.metric(metric, f"{value:.4f}")

        st.subheader('Actual vs Predicted')
        fig = plot_actual_vs_predicted(y_test, y_pred)
        st.pyplot(fig)

        st.subheader('Feature Importance')
        fig = plot_feature_importance(model.best_xgb, X.columns)
        st.pyplot(fig)

    elif page == 'Predictions':
        st.header('Make Predictions')

        # Add input fields for features
        features = {}
        for col in data.columns:
            if col not in ['ROAS', 'Date']:
                features[col] = st.number_input(f'Enter {col}', value=data[col].mean())

        if st.button('Predict'):
            X = pd.DataFrame([features])
            model = ModelTrainer().get_trained_model()
            prediction = model.predict(X)[0]
            st.success(f'Predicted ROAS: {prediction:.4f}')

if __name__ == '__main__':
    main()