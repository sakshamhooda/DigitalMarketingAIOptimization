# /home/sagemaker-user/DigitalMarketingAIOptimization/app/dashboard/app.py

import streamlit as st
import pandas as pd
import sys
import os

# Add the project root to the Python path
project_root = '/home/sagemaker-user/DigitalMarketingAIOptimization'
sys.path.insert(0, project_root)

from src.models.model_trainer import ModelTrainer
from src.models.model_evaluator import ModelEvaluator
from src.utils.visualizations_app import (
    plot_actual_vs_predicted, plot_feature_importance, 
    plot_time_series, plot_correlation_heatmap
)
from updated_data_loader import load_data

def main():
    st.title('Ad Spend Optimization Dashboard')
    data = load_data()

    st.sidebar.header('Navigation')
    page = st.sidebar.radio('Go to', ['Data Overview', 'Performance Analysis', 'Model Performance', 'Predictions'])

    if page == 'Data Overview':
        display_data_overview(data)
    elif page == 'Performance Analysis':
        display_performance_analysis(data)
    elif page == 'Model Performance':
        display_model_performance(data)
    elif page == 'Predictions':
        display_predictions(data)

def display_data_overview(data):
    st.header('Data Overview')
    st.write(data.head())
    st.write(data.describe())

    st.subheader('ROAS Over Time')
    fig = plot_time_series(data, 'Date', 'ROAS', 'ROAS Over Time')
    st.plotly_chart(fig)

    st.subheader('Correlation Heatmap')
    fig = plot_correlation_heatmap(data.drop('Date', axis=1))
    st.plotly_chart(fig)

def display_performance_analysis(data):
    st.header('Performance Analysis')

    col1, col2 = st.columns(2)
    start_date = col1.date_input('Start date', data['Date'].min())
    end_date = col2.date_input('End date', data['Date'].max())

    filtered_data = data[(data['Date'] >= pd.Timestamp(start_date)) & 
                         (data['Date'] <= pd.Timestamp(end_date))]

    st.subheader('Overall Performance')
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Spend", f"${filtered_data['Spend'].sum():,.2f}")
    col2.metric("Total Revenue", f"${filtered_data['Revenue'].sum():,.2f}")
    col3.metric("Total Conversions", f"{filtered_data['Conversions'].sum():,.0f}")
    col4.metric("Overall ROAS", f"{filtered_data['Revenue'].sum() / filtered_data['Spend'].sum():,.2f}")

    st.subheader('Performance by Source')
    source_data = filtered_data.groupby('Source').agg({
        'Spend': 'sum', 'Revenue': 'sum', 'Conversions': 'sum'
    }).reset_index()
    source_data['ROAS'] = source_data['Revenue'] / source_data['Spend']
    fig = plot_time_series(source_data, 'Source', 'ROAS', 'ROAS by Source')
    st.plotly_chart(fig)

def display_model_performance(data):
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
    st.plotly_chart(fig)

    st.subheader('Feature Importance')
    fig = plot_feature_importance(model.best_estimator_.feature_importances_, X.columns)
    st.plotly_chart(fig)

def display_predictions(data):
    st.header('Make Predictions')
    
    features = {}
    for col in data.columns:
        if col not in ['ROAS', 'Date']:
            features[col] = st.number_input(f'Enter {col}', value=data[col].mean())

    if st.button('Predict'):
        X = pd.DataFrame([features])
        model = ModelTrainer().get_trained_model()
        prediction = model.predict(X)[0]
        st.success(f'Predicted ROAS: {prediction:.4f}')

        st.subheader('What-If Analysis')
        feature_to_change = st.selectbox('Select feature to change', list(features.keys()))
        change_percentage = st.slider('Change percentage', -100, 100, 0)
        
        new_features = features.copy()
        new_features[feature_to_change] *= (1 + change_percentage / 100)
        
        new_X = pd.DataFrame([new_features])
        new_prediction = model.predict(new_X)[0]
        
        st.write(f'New prediction with {change_percentage}% change in {feature_to_change}:')
        st.write(f'New ROAS: {new_prediction:.4f}')
        st.write(f'Change in ROAS: {(new_prediction - prediction) / prediction * 100:.2f}%')

if __name__ == '__main__':
    main()