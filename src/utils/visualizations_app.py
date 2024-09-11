import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

def plot_actual_vs_predicted(y_true, y_pred):
    fig = px.scatter(x=y_true, y=y_pred, labels={'x': 'Actual', 'y': 'Predicted'})
    fig.add_trace(go.Scatter(x=[min(y_true), max(y_true)], y=[min(y_true), max(y_true)],
                             mode='lines', name='Perfect Prediction', line=dict(color='red', dash='dash')))
    fig.update_layout(title='Actual vs Predicted', xaxis_title='Actual', yaxis_title='Predicted')
    return fig

def plot_feature_importance(importance, feature_names):
    df = pd.DataFrame({'feature': feature_names, 'importance': importance})
    df = df.sort_values('importance', ascending=False)
    fig = px.bar(df, x='feature', y='importance', title='Feature Importances')
    fig.update_layout(xaxis_title='Features', yaxis_title='Importance', xaxis_tickangle=-45)
    return fig

def plot_time_series(df, date_column, value_column, title):
    fig = px.line(df, x=date_column, y=value_column, title=title)
    fig.update_layout(xaxis_title='Date', yaxis_title=value_column)
    return fig

def plot_correlation_heatmap(df):
    corr = df.corr()
    fig = px.imshow(corr, labels=dict(color="Correlation"), x=corr.columns, y=corr.columns, color_continuous_scale='RdBu_r')
    fig.update_layout(title='Correlation Heatmap', width=800, height=800)
    return fig