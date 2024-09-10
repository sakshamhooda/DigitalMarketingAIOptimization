import pandas as pd
import numpy as np
import pandas_datareader as pdr

class FeatureEngineer:
    def __init__(self):
        pass

    def create_time_features(self, df):
        df['day_of_week'] = df['Date'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['month'] = df['Date'].dt.month
        df['quarter'] = df['Date'].dt.quarter
        return df

    def create_lag_features(self, df, columns, lags):
        for col in columns:
            for lag in lags:
                df[f'{col}_lag_{lag}'] = df[col].shift(lag)
        return df

    def add_external_data(self, df):
        sp500 = pdr.get_data_yahoo('^GSPC', start=df['Date'].min(), end=df['Date'].max())
        df['SP500'] = sp500['Close']
        return df