import pandas as pd
import numpy as np
import pandas_datareader as pdr
import yfinance as yf
import logging

class FeatureEngineer:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def create_time_features(self, df):
        df['day_of_week'] = df['Date'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['month'] = df['Date'].dt.month
        df['quarter'] = df['Date'].dt.quarter
        return df

    def create_lag_features(self, df, columns, lags):
        for col in columns:
            for lag in lags:
                df[f'{col}_lag_{lag}'] = df.groupby('Source')[col].shift(lag)
        return df

    def add_external_data(self, df):
        try:
            # Try using yfinance instead of pandas_datareader
            sp500 = yf.download('^GSPC', start=df['Date'].min(), end=df['Date'].max())
            df = df.merge(sp500['Close'].reset_index(), how='left', left_on='Date', right_on='Date')
            df = df.rename(columns={'Close': 'SP500'})
        except Exception as e:
            self.logger.warning(f"Failed to fetch S&P 500 data: {str(e)}")
            self.logger.warning("Adding a dummy SP500 column instead.")
            df['SP500'] = np.nan  # Add a column of NaNs as a placeholder
        return df

    def engineer_features(self, df):
        self.logger.info("Starting feature engineering process...")
        
        df = self.create_time_features(df)
        self.logger.info("Time features created.")
        
        lag_columns = ['Impressions', 'Clicks', 'Spend', 'Conversions', 'Revenue']
        lags = [1, 7, 30]  # for example, 1 day, 1 week, and 1 month lags
        df = self.create_lag_features(df, lag_columns, lags)
        self.logger.info("Lag features created.")
        
        df = self.add_external_data(df)
        self.logger.info("External data added.")
        
        self.logger.info("Feature engineering process completed.")
        return df