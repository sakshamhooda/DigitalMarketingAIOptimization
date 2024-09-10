import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

class SyntheticDataGenerator:
    def __init__(self, start_date='2024-01-01', num_days=365):
        self.start_date = pd.to_datetime(start_date)
        self.num_days = num_days

    def generate_data(self):
        date_range = pd.date_range(start=self.start_date, periods=self.num_days)
        
        # Generate synthetic data
        impressions = np.random.randint(1000, 100000, size=self.num_days)
        clicks = np.random.randint(10, 1000, size=self.num_days)
        spend = np.random.uniform(100, 1000, size=self.num_days)
        revenue = spend * np.random.uniform(0.5, 2.5, size=self.num_days)
        
        # Create DataFrame
        df = pd.DataFrame({
            'Date': date_range,
            'Impressions': impressions,
            'Clicks': clicks,
            'Spend': spend,
            'Revenue': revenue
        })
        
        # Add some seasonality
        df['Seasonality'] = np.sin(np.arange(self.num_days) * 2 * np.pi / 365)
        df['Revenue'] *= (1 + 0.2 * df['Seasonality'])
        
        # Calculate metrics
        df['CTR'] = df['Clicks'] / df['Impressions']
        df['CPC'] = df['Spend'] / df['Clicks']
        df['ROAS'] = df['Revenue'] / df['Spend']
        
        return df

    def add_noise(self, df, noise_level=0.1):
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        noise = np.random.normal(0, noise_level, size=df[numeric_columns].shape)
        df[numeric_columns] += df[numeric_columns] * noise
        return df

    def generate_multi_channel_data(self, channels=['Google Ads', 'Facebook Ads', 'Microsoft Ads']):
        all_data = []
        for channel in channels:
            channel_data = self.generate_data()
            channel_data['Channel'] = channel
            all_data.append(channel_data)
        
        return pd.concat(all_data, ignore_index=True)