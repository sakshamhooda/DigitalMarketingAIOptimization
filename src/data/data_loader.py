import pandas as pd
import numpy as np
from datetime import datetime
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DataLoader:
    def __init__(self, data_dir='../data/raw'):
        self.data_dir = Path(data_dir)

    def load_google_ads(self):
        file_path = self.data_dir / 'googleads-performance.csv'
        df = pd.read_csv(file_path)
        df['Date'] = pd.to_datetime(df['Date'])
        df['Source'] = 'Google Ads'
        return df

    def load_meta_ads(self):
        file_path = self.data_dir / 'metaads-performance.csv'
        df = pd.read_csv(file_path)
        df['Date'] = pd.to_datetime(df['Date'])
        df['Source'] = 'Meta Ads'
        df['Campaign type'] = 'Cross-network'  # Assuming all Meta ads are cross-network
        return df

    def load_microsoft_ads(self):
        file_path = self.data_dir / 'microsoftads-performance.csv'
        df = pd.read_csv(file_path)
        df['Date'] = pd.to_datetime(df['Date'])
        df['Source'] = 'Microsoft Ads'
        return df

    def load_website_landings(self):
        file_path = self.data_dir / 'website-landings.csv'
        df = pd.read_csv(file_path)
        df['Website Landing Time'] = pd.to_datetime(df['Website Landing Time'])
        df['Date'] = pd.to_datetime(df['Website Landing Time'].dt.date)
        df['Campaign type'] = df['Campaign Type'].fillna('Unknown')
        return df

    def load_all_data(self):
        try:
            google_ads = self.load_google_ads()
            meta_ads = self.load_meta_ads()
            microsoft_ads = self.load_microsoft_ads()
            website_landings = self.load_website_landings()

            logging.info("All data loaded successfully")

            return {
                'google_ads': google_ads,
                'meta_ads': meta_ads,
                'microsoft_ads': microsoft_ads,
                'website_landings': website_landings
            }

        except Exception as e:
            logging.error(f"An error occurred while loading data: {str(e)}")
            raise

def main():
    loader = DataLoader()
    data = loader.load_all_data()

    # Display basic information about loaded data
    for name, df in data.items():
        print(f"\n{name} data:")
        print(df.info())
        print(df.head())

if __name__ == "__main__":
    main()