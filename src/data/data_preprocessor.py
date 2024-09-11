import pandas as pd
import numpy as np
from datetime import datetime
import logging

class DataPreprocessor:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    def preprocess_data(self, google_ads, meta_ads, microsoft_ads, website_landings):
        try:
            # Preprocess Google Ads data
            google_ads['Date'] = pd.to_datetime(google_ads['Date'])
            google_ads['Source'] = 'Google Ads'

            # Preprocess Meta Ads data
            meta_ads['Date'] = pd.to_datetime(meta_ads['Date'])
            meta_ads['Source'] = 'Meta Ads'
            meta_ads['Campaign type'] = 'Cross-network'  # Assuming all Meta ads are cross-network

            # Preprocess Microsoft Ads data
            microsoft_ads['Date'] = pd.to_datetime(microsoft_ads['Date'])
            microsoft_ads['Source'] = 'Microsoft Ads'

            # Combine ad performance data
            ad_performance = pd.concat([google_ads, meta_ads, microsoft_ads], ignore_index=True)

            # Ensure consistent column names
            ad_performance = ad_performance.rename(columns={
                'Impressions': 'Impressions',
                'Clicks': 'Clicks',
                'Cost': 'Spend',
                'Conversions': 'Conversions',
                'Revenue': 'Revenue'
            })

            # Clean Website Landings data
            website_landings['Website Landing Time'] = pd.to_datetime(website_landings['Website Landing Time'])
            website_landings['Date'] = website_landings['Website Landing Time'].dt.date
            website_landings['Date'] = pd.to_datetime(website_landings['Date'])
            website_landings['Campaign type'] = website_landings['Campaign Type'].fillna('Unknown')

            # Aggregate Website Landings data
            website_landings_agg = website_landings.groupby(['Date', 'Source', 'Channel', 'Campaign type']).agg({
                'User Id': 'count',
                'Is Converted': 'sum'
            }).reset_index()

            website_landings_agg = website_landings_agg.rename(columns={
                'User Id': 'Sessions',
                'Is Converted': 'Website Conversions'
            })

            # Join ad performance with website landings data
            combined_data = pd.merge(
                ad_performance,
                website_landings_agg,
                on=['Date', 'Source', 'Campaign type'],
                how='left'
            )

            # Fill NaN values with 0 for metrics that should always have a value
            metrics_to_fill = ['Impressions', 'Clicks', 'Spend', 'Conversions', 'Revenue', 'Sessions', 'Website Conversions']
            combined_data[metrics_to_fill] = combined_data[metrics_to_fill].fillna(0)

            # Calculate additional metrics
            combined_data['CTR'] = combined_data['Clicks'] / combined_data['Impressions']
            combined_data['CPC'] = combined_data['Spend'] / combined_data['Clicks']
            combined_data['CVR'] = combined_data['Conversions'] / combined_data['Clicks']
            combined_data['ROAS'] = combined_data['Revenue'] / combined_data['Spend']

            # Replace infinity and NaN with 0 in calculated metrics
            combined_data = combined_data.replace([np.inf, -np.inf], np.nan).fillna(0)

            self.logger.info("Data preprocessing completed successfully")
            return combined_data

        except Exception as e:
            self.logger.error(f"An error occurred during data preprocessing: {str(e)}")
            raise