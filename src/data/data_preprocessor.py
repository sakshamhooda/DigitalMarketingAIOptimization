import pandas as pd
import numpy as np
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_and_preprocess_data():
    try:
        # Load the data
        google_ads = pd.read_csv('DigitalMarketingAIOptimization/data/raw/googleads-performance.csv')
        meta_ads = pd.read_csv('DigitalMarketingAIOptimization/data/raw/metaads-performance.csv')
        microsoft_ads = pd.read_csv('DigitalMarketingAIOptimization/data/raw/microsoftads-performance.csv')
        website_landings = pd.read_csv('DigitalMarketingAIOptimization/data/raw/website-landings.csv')

        logging.info("Data loaded successfully")

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
        website_landings['Date'] = pd.to_datetime(website_landings['Date'])  # Convert to datetime64[ns]
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

        # Ensure 'Date' column is datetime64[ns] in both dataframes
        ad_performance['Date'] = pd.to_datetime(ad_performance['Date'])
        website_landings_agg['Date'] = pd.to_datetime(website_landings_agg['Date'])

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

        logging.info("Data preprocessing completed successfully")
        return combined_data

    except Exception as e:
        logging.error(f"An error occurred during data preprocessing: {str(e)}")
        raise

# Execute the preprocessing
try:
    preprocessed_data = load_and_preprocess_data()

    # Display the first few rows and data info
    print(preprocessed_data.head())
    print(preprocessed_data.info())

    # Save the preprocessed data
    preprocessed_data.to_csv('DigitalMarketingAIOptimization/data/processed/combined_ad_data.csv', index=False)
    logging.info("Preprocessed data saved to 'data/processed/combined_ad_data.csv'")

except Exception as e:
    logging.error(f"Failed to preprocess data: {str(e)}")