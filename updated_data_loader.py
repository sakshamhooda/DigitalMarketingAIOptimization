# updated_data_loader.py

import os
import pandas as pd

import pandas as pd
import os

def load_data():
    file_path = '/home/sagemaker-user/DigitalMarketingAIOptimization/data/processed/combined_ad_data.csv'
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist. Please check the file path.")
    
    data = pd.read_csv(file_path)
    data['Date'] = pd.to_datetime(data['Date'])
    
    if 'ROAS' not in data.columns:
        data['ROAS'] = data['Revenue'] / data['Spend']
    
    return data