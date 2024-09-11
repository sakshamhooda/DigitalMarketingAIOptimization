# app/api/main.py

from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import List
import sys
import os

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from datetime import date
import pandas as pd
from src.models.model_trainer import ModelTrainer
from updated_data_loader import load_data

app = FastAPI()

# Load the preprocessed data
data = load_data()

# Load the trained model
model = ModelTrainer().get_trained_model()

class AdPerformance(BaseModel):
    date: date
    source: str
    campaign_type: str
    spend: float
    impressions: int
    clicks: int

class PerformanceResponse(BaseModel):
    source: str
    spend: float
    revenue: float
    conversions: float
    roas: float

@app.get("/")
async def root():
    return {"message": "Welcome to the Ad Performance API"}

@app.get("/performance", response_model=List[PerformanceResponse])
async def get_performance(start_date: date = Query(...), end_date: date = Query(...)):
    filtered_data = data[(data['Date'] >= pd.Timestamp(start_date)) & 
                         (data['Date'] <= pd.Timestamp(end_date))]
    
    performance = filtered_data.groupby('Source').agg({
        'Spend': 'sum',
        'Revenue': 'sum',
        'Conversions': 'sum'
    }).reset_index()
    
    performance['ROAS'] = performance['Revenue'] / performance['Spend']
    
    return [PerformanceResponse(
        source=row['Source'],
        spend=row['Spend'],
        revenue=row['Revenue'],
        conversions=row['Conversions'],
        roas=row['ROAS']
    ) for _, row in performance.iterrows()]

@app.post("/predict")
async def predict_performance(ad: AdPerformance):
    input_data = pd.DataFrame([{
        'Date': ad.date,
        'Source': ad.source,
        'Campaign type': ad.campaign_type,
        'Spend': ad.spend,
        'Impressions': ad.impressions,
        'Clicks': ad.clicks
    }])
    
    prediction = model.predict(input_data)
    
    return {
        "predicted_roas": float(prediction[0])
    }