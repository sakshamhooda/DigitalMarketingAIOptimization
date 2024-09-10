from fastapi import FastAPI, HTTPException
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel
import pandas as pd
from src.models.model_trainer import ModelTrainer

app = FastAPI()

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

class AdData(BaseModel):
    impressions: int
    clicks: int
    spend: float
    # Add other features as needed

class Prediction(BaseModel):
    roas: float

@app.post("/predict", response_model=Prediction)
async def predict_roas(ad_data: AdData, token: str = Depends(oauth2_scheme)):
    try:
        # Convert input data to DataFrame
        input_df = pd.DataFrame([ad_data.dict()])

        # Load the trained model
        model = ModelTrainer().get_trained_model()

        # Make prediction
        prediction = model.predict(input_df)[0]

        return Prediction(roas=prediction)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "Welcome to the Ad Spend Optimization API"}