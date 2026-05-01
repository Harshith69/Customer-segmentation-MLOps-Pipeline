from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import numpy as np

from src.utils.logger import logger

app = FastAPI(title="Customer Segmentation API")

class CustomerData(BaseModel):
    recency: float
    frequency: float
    monetary: float
    quantity: float
    discount: float
    delivery_days: float
    customer_rating: float

    class Config:
        json_schema_extra = {
            "example": {
                "recency": 10,
                "frequency": 5,
                "monetary": 2000,
                "quantity": 50,
                "discount": 0.1,
                "delivery_days": 3,
                "customer_rating": 4.5
            }
        }

# Load model at startup
MODEL_PATH = "artifacts/models/kmeans_model.pkl"
SCALER_PATH = "artifacts/models/scaler.pkl"

scaler = joblib.load(SCALER_PATH)
model = joblib.load(MODEL_PATH)
logger.info("Model loaded successfully")


@app.get("/")
def home():
    return {"message": "Customer Segmentation API is running"}


@app.post("/predict")


@app.post("/predict")
def predict(data: CustomerData):
    try:
        df = pd.DataFrame([data.dict()])

        # Apply same transformations
        df = np.log1p(df)
        df_scaled = scaler.transform(df)

        prediction = model.predict(df_scaled)[0]

        return {"cluster": int(prediction)}

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        return {"error": str(e)}
    
'''
API expects:
{
  "recency": 10,
  "frequency": 5,
  "monetary": 2000,
  "quantity": 50,
  "discount": 0.1,
  "delivery_days": 3,
  "customer_rating": 4.5
}
'''