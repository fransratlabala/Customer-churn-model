# main.py
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(title="FRANS Customer Churn Prediction API")

# Load trained pipeline
model = joblib.load("customer_churn_pipeline.pkl")

# Input schema
class CustomerData(BaseModel):
    id: int
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float

@app.post("/predict")
def predict_churn(data: CustomerData):
    df = pd.DataFrame([data.dict()])
    df = df.drop(columns=["id"])  # id was dropped during training

    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]

    return {
        "churn_prediction": int(prediction),
        "churn_probability": float(probability)
    }