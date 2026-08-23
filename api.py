"""
FastAPI Backend
Exposes the 3-day AQI forecast, historical trend, SHAP feature importance,
and model comparison metrics as JSON API endpoints for the dashboard.

Run with: uvicorn api:app --reload
Then visit http://127.0.0.1:8000/docs for interactive API docs.
"""

import json
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from predict import predict_next_3_days

DATA_FILE = "aqi_training_data.csv"

app = FastAPI(
    title="Karachi AQI Prediction API",
    description="Serves 3-day Air Quality Index forecasts for Karachi",
    version="1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {"message": "AQI Prediction API is running. Go to /predict for the forecast, or /docs for API docs."}


@app.get("/predict")
def get_prediction():
    try:
        result = predict_next_3_days()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.get("/history")
def get_history(days: int = 14):
    """
    Returns daily-averaged AQI for the last N days, for the dashboard's
    trend chart. Reads the local training CSV - doesn't need to be
    live-live, just recent.
    """
    try:
        df = pd.read_csv(DATA_FILE)
        df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed")
        df = df.sort_values("timestamp")

        cutoff = df["timestamp"].max() - pd.Timedelta(days=days)
        recent = df[df["timestamp"] >= cutoff]

        daily = recent.set_index("timestamp")["aqi"].resample("D").mean().round(1)

        result = [
            {"date": str(date.date()), "aqi": float(value)}
            for date, value in daily.items()
            if not pd.isna(value)
        ]
        return {"days": days, "history": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"History fetch failed: {str(e)}")


@app.get("/feature-importance")
def get_feature_importance():
    """
    Returns pre-computed SHAP feature importance for the Day 1 model.
    Reads from feature_importance.json (generate with
    compute_feature_importance.py) rather than recomputing SHAP on
    every request.
    """
    try:
        with open("feature_importance.json", "r") as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail="feature_importance.json not found - run compute_feature_importance.py first"
        )


@app.get("/model-metrics")
def get_model_metrics():
    """
    Returns the RMSE/MAE/R2 comparison table across all models and
    horizons. Reads from model_metrics.json (generate with
    save_model_metrics.py).
    """
    try:
        with open("model_metrics.json", "r") as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail="model_metrics.json not found - run save_model_metrics.py first"
        )