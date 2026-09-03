"""
FastAPI Backend
Exposes the 3-day AQI forecast, historical trend, SHAP feature importance,
model comparison metrics, and model version history as JSON API endpoints
for the dashboard.

Run with: uvicorn api:app --reload
Then visit http://127.0.0.1:8000/docs for interactive API docs.

PERFORMANCE NOTE: Hopsworks login + model downloads now happen ONCE at
startup (see the startup event below), and /history is TTL-cached, since
the underlying feature store data only changes on the hourly pipeline
schedule anyway. This is what fixes the "takes forever to load" problem -
previously every /predict call re-logged into Hopsworks and re-downloaded
all 3 models from scratch.
"""

import os
import time
import json
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from dotenv import load_dotenv
load_dotenv()

import hopsworks

from predict import predict_next_3_days, get_cached_project, preload

HOPSWORKS_API_KEY = os.environ.get("HOPSWORKS_API_KEY")
HOPSWORKS_PROJECT = "aqi_predictor_ksa"
HOPSWORKS_HOST = "eu-west.cloud.hopsworks.ai"
FEATURE_VIEW_NAME = "aqi_feature_view"
FEATURE_VIEW_VERSION = 1

# Comma-separated list of allowed frontend origins, e.g.
#   ALLOWED_ORIGINS=https://your-dashboard.vercel.app,http://localhost:3000
# Falls back to "*" for local development if not set.
ALLOWED_ORIGINS = os.environ.get("ALLOWED_ORIGINS", "*")
origins = ["*"] if ALLOWED_ORIGINS == "*" else [o.strip() for o in ALLOWED_ORIGINS.split(",")]

# How long the /history response stays cached before re-querying the
# Feature Store. The live pipeline writes a new row once an hour, so
# there's no point recomputing this on every dashboard load.
HISTORY_CACHE_TTL_SECONDS = 5 * 60
_history_cache = {"days": None, "result": None, "fetched_at": 0.0}

app = FastAPI(
    title="Karachi AQI Prediction API",
    description="Serves 3-day Air Quality Index forecasts for Karachi",
    version="1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup():
    """
    Warms up the Hopsworks connection and downloads all 3 models ONCE
    when the API process starts, instead of on the first (or every)
    request. If this fails (e.g. Hopsworks briefly unreachable at boot),
    we don't crash the app - predict.py's per-call fallback logic will
    still try again and fall back to local files as before.
    """
    try:
        preload()
    except Exception as e:
        print(f"⚠️ Startup preload failed, will retry lazily on first request: {e}")


def connect_to_hopsworks():
    cert_dir = os.path.join(os.getcwd(), "hopsworks_certs")
    os.makedirs(cert_dir, exist_ok=True)
    return hopsworks.login(
        project=HOPSWORKS_PROJECT,
        host=HOPSWORKS_HOST,
        port=443,
        api_key_value=HOPSWORKS_API_KEY,
        cert_folder=cert_dir,
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
    trend chart. Reads from the Hopsworks Feature Store, cached for
    HISTORY_CACHE_TTL_SECONDS so repeated dashboard loads don't each
    trigger a fresh Feature Store query.
    """
    now = time.time()
    cache_fresh = (
        _history_cache["result"] is not None
        and _history_cache["days"] == days
        and (now - _history_cache["fetched_at"]) < HISTORY_CACHE_TTL_SECONDS
    )
    if cache_fresh:
        return _history_cache["result"]

    try:
        project = get_cached_project() or connect_to_hopsworks()
        fs = project.get_feature_store()
        fv = fs.get_feature_view(name=FEATURE_VIEW_NAME, version=FEATURE_VIEW_VERSION)
        df = fv.get_batch_data()

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
        response = {"days": days, "history": result}

        _history_cache["days"] = days
        _history_cache["result"] = response
        _history_cache["fetched_at"] = now

        return response
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
    horizons. Reads from model_metrics.json, generated automatically
    by train_model.py at the end of every training run.
    """
    try:
        with open("model_metrics.json", "r") as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail="model_metrics.json not found - run train_model.py first"
        )


@app.get("/model-versions")
def get_model_versions():
    """
    Returns the full version history for each model, straight from the
    Hopsworks Model Registry. Reads from model_history.json (generate
    with model_history.py).
    """
    try:
        with open("model_history.json", "r") as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail="model_history.json not found - run model_history.py first"
        )