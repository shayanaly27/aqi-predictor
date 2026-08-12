"""
Prediction Script
Takes the most recent data (live + enough history for lag/rolling features)
from the Hopsworks Feature Store, engineers the same features used in
training, loads the 3 saved models, and outputs a clean 3-day AQI forecast.

This is what your API / dashboard will call.
"""

import os
import pandas as pd
import joblib

from dotenv import load_dotenv
load_dotenv()

import hopsworks

DATA_FILE = "aqi_training_data.csv"          # fallback only
MODEL_DIR = "models"

HOPSWORKS_API_KEY = os.environ.get("HOPSWORKS_API_KEY")
HOPSWORKS_PROJECT = "aqi_predictor_ksa"
HOPSWORKS_HOST = "eu-west.cloud.hopsworks.ai"

FEATURE_VIEW_NAME = "aqi_feature_view"
FEATURE_VIEW_VERSION = 1


def connect_to_hopsworks():
    cert_dir = os.path.join(os.getcwd(), "hopsworks_certs")
    os.makedirs(cert_dir, exist_ok=True)

    project = hopsworks.login(
        project=HOPSWORKS_PROJECT,
        host=HOPSWORKS_HOST,
        port=443,
        api_key_value=HOPSWORKS_API_KEY,
        cert_folder=cert_dir,
    )
    return project


def load_recent_data_from_feature_store(hours_needed=72):
    project = connect_to_hopsworks()
    fs = project.get_feature_store()

    fv = fs.get_feature_view(name=FEATURE_VIEW_NAME, version=FEATURE_VIEW_VERSION)
    df = fv.get_batch_data()

    df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed")
    df = df.drop_duplicates(subset="timestamp").sort_values("timestamp").reset_index(drop=True)
    df = df.dropna(subset=["pm10", "o3", "no2", "so2", "co"]).reset_index(drop=True)

    recent = df.tail(hours_needed + 10).reset_index(drop=True)
    print(f"✅ Loaded recent data from Feature Store ({len(recent)} rows)")
    return recent


def load_recent_data_from_csv(hours_needed=72):
    print("⚠️ Falling back to local CSV (Hopsworks unavailable)")
    df = pd.read_csv(DATA_FILE)
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed")
    df = df.sort_values("timestamp").reset_index(drop=True)
    df = df.dropna(subset=["pm10", "o3", "no2", "so2", "co"]).reset_index(drop=True)

    recent = df.tail(hours_needed + 10).reset_index(drop=True)
    return recent


def load_recent_data(hours_needed=72):
    try:
        return load_recent_data_from_feature_store(hours_needed)
    except Exception as e:
        print(f"  -> Feature Store read failed: {e}")
        return load_recent_data_from_csv(hours_needed)


def engineer_features_for_prediction(df):
    df = df.copy()

    df["aqi_lag_1h"] = df["aqi"].shift(1)
    df["aqi_lag_3h"] = df["aqi"].shift(3)
    df["aqi_lag_24h"] = df["aqi"].shift(24)

    df["aqi_roll_mean_6h"] = df["aqi"].rolling(window=6).mean()
    df["aqi_roll_mean_24h"] = df["aqi"].rolling(window=24).mean()
    df["aqi_roll_mean_48h"] = df["aqi"].rolling(window=48).mean()
    df["aqi_roll_mean_72h"] = df["aqi"].rolling(window=72).mean()

    df["aqi_change_rate_1h"] = df["aqi"] - df["aqi_lag_1h"]

    df["pressure_lag_24h"] = df["pressure"].shift(24)
    df["pressure_change_24h"] = df["pressure"] - df["pressure_lag_24h"]

    return df


def get_feature_columns():
    return [
        "hour", "day", "month", "day_of_week",
        "temperature", "humidity", "pressure", "wind_speed",
        "aqi", "pm25", "pm10", "o3", "no2", "so2", "co",
        "aqi_lag_1h", "aqi_lag_3h", "aqi_lag_24h",
        "aqi_roll_mean_6h", "aqi_roll_mean_24h", "aqi_roll_mean_48h", "aqi_roll_mean_72h",
        "aqi_change_rate_1h",
        "pressure_lag_24h", "pressure_change_24h"
    ]


def aqi_category(aqi_value):
    if aqi_value is None:
        return "Unknown"
    if aqi_value <= 50:
        return "Good"
    elif aqi_value <= 100:
        return "Moderate"
    elif aqi_value <= 150:
        return "Unhealthy for Sensitive Groups"
    elif aqi_value <= 200:
        return "Unhealthy"
    elif aqi_value <= 300:
        return "Very Unhealthy"
    else:
        return "Hazardous"


def predict_next_3_days():
    df = load_recent_data()
    df = engineer_features_for_prediction(df)

    df = df.dropna().reset_index(drop=True)

    if len(df) == 0:
        raise ValueError("Not enough recent data to compute features for prediction.")

    latest_row = df.iloc[[-1]]
    feature_cols = get_feature_columns()
    X_latest = latest_row[feature_cols]

    predictions = {}
    for day_num, target_name in [(1, "target_day1"), (2, "target_day2"), (3, "target_day3")]:
        model_path = os.path.join(MODEL_DIR, f"{target_name}_model.pkl")
        model = joblib.load(model_path)

        pred_value = model.predict(X_latest)[0]
        pred_value = round(float(pred_value), 1)

        predictions[f"day_{day_num}"] = {
            "predicted_aqi": pred_value,
            "category": aqi_category(pred_value),
            "is_hazardous": pred_value > 150
        }

    result = {
        "based_on_timestamp": str(latest_row["timestamp"].values[0]),
        "current_aqi": float(latest_row["aqi"].values[0]),
        "current_category": aqi_category(float(latest_row["aqi"].values[0])),
        "forecast": predictions
    }

    return result


if __name__ == "__main__":
    result = predict_next_3_days()

    print("=== AQI Forecast ===")
    print(f"Based on data from: {result['based_on_timestamp']}")
    print(f"Current AQI: {result['current_aqi']} ({result['current_category']})")
    print()

    for day_key, data in result["forecast"].items():
        day_label = day_key.replace("_", " ").title()
        alert = " ⚠️ HAZARDOUS ALERT" if data["is_hazardous"] else ""
        print(f"{day_label}: {data['predicted_aqi']} ({data['category']}){alert}")  