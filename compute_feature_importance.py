"""
Computes SHAP feature importance for the Day 1 model and saves it as a
small JSON file the dashboard's /feature-importance endpoint can serve
instantly, without recomputing SHAP on every request.

Run this ONCE after training (or re-run whenever you retrain):
    python compute_feature_importance.py
"""

import pandas as pd
import joblib
import json
import shap
import numpy as np

DATA_FILE = "aqi_training_data.csv"
MODEL_PATH = "models/target_day1_model.pkl"
OUTPUT_FILE = "feature_importance.json"


def load_and_prepare_data():
    df = pd.read_csv(DATA_FILE)
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed")
    df = df.sort_values("timestamp").reset_index(drop=True)
    df = df.dropna(subset=["pm10", "o3", "no2", "so2", "co"]).reset_index(drop=True)

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

    df = df.dropna().reset_index(drop=True)
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


if __name__ == "__main__":
    print("Loading data and model...")
    df = load_and_prepare_data()
    feature_cols = get_feature_columns()
    model = joblib.load(MODEL_PATH)

    X_sample = df[feature_cols].sample(n=min(1000, len(df)), random_state=42)

    print("Computing SHAP values...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    # mean absolute SHAP value per feature = overall importance
    mean_abs_shap = np.abs(shap_values).mean(axis=0)

    importance = [
        {"feature": feat, "importance": round(float(val), 4)}
        for feat, val in zip(feature_cols, mean_abs_shap)
    ]
    importance.sort(key=lambda x: x["importance"], reverse=True)

    output = {
        "horizon": "Day 1 (24h ahead)",
        "model": type(model).__name__,
        "top_features": importance[:8],  # top 8 is plenty for a dashboard chart
    }

    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=2)

    print(f"✅ Saved top 8 feature importances to {OUTPUT_FILE}")
    for item in output["top_features"]:
        print(f"   {item['feature']}: {item['importance']}")