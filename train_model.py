"""
Training Pipeline
1. Loads the merged training dataset
2. Cleans it and engineers extra features (lag values, rolling averages, AQI change rate, pressure trend)
3. Creates 3 targets: AQI 24h, 48h, and 72h ahead
4. Trains Ridge Regression, Random Forest, and XGBoost for each target
5. Evaluates with RMSE, MAE, R²
6. Saves the best model per horizon to disk (model registry, local version)
"""

import pandas as pd
import numpy as np
import joblib
import os

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor

DATA_FILE = "aqi_training_data.csv"
MODEL_DIR = "models"

os.makedirs(MODEL_DIR, exist_ok=True)


def load_data():
    df = pd.read_csv(DATA_FILE)
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed")
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def clean_data(df):
    before = len(df)
    df = df.dropna(subset=["pm10", "o3", "no2", "so2", "co"]).reset_index(drop=True)
    print(f"Dropped {before - len(df)} incomplete rows (old WAQI-only data)")
    return df


def engineer_features(df):
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

    df["target_day1"] = df["aqi"].shift(-24)
    df["target_day2"] = df["aqi"].shift(-48)
    df["target_day3"] = df["aqi"].shift(-72)

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


def train_and_evaluate(df, target_col, feature_cols):
    X = df[feature_cols]
    y = df[target_col]

    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    results = {}

    models = {
        "Ridge Regression": Ridge(alpha=1.0),
        "Random Forest": RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1),
        "XGBoost": XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05, random_state=42, n_jobs=-1)
    }

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, preds))
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)

        results[name] = {"model": model, "rmse": rmse, "mae": mae, "r2": r2}

        print(f"  {name}: RMSE={rmse:.2f}, MAE={mae:.2f}, R2={r2:.3f}")

    best_name = min(results, key=lambda k: results[k]["rmse"])
    best_model = results[best_name]["model"]
    print(f"  -> Best model: {best_name}")

    return best_name, best_model, results


if __name__ == "__main__":
    print("Loading data...")
    df = load_data()
    print(f"  -> {len(df)} rows loaded")

    print("\nCleaning data...")
    df = clean_data(df)

    print("\nEngineering features and targets...")
    df = engineer_features(df)
    print(f"  -> {len(df)} rows remain after feature engineering")

    feature_cols = get_feature_columns()

    targets = {
        "target_day1": "Day 1 (24h ahead)",
        "target_day2": "Day 2 (48h ahead)",
        "target_day3": "Day 3 (72h ahead)"
    }

    for target_col, label in targets.items():
        print(f"\n=== Training models for {label} ===")
        best_name, best_model, results = train_and_evaluate(df, target_col, feature_cols)

        model_path = os.path.join(MODEL_DIR, f"{target_col}_model.pkl")
        joblib.dump(best_model, model_path)
        print(f"  ✅ Saved best model to {model_path}")

    print("\n✅ Training pipeline complete. All models saved in the 'models' folder.")