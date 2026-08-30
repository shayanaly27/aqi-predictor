"""
Training Pipeline
1. Fetches historical (features, targets) from the Hopsworks Feature Store
   (falls back to the local CSV if Hopsworks is unavailable)
2. Cleans it and engineers extra features (lag values, rolling averages, AQI change rate, pressure trend)
3. Creates 3 targets: AQI 24h, 48h, and 72h ahead
4. Trains Ridge Regression, Random Forest, and XGBoost for each target
5. Evaluates with RMSE, MAE, R2
6. Saves the best model per horizon locally AND to the Hopsworks Model Registry
7. Saves model_metrics.json from THIS run's real numbers - no manual typing.
   Uses "day_1"/"day_2"/"day_3" keys to match what the frontend expects.
"""

import os
import json
import pandas as pd
import numpy as np
import joblib

from dotenv import load_dotenv
load_dotenv()

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor

import hopsworks

DATA_FILE = "aqi_training_data.csv"          # fallback only
MODEL_DIR = "models"

HOPSWORKS_API_KEY = os.environ.get("HOPSWORKS_API_KEY")
HOPSWORKS_PROJECT = "aqi_predictor_ksa"
HOPSWORKS_HOST = "eu-west.cloud.hopsworks.ai"

FEATURE_VIEW_NAME = "aqi_feature_view"
FEATURE_VIEW_VERSION = 1

os.makedirs(MODEL_DIR, exist_ok=True)


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
    print(f"✅ Connected to Hopsworks project: {project.name}")
    return project


def load_data_from_feature_store():
    project = connect_to_hopsworks()
    fs = project.get_feature_store()

    fv = fs.get_feature_view(name=FEATURE_VIEW_NAME, version=FEATURE_VIEW_VERSION)
    df = fv.get_batch_data()

    df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed")
    df = df.drop_duplicates(subset="timestamp").sort_values("timestamp").reset_index(drop=True)

    print(f"✅ Loaded {len(df)} rows from Feature Store ({FEATURE_VIEW_NAME} v{FEATURE_VIEW_VERSION})")
    return project, df


def load_data_from_csv():
    print("⚠️ Falling back to local CSV (Hopsworks unavailable)")
    df = pd.read_csv(DATA_FILE)
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed")
    df = df.sort_values("timestamp").reset_index(drop=True)
    return None, df


def load_data():
    try:
        return load_data_from_feature_store()
    except Exception as e:
        print(f"  -> Feature Store read failed: {e}")
        return load_data_from_csv()


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
    """
    Trains all 3 algorithms, returns the best one PLUS the full metrics
    list for every algorithm (used to populate model_metrics.json for
    the dashboard's comparison table - not just the winner).
    """
    X = df[feature_cols]
    y = df[target_col]

    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    results = {}
    run_metrics_list = []

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
        run_metrics_list.append({
            "name": name,
            "rmse": round(float(rmse), 2),
            "mae": round(float(mae), 2),
            "r2": round(float(r2), 3),
        })

        print(f"  {name}: RMSE={rmse:.2f}, MAE={mae:.2f}, R2={r2:.3f}")

    best_name = min(results, key=lambda k: results[k]["rmse"])
    best_model = results[best_name]["model"]
    best_metrics = results[best_name]
    print(f"  -> Best model: {best_name}")

    run_metrics_list.sort(key=lambda m: m["rmse"])

    return best_name, best_model, best_metrics, run_metrics_list


def push_model_to_registry(project, model_path, target_col, best_name, metrics):
    """Store the trained model in the Hopsworks Model Registry."""
    if project is None:
        print("  ⚠️ No Hopsworks connection - skipping Model Registry upload")
        return

    try:
        mr = project.get_model_registry()

        model_meta = mr.python.create_model(
            name=f"aqi_{target_col}_model",
            metrics={
                "rmse": float(metrics["rmse"]),
                "mae": float(metrics["mae"]),
                "r2": float(metrics["r2"]),
            },
            description=f"Best model ({best_name}) for {target_col} AQI forecast",
        )
        model_meta.save(model_path)
        print(f"  ✅ Pushed {target_col} model to Hopsworks Model Registry")
    except Exception as e:
        print(f"  ⚠️ Model Registry upload failed: {e}")


if __name__ == "__main__":
    print("Loading data...")
    project, df = load_data()
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

    all_run_metrics = {}

    for target_col, label in targets.items():
        print(f"\n=== Training models for {label} ===")
        best_name, best_model, metrics, run_metrics_list = train_and_evaluate(df, target_col, feature_cols)
        all_run_metrics[target_col] = run_metrics_list

        model_path = os.path.join(MODEL_DIR, f"{target_col}_model.pkl")
        joblib.dump(best_model, model_path)
        print(f"  ✅ Saved best model locally to {model_path}")

        push_model_to_registry(project, model_path, target_col, best_name, metrics)

    # Save real metrics for the dashboard, generated from this actual run -
    # not manually typed, so it always matches what's really deployed.
    # Frontend expects keys "day_1"/"day_2"/"day_3", not "target_day1" etc.
    KEY_MAP = {
        "target_day1": "day_1",
        "target_day2": "day_2",
        "target_day3": "day_3",
    }

    dashboard_metrics = {}
    for target_col, label in targets.items():
        dashboard_key = KEY_MAP[target_col]
        dashboard_metrics[dashboard_key] = {
            "label": label,
            "models": all_run_metrics[target_col],
            "best_model": all_run_metrics[target_col][0]["name"] if all_run_metrics[target_col] else None,
        }

    with open("model_metrics.json", "w") as f:
        json.dump(dashboard_metrics, f, indent=2)
    print("\n✅ Saved model_metrics.json from this training run")

    print("\n✅ Training pipeline complete. All models saved locally and to the Model Registry.")