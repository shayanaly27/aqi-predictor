"""
Prediction Script
Takes the most recent data from the Hopsworks Feature Store, engineers the
same features used in training, loads the 3 models FROM THE HOPSWORKS MODEL
REGISTRY (not local files), and outputs a clean 3-day AQI forecast.

This is what your API / dashboard calls.

PERFORMANCE NOTE: Hopsworks login, model downloads, and feature-store reads
are expensive (network + I/O). This version caches all three in memory so
they only happen once per process (or once per TTL window), instead of on
every single prediction request. Cached models expire after
MODEL_CACHE_TTL_SECONDS so the API picks up newly-trained versions from
the daily training pipeline without needing a manual restart.
"""

import os
import time
import pandas as pd
import joblib

from dotenv import load_dotenv
load_dotenv()

import hopsworks

DATA_FILE = "aqi_training_data.csv"          # fallback only, if Hopsworks is down

HOPSWORKS_API_KEY = os.environ.get("HOPSWORKS_API_KEY")
HOPSWORKS_PROJECT = "aqi_predictor_ksa"
HOPSWORKS_HOST = "eu-west.cloud.hopsworks.ai"

FEATURE_VIEW_NAME = "aqi_feature_view"
FEATURE_VIEW_VERSION = 1

TARGET_COLS = ["target_day1", "target_day2", "target_day3"]

# How long cached "recent data" from the Feature Store stays fresh before
# we re-fetch. The live pipeline only writes a new row once an hour, so
# there's no benefit to hitting Hopsworks more often than this.
RECENT_DATA_TTL_SECONDS = 5 * 60

# How long a cached model stays in memory before we re-check the registry
# for a newer version. The daily training pipeline can produce a new
# version at any time - without this TTL, a long-running server process
# would keep serving a stale model forever.
MODEL_CACHE_TTL_SECONDS = 60 * 60

# ─────────────────────────────────────────────────────────────
# In-memory caches (module-level, live for the lifetime of the process)
# ─────────────────────────────────────────────────────────────
_project_cache = {"project": None}
_model_cache = {}  # target_col -> (model, loaded_at)
_recent_data_cache = {"df": None, "fetched_at": 0.0}


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


def get_cached_project():
    """
    Returns a cached Hopsworks project connection, logging in only once
    per process. If a previous connection is known-bad, this will retry.
    Returns None if the connection cannot be established.
    """
    if _project_cache["project"] is not None:
        return _project_cache["project"]

    try:
        project = connect_to_hopsworks()
        _project_cache["project"] = project
        print("✅ Hopsworks connection established and cached")
        return project
    except Exception as e:
        print(f"  -> Hopsworks login failed: {e}")
        return None


def invalidate_project_cache():
    """Call this if a cached connection turns out to be dead mid-use."""
    _project_cache["project"] = None


def load_recent_data_from_feature_store(project, hours_needed=72):
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
    """
    Returns (project, df). project is None if we fell back to CSV -
    downstream code uses this to decide whether it can also load models
    from the registry, or needs a local models/ fallback too.
    """
    try:
        project = get_cached_project()
        if project is None:
            raise RuntimeError("no cached/live Hopsworks connection")
        df = load_recent_data_from_feature_store(project, hours_needed)
        return project, df
    except Exception as e:
        print(f"  -> Feature Store read failed: {e}")
        invalidate_project_cache()
        return None, load_recent_data_from_csv(hours_needed)


def load_recent_data_cached(hours_needed=72, ttl_seconds=RECENT_DATA_TTL_SECONDS):
    """
    TTL-cached wrapper around load_recent_data(). Avoids hitting the
    Feature Store on every single API request - the underlying data only
    changes once an hour anyway (the feature pipeline's schedule).
    """
    now = time.time()
    cached_df = _recent_data_cache["df"]
    age = now - _recent_data_cache["fetched_at"]

    if cached_df is not None and age < ttl_seconds:
        # Still need a project handle for model loading downstream, so
        # re-fetch the cached connection (cheap - no network call).
        return get_cached_project(), cached_df

    project, df = load_recent_data(hours_needed)
    _recent_data_cache["df"] = df
    _recent_data_cache["fetched_at"] = now
    return project, df


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


def load_model_from_registry(project, target_col):
    """Downloads the LATEST version of the model from Hopsworks Model Registry."""
    mr = project.get_model_registry()

    all_versions = mr.get_models(name=f"aqi_{target_col}_model")
    if not all_versions:
        raise ValueError(f"No model found in registry for aqi_{target_col}_model")

    latest = max(all_versions, key=lambda m: m.version)
    print(f"  -> Loading aqi_{target_col}_model version {latest.version}")

    model_dir = latest.download()
    model_path = os.path.join(model_dir, f"{target_col}_model.pkl")
    return joblib.load(model_path)


def load_model_from_local(target_col):
    model_path = os.path.join("models", f"{target_col}_model.pkl")
    return joblib.load(model_path)


def get_cached_model(project, target_col, force_refresh=False):
    """
    Returns a model from the in-memory cache if it's still within
    MODEL_CACHE_TTL_SECONDS. Otherwise (or if force_refresh=True)
    re-checks the registry for the latest version - this is what lets
    the API pick up newly-trained models from the daily pipeline without
    needing a manual restart.
    """
    now = time.time()
    cached = _model_cache.get(target_col)
    if not force_refresh and cached is not None:
        model, loaded_at = cached
        if now - loaded_at < MODEL_CACHE_TTL_SECONDS:
            return model

    try:
        if project is not None:
            model = load_model_from_registry(project, target_col)
        else:
            raise RuntimeError("no Hopsworks connection - using local fallback")
    except Exception as e:
        print(f"  -> Registry load failed for {target_col}, trying local fallback: {e}")
        model = load_model_from_local(target_col)

    _model_cache[target_col] = (model, now)
    return model


def preload():
    """
    Call this once at API startup (see api.py's startup event) so the
    FIRST real request isn't the one paying for Hopsworks login + all
    3 model downloads. Safe to call multiple times - cached work is
    skipped automatically.
    """
    print("Preloading Hopsworks connection and models...")
    project = get_cached_project()
    for target_col in TARGET_COLS:
        try:
            get_cached_model(project, target_col)
        except Exception as e:
            print(f"  ⚠️ Could not preload {target_col}: {e}")
    print("✅ Preload complete")


def predict_next_3_days(force_refresh: bool = False):
    """
    Set force_refresh=True to bypass both the recent-data cache and the
    model cache - used by the dashboard's "Fetch Latest Data" button so
    it actually pulls fresh data instead of returning the same cached
    response.
    """
    if force_refresh:
        project, df = load_recent_data()
    else:
        project, df = load_recent_data_cached()

    df = engineer_features_for_prediction(df)
    df = df.dropna().reset_index(drop=True)

    if len(df) == 0:
        raise ValueError("Not enough recent data to compute features for prediction.")

    latest_row = df.iloc[[-1]]
    feature_cols = get_feature_columns()
    X_latest = latest_row[feature_cols]

    predictions = {}
    for day_num, target_name in [(1, "target_day1"), (2, "target_day2"), (3, "target_day3")]:
        model = get_cached_model(project, target_name, force_refresh=force_refresh)

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
        "pollutants": {
            "pm25": round(float(latest_row["pm25"].values[0]), 1),
            "pm10": round(float(latest_row["pm10"].values[0]), 1),
            "o3": round(float(latest_row["o3"].values[0]), 1),
            "no2": round(float(latest_row["no2"].values[0]), 1),
            "so2": round(float(latest_row["so2"].values[0]), 1),
            "co": round(float(latest_row["co"].values[0]), 2),
        },
        "forecast": predictions
    }

    return result


if __name__ == "__main__":
    result = predict_next_3_days()

    print("=== AQI Forecast ===")
    print(f"Based on data from: {result['based_on_timestamp']}")
    print(f"Current AQI: {result['current_aqi']} ({result['current_category']})")
    print(f"Pollutants: {result['pollutants']}")
    print()

    for day_key, data in result["forecast"].items():
        day_label = day_key.replace("_", " ").title()
        alert = " ⚠️ HAZARDOUS ALERT" if data["is_hazardous"] else ""
        print(f"{day_label}: {data['predicted_aqi']} ({data['category']}){alert}")