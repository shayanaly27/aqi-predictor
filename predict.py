"""
Prediction Script (v2 — live-current-reading fix)

Takes the most recent HISTORICAL data from the Hopsworks Feature Store
(used only to compute lag/rolling features), loads the 3 models FROM THE
HOPSWORKS MODEL REGISTRY, and outputs a clean 3-day AQI forecast.

FIX (Sep 2026): Hopsworks' offline materialization job has been stuck for
several days, so fv.get_batch_data() was silently returning data from
Sep 2 without ever raising an error — the dashboard was showing a
3-day-old "current AQI" as if it were live.

The fix: the CURRENT reading (today's AQI, pollutants, timestamp — what
the dashboard's hero panel shows) is now fetched LIVE and DIRECTLY from
Open-Meteo at request time, the same source feature_pipeline.py already
uses hourly. This guarantees the dashboard always shows today's real
data, independent of whether Hopsworks' offline store has caught up.

Historical Feature Store data is still used for the lag/rolling features
that feed the model (aqi_lag_24h, aqi_roll_mean_72h, etc.) — those need
real history and can tolerate being a little behind. A
data_freshness section is added to the response so the dashboard can
show a visible warning if that historical tail is stale, rather than
hiding it.

This is a resilience fallback layered on top of the existing
architecture, not a replacement for Hopsworks as the system's feature
store — once Hopsworks' materialization job recovers, this live-fetch
path simply becomes a freshness guarantee rather than a workaround.
"""

import os
import time
import pandas as pd
import joblib
import requests
from datetime import datetime

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

# Same coordinates and endpoints feature_pipeline.py uses for live ingestion.
LAT, LON = 24.8607, 67.0011
WEATHER_URL = "https://api.open-meteo.com/v1/forecast"
AQI_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"

# How long cached "recent historical data" from the Feature Store stays
# fresh before we re-fetch. This is just for the lag/rolling features.
RECENT_DATA_TTL_SECONDS = 5 * 60

# How long a cached model stays in memory before we re-check the registry
# for a newer version.
MODEL_CACHE_TTL_SECONDS = 60 * 60

# How long a live Open-Meteo reading stays cached before re-fetching.
# Kept short since this is what "current" means to the dashboard.
LIVE_READING_TTL_SECONDS = 5 * 60

# If the newest row in the Feature Store's historical data is older than
# this, we flag it as stale in the response instead of hiding it.
STALE_THRESHOLD_HOURS = 3

# ─────────────────────────────────────────────────────────────
# In-memory caches (module-level, live for the lifetime of the process)
# ─────────────────────────────────────────────────────────────
_project_cache = {"project": None}
_model_cache = {}  # target_col -> (model, loaded_at)
_recent_data_cache = {"df": None, "fetched_at": 0.0}
_live_reading_cache = {"row": None, "fetched_at": 0.0}


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
    _project_cache["project"] = None


# ─────────────────────────────────────────────────────────────
# LIVE current reading (bypasses Hopsworks entirely)
# ─────────────────────────────────────────────────────────────
def fetch_live_current_row():
    """
    Fetches the real, current AQI/weather reading directly from Open-Meteo
    — the same source feature_pipeline.py already uses hourly. This is
    what guarantees the dashboard shows TODAY's data no matter what state
    Hopsworks' offline materialization is in.
    """
    weather = requests.get(WEATHER_URL, params={
        "latitude": LAT, "longitude": LON,
        "current": "temperature_2m,relative_humidity_2m,surface_pressure,wind_speed_10m",
        "timezone": "Asia/Karachi"
    }).json()

    aqi_data = requests.get(AQI_URL, params={
        "latitude": LAT, "longitude": LON,
        "current": "us_aqi,pm2_5,pm10,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone",
        "timezone": "Asia/Karachi"
    }).json()

    # UTC, timezone-naive - matches how feature_pipeline.py's timestamps
    # land in Hopsworks (GitHub Actions runners are UTC). Using local
    # machine time here (datetime.now()) would silently shift hour/
    # day_of_week features when run outside UTC, and would also crash
    # when concatenated with tz-aware historical data below.
    now = datetime.utcnow()
    w = weather.get("current", {})
    a = aqi_data.get("current", {})

    row = {
        "timestamp": now,
        "hour": now.hour,
        "day": now.day,
        "month": now.month,
        "day_of_week": now.weekday(),
        "temperature": w.get("temperature_2m"),
        "humidity": w.get("relative_humidity_2m"),
        "pressure": w.get("surface_pressure"),
        "wind_speed": w.get("wind_speed_10m"),
        "aqi": a.get("us_aqi"),
        "pm25": a.get("pm2_5"),
        "pm10": a.get("pm10"),
        "o3": a.get("ozone"),
        "no2": a.get("nitrogen_dioxide"),
        "so2": a.get("sulphur_dioxide"),
        "co": a.get("carbon_monoxide"),
    }

    if row["aqi"] is None:
        raise ValueError("Live AQI reading came back null from Open-Meteo")

    print(f"✅ Fetched live current reading directly from Open-Meteo (aqi={row['aqi']}, ts={now})")
    return row


def get_live_current_row_cached(ttl_seconds=LIVE_READING_TTL_SECONDS, force_refresh=False):
    now = time.time()
    cached_row = _live_reading_cache["row"]
    age = now - _live_reading_cache["fetched_at"]

    if not force_refresh and cached_row is not None and age < ttl_seconds:
        return cached_row

    row = fetch_live_current_row()
    _live_reading_cache["row"] = row
    _live_reading_cache["fetched_at"] = now
    return row


# ─────────────────────────────────────────────────────────────
# Historical data from Hopsworks (for lag / rolling features only)
# ─────────────────────────────────────────────────────────────
def load_recent_data_from_feature_store(project, hours_needed=72):
    fs = project.get_feature_store()
    fv = fs.get_feature_view(name=FEATURE_VIEW_NAME, version=FEATURE_VIEW_VERSION)
    df = fv.get_batch_data()

    df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed")
    # Hopsworks returns tz-aware (UTC) timestamps; strip the tz label so
    # this column can later be concatenated with the live reading's
    # naive-UTC timestamp without pandas raising a tz-mismatch error.
    # Values are already UTC, so this only drops the label, not shifts time.
    if df["timestamp"].dt.tz is not None:
        df["timestamp"] = df["timestamp"].dt.tz_localize(None)
    df = df.drop_duplicates(subset="timestamp").sort_values("timestamp").reset_index(drop=True)
    df = df.dropna(subset=["pm10", "o3", "no2", "so2", "co"]).reset_index(drop=True)

    recent = df.tail(hours_needed + 10).reset_index(drop=True)
    print(f"✅ Loaded historical data from Feature Store ({len(recent)} rows, "
          f"newest={recent['timestamp'].max() if len(recent) else 'n/a'})")
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
    now = time.time()
    cached_df = _recent_data_cache["df"]
    age = now - _recent_data_cache["fetched_at"]

    if cached_df is not None and age < ttl_seconds:
        return get_cached_project(), cached_df

    project, df = load_recent_data(hours_needed)
    _recent_data_cache["df"] = df
    _recent_data_cache["fetched_at"] = now
    return project, df


def compute_data_freshness(historical_df):
    """Returns (hours_behind, is_stale) for the historical Feature Store data."""
    if historical_df is None or len(historical_df) == 0:
        return None, True

    newest = pd.to_datetime(historical_df["timestamp"].max())
    hours_behind = (datetime.utcnow() - newest.to_pydatetime()).total_seconds() / 3600
    return round(hours_behind, 1), hours_behind > STALE_THRESHOLD_HOURS


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
    print("Preloading Hopsworks connection and models...")
    project = get_cached_project()
    for target_col in TARGET_COLS:
        try:
            get_cached_model(project, target_col)
        except Exception as e:
            print(f"  ⚠️ Could not preload {target_col}: {e}")
    try:
        get_live_current_row_cached()
    except Exception as e:
        print(f"  ⚠️ Could not preload live reading: {e}")
    print("✅ Preload complete")


def predict_next_3_days(force_refresh: bool = False):
    """
    Set force_refresh=True to bypass all caches (recent-history, model,
    AND live reading) - used by the dashboard's "Fetch Latest Data" button.

    The row used as "current" (today's AQI, pollutants, timestamp - what
    the dashboard displays as live) is now ALWAYS the freshly-fetched
    Open-Meteo reading, not whatever Hopsworks' offline table last
    materialized. Historical Feature Store data is only used to compute
    the lag/rolling features that feed the model.
    """
    if force_refresh:
        project, historical_df = load_recent_data()
    else:
        project, historical_df = load_recent_data_cached()

    live_row = get_live_current_row_cached(force_refresh=force_refresh)

    hours_behind, is_stale = compute_data_freshness(historical_df)

    # Append the live reading as the newest row so lag/rolling features
    # are computed with it included, then engineer features on the whole
    # series and take that last (live) row for prediction.
    df = pd.concat([historical_df, pd.DataFrame([live_row])], ignore_index=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed")
    df = df.drop_duplicates(subset="timestamp").sort_values("timestamp").reset_index(drop=True)

    df = engineer_features_for_prediction(df)

    # Only require the target row (the live one) to be fully engineered;
    # earlier historical rows may lag NaNs at the edges, which is fine.
    latest_row = df.iloc[[-1]]
    feature_cols = get_feature_columns()

    missing = latest_row[feature_cols].isna().any(axis=1).iloc[0]
    if missing:
        # Lag/rolling features couldn't be computed (e.g. big gap between
        # historical data and the live row). Fall back to the last fully
        # engineered historical row rather than crash, but keep the LIVE
        # values for current_aqi/pollutants/timestamp shown on the dashboard.
        print("⚠️ Live row missing lag/rolling features (gap vs historical data) "
              "- using last complete historical row for model inputs, "
              "but keeping live values for current-state display.")
        engineered_complete = df.dropna(subset=feature_cols)
        if len(engineered_complete) == 0:
            raise ValueError("Not enough historical data to compute any complete feature row.")
        model_input_row = engineered_complete.iloc[[-1]]
    else:
        model_input_row = latest_row

    X_latest = model_input_row[feature_cols]

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
        # These come from the LIVE reading — always today's real data.
        # ISO 8601 with explicit UTC "Z" suffix - str(datetime) alone
        # (e.g. "2026-09-05 08:56:08.238158") has no timezone marker and
        # parses inconsistently across browsers (Safari in particular).
        # Appending "Z" guarantees every client interprets this the same way.
        "based_on_timestamp": live_row["timestamp"].isoformat() + "Z",
        "current_aqi": float(live_row["aqi"]),
        "current_category": aqi_category(float(live_row["aqi"])),
        "pollutants": {
            "pm25": round(float(live_row["pm25"]), 1),
            "pm10": round(float(live_row["pm10"]), 1),
            "o3": round(float(live_row["o3"]), 1),
            "no2": round(float(live_row["no2"]), 1),
            "so2": round(float(live_row["so2"]), 1),
            "co": round(float(live_row["co"]), 2),
        },
        "forecast": predictions,
        "data_freshness": {
            "current_reading_source": "live_openmeteo",
            "feature_store_hours_behind": hours_behind,
            "feature_store_stale": is_stale,
            "note": (
                "Hopsworks offline materialization is currently behind; "
                "lag/rolling model inputs use the most recent available "
                "historical data, but current AQI and pollutants above are live."
                if is_stale else
                "Feature Store data is current."
            )
        }
    }

    return result


if __name__ == "__main__":
    result = predict_next_3_days()

    print("=== AQI Forecast ===")
    print(f"Based on data from: {result['based_on_timestamp']}")
    print(f"Current AQI: {result['current_aqi']} ({result['current_category']})")
    print(f"Pollutants: {result['pollutants']}")
    print(f"Data freshness: {result['data_freshness']}")
    print()

    for day_key, data in result["forecast"].items():
        day_label = day_key.replace("_", " ").title()
        alert = " ⚠️ HAZARDOUS ALERT" if data["is_hazardous"] else ""
        print(f"{day_label}: {data['predicted_aqi']} ({data['category']}){alert}")