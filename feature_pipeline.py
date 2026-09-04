"""
Step 2 (v5): Feature Pipeline - fetches live weather + AQI from Open-Meteo
and inserts DIRECTLY into the Hopsworks Feature Store. No CSV, no git commit
of data - Hopsworks is the only persistence layer now.

Uses write_options={"wait_for_job": True} on a SINGLE attempt (no retry
loop) so a materialization failure is surfaced clearly as a failed GitHub
Action run, but doesn't multiply into a 20+ minute job by retrying a
service-side failure 3 times in a row. Hopsworks' free-tier materialization
service is known to intermittently fail or stall under load - this is a
widely-reported, cohort-wide platform issue (see project report,
Section 11), not something retries on our end can reliably fix.
"""

import os
from dotenv import load_dotenv
load_dotenv()

import requests
import pandas as pd
from datetime import datetime

import hopsworks

LAT, LON = 24.8607, 67.0011

WEATHER_URL = "https://api.open-meteo.com/v1/forecast"
AQI_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"

HOPSWORKS_API_KEY = os.environ.get("HOPSWORKS_API_KEY")
HOPSWORKS_PROJECT = "aqi_predictor_ksa"
HOPSWORKS_HOST = "eu-west.cloud.hopsworks.ai"

FEATURE_GROUP_NAME = "aqi_features"
FEATURE_GROUP_VERSION = 1


def get_current_weather():
    params = {
        "latitude": LAT, "longitude": LON,
        "current": "temperature_2m,relative_humidity_2m,surface_pressure,wind_speed_10m",
        "timezone": "Asia/Karachi"
    }
    return requests.get(WEATHER_URL, params=params).json()


def get_current_aqi():
    params = {
        "latitude": LAT, "longitude": LON,
        "current": "us_aqi,pm2_5,pm10,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone",
        "timezone": "Asia/Karachi"
    }
    return requests.get(AQI_URL, params=params).json()


def build_feature_row():
    weather = get_current_weather()
    aqi_data = get_current_aqi()
    now = datetime.now()

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
    return row


def fix_dtypes(df):
    """Match the schema push_to_hopsworks.py already established: aqi and
    humidity must be plain int64, not floats or nullable Int64."""
    for col in ["aqi", "humidity"]:
        if pd.isna(df[col].iloc[0]):
            raise ValueError(f"'{col}' came back null from the API - skipping this row rather than pushing bad data")
        df[col] = df[col].round().astype("int64")
    return df


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


def insert_row(aqi_fg, df):
    """
    Single attempt, wait_for_job=True: confirms materialization actually
    succeeded (visible failure in CI if not) without retrying a service-
    side failure 3x and ballooning an hourly job to 25+ minutes. If this
    fails, the row is still staged (Kafka insert succeeds even when
    materialization does) - the next hourly run's materialization attempt
    will pick up the backlog once Hopsworks' service recovers.
    """
    try:
        aqi_fg.insert(df, write_options={"wait_for_job": True})
        print("✅ Row inserted AND materialized into Hopsworks Feature Store")
    except Exception as e:
        print(f"  ⚠️ Materialization failed this run (Hopsworks-side): {e}")
        print("  -> Row is still staged; a future successful materialization run will pick it up.")
        raise


if __name__ == "__main__":
    row = build_feature_row()
    print("Fetched feature row:")
    for k, v in row.items():
        print(f"  {k}: {v}")

    df = pd.DataFrame([row])
    df = fix_dtypes(df)

    print("\nConnecting to Hopsworks...")
    project = connect_to_hopsworks()
    fs = project.get_feature_store()

    aqi_fg = fs.get_feature_group(name=FEATURE_GROUP_NAME, version=FEATURE_GROUP_VERSION)

    print("Inserting row into Feature Group (waiting for materialization, single attempt)...")
    insert_row(aqi_fg, df)