"""
Step 2 (v2): Feature Pipeline - now using Open-Meteo for BOTH weather and AQI.
This replaces WAQI, whose Karachi station was returning frozen/stale AQI values.
Open-Meteo's forecast endpoint gives current + near-term data, updated regularly.
"""

import os
from dotenv import load_dotenv
load_dotenv()

import requests
import csv
from datetime import datetime

LAT, LON = 24.8607, 67.0011
CSV_FILE = "aqi_features.csv"

WEATHER_URL = "https://api.open-meteo.com/v1/forecast"
AQI_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"


def get_current_weather():
    params = {
        "latitude": LAT,
        "longitude": LON,
        "current": "temperature_2m,relative_humidity_2m,surface_pressure,wind_speed_10m",
        "timezone": "Asia/Karachi"
    }
    response = requests.get(WEATHER_URL, params=params)
    return response.json()


def get_current_aqi():
    params = {
        "latitude": LAT,
        "longitude": LON,
        "current": "us_aqi,pm2_5,pm10,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone",
        "timezone": "Asia/Karachi"
    }
    response = requests.get(AQI_URL, params=params)
    return response.json()


def build_feature_row():
    weather = get_current_weather()
    aqi_data = get_current_aqi()

    now = datetime.now()

    w_current = weather.get("current", {})
    a_current = aqi_data.get("current", {})

    row = {
        "timestamp": now.isoformat(),
        "hour": now.hour,
        "day": now.day,
        "month": now.month,
        "day_of_week": now.weekday(),

        "temperature": w_current.get("temperature_2m"),
        "humidity": w_current.get("relative_humidity_2m"),
        "pressure": w_current.get("surface_pressure"),
        "wind_speed": w_current.get("wind_speed_10m"),

        "aqi": a_current.get("us_aqi"),
        "pm25": a_current.get("pm2_5"),
        "pm10": a_current.get("pm10"),
        "o3": a_current.get("ozone"),
        "no2": a_current.get("nitrogen_dioxide"),
        "so2": a_current.get("sulphur_dioxide"),
        "co": a_current.get("carbon_monoxide"),
    }
    return row


def append_to_csv(row):
    file_exists = os.path.isfile(CSV_FILE)

    with open(CSV_FILE, mode="a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


if __name__ == "__main__":
    row = build_feature_row()
    print("Fetched feature row:")
    for k, v in row.items():
        print(f"  {k}: {v}")

    append_to_csv(row)
    print(f"\n✅ Row appended to {CSV_FILE}")