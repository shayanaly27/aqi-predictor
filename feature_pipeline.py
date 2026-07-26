"""
Step 2: Feature Pipeline
Fetches raw weather + AQI data, extracts clean numeric features,
adds time-based features, and appends a row to a local CSV file.

Run this manually every so often (or on a schedule later) to build up history.
"""

import os
from dotenv import load_dotenv
load_dotenv()

import requests
import csv
from datetime import datetime

OPENWEATHER_API_KEY = os.environ.get("OPENWEATHER_API_KEY")
WAQI_TOKEN = os.environ.get("WAQI_TOKEN")

CITY = "Karachi"
LAT, LON = 24.8607, 67.0011
CSV_FILE = "aqi_features.csv"


def get_weather():
    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {"lat": LAT, "lon": LON, "appid": OPENWEATHER_API_KEY, "units": "metric"}
    response = requests.get(url, params=params)
    return response.json()


def get_aqi():
    url = "https://api.waqi.info/feed/karachi/"
    params = {"token": WAQI_TOKEN}
    response = requests.get(url, params=params)
    return response.json()


def build_feature_row():
    weather = get_weather()
    aqi_data = get_aqi()

    now = datetime.now()

    main = weather.get("main", {})
    wind = weather.get("wind", {})

    data = aqi_data.get("data", {})
    iaqi = data.get("iaqi", {})

    row = {
        "timestamp": now.isoformat(),
        "hour": now.hour,
        "day": now.day,
        "month": now.month,
        "day_of_week": now.weekday(),

        "temperature": main.get("temp"),
        "humidity": main.get("humidity"),
        "pressure": main.get("pressure"),
        "wind_speed": wind.get("speed"),

        "aqi": data.get("aqi"),
        "pm25": iaqi.get("pm25", {}).get("v"),
        "pm10": iaqi.get("pm10", {}).get("v"),
        "o3": iaqi.get("o3", {}).get("v"),
        "no2": iaqi.get("no2", {}).get("v"),
        "so2": iaqi.get("so2", {}).get("v"),
        "co": iaqi.get("co", {}).get("v"),
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