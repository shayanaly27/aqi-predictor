"""
Backfill Script: Pull 3.5 years of historical weather + AQI data from Open-Meteo
and save it as a clean CSV - this becomes the core training dataset.

Open-Meteo limits how much data you can request in a single call, so we
request it in monthly chunks and combine everything into one CSV.

Run this ONCE to build your historical dataset. Takes a while (42+ chunks).
"""

import requests
import csv
import time
from datetime import datetime, timedelta

LAT, LON = 24.8607, 67.0011
OUTPUT_FILE = "aqi_historical_backfill_v2.csv"

WEATHER_URL = "https://archive-api.open-meteo.com/v1/archive"
AQI_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"


def get_weather_chunk(start_date, end_date):
    params = {
        "latitude": LAT,
        "longitude": LON,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": "temperature_2m,relative_humidity_2m,surface_pressure,wind_speed_10m",
        "timezone": "Asia/Karachi"
    }
    response = requests.get(WEATHER_URL, params=params)
    return response.json()


def get_aqi_chunk(start_date, end_date):
    params = {
        "latitude": LAT,
        "longitude": LON,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": "us_aqi,pm2_5,pm10,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone",
        "timezone": "Asia/Karachi"
    }
    response = requests.get(AQI_URL, params=params)
    return response.json()


def generate_month_ranges(start_date, end_date):
    """Break the full date range into monthly (start, end) chunks."""
    ranges = []
    current = start_date
    while current < end_date:
        chunk_end = min(current + timedelta(days=30), end_date)
        ranges.append((current.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d")))
        current = chunk_end + timedelta(days=1)
    return ranges


def build_rows_for_chunk(start_date, end_date):
    weather = get_weather_chunk(start_date, end_date)
    aqi_data = get_aqi_chunk(start_date, end_date)

    w_hourly = weather.get("hourly", {})
    a_hourly = aqi_data.get("hourly", {})

    times = w_hourly.get("time", [])
    temps = w_hourly.get("temperature_2m", [])
    humidity = w_hourly.get("relative_humidity_2m", [])
    pressure = w_hourly.get("surface_pressure", [])
    wind = w_hourly.get("wind_speed_10m", [])

    a_times = a_hourly.get("time", [])
    aqi = a_hourly.get("us_aqi", [])
    pm25 = a_hourly.get("pm2_5", [])
    pm10 = a_hourly.get("pm10", [])
    co = a_hourly.get("carbon_monoxide", [])
    no2 = a_hourly.get("nitrogen_dioxide", [])
    so2 = a_hourly.get("sulphur_dioxide", [])
    o3 = a_hourly.get("ozone", [])

    aqi_index = {t: i for i, t in enumerate(a_times)}

    rows = []
    for i, t in enumerate(times):
        dt = datetime.fromisoformat(t)
        aqi_i = aqi_index.get(t)

        row = {
            "timestamp": t,
            "hour": dt.hour,
            "day": dt.day,
            "month": dt.month,
            "day_of_week": dt.weekday(),

            "temperature": temps[i] if i < len(temps) else None,
            "humidity": humidity[i] if i < len(humidity) else None,
            "pressure": pressure[i] if i < len(pressure) else None,
            "wind_speed": wind[i] if i < len(wind) else None,

            "aqi": aqi[aqi_i] if aqi_i is not None and aqi_i < len(aqi) else None,
            "pm25": pm25[aqi_i] if aqi_i is not None and aqi_i < len(pm25) else None,
            "pm10": pm10[aqi_i] if aqi_i is not None and aqi_i < len(pm10) else None,
            "o3": o3[aqi_i] if aqi_i is not None and aqi_i < len(o3) else None,
            "no2": no2[aqi_i] if aqi_i is not None and aqi_i < len(no2) else None,
            "so2": so2[aqi_i] if aqi_i is not None and aqi_i < len(so2) else None,
            "co": co[aqi_i] if aqi_i is not None and aqi_i < len(co) else None,
        }
        rows.append(row)

    return rows


if __name__ == "__main__":
    end_date = datetime.now()
    start_date = end_date - timedelta(days=1280)  # ~3.5 years

    chunks = generate_month_ranges(start_date, end_date)
    print(f"Will fetch {len(chunks)} monthly chunks from {chunks[0][0]} to {chunks[-1][1]}")

    all_rows = []
    for i, (chunk_start, chunk_end) in enumerate(chunks):
        print(f"Fetching chunk {i+1}/{len(chunks)}: {chunk_start} to {chunk_end} ...")
        try:
            rows = build_rows_for_chunk(chunk_start, chunk_end)
            all_rows.extend(rows)
            print(f"  -> got {len(rows)} hourly rows")
        except Exception as e:
            print(f"  -> FAILED: {e}")

        time.sleep(1)

    print(f"\nTotal rows collected: {len(all_rows)}")

    if all_rows:
        with open(OUTPUT_FILE, mode="w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_rows[0].keys())
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"✅ Saved to {OUTPUT_FILE}")
    else:
        print("⚠️ No rows collected - something went wrong.")