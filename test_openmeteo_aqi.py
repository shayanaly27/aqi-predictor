"""
Test script: Pull historical AQI + pollutant data from Open-Meteo's Air Quality API.
This is a DIFFERENT endpoint than the weather one - dedicated to air quality.
No API key needed.
"""

import requests

LAT, LON = 24.8607, 67.0011

def get_historical_air_quality(start_date, end_date):
    url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    params = {
        "latitude": LAT,
        "longitude": LON,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": "us_aqi,pm2_5,pm10,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone",
        "timezone": "Asia/Karachi"
    }
    response = requests.get(url, params=params)
    return response.json()

if __name__ == "__main__":
    data = get_historical_air_quality("2024-08-02", "2024-08-05")

    print("Status check - keys returned:", list(data.keys()))

    hourly = data.get("hourly", {})
    times = hourly.get("time", [])
    aqi = hourly.get("us_aqi", [])
    pm25 = hourly.get("pm2_5", [])

    print("\nSample of hourly AQI data:")
    for i in range(min(10, len(times))):
        print(f"  {times[i]} -> AQI: {aqi[i]}, PM2.5: {pm25[i]}")

    print(f"\nTotal hourly records returned: {len(times)}")

    unique_aqi_values = set(aqi[:24]) if aqi else set()
    print(f"\nUnique AQI values in first 24 hours: {unique_aqi_values}")
    print("(If this shows multiple different numbers, the data is real and varying - good sign!)")