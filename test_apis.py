import os
from dotenv import load_dotenv
load_dotenv()

import requests

OPENWEATHER_API_KEY = os.environ.get("OPENWEATHER_API_KEY")
WAQI_TOKEN = os.environ.get("WAQI_TOKEN")

CITY = "Karachi"
LAT, LON = 24.8607, 67.0011

def get_weather():
    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {
        "lat": LAT,
        "lon": LON,
        "appid": OPENWEATHER_API_KEY,
        "units": "metric"
    }
    response = requests.get(url, params=params)
    return response.json()

def get_aqi():
    url = "https://api.waqi.info/feed/karachi/"
    params = {"token": WAQI_TOKEN}
    response = requests.get(url, params=params)
    return response.json()

if __name__ == "__main__":
    print("=== Fetching Weather Data ===")
    weather = get_weather()
    print(weather)

    print("\n=== Fetching AQI Data ===")
    aqi = get_aqi()
    print(aqi)