"""
Quick sanity check: takes ONE point from ~3 days ago, predicts what the
AQI would be 72h later using the Day 3 model, and compares that
prediction against the AQI value ACTUALLY recorded at that real future
timestamp - found by nearest timestamp match, not by row count, since
the live pipeline has occasional gaps.
"""

import os
import pandas as pd
import joblib
from dotenv import load_dotenv
load_dotenv()

import hopsworks

HOPSWORKS_API_KEY = os.environ.get("HOPSWORKS_API_KEY")
HOPSWORKS_PROJECT = "aqi_predictor_ksa"
HOPSWORKS_HOST = "eu-west.cloud.hopsworks.ai"

cert_dir = os.path.join(os.getcwd(), "hopsworks_certs")
os.makedirs(cert_dir, exist_ok=True)
project = hopsworks.login(
    project=HOPSWORKS_PROJECT, host=HOPSWORKS_HOST, port=443,
    api_key_value=HOPSWORKS_API_KEY, cert_folder=cert_dir,
)
fs = project.get_feature_store()
fv = fs.get_feature_view(name="aqi_feature_view", version=1)
df = fv.get_batch_data()

df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed", utc=True)
df = df.drop_duplicates(subset="timestamp").sort_values("timestamp").reset_index(drop=True)
df = df.dropna(subset=["pm10", "o3", "no2", "so2", "co"]).reset_index(drop=True)

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
df = df.dropna().reset_index(drop=True)

feature_cols = [
    "hour", "day", "month", "day_of_week",
    "temperature", "humidity", "pressure", "wind_speed",
    "aqi", "pm25", "pm10", "o3", "no2", "so2", "co",
    "aqi_lag_1h", "aqi_lag_3h", "aqi_lag_24h",
    "aqi_roll_mean_6h", "aqi_roll_mean_24h", "aqi_roll_mean_48h", "aqi_roll_mean_72h",
    "aqi_change_rate_1h", "pressure_lag_24h", "pressure_change_24h",
]

# Pick a base point ~72h before the LATEST available timestamp, by actual
# elapsed time, not by row count.
latest_time = df["timestamp"].max()
target_base_time = latest_time - pd.Timedelta(hours=72)

# find the row whose timestamp is closest to that target
df["time_diff"] = (df["timestamp"] - target_base_time).abs()
base_row = df.loc[[df["time_diff"].idxmin()]]
base_time = base_row["timestamp"].iloc[0]  # .iloc keeps tz-aware Timestamp

model = joblib.load("models/target_day3_model.pkl")
predicted_aqi = float(model.predict(base_row[feature_cols])[0])

# find the row whose timestamp is closest to base_time + 72h (should be ~latest_time)
target_future_time = base_time + pd.Timedelta(hours=72)
df["future_diff"] = (df["timestamp"] - target_future_time).abs()
actual_row = df.loc[df["future_diff"].idxmin()]

actual_aqi = float(actual_row["aqi"])
actual_time = actual_row["timestamp"]
gap_hours = (actual_time - base_time).total_seconds() / 3600

print(f"Base point:      {base_time}")
print(f"Predicted 72h ahead AQI: {predicted_aqi:.1f}")
print(f"Actual point:    {actual_time}  (actual gap: {gap_hours:.1f}h - should be close to 72)")
print(f"Actual AQI was:  {actual_aqi:.1f}")
print(f"Difference:      {abs(predicted_aqi - actual_aqi):.1f} AQI points")