"""
Backtest: picks several past points in Feature Store history (by real
elapsed time, not row count - the live pipeline has occasional gaps),
simulates what the model would have predicted from each point (24h/48h/
72h ahead), and compares against the AQI value actually recorded near
that future timestamp. Reports the real gap in hours for every
comparison so you can see how close each match actually is.

Run with: python backtest.py
"""

import os
import pandas as pd
import numpy as np
import joblib
from dotenv import load_dotenv
load_dotenv()

import hopsworks

HOPSWORKS_API_KEY = os.environ.get("HOPSWORKS_API_KEY")
HOPSWORKS_PROJECT = "aqi_predictor_ksa"
HOPSWORKS_HOST = "eu-west.cloud.hopsworks.ai"
FEATURE_VIEW_NAME = "aqi_feature_view"
FEATURE_VIEW_VERSION = 1

MODEL_DIR = "models"
N_TEST_POINTS = 15
MAX_GAP_TOLERANCE_HOURS = 6  # discard a match if the real gap is off by more than this


def connect_to_hopsworks():
    cert_dir = os.path.join(os.getcwd(), "hopsworks_certs")
    os.makedirs(cert_dir, exist_ok=True)
    return hopsworks.login(
        project=HOPSWORKS_PROJECT, host=HOPSWORKS_HOST, port=443,
        api_key_value=HOPSWORKS_API_KEY, cert_folder=cert_dir,
    )


def load_full_history():
    project = connect_to_hopsworks()
    fs = project.get_feature_store()
    fv = fs.get_feature_view(name=FEATURE_VIEW_NAME, version=FEATURE_VIEW_VERSION)
    df = fv.get_batch_data()

    df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed", utc=True)
    df = df.drop_duplicates(subset="timestamp").sort_values("timestamp").reset_index(drop=True)
    df = df.dropna(subset=["pm10", "o3", "no2", "so2", "co"]).reset_index(drop=True)
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


def find_nearest_row(df, target_time):
    """Returns (row, actual_gap_hours) for the row whose timestamp is
    closest to target_time."""
    diffs = (df["timestamp"] - target_time).abs()
    idx = diffs.idxmin()
    row = df.loc[idx]
    gap_hours = (row["timestamp"] - target_time).total_seconds() / 3600
    return row, idx, gap_hours


if __name__ == "__main__":
    print("Loading full history from Feature Store...")
    df = load_full_history()
    df = engineer_features(df)
    df = df.dropna().reset_index(drop=True)
    print(f"  -> {len(df)} usable rows")

    feature_cols = get_feature_columns()

    models = {
        1: joblib.load(os.path.join(MODEL_DIR, "target_day1_model.pkl")),
        2: joblib.load(os.path.join(MODEL_DIR, "target_day2_model.pkl")),
        3: joblib.load(os.path.join(MODEL_DIR, "target_day3_model.pkl")),
    }
    horizon_hours = {1: 24, 2: 48, 3: 72}

    latest_time = df["timestamp"].max()
    earliest_usable_base = latest_time - pd.Timedelta(hours=72)  # leave room for 72h-ahead actuals
    lookback_start = earliest_usable_base - pd.Timedelta(days=30)

    # evenly spaced target base times across the last 30 usable days
    base_targets = pd.date_range(start=lookback_start, end=earliest_usable_base, periods=N_TEST_POINTS)

    results = []
    skipped = 0

    for target_base_time in base_targets:
        base_row, base_idx, base_gap = find_nearest_row(df, target_base_time)
        if abs(base_gap) > MAX_GAP_TOLERANCE_HOURS:
            skipped += 1
            continue

        base_time = base_row["timestamp"]
        X = df.loc[[base_idx], feature_cols]

        for day_num, model in models.items():
            hours_ahead = horizon_hours[day_num]
            target_future_time = base_time + pd.Timedelta(hours=hours_ahead)

            future_row, _, future_gap = find_nearest_row(df, target_future_time)
            if abs(future_gap) > MAX_GAP_TOLERANCE_HOURS:
                skipped += 1
                continue

            predicted_aqi = float(model.predict(X)[0])
            actual_aqi = float(future_row["aqi"])
            error = abs(predicted_aqi - actual_aqi)

            results.append({
                "base_time": base_time,
                "horizon": f"Day {day_num} ({hours_ahead}h)",
                "predicted": round(predicted_aqi, 1),
                "actual": round(actual_aqi, 1),
                "abs_error": round(error, 1),
                "actual_gap_hours": round(hours_ahead + future_gap - base_gap, 1),
            })

    results_df = pd.DataFrame(results)

    if skipped:
        print(f"\n⚠️ Skipped {skipped} comparisons where the nearest real timestamp was more than "
              f"{MAX_GAP_TOLERANCE_HOURS}h away from the target - these would have been misleading.")

    print("\n=== Backtest Results ===")
    print(results_df.to_string(index=False))

    print("\n=== Summary by horizon ===")
    summary = results_df.groupby("horizon")["abs_error"].agg(["mean", "median", "max", "count"]).round(1)
    print(summary)

    within_10 = (results_df["abs_error"] <= 10).mean() * 100
    within_20 = (results_df["abs_error"] <= 20).mean() * 100
    print(f"\nPredictions within 10 AQI points of actual: {within_10:.0f}%")
    print(f"Predictions within 20 AQI points of actual: {within_20:.0f}%")

    results_df.to_csv("backtest_results.csv", index=False)
    print("\n✅ Saved backtest_results.csv")