"""
Merge Script: Combines the historical backfill with the live hourly
feature data into one clean, deduplicated, strictly-hourly dataset
ready for model training.

EDA (eda.py, section 6) found 238 irregular timestamp gaps (0.77% of
rows) - including near-duplicate rows caused by the live pipeline's
timestamp coming from datetime.now() instead of the API's own reading
time, and a couple of longer gaps (up to 8.7h) from missed pipeline runs.

This version fixes that by:
1. Rounding every timestamp to the nearest hour (collapses near-duplicates)
2. Reindexing onto a strict hourly grid across the full date range
3. Interpolating ONLY short gaps (<= max_gap_hours) - longer gaps are left
   as NaN and get dropped downstream by the existing dropna() in
   train_model.py, exactly as before. We never fabricate data across a
   gap we don't actually have information for.
"""

import pandas as pd
import numpy as np

HISTORICAL_FILE = "aqi_historical_backfill_v2.csv"
LIVE_FILE = "aqi_features.csv"
OUTPUT_FILE = "aqi_training_data.csv"

MAX_GAP_HOURS = 3   # only interpolate across gaps this short or shorter

NUMERIC_COLS = [
    "temperature", "humidity", "pressure", "wind_speed",
    "aqi", "pm25", "pm10", "o3", "no2", "so2", "co",
]


def load_and_clean(filepath):
    df = pd.read_csv(filepath)
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed")
    return df


def snap_to_hourly_grid(df):
    """Round timestamps to the nearest hour and collapse duplicates."""
    df = df.copy()
    df["timestamp"] = df["timestamp"].dt.round("h")

    before = len(df)
    # if two rows land on the same hour after rounding, keep the last
    # (most recently fetched) one
    df = df.drop_duplicates(subset="timestamp", keep="last")
    after = len(df)
    print(f"  -> Collapsed {before - after} near-duplicate rows after rounding to hourly grid")

    return df


def reindex_and_interpolate(df, max_gap_hours=MAX_GAP_HOURS):
    """
    Reindex onto a strict hourly index so lag/rolling features (aqi_lag_24h,
    aqi_roll_mean_24h, etc.) are computed over actual elapsed hours, not just
    'N rows back'. Only fills gaps up to max_gap_hours; longer gaps are left
    as NaN and dropped later, same as today.
    """
    df = df.set_index("timestamp").sort_index()

    full_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq="h")
    original_len = len(df)
    df = df.reindex(full_range)
    df.index.name = "timestamp"

    n_missing_hours = len(df) - original_len
    print(f"  -> Hourly grid has {len(df)} slots ({n_missing_hours} were missing before interpolation)")

    df[NUMERIC_COLS] = df[NUMERIC_COLS].interpolate(
        method="linear", limit=max_gap_hours, limit_area="inside"
    )

    # recompute time-based features directly from the (now complete) index,
    # since these are derived from time itself, not raw sensor data
    df["hour"] = df.index.hour
    df["day"] = df.index.day
    df["month"] = df.index.month
    df["day_of_week"] = df.index.dayofweek

    df = df.reset_index()
    return df


if __name__ == "__main__":
    print("Loading historical data...")
    historical = load_and_clean(HISTORICAL_FILE)
    print(f"  -> {len(historical)} rows")

    print("Loading live pipeline data...")
    live = load_and_clean(LIVE_FILE)
    print(f"  -> {len(live)} rows")

    print("\nMerging...")
    combined = pd.concat([historical, live], ignore_index=True)

    before = len(combined)
    combined = combined.drop_duplicates(subset="timestamp", keep="last")
    after = len(combined)
    print(f"  -> Removed {before - after} exact duplicate timestamp rows")

    combined = combined.sort_values("timestamp").reset_index(drop=True)

    print("\nSnapping timestamps to hourly grid...")
    combined = snap_to_hourly_grid(combined)

    print("\nReindexing to strict hourly index and interpolating short gaps...")
    combined = reindex_and_interpolate(combined)

    print("\nMissing values per column after interpolation:")
    print(combined[NUMERIC_COLS].isnull().sum())

    print(f"\nDate range: {combined['timestamp'].min()} to {combined['timestamp'].max()}")
    print(f"Total rows in final dataset: {len(combined)}")

    combined.to_csv(OUTPUT_FILE, index=False)
    print(f"\n✅ Saved merged, hourly-regularized training dataset to {OUTPUT_FILE}")