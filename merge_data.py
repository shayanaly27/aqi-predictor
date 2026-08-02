"""
Merge Script: Combines the historical backfill with the live hourly
feature data into one clean, deduplicated dataset ready for model training.
"""

import pandas as pd

HISTORICAL_FILE = "aqi_historical_backfill_v2.csv"
LIVE_FILE = "aqi_features.csv"
OUTPUT_FILE = "aqi_training_data.csv"


def load_and_clean(filepath):
    df = pd.read_csv(filepath)
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed")
    return df


if __name__ == "__main__":
    print("Loading historical data...")
    historical = load_and_clean(HISTORICAL_FILE)
    print(f"  -> {len(historical)} rows")

    print("Loading live pipeline data...")
    live = load_and_clean(LIVE_FILE)
    print(f"  -> {len(live)} rows")

    print("Merging...")
    combined = pd.concat([historical, live], ignore_index=True)

    before = len(combined)
    combined = combined.drop_duplicates(subset="timestamp", keep="last")
    after = len(combined)
    print(f"  -> Removed {before - after} duplicate timestamp rows")

    combined = combined.sort_values("timestamp").reset_index(drop=True)

    print("\nMissing values per column:")
    print(combined.isnull().sum())

    print(f"\nDate range: {combined['timestamp'].min()} to {combined['timestamp'].max()}")
    print(f"Total rows in final dataset: {len(combined)}")

    combined.to_csv(OUTPUT_FILE, index=False)
    print(f"\n✅ Saved merged training dataset to {OUTPUT_FILE}")