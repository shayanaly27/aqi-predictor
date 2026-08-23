"""
Exploratory Data Analysis (EDA)
Investigates the merged training dataset (aqi_training_data.csv) before
modeling. Produces plots + a text summary covering:

1. Missing data summary
2. AQI distribution + outlier check
3. AQI trend over the full 3.5-year history
4. Seasonality: AQI by hour of day, by month
5. Correlation heatmap: AQI vs weather/pollutant features
6. Timestamp gap check (catches missing/irregular hourly readings -
   this directly affects lag/rolling features like aqi_lag_24h)
7. Archive vs live-source distribution check (a rough train/serving
   skew check between backfilled historical data and hourly live data)

Run with: python eda.py
Outputs saved to eda_plots/ and a summary printed to console.
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

DATA_FILE = "aqi_training_data.csv"
OUTPUT_DIR = "eda_plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

sns.set_theme(style="darkgrid")


def load_data():
    df = pd.read_csv(DATA_FILE)
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed")
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def missing_data_summary(df):
    print("\n=== 1. Missing Data Summary ===")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    summary = pd.DataFrame({"missing_count": missing, "missing_pct": missing_pct})
    summary = summary[summary["missing_count"] > 0].sort_values("missing_count", ascending=False)
    if summary.empty:
        print("  -> No missing values found.")
    else:
        print(summary)
    return summary


def aqi_distribution(df):
    print("\n=== 2. AQI Distribution & Outlier Check ===")
    print(df["aqi"].describe())

    impossible = df[(df["aqi"] < 0) | (df["aqi"] > 500)]
    print(f"\n  -> Rows with AQI outside valid 0-500 range: {len(impossible)}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.histplot(df["aqi"].dropna(), bins=50, ax=axes[0], color="#4a90d9")
    axes[0].set_title("AQI Distribution")
    axes[0].set_xlabel("US AQI")

    sns.boxplot(x=df["aqi"].dropna(), ax=axes[1], color="#4a90d9")
    axes[1].set_title("AQI Boxplot (outlier check)")
    axes[1].set_xlabel("US AQI")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "01_aqi_distribution.png"), dpi=150)
    plt.close()
    print(f"  ✅ Saved 01_aqi_distribution.png")


def aqi_trend(df):
    print("\n=== 3. AQI Trend Over Time ===")
    daily = df.set_index("timestamp")["aqi"].resample("D").mean()

    plt.figure(figsize=(16, 5))
    plt.plot(daily.index, daily.values, color="#4a90d9", linewidth=0.8)
    plt.title("Daily Average AQI - Karachi (Full History)")
    plt.xlabel("Date")
    plt.ylabel("AQI (daily mean)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "02_aqi_trend_full_history.png"), dpi=150)
    plt.close()
    print(f"  ✅ Saved 02_aqi_trend_full_history.png")

    recent = daily.last("90D")
    plt.figure(figsize=(16, 5))
    plt.plot(recent.index, recent.values, color="#e08a3c", linewidth=1.2)
    plt.title("Daily Average AQI - Last 90 Days")
    plt.xlabel("Date")
    plt.ylabel("AQI (daily mean)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "03_aqi_trend_last_90_days.png"), dpi=150)
    plt.close()
    print(f"  ✅ Saved 03_aqi_trend_last_90_days.png")


def seasonality(df):
    print("\n=== 4. Seasonality: Hour-of-Day and Month ===")

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    hourly_avg = df.groupby("hour")["aqi"].mean()
    axes[0].bar(hourly_avg.index, hourly_avg.values, color="#4a90d9")
    axes[0].set_title("Average AQI by Hour of Day")
    axes[0].set_xlabel("Hour (0-23)")
    axes[0].set_ylabel("Average AQI")

    monthly_avg = df.groupby("month")["aqi"].mean()
    axes[1].bar(monthly_avg.index, monthly_avg.values, color="#e08a3c")
    axes[1].set_title("Average AQI by Month")
    axes[1].set_xlabel("Month (1-12)")
    axes[1].set_ylabel("Average AQI")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "04_seasonality_hour_month.png"), dpi=150)
    plt.close()
    print(f"  ✅ Saved 04_seasonality_hour_month.png")

    print("\n  Hour-of-day AQI averages:")
    print(hourly_avg.round(1))
    print("\n  Month AQI averages:")
    print(monthly_avg.round(1))


def correlation_heatmap(df):
    print("\n=== 5. Correlation Heatmap ===")
    cols = ["aqi", "pm25", "pm10", "o3", "no2", "so2", "co",
            "temperature", "humidity", "pressure", "wind_speed"]
    corr_df = df[cols].dropna()
    corr = corr_df.corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                square=True, linewidths=0.5)
    plt.title("Correlation Heatmap - AQI vs Weather & Pollutant Features")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "05_correlation_heatmap.png"), dpi=150)
    plt.close()
    print(f"  ✅ Saved 05_correlation_heatmap.png")

    print("\n  Features most correlated with AQI:")
    print(corr["aqi"].drop("aqi").sort_values(key=abs, ascending=False).round(3))


def timestamp_gap_check(df):
    """
    Checks if consecutive rows are exactly ~1 hour apart. Gaps here mean
    lag features like aqi_lag_24h don't actually represent 'aqi 24 hours
    ago' - they represent 'aqi 24 rows ago', which can silently be wrong
    if the hourly pipeline missed a run.
    """
    print("\n=== 6. Timestamp Gap Check ===")
    gaps = df["timestamp"].diff().dropna()
    gap_hours = gaps.dt.total_seconds() / 3600

    expected = gap_hours.round(1) == 1.0
    n_irregular = (~expected).sum()
    pct_irregular = round(n_irregular / len(gap_hours) * 100, 2)

    print(f"  Total gaps checked: {len(gap_hours)}")
    print(f"  Irregular gaps (not ~1 hour): {n_irregular} ({pct_irregular}%)")
    print(f"  Largest gap found: {gap_hours.max():.1f} hours")
    print(f"  Smallest gap found: {gap_hours.min():.2f} hours")

    if n_irregular > 0:
        print(f"\n  ⚠️  {n_irregular} irregular gaps found. These can distort lag/rolling")
        print("     features (aqi_lag_24h, aqi_roll_mean_24h, etc). Consider resampling")
        print("     to a strict hourly index with interpolation, or flagging affected rows.")
    else:
        print("\n  ✅ All gaps are ~1 hour. Lag/rolling features are on solid ground.")

    plt.figure(figsize=(12, 4))
    plt.plot(df["timestamp"].iloc[1:], gap_hours, color="#e2584f", linewidth=0.5)
    plt.axhline(1.0, color="#5fd38d", linestyle="--", linewidth=1, label="expected (1h)")
    plt.title("Gap Between Consecutive Readings Over Time")
    plt.xlabel("Date")
    plt.ylabel("Gap (hours)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "06_timestamp_gaps.png"), dpi=150)
    plt.close()
    print(f"  ✅ Saved 06_timestamp_gaps.png")


def archive_vs_live_check(df, cutoff_days=90):
    """
    Rough train/serving skew check. The backfill came from Open-Meteo's
    ARCHIVE endpoint (verified historical data); the hourly pipeline
    pulls from the FORECAST endpoint's 'current' field (a live estimate).
    If these two sources report meaningfully different distributions for
    the same feature, the model is training on slightly different data
    than what it sees at prediction time.

    We approximate this by comparing the oldest 90% of the data (mostly
    archive-sourced) against the most recent slice (entirely from the
    live hourly pipeline).
    """
    print("\n=== 7. Archive vs Live-Source Distribution Check ===")
    cutoff = df["timestamp"].max() - pd.Timedelta(days=cutoff_days)
    older = df[df["timestamp"] < cutoff]
    recent = df[df["timestamp"] >= cutoff]

    print(f"  Older/archive-leaning slice: {len(older)} rows (before {cutoff.date()})")
    print(f"  Recent/live-pipeline slice: {len(recent)} rows (from {cutoff.date()} onward)")

    if len(recent) < 30:
        print("  -> Not enough recent rows yet for a meaningful comparison. Skipping.")
        return

    cols = ["aqi", "pm25", "temperature", "humidity", "pressure", "wind_speed"]
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    axes = axes.flatten()

    for i, col in enumerate(cols):
        sns.kdeplot(older[col].dropna(), ax=axes[i], label="older (archive)", color="#4a90d9")
        sns.kdeplot(recent[col].dropna(), ax=axes[i], label="recent (live pipeline)", color="#e08a3c")
        axes[i].set_title(col)
        axes[i].legend(fontsize=8)

    plt.suptitle("Archive-sourced vs Live-pipeline Feature Distributions")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "07_archive_vs_live_distributions.png"), dpi=150)
    plt.close()
    print(f"  ✅ Saved 07_archive_vs_live_distributions.png")
    print("  -> Visually inspect this plot: if a feature's two curves are clearly")
    print("     shifted apart, that feature may behave differently at prediction")
    print("     time than what the model was trained on.")


if __name__ == "__main__":
    print("Loading data...")
    df = load_data()
    print(f"  -> {len(df)} rows, {df['timestamp'].min()} to {df['timestamp'].max()}")

    missing_data_summary(df)
    aqi_distribution(df)
    aqi_trend(df)
    seasonality(df)
    correlation_heatmap(df)
    timestamp_gap_check(df)
    archive_vs_live_check(df)

    print(f"\n✅ EDA complete. All plots saved in '{OUTPUT_DIR}/' - ready for your report.")