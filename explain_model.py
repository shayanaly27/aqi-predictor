"""
SHAP Explainability
Loads the trained models and generates SHAP feature importance plots -
shows which features matter most for each forecast horizon, and why.

Saves plots as PNG files you can include directly in your report.
"""

import matplotlib
matplotlib.use("Agg")

import pandas as pd
import joblib
import os
import shap
import matplotlib.pyplot as plt

DATA_FILE = "aqi_training_data.csv"
MODEL_DIR = "models"
OUTPUT_DIR = "shap_plots"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_and_prepare_data():
    df = pd.read_csv(DATA_FILE)
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed")
    df = df.sort_values("timestamp").reset_index(drop=True)
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

    df["target_day1"] = df["aqi"].shift(-24)
    df["target_day2"] = df["aqi"].shift(-48)
    df["target_day3"] = df["aqi"].shift(-72)

    df = df.dropna().reset_index(drop=True)
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


def explain_model(target_col, label, df, feature_cols):
    model_path = os.path.join(MODEL_DIR, f"{target_col}_model.pkl")
    model = joblib.load(model_path)

    X = df[feature_cols]
    X_sample = X.sample(n=min(1000, len(X)), random_state=42)

    print(f"Computing SHAP values for {label} ({type(model).__name__})...")

    model_name = type(model).__name__
    if model_name in ("RandomForestRegressor", "XGBRegressor"):
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
    else:
        explainer = shap.LinearExplainer(model, X_sample)
        shap_values = explainer.shap_values(X_sample)

    plt.figure()
    shap.summary_plot(shap_values, X_sample, show=False)
    plt.title(f"SHAP Feature Importance - {label}")
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, f"{target_col}_shap_summary.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅ Saved {output_path}")

    plt.figure()
    shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
    plt.title(f"SHAP Feature Importance (Ranked) - {label}")
    plt.tight_layout()
    bar_path = os.path.join(OUTPUT_DIR, f"{target_col}_shap_bar.png")
    plt.savefig(bar_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅ Saved {bar_path}")


if __name__ == "__main__":
    print("Loading and preparing data...")
    df = load_and_prepare_data()
    feature_cols = get_feature_columns()
    print(f"  -> {len(df)} rows ready\n")

    targets = {
        "target_day1": "Day 1 (24h ahead)",
        "target_day2": "Day 2 (48h ahead)",
        "target_day3": "Day 3 (72h ahead)"
    }

    for target_col, label in targets.items():
        explain_model(target_col, label, df, feature_cols)
        print()

    print(f"✅ All SHAP plots saved in the '{OUTPUT_DIR}' folder - ready for your report.")