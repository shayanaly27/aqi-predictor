"""
Deep Learning Training Script
Adds a TensorFlow/Keras neural network to the model comparison, satisfying
the project brief's "statistical to deep learning models" requirement.

Run in its OWN virtual environment (.venv_dl), separate from the one used
for push_to_hopsworks.py / train_model.py. TensorFlow requires a newer
protobuf than the hopsworks package supports.

This script reads aqi_training_data.csv directly - no hopsworks import,
no Hopsworks connection needed. That's the whole point of the separate
venv: avoid the protobuf conflict entirely.

Run with: python train_deep_model.py
"""

import os
import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

DATA_FILE = "aqi_training_data.csv"
MODEL_DIR = "models_deep"

os.makedirs(MODEL_DIR, exist_ok=True)

np.random.seed(42)
tf.random.set_seed(42)


def load_data():
    print(f"Loading {DATA_FILE}...")
    df = pd.read_csv(DATA_FILE)
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed")
    df = df.drop_duplicates(subset="timestamp").sort_values("timestamp").reset_index(drop=True)
    print(f"✅ Loaded {len(df)} rows from {DATA_FILE}")
    return df


def clean_data(df):
    before = len(df)
    df = df.dropna(subset=["pm10", "o3", "no2", "so2", "co"]).reset_index(drop=True)
    print(f"Dropped {before - len(df)} incomplete rows (old WAQI-only data)")
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


def build_model(n_features):
    model = keras.Sequential([
        layers.Input(shape=(n_features,)),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(32, activation="relu"),
        layers.Dropout(0.1),
        layers.Dense(16, activation="relu"),
        layers.Dense(1),
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="mse",
        metrics=["mae"],
    )
    return model


def train_and_evaluate_nn(df, target_col, feature_cols, label):
    X = df[feature_cols].values
    y = df[target_col].values

    split_idx = int(len(df) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = build_model(n_features=X_train_scaled.shape[1])

    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=8, restore_best_weights=True
    )

    print(f"  Training neural network for {label}...")
    history = model.fit(
        X_train_scaled, y_train,
        validation_split=0.15,
        epochs=100,
        batch_size=64,
        callbacks=[early_stop],
        verbose=0,
    )

    epochs_ran = len(history.history["loss"])
    print(f"  -> Stopped after {epochs_ran} epochs (early stopping)")

    preds = model.predict(X_test_scaled, verbose=0).flatten()

    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    print(f"  Neural Network: RMSE={rmse:.2f}, MAE={mae:.2f}, R2={r2:.3f}")

    return model, scaler, {"rmse": rmse, "mae": mae, "r2": r2}


if __name__ == "__main__":
    print("Loading data...")
    df = load_data()

    print("\nCleaning data...")
    df = clean_data(df)

    print("\nEngineering features and targets...")
    df = engineer_features(df)
    print(f"  -> {len(df)} rows remain after feature engineering")

    feature_cols = get_feature_columns()

    targets = {
        "target_day1": "Day 1 (24h ahead)",
        "target_day2": "Day 2 (48h ahead)",
        "target_day3": "Day 3 (72h ahead)"
    }

    all_metrics = {}

    for target_col, label in targets.items():
        print(f"\n=== Training neural network for {label} ===")
        model, scaler, metrics = train_and_evaluate_nn(df, target_col, feature_cols, label)
        all_metrics[target_col] = metrics

        model_path = os.path.join(MODEL_DIR, f"{target_col}_nn_model.keras")
        scaler_path = os.path.join(MODEL_DIR, f"{target_col}_nn_scaler.pkl")

        model.save(model_path)
        joblib.dump(scaler, scaler_path)
        print(f"  ✅ Saved model to {model_path}")
        print(f"  ✅ Saved scaler to {scaler_path}")

    print("\n" + "=" * 60)
    print("SUMMARY - Neural Network vs. your existing models")
    print("=" * 60)
    for target_col, label in targets.items():
        m = all_metrics[target_col]
        print(f"  {label}: RMSE={m['rmse']:.2f}, MAE={m['mae']:.2f}, R2={m['r2']:.3f}")

    print("\n✅ Deep learning training complete. Models saved in 'models_deep/'.")