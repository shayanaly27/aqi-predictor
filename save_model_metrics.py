"""
Saves your model comparison results (RMSE/MAE/R2 per model per horizon)
as a small JSON file the dashboard's /model-metrics endpoint serves.

This is a ONE-TIME manual entry of the numbers you already have from
your train_model.py and train_deep_model.py runs - no need to re-train
just to populate this. Update the numbers below to match your latest
actual run output, then run:
    python save_model_metrics.py
"""

import json

# Fill these in from your train_model.py and train_deep_model.py output.
# The values below are from your most recent run shown in this chat -
# UPDATE if you retrain and get different numbers.
METRICS = {
    "day_1": {
        "label": "Day 1 (24h ahead)",
        "models": [
            {"name": "Ridge Regression", "rmse": 15.95, "mae": 10.26, "r2": 0.457},
            {"name": "Random Forest", "rmse": 15.25, "mae": 10.05, "r2": 0.503},
            {"name": "XGBoost", "rmse": 14.20, "mae": 9.40, "r2": 0.569},
            {"name": "Neural Network", "rmse": 16.81, "mae": 10.31, "r2": 0.398},
        ],
        "best_model": "XGBoost",
    },
    "day_2": {
        "label": "Day 2 (48h ahead)",
        "models": [
            {"name": "Ridge Regression", "rmse": 18.03, "mae": 12.85, "r2": 0.303},
            {"name": "Random Forest", "rmse": 17.85, "mae": 12.63, "r2": 0.317},
            {"name": "XGBoost", "rmse": 17.54, "mae": 12.39, "r2": 0.340},
            {"name": "Neural Network", "rmse": 18.92, "mae": 13.04, "r2": 0.238},
        ],
        "best_model": "XGBoost",
    },
    "day_3": {
        "label": "Day 3 (72h ahead)",
        "models": [
            {"name": "Ridge Regression", "rmse": 18.41, "mae": 13.43, "r2": 0.276},
            {"name": "Random Forest", "rmse": 18.52, "mae": 13.24, "r2": 0.268},
            {"name": "XGBoost", "rmse": 18.35, "mae": 13.09, "r2": 0.281},
            {"name": "Neural Network", "rmse": 19.76, "mae": 14.06, "r2": 0.170},
        ],
        "best_model": "XGBoost",
    },
}

if __name__ == "__main__":
    with open("model_metrics.json", "w") as f:
        json.dump(METRICS, f, indent=2)
    print("✅ Saved model_metrics.json")