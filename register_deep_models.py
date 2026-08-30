"""
Registers the trained Keras neural network models (already saved locally in
models_deep/) into the Hopsworks Model Registry.

Run this AFTER train_deep_model.py has already produced the .keras and
.pkl scaler files in models_deep/.

Run in your .venv_dl environment:
    python register_deep_models.py
"""

import os
import shutil
from dotenv import load_dotenv
load_dotenv()

import hopsworks

MODEL_DIR = "models_deep"

HOPSWORKS_API_KEY = os.environ.get("HOPSWORKS_API_KEY")
HOPSWORKS_PROJECT = "aqi_predictor_ksa"
HOPSWORKS_HOST = "eu-west.cloud.hopsworks.ai"

# Copy these numbers from your last train_deep_model.py console output
# (the "SUMMARY" section at the end). Update if you retrain and get
# different numbers.
NN_METRICS = {
    "target_day1": {"rmse": 16.81, "mae": 10.31, "r2": 0.398},
    "target_day2": {"rmse": 18.92, "mae": 13.04, "r2": 0.238},
    "target_day3": {"rmse": 19.76, "mae": 14.06, "r2": 0.170},
}


def connect_to_hopsworks():
    cert_dir = os.path.join(os.getcwd(), "hopsworks_certs")
    os.makedirs(cert_dir, exist_ok=True)
    project = hopsworks.login(
        project=HOPSWORKS_PROJECT,
        host=HOPSWORKS_HOST,
        port=443,
        api_key_value=HOPSWORKS_API_KEY,
        cert_folder=cert_dir,
    )
    print(f"✅ Connected to Hopsworks project: {project.name}")
    return project


if __name__ == "__main__":
    project = connect_to_hopsworks()
    mr = project.get_model_registry()

    for target_col, metrics in NN_METRICS.items():
        model_file = os.path.join(MODEL_DIR, f"{target_col}_nn_model.keras")
        scaler_file = os.path.join(MODEL_DIR, f"{target_col}_nn_scaler.pkl")

        if not os.path.exists(model_file) or not os.path.exists(scaler_file):
            print(f"⚠️ Skipping {target_col} - files not found. Run train_deep_model.py first.")
            continue

        # Bundle the model + its scaler together in one folder so they
        # travel together as one registry entry (the scaler is required
        # to use the model correctly at prediction time).
        bundle_dir = os.path.join(MODEL_DIR, f"{target_col}_nn_bundle")
        os.makedirs(bundle_dir, exist_ok=True)
        shutil.copy(model_file, bundle_dir)
        shutil.copy(scaler_file, bundle_dir)

        print(f"\nRegistering {target_col} neural network model...")
        model_meta = mr.python.create_model(
            name=f"aqi_{target_col}_nn_model",
            metrics=metrics,
            description=f"Neural network (Keras) for {target_col} AQI forecast",
        )
        model_meta.save(bundle_dir)
        print(f"  ✅ Registered aqi_{target_col}_nn_model")

    print("\n✅ All neural network models registered in the Hopsworks Model Registry.")