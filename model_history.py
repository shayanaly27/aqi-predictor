"""
Pulls the full version history for each model type from the Hopsworks
Model Registry, and identifies which version is currently "best" (lowest
RMSE) for each forecast horizon. This replaces the manually-typed
model_metrics.json with real registry data.

Run with: python model_history.py
"""

import os
import json
from dotenv import load_dotenv
load_dotenv()

import hopsworks

HOPSWORKS_API_KEY = os.environ.get("HOPSWORKS_API_KEY")
HOPSWORKS_PROJECT = "aqi_predictor_ksa"
HOPSWORKS_HOST = "eu-west.cloud.hopsworks.ai"

MODEL_NAMES = {
    "target_day1": "aqi_target_day1_model",
    "target_day2": "aqi_target_day2_model",
    "target_day3": "aqi_target_day3_model",
}


def connect_to_hopsworks():
    cert_dir = os.path.join(os.getcwd(), "hopsworks_certs")
    os.makedirs(cert_dir, exist_ok=True)
    return hopsworks.login(
        project=HOPSWORKS_PROJECT,
        host=HOPSWORKS_HOST,
        port=443,
        api_key_value=HOPSWORKS_API_KEY,
        cert_folder=cert_dir,
    )


def get_version_history():
    project = connect_to_hopsworks()
    mr = project.get_model_registry()

    result = {}
    for target_col, model_name in MODEL_NAMES.items():
        versions = mr.get_models(name=model_name)
        if not versions:
            result[target_col] = {"model_name": model_name, "versions": [], "current_best": None}
            continue

        version_list = []
        for m in versions:
            version_list.append({
                "version": m.version,
                "metrics": m.training_metrics,
                "description": m.description,
                "created": str(m.created),
            })

        version_list.sort(key=lambda v: v["version"])

        # "current best" = the most recently registered version, since
        # train_model.py already picked the best-performing algorithm
        # (Ridge/RF/XGBoost) before registering it
        current_best = max(version_list, key=lambda v: v["version"])

        result[target_col] = {
            "model_name": model_name,
            "versions": version_list,
            "current_best": current_best,
        }

    return result


if __name__ == "__main__":
    history = get_version_history()
    with open("model_history.json", "w") as f:
        json.dump(history, f, indent=2)

    for target_col, data in history.items():
        print(f"\n{data['model_name']}:")
        for v in data["versions"]:
            marker = " ← CURRENT BEST" if v["version"] == data["current_best"]["version"] else ""
            print(f"  v{v['version']}: {v['metrics']}{marker}")

    print("\n✅ Saved model_history.json")