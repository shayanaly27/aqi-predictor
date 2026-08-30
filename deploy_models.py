"""
Creates a Hopsworks Model Deployment for each of the 3 forecast horizons,
using the latest registered version of each model.

Run once: python deploy_models.py
"""

import os
from dotenv import load_dotenv
load_dotenv()

import hopsworks

HOPSWORKS_API_KEY = os.environ.get("HOPSWORKS_API_KEY")
HOPSWORKS_PROJECT = "aqi_predictor_ksa"
HOPSWORKS_HOST = "eu-west.cloud.hopsworks.ai"

TARGETS = ["target_day1", "target_day2", "target_day3"]
PREDICTOR_SCRIPT_LOCAL = "predictor.py"


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


def upload_predictor_script(project):
    """
    Hopsworks needs the predictor script to already live in its own file
    system before a deployment can reference it. It expects the FULL
    absolute path, not a relative one.
    """
    dataset_api = project.get_dataset_api()

    upload_path = "Resources"
    dataset_api.upload(PREDICTOR_SCRIPT_LOCAL, upload_path, overwrite=True)

    full_remote_path = f"/Projects/{project.name}/{upload_path}/{PREDICTOR_SCRIPT_LOCAL}"
    print(f"  ✅ Uploaded to {full_remote_path}")

    return full_remote_path


if __name__ == "__main__":
    project = connect_to_hopsworks()
    mr = project.get_model_registry()

    predictor_remote_path = upload_predictor_script(project)

    for target_col in TARGETS:
        model_name = f"aqi_{target_col}_model"

        versions = mr.get_models(name=model_name)
        latest = max(versions, key=lambda m: m.version)
        print(f"\nDeploying {model_name} v{latest.version}...")

        deployment = latest.deploy(
            name=f"aqi{target_col.replace('_', '')}deploy",
            script_file=predictor_remote_path,
        )

        deployment.start(await_running=180)
        print(f"  ✅ Deployment '{deployment.name}' is running")

    print("\n✅ All 3 deployments created and started.")