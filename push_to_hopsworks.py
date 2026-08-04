"""
Hopsworks Integration - Feature Store Upload
Connects to your Hopsworks project and pushes the merged training dataset
into a Feature Group, satisfying the "Feature Store" requirement.

Run this once to do the initial upload of your historical + live data.
"""

import os
from dotenv import load_dotenv
load_dotenv()

import pandas as pd
import hopsworks

HOPSWORKS_API_KEY = os.environ.get("HOPSWORKS_API_KEY")
HOPSWORKS_PROJECT = "aqi_predictor_ksa"
HOPSWORKS_HOST = "eu-west.cloud.hopsworks.ai"

DATA_FILE = "aqi_training_data.csv"


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


def load_data():
    df = pd.read_csv(DATA_FILE)
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed")
    df = df.drop_duplicates(subset="timestamp").reset_index(drop=True)
    return df


if __name__ == "__main__":
    print("Loading training data...")
    df = load_data()
    print(f"  -> {len(df)} rows ready to upload")

    print("\nConnecting to Hopsworks...")
    project = connect_to_hopsworks()
    fs = project.get_feature_store()

    print("\nCreating/getting feature group 'aqi_features'...")
    aqi_fg = fs.get_or_create_feature_group(
        name="aqi_features",
        version=1,
        description="Karachi AQI features - weather, pollutants, time-based and derived features",
        primary_key=["timestamp"],
        event_time="timestamp",
        time_travel_format="HUDI",
        online_enabled=False,
    )

    print("Inserting data into feature group (this may take a few minutes)...")
    aqi_fg.insert(df)

    print("\n✅ Data successfully pushed to Hopsworks Feature Store.")
    print(f"   Project: {HOPSWORKS_PROJECT}")
    print(f"   Feature Group: aqi_features (v1)")
    print(f"   Rows: {len(df)}")