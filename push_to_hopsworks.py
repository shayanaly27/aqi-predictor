"""
Hopsworks Integration - Feature Store Upload
Connects to your Hopsworks project, pushes the merged training dataset
into a Feature Group, and creates a Feature View on top of it.

The Feature Group is just storage. The Feature View is what training
and prediction scripts actually query - without it, nothing downstream
can read from the Feature Store.

Run this once to do the initial upload + feature view creation.
Safe to re-run: insert() upserts, and get_or_create_feature_view()
won't duplicate an existing view.
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

FEATURE_GROUP_NAME = "aqi_features"
FEATURE_GROUP_VERSION = 1

FEATURE_VIEW_NAME = "aqi_feature_view"
FEATURE_VIEW_VERSION = 1


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


def get_or_create_feature_group(fs):
    aqi_fg = fs.get_or_create_feature_group(
        name=FEATURE_GROUP_NAME,
        version=FEATURE_GROUP_VERSION,
        description="Karachi AQI features - weather, pollutants, time-based features",
        primary_key=["timestamp"],
        event_time="timestamp",
        time_travel_format="HUDI",
        online_enabled=False,
    )
    return aqi_fg


def get_or_create_feature_view(fs, aqi_fg):
    """
    A Feature Group is raw storage. A Feature View is the queryable object
    that training/prediction scripts actually pull data from. Without this,
    the Feature Group sits unused (this was the missing piece).
    """
    query = aqi_fg.select_all()

    feature_view = fs.get_or_create_feature_view(
        name=FEATURE_VIEW_NAME,
        version=FEATURE_VIEW_VERSION,
        description="AQI feature view - all raw features for Karachi AQI prediction",
        query=query,
    )
    print(f"✅ Feature View ready: {FEATURE_VIEW_NAME} (v{FEATURE_VIEW_VERSION})")
    return feature_view


if __name__ == "__main__":
    print("Loading training data...")
    df = load_data()
    print(f"  -> {len(df)} rows ready to upload")

    print("\nConnecting to Hopsworks...")
    project = connect_to_hopsworks()
    fs = project.get_feature_store()

    print(f"\nCreating/getting feature group '{FEATURE_GROUP_NAME}'...")
    aqi_fg = get_or_create_feature_group(fs)

    print("Inserting data into feature group (this may take a few minutes)...")
    aqi_fg.insert(df)
    print("✅ Data successfully pushed to Hopsworks Feature Store.")

    print(f"\nCreating/getting feature view '{FEATURE_VIEW_NAME}'...")
    feature_view = get_or_create_feature_view(fs, aqi_fg)

    print("\n✅ Feature Store setup complete.")
    print(f"   Project: {HOPSWORKS_PROJECT}")
    print(f"   Feature Group: {FEATURE_GROUP_NAME} (v{FEATURE_GROUP_VERSION})")
    print(f"   Feature View: {FEATURE_VIEW_NAME} (v{FEATURE_VIEW_VERSION})")
    print(f"   Rows: {len(df)}")