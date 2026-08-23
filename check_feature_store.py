"""
Quick check: query the Feature Store directly and print the row count.
This tells us the TRUTH about whether materialization actually landed,
regardless of what the job status said.

Run: python check_feature_store.py
"""

import os
from dotenv import load_dotenv
load_dotenv()

import hopsworks

HOPSWORKS_API_KEY = os.environ.get("HOPSWORKS_API_KEY")
HOPSWORKS_PROJECT = "aqi_predictor_ksa"
HOPSWORKS_HOST = "eu-west.cloud.hopsworks.ai"

FEATURE_VIEW_NAME = "aqi_feature_view"
FEATURE_VIEW_VERSION = 1

cert_dir = os.path.join(os.getcwd(), "hopsworks_certs")
os.makedirs(cert_dir, exist_ok=True)

project = hopsworks.login(
    project=HOPSWORKS_PROJECT,
    host=HOPSWORKS_HOST,
    port=443,
    api_key_value=HOPSWORKS_API_KEY,
    cert_folder=cert_dir,
)
fs = project.get_feature_store()

fv = fs.get_feature_view(name=FEATURE_VIEW_NAME, version=FEATURE_VIEW_VERSION)
df = fv.get_batch_data()

print(f"\n✅ Feature View '{FEATURE_VIEW_NAME}' returned {len(df)} rows")
print(f"   Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")  