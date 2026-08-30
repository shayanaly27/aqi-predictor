"""
Sends a real prediction request to the deployed Day 1 model to confirm
it's actually serving, not just showing 'Running' in the UI.
"""

import os
from dotenv import load_dotenv
load_dotenv()

import hopsworks

HOPSWORKS_API_KEY = os.environ.get("HOPSWORKS_API_KEY")
HOPSWORKS_PROJECT = "aqi_predictor_ksa"
HOPSWORKS_HOST = "eu-west.cloud.hopsworks.ai"

cert_dir = os.path.join(os.getcwd(), "hopsworks_certs")
os.makedirs(cert_dir, exist_ok=True)

project = hopsworks.login(
    project=HOPSWORKS_PROJECT,
    host=HOPSWORKS_HOST,
    port=443,
    api_key_value=HOPSWORKS_API_KEY,
    cert_folder=cert_dir,
)

ms = project.get_model_serving()
deployment = ms.get_deployment("aqitargetday1deploy")

# 25 features in the exact order get_feature_columns() returns them in
# train_model.py - this is just a dummy row to confirm the endpoint responds
dummy_row = [[12, 15, 8, 3,
              28.5, 55, 1005, 3.2,
              62, 11.4, 22.5, 43.0, 5.8, 4.1, 170.0,
              60, 61, 58,
              59, 60, 61, 60,
              1.0,
              1003, 2.0]]

response = deployment.predict(inputs=dummy_row)
print("Deployment response:", response)