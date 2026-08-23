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
import time
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


INTEGER_FEATURES = ["aqi", "humidity"]


def fix_dtypes(df):
    """
    The Hopsworks Feature Group schema expects aqi and humidity as plain
    integers ('bigint'/'long'). merge_data.py's gap interpolation can leave:
      (a) fractional values (e.g. 77.0 from averaging 76 and 78), and
      (b) a small number of NaN rows where a gap was too long to safely
          interpolate (these get dropped later in training anyway).
    Hopsworks' Avro-based upload also rejects pandas' nullable Int64 dtype
    even when there ARE no NaNs left, so we drop the NaN rows first and
    cast to a plain numpy int64 - the type it actually expects.
    """
    before = len(df)
    df = df.dropna(subset=INTEGER_FEATURES).reset_index(drop=True)
    dropped = before - len(df)
    if dropped:
        print(f"  -> Dropped {dropped} rows with unfillable gaps in {INTEGER_FEATURES} before upload")

    for col in INTEGER_FEATURES:
        if col in df.columns:
            df[col] = df[col].round().astype("int64")
    return df


def load_data():
    df = pd.read_csv(DATA_FILE)
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed")
    df = df.drop_duplicates(subset="timestamp").reset_index(drop=True)
    df = fix_dtypes(df)
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


def wait_for_materialization(job, timeout_seconds=600, poll_interval=10):
    """
    Different hsfs client versions expose different ways to wait for the
    background materialization job to finish. Try the known methods in
    order; if none exist, fall back to polling job.get_state() manually.

    Returns True (success), False (failed/timed out), or None (unknown -
    this hsfs version can't tell us, so we can't confirm either way).
    """
    if hasattr(job, "get_state"):
        terminal_states = {"FINISHED", "FAILED", "KILLED", "FRAMEWORK_FAILURE"}
        elapsed = 0
        while elapsed < timeout_seconds:
            state = job.get_state()
            print(f"  ...job state: {state} ({elapsed}s elapsed)")
            if state in terminal_states:
                if state != "FINISHED":
                    print(f"  ⚠️ Job ended with state '{state}', not 'FINISHED' - check the Hopsworks Jobs UI.")
                    return False
                return True
            time.sleep(poll_interval)
            elapsed += poll_interval
        print(f"  ⚠️ Timed out after {timeout_seconds}s waiting for job - check the Hopsworks Jobs UI before training.")
        return False

    if hasattr(job, "get_final_status"):
        status = job.get_final_status()
        return str(status).upper() in ("FINISHED", "SUCCEEDED", "SUCCESS")

    if hasattr(job, "await_termination"):
        job.await_termination()
        return True

    # last resort: this hsfs version doesn't expose a way to check job
    # status at all, so we genuinely don't know - don't claim success
    print("  ⚠️ This hsfs version doesn't expose job status. Waiting 90s as a fallback,")
    print("     but we cannot confirm success - check the Hopsworks Jobs UI manually.")
    time.sleep(90)
    return None


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
    job, _ = aqi_fg.insert(df)
    print("Waiting for the offline materialization job to finish...")
    print("(insert() only stages the data - the Feature View won't see it")
    print(" until this background job actually completes)")
    success = wait_for_materialization(job)

    if success is not True:
        # The job status is known to sometimes report FAILED even when the
        # data actually landed correctly (a real false-negative we hit and
        # confirmed via a manual row-count check on 2026-08-19). Rather than
        # hard-failing the whole CI/CD run on a flaky status label, do one
        # more direct check: does the row count in the Feature Store
        # actually match what we just sent?
        print("\n⚠️ Job status did not confirm success. Double-checking by directly")
        print("   querying the Feature Store row count before deciding to fail...")
        try:
            feature_view = get_or_create_feature_view(fs, aqi_fg)
            actual_df = feature_view.get_batch_data()
            actual_rows = len(actual_df)
            expected_rows = len(df)
            print(f"   Feature Store reports {actual_rows} rows (we uploaded {expected_rows}).")

            if actual_rows >= expected_rows:
                print("✅ Row count check passed - data is actually there despite the")
                print("   job status. Treating this as a successful materialization.")
                success = True
            else:
                print(f"\n❌ STOPPING - Feature Store only has {actual_rows} rows, expected")
                print(f"   at least {expected_rows}. Materialization genuinely did not complete.")
                print("   Check the Hopsworks Jobs UI logs for the real error, fix it, and re-run.")
                raise SystemExit(1)
        except SystemExit:
            raise
        except Exception as e:
            print(f"\n❌ STOPPING - Could not verify row count either: {e}")
            print("   Check the Hopsworks Jobs UI logs for the real error, fix it, and re-run.")
            raise SystemExit(1)
    else:
        print("✅ Data successfully pushed AND materialized in the Hopsworks Feature Store.")
        print(f"\nCreating/getting feature view '{FEATURE_VIEW_NAME}'...")
        feature_view = get_or_create_feature_view(fs, aqi_fg)

    print("\n✅ Feature Store setup complete.")
    print(f"   Project: {HOPSWORKS_PROJECT}")
    print(f"   Feature Group: {FEATURE_GROUP_NAME} (v{FEATURE_GROUP_VERSION})")
    print(f"   Feature View: {FEATURE_VIEW_NAME} (v{FEATURE_VIEW_VERSION})")
    print(f"   Rows: {len(df)}")