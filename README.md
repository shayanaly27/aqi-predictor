# Pearls AQI Predictor — Backend

A serverless, end-to-end machine learning system that forecasts Karachi's Air Quality Index (AQI) **24, 48, and 72 hours** into the future. Built for the 10Pearls Shine — Data Science Track.

Live API: **https://aqi-predictor-api-wodc.onrender.com**

> Note: the free-tier backend spins down after ~15 minutes of inactivity and takes 20–30s to wake up on the first request afterward — this is a Render free-tier characteristic, not a bug. See [Known Limitations](#known-limitations).

## What this repo does

This is the ML/backend half of the project (see the separate [aqi-dashboard](#) repo for the frontend). It runs two automated pipelines unattended via GitHub Actions and serves predictions through a FastAPI backend:

- **Hourly Feature Pipeline** — pulls live weather + air-quality data and writes it into a Hopsworks Feature Store
- **Daily Training Pipeline** — retrains three regression algorithms per forecast horizon, evaluates them, and registers the best-performing model per horizon to the Hopsworks Model Registry
- **FastAPI service** — loads the latest registered models and recent features, and serves live predictions, historical trends, SHAP explainability, and model performance metrics

No CSV files are used anywhere in the live serving path — Hopsworks is the single source of truth for features and models.

## Architecture

```
Open-Meteo APIs → feature_pipeline.py → Hopsworks Feature Store
                                              ↓
                                       train_model.py → Hopsworks Model Registry
                                              ↓
                                   predict.py / api.py → Next.js Dashboard
```

## Tech stack

- **Language:** Python 3.11
- **ML:** scikit-learn (Ridge, Random Forest), XGBoost, TensorFlow/Keras
- **Feature Store & Model Registry:** Hopsworks (free tier, `eu-west.cloud.hopsworks.ai`)
- **API:** FastAPI + Uvicorn
- **Automation:** GitHub Actions (hourly feature pipeline, daily training pipeline)
- **Deployment:** Render (free tier)
- **Explainability:** SHAP

## Repository structure

| File | Purpose |
|---|---|
| `feature_pipeline.py` | Hourly job — fetches live weather/AQI data, inserts into the Hopsworks Feature Group |
| `backfill_historical.py` | One-time job — pulls ~3.5 years of historical data to bootstrap the Feature Store |
| `merge_data.py` | Cleans and regularizes raw data onto a strict hourly grid |
| `eda.py` | Exploratory data analysis (distribution, trend, seasonality, correlation) |
| `train_model.py` | Trains Ridge / Random Forest / XGBoost / Keras per horizon, registers the best model |
| `model_history.py` | Generates version-history JSON for the dashboard from the live Model Registry |
| `shap_explain.py` | Full SHAP summary plots per horizon |
| `compute_feature_importance.py` | Lightweight top-8 SHAP JSON consumed by the dashboard |
| `predict.py` | Loads the latest models + recent features, produces a 3-day forecast |
| `api.py` | FastAPI service exposing `/predict`, `/history`, `/feature-importance`, `/model-metrics`, `/model-versions` |
| `backtest.py` / `quick_check.py` | Real-world accuracy validation against actual historical outcomes |
| `check_feature_store.py` | Verifies the Feature View is returning fresh rows |
| `test_deployment.py` | Sends a real inference request to a live Hopsworks Model Deployment |

## Prerequisites

- Python 3.11
- A free [Hopsworks](https://www.hopsworks.ai/) account and project, with an API key generated
- Git

## Setup

### 1. Environment

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1        # Windows PowerShell
# source .venv/bin/activate       # macOS/Linux
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```
HOPSWORKS_API_KEY=your_api_key_here
```

### 2. One-time bootstrap (only needed once, or when re-seeding history)

```bash
python backfill_historical.py     # pulls ~3.5 years of historical data
python merge_data.py              # cleans, regularizes to hourly grid
python push_to_hopsworks.py       # creates Feature Group + Feature View, uploads history
```

### 3. Train models and populate the Model Registry

```bash
python train_model.py             # trains Ridge/RF/XGBoost per horizon, registers best model
python model_history.py           # generates version-history JSON for the dashboard
python compute_feature_importance.py   # generates SHAP summary JSON for the dashboard
```

### 4. Run the API

```bash
uvicorn api:app --reload
# Visit http://127.0.0.1:8000/docs to confirm all endpoints respond
```

## Automated pipelines

Two GitHub Actions workflows live under `.github/workflows/`:

- `feature_pipeline.yml` — runs hourly (`0 * * * *`)
- `training_pipeline.yml` — runs daily at 02:00 UTC

To enable them on a fork, add `HOPSWORKS_API_KEY` as a repository secret under **Settings → Secrets and variables → Actions**. Both can also be triggered manually via `workflow_dispatch`.

## Verifying everything works

```bash
python check_feature_store.py     # confirms the Feature View returns rows
python quick_check.py             # runs a single real backtest comparison
python backtest.py                # runs the full 15-point-per-horizon backtest
python test_deployment.py         # sends a real request to a live Hopsworks deployment
```

## Key results

- **73%** of backtested 3-day forecasts landed within 10 AQI points of the actual recorded value; **93%** within 20 points, across 45 real historical comparisons
- **Best models:** XGBoost (24h, 48h horizons), Ridge Regression (72h horizon)
- **6+ model versions** tracked in the Hopsworks Model Registry across three forecast horizons
- **3 live model deployments** on Hopsworks Model Serving, tested with real inference requests

## Known limitations

- **Hopsworks free-tier materialization delays:** the background job that merges new feature rows into the queryable Feature View can stall or fail intermittently (a documented, cohort-wide free-tier issue). The API mitigates this by fetching the current AQI reading live from Open-Meteo directly, independent of Hopsworks' materialization status, and exposes a `data_freshness` field indicating how stale the historical Feature Store data is.
- **Render free-tier cold starts:** the backend spins down after ~15 minutes idle; the first request after that takes 20–30s to fully wake up (Hopsworks login + model downloads).
- **72-hour forecast accuracy** is meaningfully weaker than 24h/48h — a genuinely harder forecasting problem given the current feature set.
- Single-city scope, tuned specifically to Karachi's coordinates and pollution patterns.

Full engineering write-up, EDA, model comparisons, and an honest accounting of every issue hit during development are documented in the accompanying project report.

## License

Educational project — 10Pearls Shine, Data Science Track.
