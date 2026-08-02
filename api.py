"""
FastAPI Backend
Exposes the 3-day AQI forecast as a JSON API endpoint.
Next.js (or anything else) can call GET /predict to get the latest forecast.

Run with: uvicorn api:app --reload
Then visit http://127.0.0.1:8000/docs for interactive API docs.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from predict import predict_next_3_days

app = FastAPI(
    title="Karachi AQI Prediction API",
    description="Serves 3-day Air Quality Index forecasts for Karachi",
    version="1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {"message": "AQI Prediction API is running. Go to /predict for the forecast, or /docs for API docs."}


@app.get("/predict")
def get_prediction():
    try:
        result = predict_next_3_days()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/health")
def health_check():
    return {"status": "ok"}