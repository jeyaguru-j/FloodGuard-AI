from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


# ============================================================
# Configuration
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "floodguard_3day_random_forest.joblib"
DATA_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "ml_dataset"
    / "chennai_flood_3day_ml_dataset.csv"
)


# ============================================================
# Load model and reference dataset
# ============================================================

model = joblib.load(MODEL_PATH)
reference_data = pd.read_csv(DATA_PATH)

reference_data["date"] = pd.to_datetime(reference_data["date"])


# Exact feature order used during model training
FEATURES = list(model.feature_names_in_)


# ============================================================
# FastAPI application
# ============================================================

app = FastAPI(
    title="FloodGuard AI",
    description=(
        "Urban Flood Prediction and Early Warning System "
        "using Machine Learning and MLOps."
    ),
    version="1.0.0",
)


# ============================================================
# Request schema
# ============================================================

class PredictionRequest(BaseModel):
    date: str


# ============================================================
# Health endpoint
# ============================================================

@app.get("/")
def root():
    return {
        "system": "FloodGuard AI",
        "description": "Urban Flood Prediction and Early Warning System",
        "model": "Random Forest",
        "forecast_horizon": "Next 3 days",
        "status": "operational",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": True,
        "model_type": "RandomForestClassifier",
        "features": len(FEATURES),
    }


# ============================================================
# Model information
# ============================================================

@app.get("/model-info")
def model_info():
    return {
        "model": "FloodGuard 3-Day Random Forest",
        "algorithm": "Random Forest Classifier",
        "forecast_horizon": "3 days",
        "number_of_features": len(FEATURES),
        "features": FEATURES,
    }


# ============================================================
# Prediction endpoint
# ============================================================

@app.post("/predict")
def predict(request: PredictionRequest):

    try:
        requested_date = pd.to_datetime(request.date)

    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Invalid date format. Use YYYY-MM-DD.",
        )

    # Find the closest available feature row
    matching_rows = reference_data[
        reference_data["date"] == requested_date
    ]

    if matching_rows.empty:
        raise HTTPException(
            status_code=404,
            detail=(
                f"No feature data available for {request.date}. "
                "Try a date between 2010-01-15 and 2025-12-27."
            ),
        )

    row = matching_rows.iloc[0]

    # Extract features in EXACT training order
    X = pd.DataFrame(
        [[row[feature] for feature in FEATURES]],
        columns=FEATURES,
    )

    # Prediction probability
    probability = float(model.predict_proba(X)[0][1])

    # Classification for dashboard
    if probability >= 0.50:
        risk_level = "HIGH"
    elif probability >= 0.25:
        risk_level = "MODERATE"
    else:
        risk_level = "LOW"

    prediction = int(probability >= 0.50)

    # Identify strongest rainfall/humidity signals
    indicators = []

    if row["rainfall_14d_current_sum"] > 100:
        indicators.append("High accumulated rainfall over the previous 14 days")

    if row["rainfall_7d_current_sum"] > 50:
        indicators.append("Elevated 7-day rainfall")

    if row["humidity_7d_mean"] > 80:
        indicators.append("High average humidity")

    if row["rainy_days_14d"] >= 7:
        indicators.append("Persistent rainy conditions")

    if not indicators:
        indicators.append("No major extreme weather indicator detected")

    return {
        "date": request.date,
        "location": "Chennai",
        "forecast": "Flood risk within next 3 days",
        "flood_probability": round(probability, 4),
        "flood_probability_percent": round(probability * 100, 2),
        "risk_level": risk_level,
        "prediction": prediction,
        "key_indicators": indicators,
        "weather": {
            "rainfall_mm": round(float(row["rainfall_mm"]), 2),
            "temperature_mean_c": round(
                float(row["temperature_mean_c"]), 2
            ),
            "humidity_percent": round(
                float(row["relative_humidity_percent"]), 2
            ),
            "wind_speed_kmh": round(
                float(row["wind_speed_max_kmh"]), 2
            ),
        },
    }