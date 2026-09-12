# app/main.py
from contextlib import asynccontextmanager
from typing import List, Optional, Literal, Dict, Any
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

from app.utils import (
    ACCEL_FEATURES,
    DEPLOYED_THRESHOLD,
    INTENSIFY_THRESHOLD_KT,
    engineer_features,
    add_acceleration,
)
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="CycloCast Inference Gateway")

# Allow web browsers to make requests to this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows requests from any website domain
    allow_credentials=True,
    allow_methods=["*"],  # Allows POST, GET, OPTIONS, etc.
    allow_headers=["*"],
)
# Global dictionary to cache models in memory
models_cache: Dict[str, Any] = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- Startup: Load models once into memory ---
    horizons = ["6h", "12h", "24h"]
    for tag in horizons:
        models_cache[f"clf_{tag}"] = joblib.load(f"models/track_b_classifier_{tag}.joblib")
        models_cache[f"scaler_{tag}"] = joblib.load(f"models/track_b_scaler_{tag}.joblib")
        models_cache[f"int_clf_{tag}"] = joblib.load(f"models/track_b_intensify_clf_{tag}.joblib")
        models_cache[f"int_scaler_{tag}"] = joblib.load(f"models/track_b_intensify_scaler_{tag}.joblib")
        
        # Load regressors only for 12h and 24h
        if tag in ["12h", "24h"]:
            models_cache[f"reg_{tag}"] = joblib.load(f"models/track_b_regressor_{tag}.joblib")
            models_cache[f"reg_scaler_{tag}"] = joblib.load(f"models/track_b_reg_scaler_{tag}.joblib")
            
    print("All models successfully loaded into RAM.")
    yield
    # --- Shutdown: Cleanup if necessary ---
    models_cache.clear()

app = FastAPI(
    title="Cyclone Track & Intensity Forecasting API",
    version="3.0",
    lifespan=lifespan,
)

# Define Input Data Schemas
class TimeStepInput(BaseModel):
    ISO_TIME: str = Field(..., example="1981-10-11 18:00:00")
    LAT: float = Field(..., example=10.6)
    LON: float = Field(..., example=125.4)
    WIND_KTS: float = Field(..., example=45.0)
    PRES_MB: float = Field(..., example=1004.0)
    STORM_SPEED: float = Field(..., example=15.0)
    DIST2LAND: float = Field(..., example=21.0)

class PredictionRequest(BaseModel):
    horizon: Literal["+6h", "+12h", "+24h"] = Field(..., example="+12h")
    history: List[TimeStepInput]

# Define Output Data Schemas
class PredictionResponse(BaseModel):
    category: str
    horizon: str
    predicted_lat: Optional[float] = None
    predicted_lon: Optional[float] = None
    position_note: Optional[str] = None
    intensification_warning: bool
    intensification_probability: float
    intensification_threshold_kt: int

@app.post("/predict", response_model=PredictionResponse)
def predict_next(payload: PredictionRequest):
    horizon = payload.horizon
    tag = horizon.replace("+", "")

    # 1. Convert incoming JSON array into DataFrame
    history_data = [item.model_dump() for item in payload.history]
    df = pd.DataFrame(history_data)
    
    if len(df) < 6:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Insufficient history steps. At least 6 prior timesteps are required to generate features."
        )

    df["SID"] = "LIVE"
    
    # 2. Compute features
    try:
        engineered = add_acceleration(engineer_features(df))
        features = engineered.iloc[[-1]][ACCEL_FEATURES]
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error computing features: {str(e)}")

    if features.isna().any(axis=None):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="NaN values detected in feature matrix. Ensure chronological ordering and valid fields."
        )

    # 3. Predict Category
    clf = models_cache[f"clf_{tag}"]
    scaler = models_cache[f"scaler_{tag}"]
    features_scaled = scaler.transform(features)
    category_pred = clf.predict(features_scaled)[0]

    response = {
        "category": str(category_pred),
        "horizon": horizon
    }

    # 4. Predict Position (if applicable)
    if horizon in ["+12h", "+24h"]:
        reg = models_cache[f"reg_{tag}"]
        reg_scaler = models_cache[f"reg_scaler_{tag}"]
        lat, lon = reg.predict(reg_scaler.transform(features))[0]
        response["predicted_lat"] = round(float(lat), 2)
        response["predicted_lon"] = round(float(lon), 2)
        response["position_note"] = None
    else:
        response["predicted_lat"] = None
        response["predicted_lon"] = None
        response["position_note"] = "Not offered at +6h (no better than naive persistence)"

    # 5. Predict Intensification
    int_clf = models_cache[f"int_clf_{tag}"]
    int_scaler = models_cache[f"int_scaler_{tag}"]
    prob = int_clf.predict_proba(int_scaler.transform(features))[0, 1]
    
    response["intensification_warning"] = bool(prob >= DEPLOYED_THRESHOLD[horizon])
    response["intensification_probability"] = round(float(prob), 3)
    response["intensification_threshold_kt"] = INTENSIFY_THRESHOLD_KT[horizon]

    return response
