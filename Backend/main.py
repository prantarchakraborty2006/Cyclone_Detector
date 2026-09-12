from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import json
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from fastapi import FastAPI
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

app = FastAPI(title="CycloCast Inference Gateway")

# Step: Enable CORS so any frontend can call this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------
# PART A: IMAGE PREPROCESSING + MOCK CNN CLASSIFICATION
# ---------------------------------------------------------

def preprocess_image(file_bytes: bytes) -> np.ndarray:
    """Reads image bytes, resizes to 224x224, normalizes to [0,1]."""
    image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    image = image.resize((224, 224))
    arr = np.array(image, dtype=np.float32) / 255.0   # normalize 0-1
    arr = np.expand_dims(arr, axis=0)                  # shape -> (1,224,224,3)
    return arr


def mock_cnn_predict(image_array: np.ndarray) -> dict:
    """
    Placeholder classifier — no real trained model yet.
    Uses average pixel brightness as a fake signal just so the
    pipeline is testable end-to-end. Replace with a real model later.
    """
    brightness = float(image_array.mean())

    classes = [
        "Deep Depression",
        "Cyclonic Storm",
        "Severe Cyclonic Storm",
        "Very Severe Cyclonic Storm",
    ]

    # simple mock rule: map brightness ranges to classes
    if brightness < 0.25:
        prediction = classes[0]
    elif brightness < 0.45:
        prediction = classes[1]
    elif brightness < 0.65:
        prediction = classes[2]
    else:
        prediction = classes[3]

    confidence = round(float(np.clip(0.5 + brightness * 0.4, 0.50, 0.99)), 3)

    return {"prediction": prediction, "confidence": confidence}


# ---------------------------------------------------------
# PART B: NUMERICAL TRACK & INTENSITY PREDICTION (MOCK)
# ---------------------------------------------------------

def validate_history(history: list):
    if len(history) < 7:
        raise HTTPException(
            status_code=400,
            detail=f"'history' must contain at least 7 timesteps, got {len(history)}."
        )


def engineer_features(df: pd.DataFrame) -> dict:
    """Compute simple trend deltas and rolling averages."""
    last3 = df.tail(3)

    delta_lat = float(last3["LAT"].iloc[-1] - last3["LAT"].iloc[0])
    delta_lon = float(last3["LON"].iloc[-1] - last3["LON"].iloc[0])
    delta_wind = float(last3["WIND_KTS"].iloc[-1] - last3["WIND_KTS"].iloc[0])
    delta_pres = float(last3["PRES_MB"].iloc[-1] - last3["PRES_MB"].iloc[0])

    rolling_wind = float(df["WIND_KTS"].rolling(3).mean().iloc[-1])
    rolling_pres = float(df["PRES_MB"].rolling(3).mean().iloc[-1])

    return {
        "delta_lat": delta_lat,
        "delta_lon": delta_lon,
        "delta_wind": delta_wind,
        "delta_pres": delta_pres,
        "rolling_wind": rolling_wind,
        "rolling_pres": rolling_pres,
    }


def horizon_multiplier(horizon: str) -> float:
    """Scales the predicted shift depending on forecast horizon."""
    mapping = {"+6h": 1.0, "+12h": 2.0, "+24h": 4.0}
    if horizon not in mapping:
        raise HTTPException(status_code=400, detail=f"Invalid horizon: {horizon}")
    return mapping[horizon]


def classify_category(wind_speed: float) -> str:
    """Rough IMD-style category mapping based on predicted wind speed (knots)."""
    if wind_speed < 34:
        return "D"
    elif wind_speed < 48:
        return "DD"
    elif wind_speed < 64:
        return "CS"
    elif wind_speed < 90:
        return "SCS"
    elif wind_speed < 120:
        return "VSCS"
    elif wind_speed < 165:
        return "ESCS"
    else:
        return "SuCS"


def predict_numerical(history: list, horizon: str) -> dict:
    df = pd.DataFrame(history)
    df["ISO_TIME"] = pd.to_datetime(df["ISO_TIME"])
    df = df.sort_values("ISO_TIME").reset_index(drop=True)

    features = engineer_features(df)
    scale = horizon_multiplier(horizon)

    latest = df.iloc[-1]

    predicted_lat = round(float(latest["LAT"] + features["delta_lat"] * scale), 2)
    predicted_lon = round(float(latest["LON"] + features["delta_lon"] * scale), 2)
    predicted_wind = float(latest["WIND_KTS"] + features["delta_wind"] * scale)

    # mock intensification probability: scaled from wind trend
    intensification_probability = round(
        float(np.clip(0.5 + (features["delta_wind"] / 50), 0.0, 1.0)), 3
    )

    intensification_warning = bool(
        intensification_probability > 0.60 or features["delta_wind"] > 15
    )

    category = classify_category(predicted_wind)

    return {
        "predicted_lat": predicted_lat,
        "predicted_lon": predicted_lon,
        "category": category,
        "intensification_probability": intensification_probability,
        "intensification_warning": intensification_warning,
    }


# ---------------------------------------------------------
# PART C: THREAT LEVEL SUMMARY
# ---------------------------------------------------------

def compute_threat_level(numerical_result: dict, image_result: dict) -> str:
    if numerical_result["intensification_warning"] or image_result["confidence"] > 0.85:
        return "HIGH"
    elif numerical_result["intensification_probability"] > 0.4:
        return "MODERATE"
    else:
        return "LOW"


# ---------------------------------------------------------
# ENDPOINT
# ---------------------------------------------------------

@app.post("/predict-combined")
async def predict_combined(
    file: UploadFile = File(...),
    numeric_json: str = Form(...),
):
    # ---- Parse and validate numeric_json ----
    try:
        payload = json.loads(numeric_json)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON in 'numeric_json'.")

    horizon = payload.get("horizon")
    history = payload.get("history")

    if horizon is None or history is None:
        raise HTTPException(
            status_code=400,
            detail="'numeric_json' must contain 'horizon' and 'history'."
        )

    validate_history(history)

    # ---- Image classification ----
    file_bytes = await file.read()
    image_array = preprocess_image(file_bytes)
    image_result = mock_cnn_predict(image_array)

    # ---- Numerical prediction ----
    numerical_result = predict_numerical(history, horizon)

    # ---- Assemble response ----
    response = {
        "horizon": horizon,
        "image_classification": image_result,
        "numerical_prediction": numerical_result,
        "summary": {
            "overall_threat_level": compute_threat_level(numerical_result, image_result),
            "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        },
    }

    return response
