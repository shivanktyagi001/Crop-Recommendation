
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.joblib"

model = joblib.load(MODEL_PATH)

FEATURES = [
    "N",
    "P",
    "K",
    "temperature",
    "humidity",
    "ph",
    "rainfall",
]

app = FastAPI(
    title="Crop Recommendation API",
    description="Recommend a crop using soil and climate features.",
    version="1.0.0",
)


class CropInput(BaseModel):
    N: float = Field(ge=0, le=140)
    P: float = Field(ge=0, le=145)
    K: float = Field(ge=0, le=205)
    temperature: float = Field(ge=0, le=60)
    humidity: float = Field(ge=0, le=100)
    ph: float = Field(ge=0, le=14)
    rainfall: float = Field(ge=0, le=500)


@app.get("/")
def home():
    return {
        "message": "Crop Recommendation API is running",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": True}


@app.post("/predict")
def predict_crop(data: CropInput):
    input_df = pd.DataFrame(
        [data.model_dump()],
        columns=FEATURES,
    )

    prediction = model.predict(input_df)[0]

    probabilities = model.predict_proba(input_df)[0]
    confidence = float(probabilities.max())

    return {
        "recommended_crop": str(prediction),
        "confidence": round(confidence, 4),
    }
