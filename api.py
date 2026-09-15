"""FastAPI service returning per-antibiotic-class resistance probabilities."""
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict

MODEL_DIR = Path(__file__).parent.parent / "models"
app = FastAPI(title="Antimicrobial Resistance Prediction API", version="1.0.0")
_bundle = None


def _load():
    global _bundle
    if _bundle is None:
        _bundle = joblib.load(MODEL_DIR / "amr_multilabel.joblib")
    return _bundle


class PatientFeatures(BaseModel):
    prior_abx_90d: int = 0
    recent_hosp: int = 0
    nursing_home: int = 0
    device_catheter: int = 0
    device_central_line: int = 0
    device_ventilator: int = 0
    icu_admission: int = 0
    note_flag_prior_mdro: int = 0
    note_flag_recent_travel: int = 0
    note_flag_immunosuppressed: int = 0
    month: int = 24


class PredictionResponse(BaseModel):
    resistance_probabilities: Dict[str, float]
    likely_mdro: bool


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(x: PatientFeatures):
    bundle = _load()
    row = pd.DataFrame([x.dict()])[bundle["feature_cols"]]
    probs = {cls: round(float(bundle["models"][cls].predict_proba(row)[0, 1]), 4) for cls in bundle["classes"]}
    likely_mdro = sum(p >= 0.5 for p in probs.values()) >= 3
    return PredictionResponse(resistance_probabilities=probs, likely_mdro=likely_mdro)
