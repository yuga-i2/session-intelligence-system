"""
arvyax FastAPI — local inference API
=====================================
run:  uvicorn app:app --reload
docs: http://localhost:8000/docs
"""

from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# import core pipeline from arvyax_system
from arvyax_system import (
    TRAINING_RECORDS, COLS_TRAIN,
    extract_text_features, compute_uncertainty,
    build_features, decide, train_models
)

# ── startup: train model once on launch ──────────────────────
app = FastAPI(
    title="Arvyax Session Intelligence API",
    description="interpretation → judgment → decision",
    version="1.0.0"
)

print("Training models on startup...")
df_train = pd.DataFrame(TRAINING_RECORDS, columns=COLS_TRAIN)
clf_es, le_es, reg_int, feature_cols = train_models(df_train)
print("Models ready.")


# ── request / response schemas ───────────────────────────────

class SessionInput(BaseModel):
    id:                  Optional[int]   = None
    journal_text:        str
    ambience_type:       Optional[str]   = None
    duration_min:        Optional[float] = None
    sleep_hours:         Optional[float] = None
    energy_level:        Optional[float] = None
    stress_level:        Optional[float] = None
    time_of_day:         Optional[str]   = None
    previous_day_mood:   Optional[str]   = None
    face_emotion_hint:   Optional[str]   = None
    reflection_quality:  Optional[str]   = None

class SessionOutput(BaseModel):
    id:                    Optional[int]
    predicted_state:       str
    predicted_intensity:   int
    confidence:            float
    uncertain_flag:        int
    what_to_do:            str
    when_to_do:            str
    supportive_message:    str
    uncertainty_reasons:   list


# ── endpoints ─────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "service": "Arvyax Session Intelligence API",
        "status":  "running",
        "endpoints": {
            "POST /predict":       "Single session prediction",
            "POST /predict/batch": "Batch predictions",
            "GET  /health":        "Health check",
            "GET  /docs":          "Interactive API docs",
        }
    }


@app.get("/health")
def health():
    return {"status": "ok", "model": "RandomForest", "training_records": len(df_train)}


@app.post("/predict", response_model=SessionOutput)
def predict(session: SessionInput):
    row = session.dict()
    row_series = pd.Series(row)

    text_feats = extract_text_features(str(row.get('journal_text', '')))
    unc        = compute_uncertainty(row_series, text_feats)

    sample   = pd.DataFrame([row])
    X_sample = build_features(sample)
    X_sample = X_sample.reindex(columns=feature_cols, fill_value=0)

    # emotional state
    es_probs    = clf_es.predict_proba(X_sample)[0]
    es_idx      = np.argmax(es_probs)
    pred_state  = le_es.inverse_transform([es_idx])[0]
    model_conf  = float(es_probs[es_idx])
    final_conf  = round(model_conf * (0.5 + 0.5 * (1 - unc['uncertain_flag'] * 0.3)), 3)

    # intensity
    pred_intensity = int(round(np.clip(reg_int.predict(X_sample)[0], 1, 5)))

    # decision
    what, when, message = decide(
        pred_state, pred_intensity,
        row.get('stress_level'), row.get('energy_level'),
        row.get('time_of_day'), text_feats, final_conf
    )

    return SessionOutput(
        id                  = row.get('id'),
        predicted_state     = pred_state,
        predicted_intensity = pred_intensity,
        confidence          = final_conf,
        uncertain_flag      = unc['uncertain_flag'],
        what_to_do          = what,
        when_to_do          = when,
        supportive_message  = message,
        uncertainty_reasons = unc['reasons'],
    )


@app.post("/predict/batch")
def predict_batch(sessions: list[SessionInput]):
    return [predict(s) for s in sessions]
