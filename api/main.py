"""Inference service for the breast-cancer classifier.

Endpoints
  GET  /healthz  liveness: the process is up
  GET  /readyz   readiness: 200 only when the model is loaded, else 503
  GET  /version  app version + model provenance
  POST /predict  {"features": [30 floats]}
Config comes from environment variables (a ConfigMap in Kubernetes).
"""
from __future__ import annotations

import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

REPO_ROOT = Path(__file__).resolve().parents[1]
APP_VERSION = os.getenv("APP_VERSION", "dev")
MODEL_PATH = Path(os.getenv("MODEL_PATH", str(REPO_ROOT / "serving" / "model" / "model.joblib")))
METADATA_PATH = Path(os.getenv("MODEL_METADATA_PATH", str(MODEL_PATH.with_name("metadata.json"))))

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
log = logging.getLogger("api")

state: dict = {"model": None, "metadata": {}, "load_error": None}


def load_model() -> None:
    """Load the model; on failure stay alive but not ready (visible to probes)."""
    state.update(model=None, metadata={}, load_error=None)
    try:
        state["model"] = joblib.load(MODEL_PATH)
        if METADATA_PATH.exists():
            state["metadata"] = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
        log.info("model loaded from %s (app_version=%s)", MODEL_PATH, APP_VERSION)
    except Exception as exc:  # noqa: BLE001 - we want to report any load failure
        state["load_error"] = f"{type(exc).__name__}: {exc}"
        log.error("model load failed from %s: %s", MODEL_PATH, state["load_error"])


@asynccontextmanager
async def lifespan(_: FastAPI):
    load_model()
    yield


app = FastAPI(title="Breast cancer inference API", version=APP_VERSION, lifespan=lifespan)


class PredictRequest(BaseModel):
    features: list[float] = Field(min_length=1, max_length=1000)


class PredictResponse(BaseModel):
    prediction: int
    confidence: float
    model_name: str | None = None
    dataset_version: str | None = None
    app_version: str


@app.get("/healthz")
def healthz():
    return {"status": "alive"}


@app.get("/readyz")
def readyz():
    if state["model"] is None:
        return JSONResponse(status_code=503, content={"status": "not_ready", "reason": state["load_error"]})
    return {"status": "ready"}


@app.get("/version")
def version():
    return {
        "app_version": APP_VERSION,
        "model_loaded": state["model"] is not None,
        "model": state["metadata"],
        "load_error": state["load_error"],
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    model = state["model"]
    if model is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    expected = state["metadata"].get("n_features") or getattr(model, "n_features_in_", None)
    if expected is not None and len(req.features) != expected:
        raise HTTPException(status_code=422, detail=f"expected {expected} features, got {len(req.features)}")
    x = np.asarray([req.features], dtype=float)
    pred = int(model.predict(x)[0])
    confidence = float(np.max(model.predict_proba(x)[0]))
    meta = state["metadata"]
    return PredictResponse(
        prediction=pred,
        confidence=round(confidence, 4),
        model_name=meta.get("model_name"),
        dataset_version=meta.get("dataset_version"),
        app_version=APP_VERSION,
    )