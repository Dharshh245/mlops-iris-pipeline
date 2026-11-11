"""Q2 (a) - Iris model API with /predict and /health endpoints.

Run locally:  uvicorn api.main:app --reload --port 8000
Docs:         http://localhost:8000/docs
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from api.model_service import ModelService

service = ModelService()


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        service.load()
    except Exception as exc:  # keep running so /health can report the failure
        print(f"Failed to load model from {service.model_path}: {exc}")
    yield


app = FastAPI(title="Iris Classifier API", version="1.0.0", lifespan=lifespan)


class IrisFeatures(BaseModel):
    sepal_length: float = Field(..., ge=0, examples=[5.1])
    sepal_width: float = Field(..., ge=0, examples=[3.5])
    petal_length: float = Field(..., ge=0, examples=[1.4])
    petal_width: float = Field(..., ge=0, examples=[0.2])


class PredictRequest(BaseModel):
    instances: list[IrisFeatures] = Field(..., min_length=1)


@app.get("/health")
def health():
    """Used by Kubernetes liveness/readiness probes. 503 until the model is loaded."""
    body = {
        "status": "ok" if service.ready else "unavailable",
        "model_loaded": service.ready,
        "model_version": service.version,
    }
    return JSONResponse(body, status_code=200 if service.ready else 503)


@app.post("/predict")
def predict(req: PredictRequest):
    if not service.ready:
        raise HTTPException(status_code=503, detail="Model not loaded")
    rows = [inst.model_dump() for inst in req.instances]
    return {"model_version": service.version, "predictions": service.predict(rows)}
