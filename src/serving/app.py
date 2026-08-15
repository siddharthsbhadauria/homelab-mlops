"""
FastAPI Model Serving Engine with Prometheus Observability & Health Probes.
"""
import os
import json
import logging
import time
from contextlib import asynccontextmanager
from typing import Optional

import joblib
from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

from src.config import Config
from src.features.feature_engineer import FeatureEngineer

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Prometheus metrics
PREDICTION_COUNT = Counter("model_predictions_total", "Total number of predictions", ["result"])
PREDICTION_LATENCY = Histogram("model_prediction_latency_seconds", "Latency of predictions in seconds")
ANOMALY_SCORE = Gauge("model_anomaly_score", "Anomaly score of the latest prediction")

# Pydantic models
class SystemMetrics(BaseModel):
    cpu_percent: float = Field(..., ge=0.0, le=100.0)
    ram_total_gb: Optional[float] = 32.0
    ram_used_gb: Optional[float] = 8.0
    ram_percent: float = Field(..., ge=0.0, le=100.0)
    disk_total_gb: Optional[float] = 1000.0
    disk_used_gb: Optional[float] = 200.0
    disk_percent: float = Field(..., ge=0.0, le=100.0)

class DiskIO(BaseModel):
    read_bytes: Optional[int] = 0
    write_bytes: Optional[int] = 0

class TelemetryInput(BaseModel):
    timestamp: str
    system: SystemMetrics
    disk_io: Optional[DiskIO] = Field(default_factory=DiskIO)
    status: Optional[str] = "HEALTHY"

class PredictionResponse(BaseModel):
    anomaly: bool
    is_anomaly: bool
    anomaly_score: float
    timestamp: str
    model_type: str
    model_version: str

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_type: str = "unknown"
    model_version: str = "unknown"
    last_trained: str = "unknown"


def init_app_state(target_app: FastAPI):
    """Initializes app.state defaults."""
    config = Config()
    target_app.state.config = config
    target_app.state.model = None
    target_app.state.metadata = {}
    target_app.state.model_loaded = False
    target_app.state.feature_engineer = FeatureEngineer(config)

    metadata_path = os.path.join(config.MODEL_DIR, "metadata.json")
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                target_app.state.metadata = json.load(f)
        except Exception as e:
            logger.warning(f"Failed to parse metadata: {e}")

    candidate_paths = [
        os.path.join(config.MODEL_DIR, "primary_model.joblib"),
        os.path.join(config.MODEL_DIR, "isolation_forest.joblib"),
        os.path.join(config.MODEL_DIR, "local_outlier_factor.joblib"),
    ]

    for model_path in candidate_paths:
        if os.path.exists(model_path):
            try:
                target_app.state.model = joblib.load(model_path)
                target_app.state.model_loaded = True
                logger.info(f"Successfully loaded model from {model_path}")
                break
            except Exception as e:
                logger.error(f"Error loading model from {model_path}: {e}")


@asynccontextmanager
async def lifespan(target_app: FastAPI):
    logger.info("Initializing Homelab MLOps Serving Engine...")
    init_app_state(target_app)
    yield
    logger.info("Shutting down Homelab MLOps Serving Engine...")
    target_app.state.model = None


app = FastAPI(
    title="Homelab MLOps Anomaly Detection Engine",
    description="Real-time telemetry inference and anomaly detection microservice with Prometheus telemetry.",
    version="1.0.0",
    lifespan=lifespan
)

# Eagerly initialize state so tests and standalone clients have valid state
init_app_state(app)


@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/docs")


@app.get("/health", response_model=HealthResponse)
async def health():
    if not getattr(app.state, "model_loaded", False):
        return HealthResponse(
            status="ok",
            model_loaded=False,
            model_type="none",
            model_version="none",
            last_trained="none"
        )

    metadata = getattr(app.state, "metadata", {})
    return HealthResponse(
        status="ok",
        model_loaded=True,
        model_type=metadata.get("model_type", "IsolationForest"),
        model_version=str(metadata.get("mlflow_run_id", "local-v1")),
        last_trained=str(metadata.get("trained_at", "recent"))
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(data: TelemetryInput):
    if not getattr(app.state, "model_loaded", False) or app.state.model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded. Train a model first.")

    start_time = time.time()
    try:
        sys_dict = data.system.model_dump() if hasattr(data.system, "model_dump") else data.system.dict()
        io_dict = data.disk_io.model_dump() if (data.disk_io and hasattr(data.disk_io, "model_dump")) else (data.disk_io.dict() if data.disk_io else {})

        snapshot = {
            "timestamp": data.timestamp,
            "system": sys_dict,
            "disk_io": io_dict,
            "status": data.status or "HEALTHY"
        }

        # Extract features
        features = app.state.feature_engineer.compute_single_point_features(snapshot)

        # Prepare feature vector
        metadata = getattr(app.state, "metadata", {})
        feature_columns = metadata.get("feature_columns", app.state.config.FEATURE_COLUMNS)
        feature_vector = [[features.get(col, 0.0) for col in feature_columns]]

        # Predict
        model = app.state.model
        if hasattr(model, "decision_function"):
            score = float(model.decision_function(feature_vector)[0])
        elif hasattr(model, "score_samples"):
            score = float(model.score_samples(feature_vector)[0])
        else:
            score = 0.0

        label = int(model.predict(feature_vector)[0])
        is_anomaly = (label == -1)

        PREDICTION_COUNT.labels(result="anomaly" if is_anomaly else "normal").inc()
        ANOMALY_SCORE.set(score)

        latency = time.time() - start_time
        PREDICTION_LATENCY.observe(latency)

        return PredictionResponse(
            anomaly=is_anomaly,
            is_anomaly=is_anomaly,
            anomaly_score=score,
            timestamp=data.timestamp,
            model_type=metadata.get("model_type", "IsolationForest"),
            model_version=str(metadata.get("mlflow_run_id", "local-v1"))
        )
    except Exception as e:
        logger.error(f"Error during prediction: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


if __name__ == "__main__":
    import uvicorn
    cfg = Config()
    uvicorn.run(app, host=cfg.API_HOST, port=cfg.API_PORT)
