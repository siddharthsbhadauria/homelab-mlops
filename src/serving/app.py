"""
FastAPI Model Serving Engine with Prometheus Observability & Health Probes.
"""
import os
import json
import logging
import time
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional

import joblib
from fastapi import FastAPI, HTTPException, Request, Response
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

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Initializing Homelab MLOps Serving Engine...")
    config = Config()
    app.state.config = config
    app.state.model = None
    app.state.metadata = {}
    app.state.model_loaded = False
    app.state.feature_engineer = FeatureEngineer(config)

    metadata_path = os.path.join(config.MODEL_DIR, "metadata.json")
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, "r") as f:
                app.state.metadata = json.load(f)
        except Exception as e:
            logger.warning(f"Failed to parse metadata: {e}")

    # Check candidate model paths in priority order
    candidate_paths = [
        os.path.join(config.MODEL_DIR, "primary_model.joblib"),
        os.path.join(config.MODEL_DIR, "isolation_forest.joblib"),
        os.path.join(config.MODEL_DIR, "local_outlier_factor.joblib"),
    ]

    for model_path in candidate_paths:
        if os.path.exists(model_path):
            try:
                app.state.model = joblib.load(model_path)
                app.state.model_loaded = True
                logger.info(f"Successfully loaded model from {model_path}")
                break
            except Exception as e:
                logger.error(f"Error loading model from {model_path}: {e}")

    if not app.state.model_loaded:
        logger.warning("No pre-trained model file found. Serving in uninitialized mode until first training run.")

    yield
    # Shutdown
    logger.info("Shutting down Homelab MLOps Serving Engine...")
    app.state.model = None

app = FastAPI(
    title="Homelab MLOps Anomaly Detection Engine",
    description="Real-time telemetry inference and anomaly detection microservice with Prometheus telemetry.",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/docs")

@app.get("/health", response_model=HealthResponse)
async def health():
    if not app.state.model_loaded:
        return HealthResponse(
            status="ok",
            model_loaded=False,
            model_type="none",
            model_version="none",
            last_trained="none"
        )

    return HealthResponse(
        status="ok",
        model_loaded=True,
        model_type=app.state.metadata.get("model_type", "IsolationForest"),
        model_version=str(app.state.metadata.get("mlflow_run_id", "local-v1")),
        last_trained=str(app.state.metadata.get("trained_at", "recent"))
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict(data: TelemetryInput):
    if not app.state.model_loaded or app.state.model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded. Train a model first.")

    start_time = time.time()
    try:
        # Prepare snapshot dict
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

        # Prepare feature vector based on expected columns
        feature_columns = app.state.metadata.get("feature_columns", app.state.config.FEATURE_COLUMNS)
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

        # In sklearn anomaly detection, -1 is anomaly, 1 is normal
        is_anomaly = (label == -1)

        # Update metrics
        PREDICTION_COUNT.labels(result="anomaly" if is_anomaly else "normal").inc()
        ANOMALY_SCORE.set(score)

        latency = time.time() - start_time
        PREDICTION_LATENCY.observe(latency)

        return PredictionResponse(
            anomaly=is_anomaly,
            is_anomaly=is_anomaly,
            anomaly_score=score,
            timestamp=data.timestamp,
            model_type=app.state.metadata.get("model_type", "IsolationForest"),
            model_version=str(app.state.metadata.get("mlflow_run_id", "local-v1"))
        )
    except Exception as e:
        logger.error(f"Error during prediction: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

if __name__ == "__main__":
    import uvicorn
    config = Config()
    uvicorn.run(app, host=config.API_HOST, port=config.API_PORT)
