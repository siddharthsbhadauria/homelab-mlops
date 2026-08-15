"""
Training module for homelab-mlops.
Trains Isolation Forest and Local Outlier Factor models, logs runs to MLflow, and registers the primary model.
"""
import os
import json
import time
import logging
from datetime import datetime
from typing import Optional, Dict, Any, Union

import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import silhouette_score
import mlflow

from src.config import Config

logger = logging.getLogger(__name__)


class AnomalyTrainer:
    """Trains, benchmarks, and persists anomaly detection models."""

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()

    def _evaluate_model(self, model: Any, X: pd.DataFrame, model_name: str) -> Dict[str, Any]:
        """Helper to compute predictions and evaluation metrics for a candidate model."""
        predictions = model.predict(X)
        anomalies = (predictions == -1).astype(int)

        anomaly_rate = float(anomalies.mean())

        metrics = {
            "anomaly_rate": anomaly_rate,
            "total_anomalies": int(anomalies.sum())
        }

        if 0 < anomalies.sum() < len(anomalies):
            sample_size = min(len(X), 5000)
            if sample_size < len(X):
                idx = np.random.choice(len(X), sample_size, replace=False)
                X_sample = X.iloc[idx]
                labels_sample = anomalies[idx]
                if 0 < labels_sample.sum() < len(labels_sample):
                    metrics["silhouette_score"] = float(silhouette_score(X_sample, labels_sample))
            else:
                metrics["silhouette_score"] = float(silhouette_score(X, anomalies))

        return metrics

    def train(self, features: Optional[Union[str, pd.DataFrame]] = None) -> Dict[str, Any]:
        """
        Trains IsolationForest and LocalOutlierFactor models, logs to MLflow,
        and saves the best performing model locally.
        """
        if isinstance(features, pd.DataFrame):
            df = features
        else:
            path = features or os.path.join(self.config.FEATURE_OUTPUT_DIR, "features.parquet")
            if not os.path.exists(path):
                raise FileNotFoundError(f"Features file not found at {path}")
            logger.info(f"Reading features from {path}")
            df = pd.read_parquet(path)

        if len(df) < self.config.MIN_SAMPLES_FOR_TRAINING:
            raise ValueError(
                f"Not enough samples for training. Found {len(df)}, "
                f"require at least {self.config.MIN_SAMPLES_FOR_TRAINING}."
            )

        # Subset required features
        X = df[self.config.FEATURE_COLUMNS]

        logger.info(f"Training on {len(X)} samples with {len(X.columns)} features.")

        start_time = time.time()

        # 1. Isolation Forest
        logger.info("Fitting Isolation Forest model...")
        iso_forest = IsolationForest(
            n_estimators=200,
            contamination=self.config.CONTAMINATION,
            random_state=42
        )
        iso_forest.fit(X)
        iso_metrics = self._evaluate_model(iso_forest, X, "IsolationForest")

        # 2. Local Outlier Factor (novelty=True for online inference)
        logger.info("Fitting Local Outlier Factor model...")
        lof = LocalOutlierFactor(
            n_neighbors=20,
            contamination=self.config.CONTAMINATION,
            novelty=True
        )
        lof.fit(X)
        lof_metrics = self._evaluate_model(lof, X, "LocalOutlierFactor")

        training_duration = time.time() - start_time

        # Select primary model
        if iso_metrics["anomaly_rate"] <= lof_metrics["anomaly_rate"]:
            best_model = iso_forest
            best_model_name = "IsolationForest"
            best_anomaly_rate = iso_metrics["anomaly_rate"]
        else:
            best_model = lof
            best_model_name = "LocalOutlierFactor"
            best_anomaly_rate = lof_metrics["anomaly_rate"]

        logger.info(f"Selected {best_model_name} as primary model (anomaly rate: {best_anomaly_rate:.4f})")

        # Persist models locally
        os.makedirs(self.config.MODEL_DIR, exist_ok=True)
        iso_path = os.path.join(self.config.MODEL_DIR, "isolation_forest.joblib")
        lof_path = os.path.join(self.config.MODEL_DIR, "local_outlier_factor.joblib")
        primary_path = os.path.join(self.config.MODEL_DIR, "primary_model.joblib")

        joblib.dump(iso_forest, iso_path)
        joblib.dump(lof, lof_path)
        joblib.dump(best_model, primary_path)

        metadata = {
            "model_type": best_model_name,
            "algorithm": best_model_name,
            "trained_at": datetime.utcnow().isoformat() + "Z",
            "n_samples": len(X),
            "features": self.config.FEATURE_COLUMNS,
            "feature_columns": self.config.FEATURE_COLUMNS,
            "anomaly_rate": best_anomaly_rate,
            "mlflow_run_id": None
        }

        # Log run to MLflow
        try:
            mlflow.set_tracking_uri(self.config.MLFLOW_TRACKING_URI)
            mlflow.set_experiment(self.config.EXPERIMENT_NAME)

            with mlflow.start_run() as run:
                mlflow.log_params({
                    "contamination": self.config.CONTAMINATION,
                    "n_estimators": 200,
                    "n_neighbors": 20,
                    "n_samples": len(X),
                    "n_features": len(X.columns),
                    "feature_columns": self.config.FEATURE_COLUMNS
                })

                mlflow.log_metrics({
                    "isolation_forest_anomaly_rate": iso_metrics["anomaly_rate"],
                    "lof_anomaly_rate": lof_metrics["anomaly_rate"],
                    "training_duration_seconds": training_duration
                })

                mlflow.log_artifact(iso_path, "models")
                mlflow.log_artifact(lof_path, "models")
                mlflow.log_artifact(primary_path, "models")

                feat_json_path = os.path.join(self.config.MODEL_DIR, "feature_columns.json")
                with open(feat_json_path, "w") as f:
                    json.dump(self.config.FEATURE_COLUMNS, f, indent=2)
                mlflow.log_artifact(feat_json_path, "metadata")

                mlflow.set_tag("model_type", best_model_name)
                mlflow.set_tag("phase", "cpu-only")

                metadata["mlflow_run_id"] = run.info.run_id

        except Exception as e:
            logger.warning(f"MLflow tracking skipped / unavailable: {e}")

        meta_path = os.path.join(self.config.MODEL_DIR, "metadata.json")
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)

        return {
            "best_model": best_model_name,
            "model_path": primary_path,
            "anomaly_rate": best_anomaly_rate,
            "metrics": {
                "isolation_forest": iso_metrics,
                "local_outlier_factor": lof_metrics,
                "training_duration_seconds": training_duration
            },
            "metadata": metadata
        }
