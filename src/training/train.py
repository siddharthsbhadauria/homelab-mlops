"""
Training module for homelab-mlops.
Trains Isolation Forest and Local Outlier Factor models, logs runs to MLflow, and registers the primary model.
"""
import os
import json
import time
import logging
from datetime import UTC, datetime
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

    HOLDOUT_FRACTION = 0.1  # newest 10% of rows is never trained on during model selection

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()

    @staticmethod
    def inject_anomalies(holdout: pd.DataFrame) -> pd.DataFrame:
        """Copies of real holdout rows with known failures: CPU pinned, RAM pinned, disk jumping 20 points."""
        cpu = holdout.copy()
        cpu["cpu_rate_of_change"] = 100.0 - cpu["cpu_percent"]
        cpu["cpu_percent"] = 100.0

        ram = holdout.copy()
        ram["ram_rate_of_change"] = 98.0 - ram["ram_percent"]
        ram["ram_percent"] = 98.0

        disk = holdout.copy()
        disk["disk_percent"] = (disk["disk_percent"] + 20.0).clip(upper=100.0)
        disk["disk_rate_of_change"] = 20.0

        return pd.concat([cpu, ram, disk], ignore_index=True)

    @staticmethod
    def holdout_scores(model: Any, holdout: pd.DataFrame, injected: pd.DataFrame) -> Dict[str, float]:
        """False alarms on unseen real data, recall on injected failures, and score = recall - false alarms."""
        false_alarm_rate = float((model.predict(holdout) == -1).mean())
        recall = float((model.predict(injected) == -1).mean())
        return {"false_alarm_rate": false_alarm_rate, "recall": recall, "score": recall - false_alarm_rate}

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

        def make_models():
            return {
                "IsolationForest": IsolationForest(
                    n_estimators=200, contamination=self.config.CONTAMINATION, random_state=42
                ),
                # novelty=True so the fitted model can score new rows online
                "LocalOutlierFactor": LocalOutlierFactor(
                    n_neighbors=20, contamination=self.config.CONTAMINATION, novelty=True
                ),
            }

        # 1. Model selection on data the models never saw. Rows arrive oldest first from the feature SQL,
        # so the newest 10% is a time-based holdout. Scoring on training data favoured LOF, because with
        # novelty=True every training row is its own nearest neighbour.
        split = int(len(X) * (1 - self.HOLDOUT_FRACTION))
        X_train, X_holdout = X.iloc[:split], X.iloc[split:]
        injected = self.inject_anomalies(X_holdout)
        holdout = {}
        for name, model in make_models().items():
            logger.info(f"Fitting {name} on the first {len(X_train)} rows for selection...")
            model.fit(X_train)
            holdout[name] = self.holdout_scores(model, X_holdout, injected)
            logger.info(
                f"{name}: recall {holdout[name]['recall']:.3f} on {len(injected)} injected failures, "
                f"false alarms {holdout[name]['false_alarm_rate']:.4f} on {len(X_holdout)} unseen rows"
            )
        # Highest recall minus false alarms wins; a tie goes to Isolation Forest (listed first).
        best_model_name = max(holdout, key=lambda n: holdout[n]["score"])

        # 2. Refit both on all rows for serving.
        models = make_models()
        for model in models.values():
            model.fit(X)
        iso_forest, lof = models["IsolationForest"], models["LocalOutlierFactor"]
        iso_metrics = {**self._evaluate_model(iso_forest, X, "IsolationForest"), **holdout["IsolationForest"]}
        lof_metrics = {**self._evaluate_model(lof, X, "LocalOutlierFactor"), **holdout["LocalOutlierFactor"]}
        best_model = models[best_model_name]
        best_anomaly_rate = (iso_metrics if best_model_name == "IsolationForest" else lof_metrics)["anomaly_rate"]

        training_duration = time.time() - start_time

        logger.info(
            f"Selected {best_model_name} as primary model "
            f"(holdout score {holdout[best_model_name]['score']:.3f}, anomaly rate {best_anomaly_rate:.4f})"
        )

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
            "trained_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "n_samples": int(len(X)),
            "features": list(self.config.FEATURE_COLUMNS),
            "feature_columns": list(self.config.FEATURE_COLUMNS),
            "anomaly_rate": float(best_anomaly_rate),
            "holdout_recall": holdout[best_model_name]["recall"],
            "holdout_false_alarm_rate": holdout[best_model_name]["false_alarm_rate"],
            "selection": {name: holdout[name] for name in holdout},
            "mlflow_run_id": "local-run"
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
                    "isolation_forest_holdout_recall": iso_metrics["recall"],
                    "isolation_forest_holdout_false_alarm_rate": iso_metrics["false_alarm_rate"],
                    "lof_holdout_recall": lof_metrics["recall"],
                    "lof_holdout_false_alarm_rate": lof_metrics["false_alarm_rate"],
                    "training_duration_seconds": training_duration
                })

                mlflow.log_artifact(iso_path, "models")
                mlflow.log_artifact(lof_path, "models")
                mlflow.log_artifact(primary_path, "models")

                feat_json_path = os.path.join(self.config.MODEL_DIR, "feature_columns.json")
                with open(feat_json_path, "w", encoding="utf-8") as f:
                    json.dump(self.config.FEATURE_COLUMNS, f, indent=2)
                mlflow.log_artifact(feat_json_path, "metadata")

                mlflow.set_tag("model_type", best_model_name)
                mlflow.set_tag("phase", "cpu-only")

                if run and hasattr(run, "info") and hasattr(run.info, "run_id"):
                    metadata["mlflow_run_id"] = str(run.info.run_id)

        except Exception as e:
            logger.warning(f"MLflow tracking skipped / unavailable: {e}")

        meta_path = os.path.join(self.config.MODEL_DIR, "metadata.json")
        with open(meta_path, "w", encoding="utf-8") as f:
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
