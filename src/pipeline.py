"""
Pipeline Orchestrator Entry Point
Runs feature engineering, model training, and monitoring as a daemon.
"""

import os
import sys
import time
import logging
import threading

from src.config import Config
from src.features.feature_engineer import FeatureEngineer
from src.training.train import AnomalyTrainer
from src.monitoring.anomaly_monitor import AnomalyMonitor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run_monitor_daemon(config: Config):
    """Runs the anomaly monitor every MONITOR_INTERVAL_SECONDS (default 15 min)."""
    monitor = AnomalyMonitor(config)
    logger.info(f"Starting anomaly monitor daemon (interval: {config.MONITOR_INTERVAL_SECONDS}s)...")
    while True:
        try:
            result = monitor.run_check()
            logger.info(f"Monitor check result: {result}")
        except Exception as e:
            logger.error(f"Error in anomaly monitor: {e}", exc_info=True)
        time.sleep(config.MONITOR_INTERVAL_SECONDS)


def run_pipeline_daemon():
    """Runs the main pipeline: feature extraction → training → metrics publishing."""
    config = Config()
    pipeline_interval = int(os.environ.get("PIPELINE_INTERVAL_SECONDS", 21600))

    logger.info(f"Starting pipeline daemon (interval: {pipeline_interval}s / {pipeline_interval // 3600}h)...")

    engineer = FeatureEngineer(config)
    trainer = AnomalyTrainer(config)

    while True:
        try:
            logger.info("=" * 60)
            logger.info("Starting pipeline cycle...")

            # 1. Feature Engineering
            logger.info("[1/3] Extracting features from auto-datapulse telemetry...")
            df_features = engineer.extract_features()

            if df_features is None or df_features.empty:
                logger.warning("No features extracted. Skipping this cycle.")
                time.sleep(pipeline_interval)
                continue

            # Save features to Parquet
            features_path = os.path.join(config.FEATURE_OUTPUT_DIR, "features.parquet")
            engineer.save_features(df_features, features_path)
            logger.info(f"Saved {len(df_features)} feature rows to {features_path}")

            # 2. Check data sufficiency and train
            if len(df_features) < config.MIN_SAMPLES_FOR_TRAINING:
                logger.warning(
                    f"Not enough data for training: {len(df_features)} samples, "
                    f"need {config.MIN_SAMPLES_FOR_TRAINING}. Skipping training."
                )
            else:
                logger.info("[2/3] Training anomaly detection models...")
                results = trainer.train(features_path)
                logger.info(
                    f"Training complete — best model: {results['best_model']}, "
                    f"anomaly rate: {results['anomaly_rate']:.4f}"
                )

            # 3. Publish metrics to GitHub
            logger.info("[3/3] Publishing model metrics...")
            try:
                from scripts.publish_metrics import publish_model_metrics
                publish_model_metrics(config)
            except Exception as e:
                logger.warning(f"Could not publish metrics to GitHub: {e}")

            logger.info("Pipeline cycle completed successfully.")

        except Exception as e:
            logger.error(f"Error in pipeline cycle: {e}", exc_info=True)

        logger.info(f"Sleeping for {pipeline_interval}s ({pipeline_interval // 3600}h)...")
        time.sleep(pipeline_interval)


if __name__ == "__main__":
    config = Config()

    # Start the anomaly monitor in a background thread
    monitor_thread = threading.Thread(target=run_monitor_daemon, args=(config,), daemon=True)
    monitor_thread.start()

    # Run the main pipeline in the foreground
    run_pipeline_daemon()
