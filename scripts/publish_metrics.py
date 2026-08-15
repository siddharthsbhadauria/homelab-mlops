"""
Metrics Publisher Engine.
Extracts model metadata and training evaluation stats, formats a JSON operational report,
and commits it to the GitHub repository using the GitHub REST API.
"""
import os
import json
import base64
import logging
from datetime import datetime, timezone
from typing import Optional

import requests

from src.config import Config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def publish_model_metrics(config: Optional[Config] = None) -> bool:
    """Publishes latest model performance and metadata to GitHub repository."""
    cfg = config or Config()
    token = cfg.GITHUB_TOKEN or os.environ.get("GITHUB_TOKEN", "")
    repo = cfg.GITHUB_REPO or os.environ.get("GITHUB_REPO", "siddharthsbhadauria/homelab-mlops")

    if not token or not repo:
        logger.warning("[METRICS NOTICE] GITHUB_TOKEN or GITHUB_REPO not set. Skipping GitHub commit.")
        return False

    metadata_path = os.path.join(cfg.MODEL_DIR, "metadata.json")
    if not os.path.exists(metadata_path):
        logger.warning(f"Metadata file not found at {metadata_path}")
        return False

    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
    except Exception as e:
        logger.error(f"Error reading model metadata: {e}")
        return False

    now_utc = datetime.now(timezone.utc)
    date_str = now_utc.strftime("%Y-%m-%d")

    report = {
        "last_updated": now_utc.isoformat(),
        "model_type": metadata.get("model_type", "IsolationForest"),
        "algorithm": metadata.get("algorithm", "IsolationForest"),
        "anomaly_rate": metadata.get("anomaly_rate", 0.0),
        "n_samples_trained": metadata.get("n_samples", 0),
        "features_count": len(metadata.get("features", metadata.get("feature_columns", []))),
        "mlflow_run_id": metadata.get("mlflow_run_id", "local"),
        "status": "HEALTHY"
    }

    file_path = f"REPORTS/model_metrics_{date_str}.json"
    url = f"https://api.github.com/repos/{repo}/contents/{file_path}"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github.v3+json"
    }

    # Check for existing file SHA
    sha = None
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        if resp.status_code == 200:
            sha = resp.json().get("sha")
    except Exception as e:
        logger.warning(f"Could not check existing file SHA: {e}")

    content_str = json.dumps(report, indent=2)
    content_b64 = base64.b64encode(content_str.encode("utf-8")).decode("utf-8")

    payload = {
        "message": f"chore(mlops): automated daily model metrics update [{date_str}]",
        "content": content_b64
    }
    if sha:
        payload["sha"] = sha

    try:
        put_resp = requests.put(url, headers=headers, json=payload, timeout=15)
        if put_resp.status_code in (200, 201):
            logger.info(f"[GIT API SUCCESS] Successfully committed {file_path} to {repo}")
            return True
        else:
            logger.error(f"[GIT API ERROR] Status {put_resp.status_code}: {put_resp.text}")
            return False
    except Exception as e:
        logger.error(f"Error publishing metrics: {e}")
        return False


def main():
    publish_model_metrics()


if __name__ == "__main__":
    main()
