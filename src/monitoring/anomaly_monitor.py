"""
Automated Anomaly Monitor & Incident Alerting Daemon.
Scans latest DuckDB telemetry snapshots, invokes the prediction API, and logs GitHub Issues.
"""
import os
import time
import json
import logging
from typing import Dict, Any, Optional

import requests
import duckdb

from src.config import Config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class AnomalyMonitor:
    """Daemon that continuously queries recent telemetry and posts alerts on anomaly detection."""

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.api_url = self.config.API_URL

    def run_check(self) -> Dict[str, Any]:
        """Runs a single telemetry evaluation pass against the serving API."""
        logger.info("Executing telemetry anomaly health check...")
        result = {"status": "success", "anomaly_detected": False}

        try:
            db_path = self.config.TELEMETRY_DB_PATH
            if not os.path.exists(db_path):
                logger.warning(f"Telemetry DB not found at {db_path}")
                result["status"] = "no_data"
                return result

            conn = duckdb.connect(db_path, read_only=True)
            tables = [t[0] for t in conn.execute("SHOW TABLES").fetchall()]
            if "telemetry_snapshots" not in tables:
                conn.close()
                logger.warning("Table 'telemetry_snapshots' not found.")
                result["status"] = "no_data"
                return result

            df = conn.execute("SELECT * FROM telemetry_snapshots ORDER BY timestamp DESC LIMIT 1").df()
            conn.close()

            if df.empty:
                logger.warning("No telemetry snapshots found in database.")
                result["status"] = "no_data"
                return result

            row = df.iloc[0]

            # Reconstruct the snapshot from DuckDB flat columns
            ts_str = str(row["timestamp"])
            snapshot = {
                "timestamp": ts_str,
                "system": {
                    "cpu_percent": float(row.get("cpu_percent", 0.0)),
                    "ram_total_gb": float(row.get("ram_total_gb", 32.0)),
                    "ram_used_gb": float(row.get("ram_used_gb", 8.0)),
                    "ram_percent": float(row.get("ram_percent", 0.0)),
                    "disk_total_gb": float(row.get("disk_total_gb", 1000.0)),
                    "disk_used_gb": float(row.get("disk_used_gb", 200.0)),
                    "disk_percent": float(row.get("disk_percent", 0.0))
                },
                "disk_io": {
                    "read_bytes": 0,
                    "write_bytes": 0
                },
                "status": str(row.get("status", "HEALTHY"))
            }

            # Query the serving API
            logger.info(f"Sending telemetry snapshot to {self.api_url}")
            response = requests.post(self.api_url, json=snapshot, timeout=10)

            if response.status_code == 200:
                prediction = response.json()
                logger.info(f"Prediction received: {prediction}")

                if prediction.get("anomaly", False) or prediction.get("is_anomaly", False):
                    result["anomaly_detected"] = True
                    self.create_github_issue(prediction, snapshot)
            else:
                logger.error(f"Serving API returned status {response.status_code}: {response.text}")
                result["status"] = "api_error"

        except Exception as e:
            logger.error(f"Error during anomaly check: {e}", exc_info=True)
            result["status"] = "error"
            result["error"] = str(e)

        return result

    def create_github_issue(self, prediction: Dict[str, Any], snapshot: Dict[str, Any]):
        """Dispatches an incident alert issue to GitHub via REST API."""
        token = self.config.GITHUB_TOKEN
        repo = self.config.GITHUB_REPO

        if not token or not repo:
            logger.info("GITHUB_TOKEN or GITHUB_REPO not configured. Skipping GitHub issue dispatch.")
            return

        timestamp = prediction.get("timestamp", "unknown")
        title = f"🚨 Homelab Anomaly Detected — {timestamp}"

        headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github.v3+json"
        }

        # Prevent duplicate issues
        try:
            search_url = f"https://api.github.com/repos/{repo}/issues?state=open"
            issues_resp = requests.get(search_url, headers=headers, timeout=10)
            if issues_resp.status_code == 200:
                open_issues = issues_resp.json()
                if any(issue.get("title") == title for issue in open_issues):
                    logger.info("GitHub Issue for this incident already exists, skipping duplicate.")
                    return
        except Exception as e:
            logger.warning(f"Failed to check duplicate GitHub issues: {e}")

        body = f"""# 🚨 Automated Homelab Incident Alert

An operational anomaly was detected on the **UGREEN NAS** cluster by the `{prediction.get('model_type', 'IsolationForest')}` model.

---

### 📊 Model Inference
- **Anomaly Status**: `FLAGGED ⚠️`
- **Anomaly Score**: `{prediction.get('anomaly_score', 'N/A')}`
- **Model Version**: `{prediction.get('model_version', 'local-v1')}`
- **Event Timestamp**: `{timestamp}`

---

### 🔍 System Telemetry Snapshot
```json
{json.dumps(snapshot, indent=2)}
```

---

### 🛠️ Remediation Playbook
1. Check running Docker containers (`docker ps`) for runaway processes.
2. Review disk I/O spikes or memory leaks in Grafana (`http://<nas-ip>:3000`).
3. Verify temperatures and fan speeds via UGREEN control dashboard.
"""

        issue_data = {
            "title": title,
            "body": body,
            "labels": ["anomaly", "automated-alert", "homelab"]
        }

        try:
            url = f"https://api.github.com/repos/{repo}/issues"
            resp = requests.post(url, headers=headers, json=issue_data, timeout=10)
            if resp.status_code in (200, 201):
                logger.info(f"Successfully created GitHub issue: {resp.json().get('html_url')}")
            else:
                logger.error(f"Failed to create GitHub issue: {resp.status_code} - {resp.text}")
        except Exception as e:
            logger.error(f"Error creating GitHub issue: {e}")

    def start_daemon(self):
        """Starts continuous polling loop."""
        logger.info(f"Starting AnomalyMonitor daemon (interval: {self.config.MONITOR_INTERVAL_SECONDS}s)...")
        while True:
            try:
                self.run_check()
            except Exception as e:
                logger.error(f"Unhandled exception in daemon iteration: {e}")

            time.sleep(self.config.MONITOR_INTERVAL_SECONDS)


if __name__ == "__main__":
    monitor = AnomalyMonitor()
    monitor.start_daemon()
