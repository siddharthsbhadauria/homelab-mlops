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
        self.consecutive_anomalies: int = 0
        self.last_alert_time: float = 0.0

    def is_actionable_anomaly(self, prediction: Dict[str, Any], snapshot: Dict[str, Any]) -> bool:
        """
        Determines whether an inference result is a true actionable anomaly.
        Applies score margin threshold and safe resource guardrails to eliminate false positives.
        """
        is_flagged = prediction.get("anomaly", False) or prediction.get("is_anomaly", False)
        if not is_flagged:
            return False

        score = float(prediction.get("anomaly_score", 0.0))

        # 1. Score threshold filter: Ignore marginal boundary points (e.g. -0.01)
        if score > self.config.ANOMALY_SCORE_THRESHOLD:
            logger.info(
                f"Model flagged anomaly but score {score:.4f} is above threshold {self.config.ANOMALY_SCORE_THRESHOLD:.4f}. "
                "Classifying as normal operational variance."
            )
            return False

        # 2. Resource safety guardrails: Don't panic if resource utilization is safely low
        sys_data = snapshot.get("system", {}) if isinstance(snapshot.get("system"), dict) else {}
        cpu = float(sys_data.get("cpu_percent", snapshot.get("cpu_percent", 0.0)))
        ram = float(sys_data.get("ram_percent", snapshot.get("ram_percent", 0.0)))
        disk = float(sys_data.get("disk_percent", snapshot.get("disk_percent", 0.0)))

        is_resource_safe = (
            cpu < self.config.RESOURCE_GUARD_CPU_PERCENT
            and ram < self.config.RESOURCE_GUARD_RAM_PERCENT
            and disk < self.config.RESOURCE_GUARD_DISK_PERCENT
        )

        # If system resources are within safe limits and anomaly is not severe (score > -0.25), suppress alert
        if is_resource_safe and score > -0.25:
            logger.info(
                f"System resources healthy (CPU: {cpu}%, RAM: {ram}%, Disk: {disk}%) and score {score:.4f} is moderate. "
                "Suppressing false alert."
            )
            return False

        return True

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

                if self.is_actionable_anomaly(prediction, snapshot):
                    self.consecutive_anomalies += 1
                    logger.warning(
                        f"Actionable anomaly detected ({self.consecutive_anomalies}/{self.config.CONSECUTIVE_ANOMALIES_REQUIRED} "
                        f"consecutive checks, score: {prediction.get('anomaly_score')})"
                    )

                    if self.consecutive_anomalies >= self.config.CONSECUTIVE_ANOMALIES_REQUIRED:
                        result["anomaly_detected"] = True
                        self.create_or_update_github_issue(prediction, snapshot)
                else:
                    if self.consecutive_anomalies > 0:
                        logger.info("Telemetry returned to normal baseline. Resetting consecutive anomaly counter.")
                    self.consecutive_anomalies = 0

            else:
                logger.error(f"Serving API returned status {response.status_code}: {response.text}")
                result["status"] = "api_error"

        except Exception as e:
            logger.error(f"Error during anomaly check: {e}", exc_info=True)
            result["status"] = "error"
            result["error"] = str(e)

        return result

    def create_or_update_github_issue(self, prediction: Dict[str, Any], snapshot: Dict[str, Any]):
        """Dispatches or updates an incident alert issue on GitHub with deduplication and cooldown."""
        token = self.config.GITHUB_TOKEN
        repo = self.config.GITHUB_REPO

        if not token or not repo:
            logger.info("GITHUB_TOKEN or GITHUB_REPO not configured. Skipping GitHub issue dispatch.")
            return

        timestamp = prediction.get("timestamp", "unknown")
        headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "Homelab-Anomaly-Monitor"
        }

        # Check for active open incident issues to avoid duplicate spam
        existing_issue_id = None
        try:
            search_url = f"https://api.github.com/repos/{repo}/issues?state=open&labels=automated-alert"
            issues_resp = requests.get(search_url, headers=headers, timeout=10)
            if issues_resp.status_code == 200:
                open_issues = issues_resp.json()
                if open_issues:
                    existing_issue_id = open_issues[0].get("number")
        except Exception as e:
            logger.warning(f"Failed to query existing open issues: {e}")

        # If an open incident already exists, append a comment timeline update instead of creating a new issue
        if existing_issue_id:
            now = time.time()
            cooldown_seconds = self.config.ALERT_COOLDOWN_HOURS * 3600
            if (now - self.last_alert_time) < 1800:  # Comment max once every 30 mins
                logger.info(f"Open incident #{existing_issue_id} exists. Update suppressed within comment interval.")
                return

            comment_body = f"""### ⏱️ Incident Update — `{timestamp}`
- **Consecutive Anomalous Checks**: `{self.consecutive_anomalies}`
- **Anomaly Score**: `{prediction.get('anomaly_score', 'N/A')}`
- **Model**: `{prediction.get('model_type', 'IsolationForest')}`

```json
{json.dumps(snapshot, indent=2)}
```
"""
            try:
                comment_url = f"https://api.github.com/repos/{repo}/issues/{existing_issue_id}/comments"
                c_resp = requests.post(comment_url, headers=headers, json={"body": comment_body}, timeout=10)
                if c_resp.status_code in (200, 201):
                    logger.info(f"Appended status update to existing incident issue #{existing_issue_id}")
                    self.last_alert_time = now
            except Exception as e:
                logger.error(f"Failed to comment on existing incident #{existing_issue_id}: {e}")
            return

        # Otherwise create a new incident issue
        title = f"🚨 Homelab Incident Alert — Anomaly Detected ({timestamp})"
        body = f"""# 🚨 Automated Homelab Incident Alert

An operational anomaly was confirmed on the **UGREEN NAS** cluster by the `{prediction.get('model_type', 'IsolationForest')}` model after **{self.consecutive_anomalies} consecutive intervals**.

---

### 📊 Model Inference
- **Anomaly Status**: `CONFIRMED ⚠️`
- **Anomaly Score**: `{prediction.get('anomaly_score', 'N/A')}` (Threshold: `{self.config.ANOMALY_SCORE_THRESHOLD}`)
- **Consecutive Detections**: `{self.consecutive_anomalies}`
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
2. Review disk I/O spikes or memory leaks in Grafana (`http://<nas-ip>:3030`).
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
                self.last_alert_time = time.time()
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
