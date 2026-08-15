"""
Configuration for homelab-mlops.
"""
import os
from typing import List

class Config:
    TELEMETRY_DB_PATH: str = os.getenv("TELEMETRY_DB_PATH", "/data/auto-datapulse/telemetry.duckdb")
    FEATURE_OUTPUT_DIR: str = os.getenv("FEATURE_OUTPUT_DIR", "/data/homelab-mlops/features")
    MODEL_DIR: str = os.getenv("MODEL_DIR", "/data/homelab-mlops/models")
    MLFLOW_TRACKING_URI: str = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    EXPERIMENT_NAME: str = "homelab-anomaly-detection"
    CONTAMINATION: float = float(os.getenv("CONTAMINATION", "0.05"))
    ROLLING_WINDOW_SHORT: int = 4   # 4 samples = 1 hour at 15-min intervals
    ROLLING_WINDOW_LONG: int = 24   # 24 samples = 6 hours
    MIN_SAMPLES_FOR_TRAINING: int = 96  # 24 hours of data minimum
    FEATURE_COLUMNS: List[str] = [
        "cpu_percent", "ram_percent", "disk_percent", 
        "cpu_rolling_mean_1h", "cpu_rolling_std_1h", 
        "ram_rolling_mean_1h", "ram_rolling_std_1h", 
        "disk_rolling_mean_1h", "disk_rolling_std_1h", 
        "cpu_rolling_mean_6h", "cpu_rolling_std_6h", 
        "ram_rolling_mean_6h", "ram_rolling_std_6h", 
        "cpu_rate_of_change", "ram_rate_of_change", "disk_rate_of_change", 
        "hour_of_day", "day_of_week"
    ]
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", "8000"))
    API_URL: str = os.getenv("API_URL", f"http://{os.getenv('API_HOST', 'api')}:{os.getenv('API_PORT', '8000')}/predict")
    MONITOR_INTERVAL_SECONDS: int = int(os.getenv("MONITOR_INTERVAL_SECONDS", "900"))
    PIPELINE_INTERVAL_SECONDS: int = int(os.getenv("PIPELINE_INTERVAL_SECONDS", "21600"))
    GITHUB_TOKEN: str = os.getenv("GITHUB_TOKEN", "")
    GITHUB_REPO: str = os.getenv("GITHUB_REPO", "siddharthsbhadauria/homelab-mlops")

config = Config()
