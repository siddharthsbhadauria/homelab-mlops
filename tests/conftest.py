"""
Shared pytest fixtures for homelab-mlops test suite.
"""
import os
import json
import pytest
import duckdb
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import IsolationForest

from src.config import Config

FEATURE_COLUMNS = [
    "cpu_percent", "ram_percent", "disk_percent", 
    "cpu_rolling_mean_1h", "cpu_rolling_std_1h", 
    "ram_rolling_mean_1h", "ram_rolling_std_1h", 
    "disk_rolling_mean_1h", "disk_rolling_std_1h", 
    "cpu_rolling_mean_6h", "cpu_rolling_std_6h", 
    "ram_rolling_mean_6h", "ram_rolling_std_6h", 
    "cpu_rate_of_change", "ram_rate_of_change", "disk_rate_of_change", 
    "hour_of_day", "day_of_week"
]

@pytest.fixture
def sample_snapshot():
    return {
        "timestamp": "2026-08-15T22:04:31+01:00",
        "system": {
            "cpu_percent": 15.2,
            "ram_total_gb": 32.0,
            "ram_used_gb": 16.0,
            "ram_percent": 50.0,
            "disk_total_gb": 1000.0,
            "disk_used_gb": 500.0,
            "disk_percent": 50.0
        },
        "disk_io": {
            "read_bytes": 1024,
            "write_bytes": 2048
        },
        "status": "HEALTHY"
    }

@pytest.fixture
def sample_features_df():
    np.random.seed(42)
    n_rows = 200
    data = {
        "timestamp": [f"2026-08-15T{i%24:02d}:00:00Z" for i in range(n_rows)],
        "cpu_percent": np.random.uniform(5, 80, n_rows),
        "ram_percent": np.random.uniform(20, 60, n_rows),
        "disk_percent": np.random.uniform(15, 40, n_rows),
        "cpu_rolling_mean_1h": np.random.uniform(10, 75, n_rows),
        "cpu_rolling_std_1h": np.random.uniform(1, 10, n_rows),
        "ram_rolling_mean_1h": np.random.uniform(25, 55, n_rows),
        "ram_rolling_std_1h": np.random.uniform(0.5, 5, n_rows),
        "disk_rolling_mean_1h": np.random.uniform(15, 40, n_rows),
        "disk_rolling_std_1h": np.random.uniform(0.1, 2, n_rows),
        "cpu_rolling_mean_6h": np.random.uniform(15, 70, n_rows),
        "cpu_rolling_std_6h": np.random.uniform(2, 12, n_rows),
        "ram_rolling_mean_6h": np.random.uniform(30, 50, n_rows),
        "ram_rolling_std_6h": np.random.uniform(1, 6, n_rows),
        "cpu_rate_of_change": np.random.normal(0, 2, n_rows),
        "ram_rate_of_change": np.random.normal(0, 1, n_rows),
        "disk_rate_of_change": np.random.normal(0, 0.5, n_rows),
        "hour_of_day": np.random.randint(0, 24, n_rows).astype(float),
        "day_of_week": np.random.randint(0, 7, n_rows).astype(float)
    }
    return pd.DataFrame(data)

@pytest.fixture
def tmp_duckdb(tmp_path):
    db_path = str(tmp_path / "test_telemetry.db")
    conn = duckdb.connect(db_path)
    conn.execute('''
        CREATE TABLE telemetry_snapshots (
            timestamp VARCHAR,
            cpu_percent DOUBLE,
            ram_used_gb DOUBLE,
            ram_total_gb DOUBLE,
            ram_percent DOUBLE,
            disk_used_gb DOUBLE,
            disk_total_gb DOUBLE,
            disk_percent DOUBLE,
            status VARCHAR
        )
    ''')
    
    # Insert 200 rows of valid synthetic data
    for i in range(200):
        ts = f"2026-08-15T{i%24:02d}:{i%60:02d}:00Z"
        conn.execute(f'''
            INSERT INTO telemetry_snapshots VALUES (
                '{ts}', {np.random.uniform(5, 80)}, 16.0, 32.0, {np.random.uniform(20, 60)}, 
                500.0, 1000.0, {np.random.uniform(15, 40)}, 'HEALTHY'
            )
        ''')
    conn.close()
    return db_path

@pytest.fixture
def tmp_model_dir(tmp_path, sample_features_df):
    model_dir = tmp_path / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Train and dump quick models
    clf = IsolationForest(n_estimators=10, contamination=0.05, random_state=42)
    X = sample_features_df[FEATURE_COLUMNS]
    clf.fit(X)
    
    primary_path = model_dir / "primary_model.joblib"
    joblib.dump(clf, primary_path)
    
    metadata = {
        "model_type": "IsolationForest",
        "algorithm": "IsolationForest",
        "features": FEATURE_COLUMNS,
        "feature_columns": FEATURE_COLUMNS,
        "contamination": 0.05,
        "mlflow_run_id": "test-run-123",
        "trained_at": "2026-08-15T22:00:00Z"
    }
    with open(model_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f)
        
    return str(model_dir)

@pytest.fixture
def mock_config(tmp_path, tmp_duckdb, tmp_model_dir):
    cfg = Config()
    cfg.TELEMETRY_DB_PATH = tmp_duckdb
    cfg.MODEL_DIR = tmp_model_dir
    cfg.FEATURE_OUTPUT_DIR = str(tmp_path / "features")
    os.makedirs(cfg.FEATURE_OUTPUT_DIR, exist_ok=True)
    return cfg
