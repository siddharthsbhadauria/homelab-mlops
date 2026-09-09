"""
Unit tests for AnomalyMonitor and alerting debounce logic.
"""
from unittest.mock import patch, MagicMock
from src.config import Config
from src.monitoring.anomaly_monitor import AnomalyMonitor


def test_is_actionable_anomaly_filters_marginal_score(mock_config, sample_snapshot):
    monitor = AnomalyMonitor(mock_config)
    
    # Marginal anomaly score (e.g. -0.02 when threshold is -0.10)
    prediction = {
        "anomaly": True,
        "is_anomaly": True,
        "anomaly_score": -0.02,
        "model_type": "IsolationForest"
    }
    
    assert monitor.is_actionable_anomaly(prediction, sample_snapshot) is False


def test_is_actionable_anomaly_passes_severe_anomaly(mock_config, sample_snapshot):
    monitor = AnomalyMonitor(mock_config)
    
    # Severe anomaly score
    prediction = {
        "anomaly": True,
        "is_anomaly": True,
        "anomaly_score": -0.30,
        "model_type": "IsolationForest"
    }
    
    assert monitor.is_actionable_anomaly(prediction, sample_snapshot) is True


def test_consecutive_debounce_counter(mock_config):
    monitor = AnomalyMonitor(mock_config)
    assert monitor.consecutive_anomalies == 0
    
    prediction = {
        "anomaly": True,
        "is_anomaly": True,
        "anomaly_score": -0.25,
        "model_type": "IsolationForest"
    }
    
    # Resource safe snapshot but severe score -> actionable
    snapshot = {
        "timestamp": "2026-08-15T22:00:00Z",
        "system": {"cpu_percent": 85.0, "ram_percent": 90.0, "disk_percent": 50.0}
    }
    
    assert monitor.is_actionable_anomaly(prediction, snapshot) is True
