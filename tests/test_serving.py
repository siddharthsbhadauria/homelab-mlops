"""
Unit tests for FastAPI model serving and health endpoints.
"""
from fastapi.testclient import TestClient
from unittest.mock import patch
from src.serving.app import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "model_loaded" in data


@patch('src.features.feature_engineer.FeatureEngineer.compute_single_point_features')
def test_predict_endpoint_valid_input(mock_compute, sample_snapshot):
    from tests.conftest import FEATURE_COLUMNS
    mock_features = {col: 0.0 for col in FEATURE_COLUMNS}
    mock_compute.return_value = mock_features

    # Mock model in app.state
    class MockModel:
        def predict(self, X):
            import numpy as np
            return np.array([1])  # 1 = normal, -1 = anomaly

        def decision_function(self, X):
            import numpy as np
            return np.array([0.15])

    app.state.model = MockModel()
    app.state.model_loaded = True
    app.state.metadata = {"model_type": "IsolationForest", "mlflow_run_id": "test-run"}

    response = client.post("/predict", json=sample_snapshot)
    assert response.status_code == 200
    data = response.json()
    assert "anomaly" in data or "is_anomaly" in data
    assert "anomaly_score" in data
    assert data["anomaly"] is False


def test_predict_endpoint_invalid_input():
    invalid_data = {"random": "garbage"}
    response = client.post("/predict", json=invalid_data)
    assert response.status_code == 422


def test_metrics_endpoint():
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]
