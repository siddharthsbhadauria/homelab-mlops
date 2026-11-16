"""
Unit tests for AnomalyTrainer module.
"""
import os
import json
import joblib
import numpy as np
import pytest
from unittest.mock import patch

from src.config import Config
from src.training.train import AnomalyTrainer


@patch('src.training.train.mlflow')
def test_train_returns_results_dict(mock_mlflow, mock_config, sample_features_df):
    trainer = AnomalyTrainer(mock_config)
    results = trainer.train(sample_features_df)

    assert isinstance(results, dict)
    assert "model_path" in results
    assert "metrics" in results
    assert "best_model" in results


@patch('src.training.train.mlflow')
def test_train_saves_model_file(mock_mlflow, mock_config, sample_features_df):
    trainer = AnomalyTrainer(mock_config)
    results = trainer.train(sample_features_df)

    assert os.path.exists(results["model_path"])
    assert results["model_path"].endswith(".joblib")


@patch('src.training.train.mlflow')
def test_train_saves_metadata(mock_mlflow, mock_config, sample_features_df):
    trainer = AnomalyTrainer(mock_config)
    results = trainer.train(sample_features_df)

    metadata_path = os.path.join(mock_config.MODEL_DIR, "metadata.json")
    assert os.path.exists(metadata_path)

    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
        assert "features" in metadata or "feature_columns" in metadata
        assert "model_type" in metadata or "algorithm" in metadata


@patch('src.training.train.mlflow')
def test_train_insufficient_data_raises(mock_mlflow, mock_config, sample_features_df):
    trainer = AnomalyTrainer(mock_config)
    tiny_df = sample_features_df.head(5)

    with pytest.raises(ValueError):
        trainer.train(tiny_df)


@patch('src.training.train.mlflow')
def test_trained_model_can_predict(mock_mlflow, mock_config, sample_features_df):
    trainer = AnomalyTrainer(mock_config)
    results = trainer.train(sample_features_df)

    model = joblib.load(results["model_path"])
    X = sample_features_df[Config.FEATURE_COLUMNS]
    predictions = model.predict(X)

    assert predictions.shape == (len(sample_features_df),)
    assert set(np.unique(predictions)).issubset({1, -1})


@patch('src.training.train.mlflow')
def test_selection_uses_holdout_and_injected_failures(mock_mlflow, mock_config, sample_features_df):
    results = AnomalyTrainer(mock_config).train(sample_features_df)

    iso = results["metrics"]["isolation_forest"]
    lof = results["metrics"]["local_outlier_factor"]
    for m in (iso, lof):
        assert 0.0 <= m["recall"] <= 1.0
        assert 0.0 <= m["false_alarm_rate"] <= 1.0
        assert m["score"] == pytest.approx(m["recall"] - m["false_alarm_rate"])
    best = iso if results["best_model"] == "IsolationForest" else lof
    assert best["score"] == max(iso["score"], lof["score"])
    assert results["metadata"]["holdout_recall"] == best["recall"]


def test_injected_anomalies_are_real_failures(sample_features_df):
    holdout = sample_features_df[Config.FEATURE_COLUMNS].tail(20)
    injected = AnomalyTrainer.inject_anomalies(holdout)

    assert len(injected) == 3 * len(holdout)
    assert (injected["cpu_percent"].iloc[:20] == 100.0).all()
    assert (injected["ram_percent"].iloc[20:40] == 98.0).all()
    assert (injected["disk_percent"].iloc[40:] <= 100.0).all()
    assert (injected["disk_rate_of_change"].iloc[40:] == 20.0).all()
