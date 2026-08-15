"""
Unit tests for AnomalyTrainer module.
"""
import os
import json
import joblib
import numpy as np
import pytest
from unittest.mock import patch

from src.training.train import AnomalyTrainer
from tests.conftest import FEATURE_COLUMNS


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
    X = sample_features_df[FEATURE_COLUMNS]
    predictions = model.predict(X)
    
    assert predictions.shape == (len(sample_features_df),)
    assert set(np.unique(predictions)).issubset({1, -1})
