"""
Unit tests for FeatureEngineer module.
"""
import os
import pytest
import pandas as pd
import duckdb

from src.config import Config
from src.features.feature_engineer import FeatureEngineer


def test_extract_features_returns_dataframe(mock_config):
    fe = FeatureEngineer(mock_config)
    df = fe.extract_features()
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0


def test_extract_features_no_nulls(mock_config):
    fe = FeatureEngineer(mock_config)
    df = fe.extract_features()
    assert df[Config.FEATURE_COLUMNS].isnull().sum().sum() == 0


def test_extract_features_correct_column_count(mock_config):
    fe = FeatureEngineer(mock_config)
    df = fe.extract_features()
    for col in Config.FEATURE_COLUMNS:
        assert col in df.columns


def test_extract_features_empty_db_raises(mock_config, tmp_path):
    empty_db_path = str(tmp_path / "empty.db")
    conn = duckdb.connect(empty_db_path)
    conn.execute('''
        CREATE TABLE telemetry_snapshots (
            timestamp VARCHAR, cpu_percent DOUBLE, ram_used_gb DOUBLE, ram_total_gb DOUBLE,
            ram_percent DOUBLE, disk_used_gb DOUBLE, disk_total_gb DOUBLE, disk_percent DOUBLE, status VARCHAR
        )
    ''')
    conn.close()
    
    mock_config.TELEMETRY_DB_PATH = empty_db_path
    fe = FeatureEngineer(mock_config)
    
    with pytest.raises(Exception):
        fe.extract_features()


def test_save_features_creates_parquet(mock_config):
    fe = FeatureEngineer(mock_config)
    df = fe.extract_features()
    out_path = fe.save_features(df)
    assert os.path.exists(out_path)
    assert out_path.endswith('.parquet')


def test_compute_single_point_features_returns_dict(mock_config, sample_snapshot):
    fe = FeatureEngineer(mock_config)
    features = fe.compute_single_point_features(sample_snapshot)
    assert isinstance(features, dict)
    assert "cpu_percent" in features
    assert "hour_of_day" in features
    for col in Config.FEATURE_COLUMNS:
        assert col in features
