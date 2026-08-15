"""
Feature engineering module for homelab-mlops.
Extracts sliding window metrics, rates of change, and cyclical temporal features from auto-datapulse DuckDB.
"""
import os
import logging
import datetime
from typing import Optional, Dict, Any

import duckdb
import pandas as pd

from src.config import Config

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Extracts and engineers features from auto-datapulse telemetry."""

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()

    def extract_features(self, lookback_hours: Optional[int] = None) -> pd.DataFrame:
        """
        Connects to auto-datapulse's DuckDB (read-only) and computes rolling stats.
        Returns a cleaned pandas DataFrame with all required FEATURE_COLUMNS.
        """
        if not os.path.exists(self.config.TELEMETRY_DB_PATH):
            raise FileNotFoundError(f"Telemetry DB not found at {self.config.TELEMETRY_DB_PATH}")

        logger.info(f"Connecting to DuckDB at {self.config.TELEMETRY_DB_PATH}")

        try:
            conn = duckdb.connect(self.config.TELEMETRY_DB_PATH, read_only=True)

            # Check if telemetry_snapshots table exists
            tables = conn.execute("SHOW TABLES").fetchall()
            table_names = [t[0] for t in tables]
            if "telemetry_snapshots" not in table_names:
                conn.close()
                raise ValueError("Table 'telemetry_snapshots' does not exist in the database.")

            query = f"""
            WITH base AS (
                SELECT 
                    timestamp,
                    CAST(cpu_percent AS DOUBLE) as cpu_percent,
                    CAST(ram_percent AS DOUBLE) as ram_percent,
                    CAST(disk_percent AS DOUBLE) as disk_percent
                FROM telemetry_snapshots
                ORDER BY timestamp ASC
            )
            SELECT 
                timestamp,
                cpu_percent,
                ram_percent,
                disk_percent,
                AVG(cpu_percent) OVER w_1h AS cpu_rolling_mean_1h,
                COALESCE(STDDEV_POP(cpu_percent) OVER w_1h, 0.0) AS cpu_rolling_std_1h,
                AVG(ram_percent) OVER w_1h AS ram_rolling_mean_1h,
                COALESCE(STDDEV_POP(ram_percent) OVER w_1h, 0.0) AS ram_rolling_std_1h,
                AVG(disk_percent) OVER w_1h AS disk_rolling_mean_1h,
                COALESCE(STDDEV_POP(disk_percent) OVER w_1h, 0.0) AS disk_rolling_std_1h,
                
                AVG(cpu_percent) OVER w_6h AS cpu_rolling_mean_6h,
                COALESCE(STDDEV_POP(cpu_percent) OVER w_6h, 0.0) AS cpu_rolling_std_6h,
                AVG(ram_percent) OVER w_6h AS ram_rolling_mean_6h,
                COALESCE(STDDEV_POP(ram_percent) OVER w_6h, 0.0) AS ram_rolling_std_6h,
                
                COALESCE(cpu_percent - LAG(cpu_percent, 1) OVER (ORDER BY timestamp ASC), 0.0) AS cpu_rate_of_change,
                COALESCE(ram_percent - LAG(ram_percent, 1) OVER (ORDER BY timestamp ASC), 0.0) AS ram_rate_of_change,
                COALESCE(disk_percent - LAG(disk_percent, 1) OVER (ORDER BY timestamp ASC), 0.0) AS disk_rate_of_change,
                
                EXTRACT(HOUR FROM CAST(timestamp AS TIMESTAMP)) AS hour_of_day,
                EXTRACT(DOW FROM CAST(timestamp AS TIMESTAMP)) AS day_of_week
            FROM base
            WINDOW 
                w_1h AS (ORDER BY timestamp ASC ROWS BETWEEN {max(0, self.config.ROLLING_WINDOW_SHORT - 1)} PRECEDING AND CURRENT ROW),
                w_6h AS (ORDER BY timestamp ASC ROWS BETWEEN {max(0, self.config.ROLLING_WINDOW_LONG - 1)} PRECEDING AND CURRENT ROW)
            """

            df = conn.execute(query).df()
            conn.close()

            if df.empty:
                raise ValueError("No data returned from telemetry database.")

            # Drop any remaining null rows if any
            df = df.dropna()

            if df.empty:
                raise ValueError("All rows were dropped after cleaning nulls.")

            # Ensure all required feature columns are present and strictly typed
            for col in self.config.FEATURE_COLUMNS:
                if col not in df.columns:
                    df[col] = 0.0
                else:
                    df[col] = df[col].astype(float)

            logger.info(f"Successfully extracted {len(df)} feature rows.")
            return df

        except Exception as e:
            logger.error(f"Failed to extract features: {e}")
            raise

    def save_features(self, df: pd.DataFrame, output_path: Optional[str] = None) -> str:
        """Saves features DataFrame to Parquet and returns the file path."""
        path = output_path or os.path.join(self.config.FEATURE_OUTPUT_DIR, "features.parquet")
        os.makedirs(os.path.dirname(path), exist_ok=True)

        df.to_parquet(path, index=False)
        logger.info(f"Successfully saved {len(df)} rows to {path}")
        return path

    def compute_single_point_features(self, snapshot: Dict[str, Any]) -> Dict[str, Any]:
        """
        Computes features for a single real-time snapshot for online API prediction.
        Gracefully handles nested auto-datapulse format or flat dictionaries.
        """
        sys_data = snapshot.get("system", {}) if isinstance(snapshot.get("system"), dict) else {}

        cpu_val = float(sys_data.get("cpu_percent", snapshot.get("cpu_percent", 0.0)))
        ram_val = float(sys_data.get("ram_percent", snapshot.get("ram_percent", 0.0)))
        disk_val = float(sys_data.get("disk_percent", snapshot.get("disk_percent", 0.0)))

        df_hist = pd.DataFrame()
        if os.path.exists(self.config.TELEMETRY_DB_PATH):
            try:
                conn = duckdb.connect(self.config.TELEMETRY_DB_PATH, read_only=True)
                query = f"""
                    SELECT 
                        CAST(cpu_percent AS DOUBLE) as cpu_percent,
                        CAST(ram_percent AS DOUBLE) as ram_percent,
                        CAST(disk_percent AS DOUBLE) as disk_percent
                    FROM telemetry_snapshots
                    ORDER BY timestamp DESC
                    LIMIT {self.config.ROLLING_WINDOW_LONG}
                """
                df_hist = conn.execute(query).df()
                conn.close()
                # Sort chronological
                if not df_hist.empty:
                    df_hist = df_hist.iloc[::-1].reset_index(drop=True)
            except Exception as e:
                logger.warning(f"Could not read historical data from DuckDB: {e}")

        # Append new point
        new_row = pd.DataFrame([{
            'cpu_percent': cpu_val,
            'ram_percent': ram_val,
            'disk_percent': disk_val
        }])

        df_combined = pd.concat([df_hist, new_row], ignore_index=True) if not df_hist.empty else new_row

        features: Dict[str, float] = {
            'cpu_percent': cpu_val,
            'ram_percent': ram_val,
            'disk_percent': disk_val
        }

        # 1-hour rolling window
        df_1h = df_combined.tail(self.config.ROLLING_WINDOW_SHORT)
        features['cpu_rolling_mean_1h'] = float(df_1h['cpu_percent'].mean())
        features['cpu_rolling_std_1h'] = float(df_1h['cpu_percent'].std(ddof=0)) if len(df_1h) > 1 else 0.0
        features['ram_rolling_mean_1h'] = float(df_1h['ram_percent'].mean())
        features['ram_rolling_std_1h'] = float(df_1h['ram_percent'].std(ddof=0)) if len(df_1h) > 1 else 0.0
        features['disk_rolling_mean_1h'] = float(df_1h['disk_percent'].mean())
        features['disk_rolling_std_1h'] = float(df_1h['disk_percent'].std(ddof=0)) if len(df_1h) > 1 else 0.0

        # 6-hour rolling window
        df_6h = df_combined.tail(self.config.ROLLING_WINDOW_LONG)
        features['cpu_rolling_mean_6h'] = float(df_6h['cpu_percent'].mean())
        features['cpu_rolling_std_6h'] = float(df_6h['cpu_percent'].std(ddof=0)) if len(df_6h) > 1 else 0.0
        features['ram_rolling_mean_6h'] = float(df_6h['ram_percent'].mean())
        features['ram_rolling_std_6h'] = float(df_6h['ram_percent'].std(ddof=0)) if len(df_6h) > 1 else 0.0

        # Rate of change
        if len(df_hist) > 0:
            last_hist = df_hist.iloc[-1]
            features['cpu_rate_of_change'] = float(cpu_val - last_hist['cpu_percent'])
            features['ram_rate_of_change'] = float(ram_val - last_hist['ram_percent'])
            features['disk_rate_of_change'] = float(disk_val - last_hist['disk_percent'])
        else:
            features['cpu_rate_of_change'] = 0.0
            features['ram_rate_of_change'] = 0.0
            features['disk_rate_of_change'] = 0.0

        # Temporal features
        ts_val = snapshot.get('timestamp')
        if ts_val:
            try:
                dt = pd.to_datetime(ts_val)
                features['hour_of_day'] = float(dt.hour)
                features['day_of_week'] = float(dt.dayofweek)
            except Exception:
                now = datetime.datetime.now(datetime.timezone.utc)
                features['hour_of_day'] = float(now.hour)
                features['day_of_week'] = float(now.weekday())
        else:
            now = datetime.datetime.now(datetime.timezone.utc)
            features['hour_of_day'] = float(now.hour)
            features['day_of_week'] = float(now.weekday())

        # Return dict aligned to FEATURE_COLUMNS
        return {k: float(features.get(k, 0.0)) for k in self.config.FEATURE_COLUMNS}
