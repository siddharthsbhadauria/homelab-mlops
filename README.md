# homelab-mlops

[![CI](https://github.com/siddharthsbhadauria/homelab-mlops/actions/workflows/ci.yml/badge.svg)](https://github.com/siddharthsbhadauria/homelab-mlops/actions/workflows/ci.yml)
[![Pages](https://github.com/siddharthsbhadauria/homelab-mlops/actions/workflows/deploy_pages.yml/badge.svg)](https://github.com/siddharthsbhadauria/homelab-mlops/actions/workflows/deploy_pages.yml)

Anomaly detection for my home NAS. It reads the telemetry that [auto-datapulse](https://github.com/siddharthsbhadauria/auto-datapulse) writes to DuckDB every 15 minutes, builds features in SQL, trains two scikit-learn models, serves the better one over FastAPI and opens a GitHub issue when the NAS behaves oddly. Everything runs on CPU in Docker on the NAS.

**Dashboard:** https://siddharthsbhadauria.github.io/homelab-mlops/ (one point per daily training run, built from `REPORTS/`)

## How it works

```
auto-datapulse telemetry.duckdb
  -> 18 features in DuckDB window SQL      src/features/feature_engineer.py  (saved as Parquet)
  -> IsolationForest and LOF               src/training/train.py             (runs logged to MLflow)
  -> better model saved as primary_model   models/primary_model.joblib
  -> FastAPI /predict with Prometheus      src/serving/app.py
  -> monitor every 15 min                  src/monitoring/anomaly_monitor.py
  -> GitHub issue on a sustained anomaly
```

- **Training** runs every 6 hours (`PIPELINE_INTERVAL_SECONDS`). It needs at least 24 hours of data. Contamination defaults to 1%.
- **Model choice:** both models are trained on the same features. The one that flags fewer training readings as anomalies is saved as the primary model, and silhouette scores are logged to MLflow for comparison. Over the 94 daily reports so far, LOF has been picked every time.
- **Alerting:** an issue opens only after 2 anomalous readings in a row, and at most once every 4 hours, so a single spike doesn't page anyone.
- **Monitoring:** the API exports prediction counts, latency and anomaly scores to Prometheus. Grafana dashboards are provisioned from `grafana/`.

## Features

| Group | Columns |
| :--- | :--- |
| Current reading | cpu_percent, ram_percent, disk_percent |
| 1-hour rolling (4 samples) | mean and std of CPU, RAM and disk |
| 6-hour rolling (24 samples) | mean and std of CPU and RAM |
| Change since last reading | CPU, RAM and disk |
| Time | hour_of_day, day_of_week |

## Deploy on the NAS (Portainer)

1. In Portainer, go to **Stacks** > **Add stack** and choose **Repository**.
2. Repository URL `https://github.com/siddharthsbhadauria/homelab-mlops`, reference `refs/heads/main`, compose path `docker-compose.yml`.
3. Add environment variables (see `.env.example` for the rest):
   - `GITHUB_TOKEN` and `GITHUB_REPO` for incident issues
   - `DATAPULSE_DATA_PATH`: the folder holding auto-datapulse's `telemetry.duckdb`
   - `CONTAMINATION`: default `0.01`
4. Deploy.

Locally: `cp .env.example .env && docker compose up -d --build`, then

| Service | URL |
| :--- | :--- |
| FastAPI docs | http://localhost:8000/docs |
| MLflow | http://localhost:5000 |
| Grafana | http://localhost:3000 |
| Prometheus | http://localhost:9090 |

## Develop

```bash
pip install -r requirements.txt pytest httpx ruff
ruff check .
pytest -q
```

## License

MIT
