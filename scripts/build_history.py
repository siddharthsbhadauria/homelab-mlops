"""Roll the daily REPORTS/model_metrics_*.json files into site/model_history.js for the dashboard."""
import glob
import json
import os

ROOT = os.path.join(os.path.dirname(__file__), "..")


def main() -> None:
    rows = []
    for path in sorted(glob.glob(os.path.join(ROOT, "REPORTS", "model_metrics_*.json"))):
        with open(path, encoding="utf-8") as f:
            r = json.load(f)
        rows.append({
            "date": r["last_updated"][:10],
            "model": r.get("model_type", ""),
            "anomaly_rate": round(float(r.get("anomaly_rate", 0.0)), 5),
            "samples": int(r.get("n_samples_trained", 0)),
            "features": int(r.get("features_count", 0)),
            "status": r.get("status", ""),
        })
    os.makedirs(os.path.join(ROOT, "site"), exist_ok=True)
    with open(os.path.join(ROOT, "site", "model_history.js"), "w", encoding="utf-8") as f:
        f.write("window.MODEL_HISTORY = " + json.dumps(rows, separators=(",", ":")) + ";\n")
    print(f"[OK] {len(rows)} daily reports -> site/model_history.js")


if __name__ == "__main__":
    main()
