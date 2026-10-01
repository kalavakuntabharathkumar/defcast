"""Metrics logging with a local JSON fallback and optional MLflow."""
import json
from pathlib import Path
from datetime import datetime, timezone

def log_metrics(run_name, metrics, output_dir="artifacts/experiments"):
    record={"run_name":run_name,"timestamp_utc":datetime.now(timezone.utc).isoformat(),"metrics":metrics}
    try:
        import mlflow
        with mlflow.start_run(run_name=run_name):
            for key,value in metrics.items():
                if isinstance(value,(int,float)): mlflow.log_metric(key,float(value))
            mlflow.set_tag("project","defect-demand-forecasting")
    except ImportError:
        pass
    out=Path(output_dir); out.mkdir(parents=True,exist_ok=True)
    path=out/f"{run_name.replace(' ','_')}.json"
    path.write_text(json.dumps(record,indent=2),encoding="utf-8")
    return path
