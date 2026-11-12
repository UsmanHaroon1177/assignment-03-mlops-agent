"""List MLflow experiments and runs with their dataset provenance tags."""
import os

import mlflow
from dotenv import load_dotenv
from mlflow import MlflowClient

load_dotenv()
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
client = MlflowClient()

for exp in client.search_experiments():
    print(f"experiment {exp.experiment_id}: {exp.name}")
    runs = client.search_runs([exp.experiment_id], order_by=["attributes.start_time ASC"])
    for r in runs:
        tags, met = r.data.tags, r.data.metrics
        print(
            f"  {r.info.run_id[:8]}  {tags.get('mlflow.runName', '-'):12} "
            f"f1={met.get('f1', float('nan')):.4f}  "
            f"data={tags.get('dataset.version', '-'):9} "
            f"md5={tags.get('dataset.dvc_md5', '')[:8]}  "
            f"data_commit={tags.get('dataset.git_commit', '')[:7]}  "
            f"dvc={tags.get('dataset.dvc_status', '-')}  "
            f"dirty={tags.get('code.git_dirty', '-')}"
        )
