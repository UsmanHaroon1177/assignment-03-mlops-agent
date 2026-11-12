"""Soft-delete FAILED runs in the tracking server (keeps the experiment clean)."""
import os

import mlflow
from dotenv import load_dotenv
from mlflow import MlflowClient

load_dotenv()
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
client = MlflowClient()

removed = 0
for exp in client.search_experiments():
    for run in client.search_runs([exp.experiment_id], filter_string="attributes.status = 'FAILED'"):
        name = run.data.tags.get("mlflow.runName", "-")
        client.delete_run(run.info.run_id)
        removed += 1
        print(f"deleted FAILED run {run.info.run_id[:8]} ({name})")
print(f"{removed} failed run(s) deleted")
