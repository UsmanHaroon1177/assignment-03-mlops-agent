"""Training pipeline: validate -> preprocess -> train -> evaluate/log -> quality gate.

* validate_dataset   fails the run (no retry) if the DVC-tracked dataset is unusable.
* preprocess_data    de-duplicates and records a summary (training/train.py applies its own
                     sklearn pipeline, so this stage verifies and documents, it does not mutate data).
* train_model        runs `python -m training.train` (list args, no shell); retries on failure.
* evaluate_and_log   reads the finished MLflow run, tags it with Airflow provenance.
* quality_gate       fails the run if F1 < min_f1. Nothing is promoted automatically.
"""
from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

from airflow.exceptions import AirflowFailException
from airflow.sdk import Param, dag, get_current_context, task

log = logging.getLogger(__name__)

PROJECT_DIR = Path(os.environ.get("PROJECT_DIR", "/opt/airflow/project"))
WORK_DIR = Path(os.environ.get("PIPELINE_WORK_DIR", "/opt/airflow/state/work"))
DATASET_REL = "data/dataset.csv"
EXPERIMENT = "breast-cancer-classifier"
MIN_ROWS = 100


def _safe(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "-", text)


def _target_column(df):
    for name in ("target", "diagnosis", "label", "class"):
        if name in df.columns:
            return name
    return df.columns[-1]


@task(retries=0)
def validate_dataset() -> dict:
    import pandas as pd

    path = PROJECT_DIR / DATASET_REL
    if not path.is_file():
        raise AirflowFailException(f"Dataset not found: {DATASET_REL}")
    df = pd.read_csv(path)
    target = _target_column(df)
    problems = []
    if len(df) < MIN_ROWS:
        problems.append(f"only {len(df)} rows (minimum {MIN_ROWS})")
    null_cells = int(df.isna().sum().sum())
    if null_cells:
        problems.append(f"{null_cells} missing values")
    if df[target].nunique() != 2:
        problems.append(f"target '{target}' is not binary ({df[target].nunique()} classes)")
    else:
        share = float(df[target].value_counts(normalize=True).min())
        if share < 0.10:
            problems.append(f"minority class is only {share:.1%} of rows")
    non_numeric = [c for c in df.columns if c != target and not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        problems.append(f"non-numeric feature columns: {non_numeric[:5]}")
    if problems:
        raise AirflowFailException("Dataset validation failed: " + "; ".join(problems))
    summary = {"rows": int(len(df)), "n_features": int(df.shape[1] - 1), "target": target}
    log.info("Dataset validation passed: %s", summary)
    return summary


@task
def preprocess_data(validation: dict) -> dict:
    import pandas as pd

    ctx = get_current_context()
    df = pd.read_csv(PROJECT_DIR / DATASET_REL)
    before = int(len(df))
    after = int(len(df.drop_duplicates()))
    summary = {
        "rows_in": before,
        "rows_after_dedup": after,
        "duplicates_found": before - after,
        "n_features": validation["n_features"],
    }
    out = WORK_DIR / _safe(ctx["run_id"])
    out.mkdir(parents=True, exist_ok=True)
    (out / "preprocess_summary.json").write_text(json.dumps(summary, indent=2))
    log.info("Preprocessing summary: %s", summary)
    return summary


@task(retries=2, retry_delay=timedelta(seconds=20))
def train_model(preprocess_summary: dict) -> dict:
    ctx = get_current_context()
    model = ctx["params"]["model"]
    run_name = f"airflow-{model}-{_safe(ctx['run_id'])}"
    cmd = [sys.executable, "-m", "training.train", "--model", model, "--run-name", run_name]
    log.info("Training attempt %s: %s", ctx["ti"].try_number, " ".join(cmd))
    proc = subprocess.run(cmd, cwd=PROJECT_DIR, capture_output=True, text=True, timeout=900)
    log.info("train stdout (tail):\n%s", proc.stdout[-2000:])
    if proc.returncode != 0:
        raise RuntimeError(f"training exited with code {proc.returncode}: {proc.stderr[-1500:]}")
    return {"run_name": run_name, "model": model}


@task(retries=2, retry_delay=timedelta(seconds=20))
def evaluate_and_log(trained: dict) -> dict:
    from mlflow import MlflowClient

    ctx = get_current_context()
    client = MlflowClient()  # MLFLOW_TRACKING_URI comes from the container environment
    exp = client.get_experiment_by_name(EXPERIMENT)
    if exp is None:
        raise RuntimeError(f"MLflow experiment '{EXPERIMENT}' not found")
    runs = client.search_runs(
        [exp.experiment_id],
        filter_string=f"attributes.run_name = '{trained['run_name']}' and attributes.status = 'FINISHED'",
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )
    if not runs:
        raise RuntimeError(f"No FINISHED MLflow run named {trained['run_name']}")
    run = runs[0]
    client.set_tag(run.info.run_id, "airflow.dag_id", ctx["dag"].dag_id)
    client.set_tag(run.info.run_id, "airflow.dag_run_id", ctx["run_id"])
    metrics = {k: float(v) for k, v in run.data.metrics.items()}
    log.info("Evaluated MLflow run %s: %s", run.info.run_id, metrics)
    return {"mlflow_run_id": run.info.run_id, "model": trained["model"], "metrics": metrics}


@task(retries=0)
def quality_gate(evaluated: dict) -> dict:
    from mlflow import MlflowClient

    ctx = get_current_context()
    min_f1 = float(ctx["params"]["min_f1"])
    f1 = evaluated["metrics"].get("f1")
    passed = f1 is not None and f1 >= min_f1
    client = MlflowClient()
    client.set_tag(evaluated["mlflow_run_id"], "quality_gate", "passed" if passed else "failed")
    client.set_tag(evaluated["mlflow_run_id"], "quality_gate.min_f1", str(min_f1))
    if not passed:
        raise AirflowFailException(f"Quality gate FAILED: f1={f1} < min_f1={min_f1}. Nothing was promoted.")
    log.info("Quality gate passed: f1=%s >= %s. No automatic promotion is performed.", f1, min_f1)
    return {"passed": True, "f1": f1, "min_f1": min_f1}


@dag(
    dag_id="training_pipeline",
    schedule=None,
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,
    dagrun_timeout=timedelta(minutes=30),
    default_args={"owner": "mlops", "retries": 1, "retry_delay": timedelta(seconds=15)},
    params={
        "model": Param("logreg", type="string", enum=["logreg", "rf", "gbm"]),
        "min_f1": Param(0.90, type="number", minimum=0, maximum=1),
    },
    tags=["mlops", "training"],
    doc_md=__doc__,
)
def training_pipeline():
    validated = validate_dataset()
    prepared = preprocess_data(validated)
    trained = train_model(prepared)
    evaluated = evaluate_and_log(trained)
    quality_gate(evaluated)


training_pipeline()