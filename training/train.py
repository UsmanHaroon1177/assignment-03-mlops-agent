"""Train a classifier on the DVC-tracked dataset and log everything to MLflow.

Run from the repo root:  python -m training.train --model rf --n-estimators 100
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
import requests
from dotenv import load_dotenv
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from training import provenance

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "dataset.csv"
DEFAULT_EXPERIMENT = "breast-cancer-classifier"
REGISTERED_MODEL = "breast-cancer-clf"
# skops must be told to trust the sklearn tree type to save tree models.
# Safe here: we train and save these models ourselves.
SKOPS_TRUSTED = ["sklearn.tree._tree.Tree"]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=["logreg", "rf", "gbm"], default="logreg")
    p.add_argument("--C", type=float, default=1.0)
    p.add_argument("--n-estimators", type=int, default=100)
    p.add_argument("--max-depth", type=int, default=None)
    p.add_argument("--learning-rate", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--experiment", default=DEFAULT_EXPERIMENT)
    p.add_argument("--run-name", default=None)
    p.add_argument("--register", action="store_true", help="register model in MLflow registry")
    p.add_argument("--tracking-uri", default=None)
    return p.parse_args(argv)


def build_model(a):
    if a.model == "logreg":
        model = make_pipeline(StandardScaler(), LogisticRegression(C=a.C, max_iter=2000))
        return model, {"C": a.C}
    if a.model == "rf":
        model = RandomForestClassifier(
            n_estimators=a.n_estimators, max_depth=a.max_depth, random_state=a.seed
        )
        return model, {"n_estimators": a.n_estimators, "max_depth": a.max_depth}
    model = GradientBoostingClassifier(
        n_estimators=a.n_estimators,
        learning_rate=a.learning_rate,
        max_depth=a.max_depth or 3,
        random_state=a.seed,
    )
    return model, {
        "n_estimators": a.n_estimators,
        "learning_rate": a.learning_rate,
        "max_depth": a.max_depth or 3,
    }


def check_tracking(uri: str) -> None:
    if uri.startswith("http"):
        requests.get(uri.rstrip("/") + "/health", timeout=5).raise_for_status()


def main(argv=None) -> int:
    load_dotenv(ROOT / ".env")
    a = parse_args(argv)
    uri = a.tracking_uri or os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")

    if not DATA_PATH.exists():
        print("ERROR: data/dataset.csv missing. Run: dvc checkout")
        return 2
    try:
        check_tracking(uri)
    except Exception as exc:  # connection refused, timeout, bad status
        print(f"ERROR: MLflow tracking server not reachable at {uri}: {type(exc).__name__}")
        return 2

    prov = provenance.collect()
    df = pd.read_csv(DATA_PATH)
    X, y = df.drop(columns=["target"]), df["target"]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=a.test_size, random_state=a.seed, stratify=y
    )
    model, hyper = build_model(a)

    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(a.experiment)
    with mlflow.start_run(run_name=a.run_name) as run:
        mlflow.log_params(
            {
                "model": a.model,
                "seed": a.seed,
                "test_size": a.test_size,
                "n_rows": len(df),
                "n_features": X.shape[1],
                **hyper,
            }
        )
        mlflow.set_tags(prov["tags"])

        t0 = time.time()
        model.fit(X_tr, y_tr)
        train_seconds = time.time() - t0

        pred = model.predict(X_te)
        proba = model.predict_proba(X_te)[:, 1]
        metrics = {
            "accuracy": accuracy_score(y_te, pred),
            "precision": precision_score(y_te, pred),
            "recall": recall_score(y_te, pred),
            "f1": f1_score(y_te, pred),
            "roc_auc": roc_auc_score(y_te, proba),
            "train_seconds": train_seconds,
        }
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(ROOT / provenance.DVC_FILE), artifact_path="provenance")
        mlflow.sklearn.log_model(
            model,
            name="model",
            registered_model_name=REGISTERED_MODEL if a.register else None,
            skops_trusted_types=SKOPS_TRUSTED,
        )
        run_id = run.info.run_id

    print(
        json.dumps(
            {
                "run_id": run_id,
                "run_name": a.run_name,
                "experiment": a.experiment,
                "metrics": {k: round(v, 4) for k, v in metrics.items()},
                "provenance": prov["tags"],
                "registered": a.register,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
