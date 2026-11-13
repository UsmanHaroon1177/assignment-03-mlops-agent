"""Export the serving model and its metadata from the DVC-tracked dataset.

Run from the repo root:  python scripts/export_model.py
Writes serving/model/model.joblib (gitignored), serving/model/metadata.json
and serving/requirements.txt (pinned to the versions installed here, so the
container unpickles with the same scikit-learn that trained the model).
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "dataset.csv"
DVC_FILE = ROOT / "data" / "dataset.csv.dvc"
OUT_DIR = ROOT / "serving" / "model"
REQ_FILE = ROOT / "serving" / "requirements.txt"
SERVING_PACKAGES = ["fastapi", "uvicorn", "scikit-learn", "numpy", "joblib", "pydantic"]
ROWS_TO_VERSION = {400: "v1", 569: "v2"}


def git(*args: str) -> str:
    r = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, timeout=30)
    return r.stdout.strip() if r.returncode == 0 else "unknown"


def dvc_md5() -> str:
    m = re.search(r"md5:\s*([0-9a-f]{32})", DVC_FILE.read_text(encoding="utf-8"))
    return m.group(1) if m else "unknown"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-version", default=None, help="override v1/v2 detection")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-estimators", type=int, default=100)
    a = p.parse_args()

    df = pd.read_csv(DATA)
    target = "target" if "target" in df.columns else df.columns[-1]
    X, y = df.drop(columns=[target]), df[target]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=a.seed, stratify=y
    )
    model = RandomForestClassifier(n_estimators=a.n_estimators, random_state=a.seed)
    model.fit(X_tr.values, y_tr)
    f1 = float(f1_score(y_te, model.predict(X_te.values)))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, OUT_DIR / "model.joblib", compress=3)

    meta = {
        "model_name": "breast-cancer-clf",
        "model_type": "RandomForestClassifier",
        "dataset_version": a.dataset_version or ROWS_TO_VERSION.get(len(df), "unknown"),
        "dataset_dvc_md5": dvc_md5(),
        "code_git_commit": git("rev-parse", "--short", "HEAD"),
        "n_rows": int(len(df)),
        "n_features": int(X.shape[1]),
        "target_column": target,
        "holdout_f1": round(f1, 4),
        "sklearn_version": metadata.version("scikit-learn"),
        "exported_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "exported_by": "scripts/export_model.py (retrained locally; not an MLflow artifact)",
    }
    (OUT_DIR / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    REQ_FILE.write_text(
        "\n".join(f"{n}=={metadata.version(n)}" for n in SERVING_PACKAGES) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(meta, indent=2))
    print(f"\nwrote {OUT_DIR / 'model.joblib'} and {REQ_FILE}")


if __name__ == "__main__":
    main()