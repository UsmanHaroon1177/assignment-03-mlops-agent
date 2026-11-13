"""Smoke test for the inference API (local uvicorn or a kubectl port-forward).
Usage: python scripts/try_api.py [--url http://127.0.0.1:8001]
"""
import argparse
import json
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://127.0.0.1:8001")
    a = p.parse_args()

    for path in ("/healthz", "/readyz", "/version"):
        r = requests.get(a.url + path, timeout=5)
        print(f"GET {path} -> {r.status_code} {json.dumps(r.json())[:300]}")

    df = pd.read_csv(ROOT / "data" / "dataset.csv")
    target = "target" if "target" in df.columns else df.columns[-1]
    row = df.drop(columns=[target]).iloc[0].astype(float).tolist()
    r = requests.post(a.url + "/predict", json={"features": row}, timeout=5)
    print(f"POST /predict -> {r.status_code} {r.text}  (true label: {df[target].iloc[0]})")
    r = requests.post(a.url + "/predict", json={"features": [1.0, 2.0]}, timeout=5)
    print(f"POST /predict (bad length) -> {r.status_code} {r.text}")


if __name__ == "__main__":
    main()