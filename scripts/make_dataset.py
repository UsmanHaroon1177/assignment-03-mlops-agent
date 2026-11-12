"""Create the versioned dataset used for training (deterministic).

v1 = 400 rows, v2 = all 569 rows of the same shuffled data, so v2 is a superset of v1.
Usage: python scripts/make_dataset.py --version v1
"""
import argparse
from pathlib import Path

from sklearn.datasets import load_breast_cancer

SEED = 42
ROWS = {"v1": 400, "v2": 569}
OUT = Path(__file__).resolve().parents[1] / "data" / "dataset.csv"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", choices=sorted(ROWS), required=True)
    args = parser.parse_args()

    df = load_breast_cancer(as_frame=True).frame
    df = df.sample(frac=1.0, random_state=SEED).reset_index(drop=True).head(ROWS[args.version])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    # LF line endings so the DVC hash is the same on Windows and Linux
    df.to_csv(OUT, index=False, lineterminator="\n")
    print(f"wrote {OUT} version={args.version} rows={len(df)} cols={df.shape[1]}")


if __name__ == "__main__":
    main()
