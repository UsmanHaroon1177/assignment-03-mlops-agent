"""Collect dataset and code provenance for a training run (git + DVC), read-only."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
DVC_FILE = "data/dataset.csv.dvc"


def _run(cmd: list[str], timeout: int = 60) -> tuple[int, str]:
    try:
        res = subprocess.run(
            cmd, cwd=ROOT, capture_output=True, text=True, timeout=timeout, shell=False
        )
    except (OSError, subprocess.TimeoutExpired):
        return 1, ""
    return res.returncode, res.stdout.strip()


def dvc_md5() -> str | None:
    path = ROOT / DVC_FILE
    if not path.exists():
        return None
    doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    outs = doc.get("outs") or []
    return outs[0].get("md5") if outs else None


def dataset_commit() -> str:
    """Last commit that changed the .dvc file = the commit that defines the dataset version."""
    code, out = _run(["git", "log", "-1", "--format=%H", "--", DVC_FILE])
    return out if code == 0 and out else "uncommitted"


def dataset_version(commit: str) -> str:
    if commit == "uncommitted":
        return "unlabeled"
    code, out = _run(["git", "tag", "--points-at", commit, "--list", "data-v*"])
    tags = sorted(t for t in out.splitlines() if t)
    return tags[0].removeprefix("data-") if code == 0 and tags else "unlabeled"


def git_head() -> str:
    code, out = _run(["git", "rev-parse", "HEAD"])
    return out if code == 0 and out else "unknown"


def git_dirty() -> bool:
    code, out = _run(["git", "status", "--porcelain", "--untracked-files=no"])
    return code == 0 and bool(out)


def _dvc_status_impl() -> str:
    """'up-to-date' if the data file matches the .dvc record, 'modified' if not."""
    code, out = _run([sys.executable, "-m", "dvc", "status", DVC_FILE, "--json"], timeout=120)
    if code != 0:
        return "unknown"
    return "up-to-date" if out in ("", "{}") else "modified"


def collect() -> dict:
    commit = dataset_commit()
    tags = {
        "dataset.path": "data/dataset.csv",
        "dataset.dvc_file": DVC_FILE,
        "dataset.dvc_md5": dvc_md5() or "unknown",
        "dataset.version": dataset_version(commit),
        "dataset.git_commit": commit,
        "dataset.dvc_status": dvc_status(),
        "code.git_commit": git_head(),
        "code.git_dirty": str(git_dirty()).lower(),
    }
    return {"tags": tags}

def dvc_status() -> str:
    """DVC refuses nested dvc calls inside `dvc repro` stages (it sets DVC_ROOT); skip there."""
    import os as _os

    if _os.environ.get("DVC_ROOT"):
        return "unknown"
    return _dvc_status_impl()
