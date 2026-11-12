"""DVC data-provenance tools (read-only).

Commands run WITHOUT a shell and only from an allowlist. Refs and paths are validated, so a
request can never smuggle in an option such as --output or a path outside the repo.
Nothing here modifies the working tree: reproduction is explained, not executed.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

from agent.tools import mlflow_tools
from agent.tools.base import ToolInputError, ToolSpec, ToolUnavailableError

ROOT = Path(os.getenv("REPO_ROOT") or Path(__file__).resolve().parents[2])
DEFAULT_DATASET = "data/dataset.csv"
GIT_ALLOWED = {"log", "show", "tag", "rev-parse", "ls-files", "status"}
DVC_ALLOWED = {"status", "diff"}
_REF_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/\-]{0,63}$")
_PATH_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_./\-]{0,100}$")


def _exec(cmd: list[str], timeout: int) -> tuple[int, str, str]:
    try:
        res = subprocess.run(
            cmd,
            cwd=ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            shell=False,
        )
    except FileNotFoundError as exc:
        raise ToolUnavailableError(f"command not found: {cmd[0]}") from exc
    except subprocess.TimeoutExpired as exc:
        raise ToolUnavailableError(f"command timed out after {timeout}s") from exc
    return res.returncode, res.stdout.strip(), res.stderr.strip()


def _git(*args: str, timeout: int = 60) -> tuple[int, str, str]:
    if not args or args[0] not in GIT_ALLOWED:
        raise ToolInputError(f"git subcommand not allowed: {args[:1]}")
    return _exec(["git", *args], timeout)


def _dvc(*args: str, timeout: int = 120) -> tuple[int, str, str]:
    if not args or args[0] not in DVC_ALLOWED:
        raise ToolInputError(f"dvc subcommand not allowed: {args[:1]}")
    return _exec([sys.executable, "-m", "dvc", *args], timeout)


def _check_ref(ref: Any) -> str:
    """Validate a git ref (tag, branch or commit) and return the full commit hash."""
    if not isinstance(ref, str) or not _REF_RE.fullmatch(ref) or ".." in ref:
        raise ToolInputError(f"invalid git ref: {str(ref)[:60]!r}")
    code, out, _ = _git("rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}")
    if code != 0 or not out:
        raise ToolInputError(f"git ref {ref!r} does not exist in this repository")
    return out


def _dataset(dataset: Any) -> str:
    """Validate a dataset path: must be tracked by DVC (has a .dvc file) inside the repo."""
    dataset = DEFAULT_DATASET if dataset in (None, "") else dataset
    if not isinstance(dataset, str) or not _PATH_RE.fullmatch(dataset) or ".." in dataset:
        raise ToolInputError(f"invalid dataset path: {str(dataset)[:60]!r}")
    dvc_file = (ROOT / f"{dataset}.dvc").resolve()
    if not dvc_file.is_relative_to(ROOT.resolve()) or not dvc_file.exists():
        raise ToolInputError(f"{dataset!r} is not tracked by DVC (no {dataset}.dvc file)")
    return dataset


def _parse_dvc(text: str) -> dict[str, Any]:
    doc = yaml.safe_load(text) or {}
    outs = doc.get("outs") or [{}]
    out = outs[0]
    return {"md5": out.get("md5"), "size_bytes": out.get("size"), "path": out.get("path")}


def _dvc_at(commit: str, dataset: str) -> dict[str, Any] | None:
    code, out, _ = _git("show", f"{commit}:{dataset}.dvc")
    return _parse_dvc(out) if code == 0 and out else None


def _tags_at(commit: str) -> list[str]:
    code, out, _ = _git("tag", "--points-at", commit, "--list", "data-v*")
    return sorted(t for t in out.splitlines() if t) if code == 0 else []


def _dataset_commit(dataset: str) -> tuple[str, str, str]:
    code, out, _ = _git("log", "-1", "--format=%H%x09%cI%x09%s", "--", f"{dataset}.dvc")
    if code != 0 or not out:
        return "uncommitted", "", ""
    commit, date, subject = (out.split("\t") + ["", ""])[:3]
    return commit, date, subject


def _version_label(tags: list[str]) -> str:
    return tags[0].removeprefix("data-") if tags else "unlabeled"


# ----------------------------------------------------------------------------- tools


def list_tracked_datasets() -> dict[str, Any]:
    code, out, err = _git("ls-files", "--", "*.dvc")
    if code != 0:
        raise RuntimeError(f"git ls-files failed: {err[:200]}")
    datasets = []
    for rel in sorted(line for line in out.splitlines() if line):
        data_path = rel[: -len(".dvc")]
        info = _parse_dvc((ROOT / rel).read_text(encoding="utf-8"))
        datasets.append(
            {
                "dataset": data_path,
                "dvc_file": rel,
                "dvc_md5": info["md5"],
                "size_bytes": info["size_bytes"],
                "present_in_workspace": (ROOT / data_path).exists(),
            }
        )
    return {"datasets": datasets, "note": "empty list means no .dvc files are tracked in git"}


def get_dataset_version(dataset: str | None = None) -> dict[str, Any]:
    dataset = _dataset(dataset)
    info = _parse_dvc((ROOT / f"{dataset}.dvc").read_text(encoding="utf-8"))
    commit, date, subject = _dataset_commit(dataset)
    tags = _tags_at(commit) if commit != "uncommitted" else []
    _, head, _ = _git("rev-parse", "HEAD")
    return {
        "dataset": dataset,
        "version": _version_label(tags),
        "tags": tags,
        "dvc_md5": info["md5"],
        "size_bytes": info["size_bytes"],
        "dataset_git_commit": commit,
        "dataset_commit_date": date,
        "dataset_commit_subject": subject,
        "head_commit": head,
    }


def check_data_changes(dataset: str | None = None) -> dict[str, Any]:
    dataset = _dataset(dataset)
    code, out, err = _dvc("status", f"{dataset}.dvc", "--json")
    if code != 0:
        raise RuntimeError(f"dvc status failed: {err[:300]}")
    parsed = json.loads(out) if out else {}
    _, porcelain, _ = _git("status", "--porcelain", "--", f"{dataset}.dvc")
    return {
        "dataset": dataset,
        "data_changed": bool(parsed),
        "dvc_status": parsed,
        "dvc_file_modified_in_git": bool(porcelain),
        "meaning": "data_changed=true means the data file no longer matches the md5 recorded in its .dvc file",
    }


def compare_dataset_versions(version_a: str, version_b: str, dataset: str | None = None) -> dict[str, Any]:
    dataset = _dataset(dataset)
    commit_a, commit_b = _check_ref(version_a), _check_ref(version_b)
    info_a, info_b = _dvc_at(commit_a, dataset), _dvc_at(commit_b, dataset)
    if info_a is None or info_b is None:
        raise ToolInputError(f"{dataset}.dvc does not exist at one of the two versions")
    code, out, err = _dvc("diff", version_a, version_b, "--json")
    if code != 0:
        raise RuntimeError(f"dvc diff failed: {err[:300]}")
    size_a, size_b = info_a["size_bytes"], info_b["size_bytes"]
    return {
        "dataset": dataset,
        "version_a": {"ref": version_a, "commit": commit_a, **info_a},
        "version_b": {"ref": version_b, "commit": commit_b, **info_b},
        "identical": info_a["md5"] == info_b["md5"],
        "size_delta_bytes": (size_b - size_a) if isinstance(size_a, int) and isinstance(size_b, int) else None,
        "dvc_diff": json.loads(out) if out else {},
    }


def get_run_dataset_version(run_ref: str) -> dict[str, Any]:
    """Which dataset version trained a run? Read from MLflow, then cross-checked against git."""
    run = mlflow_tools.resolve_run(mlflow_tools._client(), run_ref)
    tags = run.data.tags
    result: dict[str, Any] = {
        "run_id": run.info.run_id,
        "run_name": tags.get("mlflow.runName"),
        "dataset_version": tags.get("dataset.version"),
        "dvc_md5": tags.get("dataset.dvc_md5"),
        "dataset_git_commit": tags.get("dataset.git_commit"),
        "dvc_status_at_training": tags.get("dataset.dvc_status"),
        "code_git_commit": tags.get("code.git_commit"),
        "code_git_dirty": tags.get("code.git_dirty"),
        "provenance_recorded": bool(tags.get("dataset.dvc_md5")),
        "verified_against_git": None,
        "verification_note": "",
    }
    if not result["provenance_recorded"]:
        result["verification_note"] = "This run has no recorded dataset provenance tags."
        return result
    dvc_file = tags.get("dataset.dvc_file") or f"{DEFAULT_DATASET}.dvc"
    dataset = dvc_file[: -len(".dvc")] if dvc_file.endswith(".dvc") else dvc_file
    try:
        commit = _check_ref(tags.get("dataset.git_commit"))
        info = _dvc_at(commit, _dataset(dataset))
    except ToolInputError as exc:
        result["verification_note"] = f"could not verify locally: {exc}"
        return result
    if info is None:
        result["verification_note"] = "the .dvc file does not exist at the recorded commit"
        result["verified_against_git"] = False
    else:
        match = info["md5"] == tags.get("dataset.dvc_md5")
        result["verified_against_git"] = match
        result["verification_note"] = (
            "recorded md5 matches the .dvc file at the recorded commit"
            if match
            else "recorded md5 does NOT match the .dvc file at the recorded commit"
        )
    return result


def explain_reproduce_version(version: str, dataset: str | None = None) -> dict[str, Any]:
    """Explain (not execute) how to restore a past dataset version."""
    dataset = _dataset(dataset)
    commit = _check_ref(version)
    info = _dvc_at(commit, dataset)
    if info is None:
        raise ToolInputError(f"{dataset}.dvc does not exist at {version!r}")
    md5 = info["md5"] or ""
    cache_file = ROOT / ".dvc" / "cache" / "files" / "md5" / md5[:2] / md5[2:]
    in_cache = cache_file.exists() if len(md5) > 2 else None
    steps = [
        f"git checkout {version} -- {dataset}.dvc",
        f"dvc checkout {dataset}.dvc   (use 'dvc pull {dataset}.dvc' if the data is not in the local cache)",
        f"dvc status {dataset}.dvc   (expect: no changes)",
        f"git checkout HEAD -- {dataset}.dvc ; dvc checkout {dataset}.dvc   (return to the current version)",
    ]
    return {
        "dataset": dataset,
        "version_ref": version,
        "commit": commit,
        "expected_md5": md5,
        "available_in_local_cache": in_cache,
        "steps": steps,
        "executed": False,
        "note": "Read-only: nothing was changed in the working tree.",
    }


_REF_HELP = "git tag or commit, e.g. data-v1"
TOOLS = [
    ToolSpec(
        "dvc_list_tracked_datasets",
        "List datasets tracked by DVC (from .dvc files in git).",
        {},
        list_tracked_datasets,
    ),
    ToolSpec(
        "dvc_get_dataset_version",
        "Current dataset version (tag), md5 and the Git commit that defines it.",
        {"dataset": "optional path, default data/dataset.csv"},
        get_dataset_version,
    ),
    ToolSpec(
        "dvc_check_data_changes",
        "Detect whether tracked data differs from what its .dvc file records.",
        {"dataset": "optional path"},
        check_data_changes,
    ),
    ToolSpec(
        "dvc_compare_dataset_versions",
        "Compare two dataset versions (md5, size, dvc diff).",
        {"version_a": _REF_HELP, "version_b": _REF_HELP, "dataset": "optional path"},
        compare_dataset_versions,
    ),
    ToolSpec(
        "dvc_get_run_dataset_version",
        "Which dataset version (and Git commit) was used to train an MLflow run, verified against git.",
        {"run_ref": "run id, 8+ char id prefix, or run name"},
        get_run_dataset_version,
    ),
    ToolSpec(
        "dvc_explain_reproduce_version",
        "Explain the exact commands to restore a previous dataset version (read-only, not executed).",
        {"version": _REF_HELP, "dataset": "optional path"},
        explain_reproduce_version,
    ),
]
