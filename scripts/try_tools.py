"""Smoke test: call every Phase 3 tool against the REAL MLflow server and DVC repo.

Usage: python scripts/try_tools.py [--full]
Each line shows status and duration. Two calls are deliberately invalid and must be 'rejected'.
JSON log lines (one per tool call) go to stderr; add 2>$null in PowerShell to hide them.
"""
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

# make "agent" importable when run as: python scripts/try_tools.py
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.logging_utils import configure, new_request_id  # noqa: E402
from agent.tools import dvc_tools, mlflow_tools  # noqa: E402
from agent.tools.base import execute  # noqa: E402

load_dotenv()

CALLS = [
    ("mlflow_list_experiments", {}, "ok"),
    ("mlflow_list_runs", {"max_results": 3}, "ok"),
    ("mlflow_find_best_run", {"metric": "f1", "recent_n": 5}, "ok"),
    ("mlflow_compare_runs", {"run_ids": ["v1-logreg", "v2-logreg"]}, "ok"),
    ("mlflow_get_run_details", {"run_id": "v1-gbm"}, "ok"),
    ("mlflow_get_model_registry", {}, "ok"),
    ("dvc_list_tracked_datasets", {}, "ok"),
    ("dvc_get_dataset_version", {}, "ok"),
    ("dvc_check_data_changes", {}, "ok"),
    ("dvc_compare_dataset_versions", {"version_a": "data-v1", "version_b": "data-v2"}, "ok"),
    ("dvc_get_run_dataset_version", {"run_ref": "v1-rf"}, "ok"),
    ("dvc_explain_reproduce_version", {"version": "data-v1"}, "ok"),
    ("dvc_explain_reproduce_version", {"version": "--output=x"}, "rejected"),
    ("mlflow_get_run_details", {"run_id": "no-such-run"}, "rejected"),
]


def main() -> int:
    configure()  # JSON log line per tool call on stderr
    full = "--full" in sys.argv
    specs = {s.name: s for s in mlflow_tools.TOOLS + dvc_tools.TOOLS}
    request_id = new_request_id()
    bad = 0
    for name, args, expected in CALLS:
        res = execute(name, specs[name].fn, request_id=request_id, **args)
        flag = "PASS" if res.status == expected else "FAIL"
        bad += flag == "FAIL"
        print(f"[{flag}] {name:34} {res.status:11} {res.duration_ms:5} ms  args={json.dumps(args)}")
        body = res.error if res.error else json.dumps(res.data, indent=2 if full else None)
        print("       " + (body if full else body[:230]))
    print(f"\nrequest_id={request_id}  failures={bad}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
