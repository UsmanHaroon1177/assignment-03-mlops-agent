"""Airflow 3 workflow tools (authenticated REST API, v2).

Scope is fixed to the approved DAG `training_pipeline`: no tool accepts a dag_id. The only
mutating tool (airflow_trigger_training) accepts just `model` and `min_f1`, both validated.
Credentials come from environment variables (fallback: the gitignored .env.airflow file).
"""
from __future__ import annotations

import os
import re
import time
from pathlib import Path
from typing import Any
from urllib.parse import quote

import requests

from agent.tools.base import ToolInputError, ToolSpec, ToolUnavailableError

ROOT = Path(os.getenv("REPO_ROOT") or Path(__file__).resolve().parents[2])
APPROVED_DAG = "training_pipeline"
TASK_IDS = ("validate_dataset", "preprocess_data", "train_model", "evaluate_and_log", "quality_gate")
MODELS = ("logreg", "rf", "gbm")
FAILED_STATES = {"failed", "upstream_failed"}
TIMEOUT = 10
TOKEN_TTL = 45 * 60
LOG_TAIL_CHARS = 3000
_RUN_ID_RE = re.compile(r"^[A-Za-z0-9_.:+\-]{1,200}$")
_token: dict[str, Any] = {"value": None, "expires": 0.0}


# ---------------------------------------------------------------- HTTP plumbing
def _base_url() -> str:
    return os.getenv("AIRFLOW_BASE_URL", "http://localhost:8080").rstrip("/")


def _credentials() -> tuple[str, str]:
    user, pw = os.getenv("AIRFLOW_USERNAME"), os.getenv("AIRFLOW_PASSWORD")
    if not (user and pw):
        env_file = ROOT / ".env.airflow"
        if env_file.is_file():
            for line in env_file.read_text(encoding="utf-8").splitlines():
                key, _, value = line.partition("=")
                if key.strip() == "AIRFLOW_USERNAME" and not user:
                    user = value.strip()
                if key.strip() == "AIRFLOW_PASSWORD" and not pw:
                    pw = value.strip()
    if not (user and pw):
        raise ToolUnavailableError("Airflow credentials are not configured (AIRFLOW_USERNAME / AIRFLOW_PASSWORD)")
    return user, pw


def _send(method: str, url: str, *, headers=None, params=None, json_body=None):
    try:
        return requests.request(method, url, headers=headers, params=params, json=json_body, timeout=TIMEOUT)
    except requests.Timeout as exc:
        raise ToolUnavailableError(f"Airflow request timed out after {TIMEOUT}s") from exc
    except requests.ConnectionError as exc:
        raise ToolUnavailableError(f"Airflow is not reachable at {_base_url()}") from exc


def _get_token(force: bool = False) -> str:
    now = time.time()
    if not force and _token["value"] and now < _token["expires"]:
        return _token["value"]
    user, pw = _credentials()
    resp = _send("POST", f"{_base_url()}/auth/token", json_body={"username": user, "password": pw})
    if resp.status_code not in (200, 201):
        raise ToolUnavailableError(f"Airflow authentication failed (HTTP {resp.status_code})")
    token = (resp.json() or {}).get("access_token")
    if not token:
        raise ToolUnavailableError("Airflow authentication returned no token")
    _token.update(value=token, expires=now + TOKEN_TTL)
    return token


def _detail(resp) -> str:
    try:
        d = resp.json().get("detail")
        if d:
            return str(d)[:300]
    except Exception:  # noqa: BLE001
        pass
    return (getattr(resp, "text", "") or "")[:300]


def _api(method: str, path: str, *, params=None, json_body=None) -> Any:
    url = f"{_base_url()}/api/v2{path}"
    resp = None
    for attempt in (0, 1):
        headers = {"Authorization": f"Bearer {_get_token(force=attempt == 1)}"}
        resp = _send(method, url, headers=headers, params=params, json_body=json_body)
        if resp.status_code != 401:
            break
    code = resp.status_code
    if code < 300:
        return resp.json()
    if code in (400, 404, 409, 422):
        raise ToolInputError(f"Airflow rejected the request (HTTP {code}): {_detail(resp)}")
    if code in (401, 403):
        raise ToolUnavailableError(f"Airflow denied access (HTTP {code})")
    raise ToolUnavailableError(f"Airflow server error (HTTP {code}): {_detail(resp)}")


# ---------------------------------------------------------------- helpers
def _dag_path() -> str:
    return f"/dags/{APPROVED_DAG}"


def _check_run_id(run_id: Any) -> str:
    if not isinstance(run_id, str) or not _RUN_ID_RE.fullmatch(run_id):
        raise ToolInputError(f"invalid dag_run_id: {str(run_id)[:60]!r}")
    return run_id


def _check_task(task_id: Any) -> str:
    if task_id not in TASK_IDS:
        raise ToolInputError(f"unknown task_id {str(task_id)[:40]!r}; valid: {list(TASK_IDS)}")
    return task_id


def _check_limit(limit: Any, default: int = 5) -> int:
    limit = default if limit in (None, "") else limit
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 20:
        raise ToolInputError("limit must be an integer between 1 and 20")
    return limit


def _run_summary(r: dict[str, Any]) -> dict[str, Any]:
    return {
        "dag_run_id": r.get("dag_run_id"),
        "state": r.get("state"),
        "run_type": r.get("run_type"),
        "start_date": r.get("start_date"),
        "end_date": r.get("end_date"),
        "conf": r.get("conf") or {},
    }


def _runs() -> list[dict[str, Any]]:
    data = _api("GET", f"{_dag_path()}/dagRuns", params={"limit": 100})
    runs = data.get("dag_runs", [])
    runs.sort(key=lambda r: r.get("run_after") or r.get("start_date") or r.get("logical_date") or "", reverse=True)
    return runs


def _resolve_run(dag_run_id: Any) -> dict[str, Any]:
    if dag_run_id in (None, ""):
        runs = _runs()
        if not runs:
            raise ToolInputError(f"no runs exist for {APPROVED_DAG} yet")
        return runs[0]
    return _api("GET", f"{_dag_path()}/dagRuns/{quote(_check_run_id(dag_run_id), safe='')}")


def _tasks(run_id: str) -> list[dict[str, Any]]:
    data = _api("GET", f"{_dag_path()}/dagRuns/{quote(run_id, safe='')}/taskInstances")
    out = []
    for t in data.get("task_instances", []):
        out.append({k: t.get(k) for k in ("task_id", "state", "try_number", "max_tries", "start_date", "end_date", "duration")})
    order = {name: i for i, name in enumerate(TASK_IDS)}
    out.sort(key=lambda t: order.get(t["task_id"], 99))
    return out


def _log_text(run_id: str, task_id: str, try_number: int) -> str:
    path = f"{_dag_path()}/dagRuns/{quote(run_id, safe='')}/taskInstances/{task_id}/logs/{int(try_number)}"
    data = _api("GET", path, params={"full_content": "true"})
    content = data.get("content", "") if isinstance(data, dict) else data
    if isinstance(content, list):
        lines = []
        for item in content:
            lines.append(str(item.get("event", item)) if isinstance(item, dict) else str(item))
        content = "\n".join(lines)
    return str(content)[-LOG_TAIL_CHARS:]


# ---------------------------------------------------------------- tools
def list_dags() -> dict[str, Any]:
    data = _api("GET", "/dags", params={"limit": 100})
    dags = []
    for d in data.get("dags", []):
        tags = [t.get("name") if isinstance(t, dict) else t for t in d.get("tags", [])]
        dags.append({"dag_id": d.get("dag_id"), "is_paused": d.get("is_paused"), "tags": tags,
                     "approved_for_trigger": d.get("dag_id") == APPROVED_DAG})
    return {"dags": dags, "count": len(dags), "approved_dag": APPROVED_DAG}


def list_dag_runs(limit: Any = None) -> dict[str, Any]:
    n = _check_limit(limit)
    runs = _runs()[:n]
    return {"dag_id": APPROVED_DAG, "runs": [_run_summary(r) for r in runs], "returned": len(runs)}


def get_run_status(dag_run_id: Any = None) -> dict[str, Any]:
    run = _resolve_run(dag_run_id)
    return {"dag_id": APPROVED_DAG, "run": _run_summary(run), "tasks": _tasks(run["dag_run_id"]),
            "resolved_latest": dag_run_id in (None, "")}


def get_failed_tasks(dag_run_id: Any = None) -> dict[str, Any]:
    run = _resolve_run(dag_run_id)
    run_id = run["dag_run_id"]
    failed = []
    for t in _tasks(run_id):
        if t["state"] in FAILED_STATES:
            item = dict(t)
            if t["state"] == "failed" and t.get("try_number"):
                try:
                    item["log_tail"] = _log_text(run_id, t["task_id"], t["try_number"])
                except Exception as exc:  # noqa: BLE001 - diagnostics are best effort
                    item["log_tail"] = None
                    item["log_error"] = f"{type(exc).__name__}: {exc}"[:200]
            failed.append(item)
    return {"dag_id": APPROVED_DAG, "dag_run_id": run_id, "run_state": run.get("state"),
            "failed_tasks": failed,
            "note": None if failed else "no failed tasks recorded for this run"}


def get_task_logs(task_id: Any, dag_run_id: Any = None, try_number: Any = None) -> dict[str, Any]:
    task_id = _check_task(task_id)
    run = _resolve_run(dag_run_id)
    run_id = run["dag_run_id"]
    if try_number in (None, ""):
        ti = _api("GET", f"{_dag_path()}/dagRuns/{quote(run_id, safe='')}/taskInstances/{task_id}")
        try_number = ti.get("try_number") or 1
    if isinstance(try_number, bool) or not isinstance(try_number, int) or not 1 <= try_number <= 10:
        raise ToolInputError("try_number must be an integer between 1 and 10")
    return {"dag_run_id": run_id, "task_id": task_id, "try_number": try_number,
            "log_tail": _log_text(run_id, task_id, try_number), "tail_chars": LOG_TAIL_CHARS}


def trigger_training(model: Any = "logreg", min_f1: Any = 0.90) -> dict[str, Any]:
    model = "logreg" if model in (None, "") else model
    min_f1 = 0.90 if min_f1 in (None, "") else min_f1
    if model not in MODELS:
        raise ToolInputError(f"model must be one of {list(MODELS)}")
    if isinstance(min_f1, bool) or not isinstance(min_f1, (int, float)) or not 0 <= min_f1 <= 1:
        raise ToolInputError("min_f1 must be a number between 0 and 1")
    run = _api("POST", f"{_dag_path()}/dagRuns", json_body={"logical_date": None, "conf": {"model": model, "min_f1": float(min_f1)}})
    return {"dag_id": APPROVED_DAG, "dag_run_id": run.get("dag_run_id"), "state": run.get("state"),
            "conf": {"model": model, "min_f1": float(min_f1)},
            "note": "triggered only; completion not awaited. Use airflow_get_run_status to check the result."}


TOOLS: list[ToolSpec] = [
    ToolSpec("airflow_list_dags", "List Airflow DAGs and whether each is paused / approved for triggering.", {}, list_dags),
    ToolSpec("airflow_list_dag_runs", "List recent runs of the training_pipeline DAG (newest first).",
             {"limit": "optional int 1-20, default 5"}, list_dag_runs),
    ToolSpec("airflow_get_run_status", "State of a training_pipeline run and each task's state. Defaults to the latest run.",
             {"dag_run_id": "optional run id; latest if omitted"}, get_run_status),
    ToolSpec("airflow_get_failed_tasks", "Failed / upstream_failed tasks of a run with log tail diagnostics. Defaults to the latest run.",
             {"dag_run_id": "optional run id; latest if omitted"}, get_failed_tasks),
    ToolSpec("airflow_get_task_logs", "Log tail of one training_pipeline task.",
             {"task_id": "one of " + ", ".join(TASK_IDS), "dag_run_id": "optional; latest if omitted",
              "try_number": "optional attempt number; current attempt if omitted"}, get_task_logs),
    ToolSpec("airflow_trigger_training", "Trigger a training_pipeline run (needs explicit user confirmation).",
             {"model": "optional logreg|rf|gbm, default logreg", "min_f1": "optional quality-gate threshold 0-1, default 0.90"},
             trigger_training, mutating=True),
]