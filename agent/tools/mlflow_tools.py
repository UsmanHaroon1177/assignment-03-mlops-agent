"""MLflow tools (read-only). Uses MlflowClient, never the web UI.

Only FINISHED runs are used for model selection: a FAILED run can still have logged
metrics before it crashed (see PROGRESS.md bug log).
"""
from __future__ import annotations

import os
import re
from typing import Any

from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from agent.tools.base import ToolInputError, ToolSpec

DEFAULT_EXPERIMENT = "breast-cancer-classifier"
MAX_RUNS = 50
CLOSE_GAP = 0.01
PROVENANCE_TAGS = (
    "dataset.version",
    "dataset.dvc_md5",
    "dataset.git_commit",
    "dataset.dvc_status",
    "code.git_commit",
    "code.git_dirty",
)
_NAME_RE = re.compile(r"^[A-Za-z0-9_.\- ]{1,100}$")
_METRIC_RE = re.compile(r"^[A-Za-z0-9_.\-]{1,64}$")
_FULL_ID_RE = re.compile(r"^[0-9a-f]{32}$")
_PREFIX_RE = re.compile(r"^[0-9a-f]{8,31}$")
_STATUSES = ("FINISHED", "FAILED", "RUNNING", "ALL")


def _client() -> MlflowClient:
    # short timeout and one retry so a dead server fails fast instead of hanging the agent
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "10")
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "1")
    uri = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    return MlflowClient(tracking_uri=uri)


def _check(pattern: re.Pattern[str], value: Any, label: str) -> str:
    if not isinstance(value, str) or not pattern.fullmatch(value):
        raise ToolInputError(f"invalid {label}: {str(value)[:60]!r}")
    return value


def _clamp(value: Any, low: int, high: int, default: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return default
    return max(low, min(high, number))


def _experiment(client: MlflowClient, name: str | None):
    name = _check(_NAME_RE, name or DEFAULT_EXPERIMENT, "experiment_name")
    exp = client.get_experiment_by_name(name)
    if exp is None:
        raise ToolInputError(f"experiment {name!r} not found")
    return exp


def _runs(client: MlflowClient, exp_id: str, status: str, limit: int):
    status = status.upper()
    if status not in _STATUSES:
        raise ToolInputError(f"status must be one of {_STATUSES}")
    flt = "" if status == "ALL" else f"attributes.status = '{status}'"
    return client.search_runs(
        [exp_id], filter_string=flt, order_by=["attributes.start_time DESC"], max_results=limit
    )


def _summary(run) -> dict[str, Any]:
    tags = run.data.tags
    return {
        "run_id": run.info.run_id,
        "run_name": tags.get("mlflow.runName"),
        "status": run.info.status,
        "start_time_ms": run.info.start_time,
        "metrics": {k: round(v, 4) for k, v in run.data.metrics.items()},
        "params": dict(run.data.params),
        "provenance": {k: tags.get(k) for k in PROVENANCE_TAGS},
    }


def _registered_versions(client: MlflowClient, run_id: str) -> list[dict[str, Any]]:
    versions = client.search_model_versions(f"run_id='{run_id}'")
    return [{"name": v.name, "version": str(v.version)} for v in versions]


def resolve_run(client: MlflowClient, ref: Any):
    """Find a run by full id, id prefix (8+ hex chars) or run name."""
    ref = str(ref).strip()
    if not ref or len(ref) > 100:
        raise ToolInputError("run reference is empty or too long")
    if _FULL_ID_RE.fullmatch(ref):
        try:
            return client.get_run(ref)
        except MlflowException as exc:
            if getattr(exc, "error_code", "") == "RESOURCE_DOES_NOT_EXIST":
                raise ToolInputError(f"run {ref[:8]} not found") from exc
            raise
    exp_ids = [e.experiment_id for e in client.search_experiments()]
    runs = client.search_runs(exp_ids, max_results=1000) if exp_ids else []
    matches = []
    for run in runs:
        name = run.data.tags.get("mlflow.runName")
        if name == ref or (_PREFIX_RE.fullmatch(ref) and run.info.run_id.startswith(ref)):
            matches.append(run)
    if not matches:
        raise ToolInputError(f"no run matches {ref!r} (use run id, 8+ char id prefix, or run name)")
    if len(matches) > 1:
        ids = [m.info.run_id[:8] for m in matches[:5]]
        raise ToolInputError(f"{ref!r} is ambiguous; matching run ids: {ids}")
    return matches[0]


# ----------------------------------------------------------------------------- tools


def list_experiments() -> dict[str, Any]:
    client = _client()
    out = []
    for exp in client.search_experiments():
        finished = client.search_runs(
            [exp.experiment_id], filter_string="attributes.status = 'FINISHED'", max_results=1000
        )
        out.append(
            {
                "experiment_id": exp.experiment_id,
                "name": exp.name,
                "lifecycle_stage": exp.lifecycle_stage,
                "finished_runs": len(finished),
            }
        )
    return {"experiments": out}


def list_runs(experiment_name: str | None = None, max_results: int = 10, status: str = "FINISHED") -> dict[str, Any]:
    client = _client()
    exp = _experiment(client, experiment_name)
    limit = _clamp(max_results, 1, MAX_RUNS, 10)
    runs = _runs(client, exp.experiment_id, status, limit)
    return {
        "experiment": exp.name,
        "status_filter": status.upper(),
        "order": "most recent first",
        "runs": [_summary(r) for r in runs],
    }


def get_run_details(run_id: str) -> dict[str, Any]:
    client = _client()
    run = resolve_run(client, run_id)
    details = _summary(run)
    details["tags"] = {
        k: v for k, v in run.data.tags.items() if not k.startswith("mlflow.") or k == "mlflow.runName"
    }
    details["artifacts"] = [
        {"path": a.path, "is_dir": a.is_dir} for a in client.list_artifacts(run.info.run_id)
    ]
    details["registered_versions"] = _registered_versions(client, run.info.run_id)
    return details


def find_best_run(
    metric: str = "f1",
    experiment_name: str | None = None,
    recent_n: int | None = None,
    higher_is_better: bool = True,
) -> dict[str, Any]:
    client = _client()
    _check(_METRIC_RE, metric, "metric")
    exp = _experiment(client, experiment_name)
    limit = _clamp(recent_n, 1, MAX_RUNS, MAX_RUNS)
    runs = _runs(client, exp.experiment_id, "FINISHED", limit)
    scored = [r for r in runs if metric in r.data.metrics]
    if not scored:
        available = sorted({m for r in runs for m in r.data.metrics})
        raise ToolInputError(f"no FINISHED run has metric {metric!r}; available metrics: {available}")
    ranked = sorted(scored, key=lambda r: r.data.metrics[metric], reverse=bool(higher_is_better))
    best = ranked[0]
    best_summary = _summary(best)
    best_summary["registered_versions"] = _registered_versions(client, best.info.run_id)

    caveats = [
        f"{metric} comes from one hold-out split per run; the highest value alone does not make "
        "a model production-ready."
    ]
    versions = {r.data.tags.get("dataset.version") or "unknown" for r in ranked}
    if len(versions) > 1:
        caveats.append(
            f"Candidates were trained on different dataset versions ({sorted(versions)}); "
            "scores are not directly comparable."
        )
    if len(ranked) >= 2:
        gap = abs(ranked[0].data.metrics[metric] - ranked[1].data.metrics[metric])
        if gap < CLOSE_GAP:
            caveats.append(
                f"Top two candidates differ by only {gap:.4f} in {metric}; this may be within noise."
            )
    prov = best_summary["provenance"]
    if prov.get("dataset.dvc_status") != "up-to-date" or prov.get("code.git_dirty") == "true":
        caveats.append("Best run's provenance is missing, modified or from uncommitted code.")
    if not best_summary["registered_versions"]:
        caveats.append("Best run is not registered in the model registry.")

    return {
        "experiment": exp.name,
        "metric": metric,
        "higher_is_better": bool(higher_is_better),
        "runs_considered": len(ranked),
        "runs_without_metric": len(runs) - len(scored),
        "best": best_summary,
        "ranking": [
            {
                "run_id": r.info.run_id,
                "run_name": r.data.tags.get("mlflow.runName"),
                "value": round(r.data.metrics[metric], 4),
                "dataset_version": r.data.tags.get("dataset.version"),
            }
            for r in ranked
        ],
        "caveats": caveats,
        "is_deployment_decision": False,
    }


def compare_runs(run_ids: list[str]) -> dict[str, Any]:
    if not isinstance(run_ids, list) or not 2 <= len(run_ids) <= 5:
        raise ToolInputError("run_ids must be a list of 2 to 5 run references")
    client = _client()
    runs = [resolve_run(client, ref) for ref in run_ids]
    names = [r.data.tags.get("mlflow.runName") or r.info.run_id[:8] for r in runs]
    metric_names = sorted({m for r in runs for m in r.data.metrics})
    metrics = {m: {n: r.data.metrics.get(m) for n, r in zip(names, runs, strict=True)} for m in metric_names}
    param_names = sorted({p for r in runs for p in r.data.params})
    differing = {
        p: {n: r.data.params.get(p) for n, r in zip(names, runs, strict=True)}
        for p in param_names
        if len({r.data.params.get(p) for r in runs}) > 1
    }
    winners = {}
    for m, vals in metrics.items():
        present = {n: v for n, v in vals.items() if v is not None}
        if present and m != "train_seconds":
            winners[m] = max(present, key=present.get)
    return {
        "runs": [_summary(r) for r in runs],
        "metrics": {m: {n: (round(v, 4) if v is not None else None) for n, v in d.items()} for m, d in metrics.items()},
        "differing_params": differing,
        "highest_per_metric": winners,
        "note": "highest_per_metric is a raw comparison, not a recommendation.",
    }


def compare_experiments(experiment_names: list[str], metric: str = "f1") -> dict[str, Any]:
    if not isinstance(experiment_names, list) or not 1 <= len(experiment_names) <= 5:
        raise ToolInputError("experiment_names must be a list of 1 to 5 names")
    _check(_METRIC_RE, metric, "metric")
    client = _client()
    out = []
    for name in experiment_names:
        exp = _experiment(client, name)
        runs = [r for r in _runs(client, exp.experiment_id, "FINISHED", MAX_RUNS) if metric in r.data.metrics]
        entry: dict[str, Any] = {"experiment": exp.name, "runs_with_metric": len(runs)}
        if runs:
            values = [r.data.metrics[metric] for r in runs]
            top = max(runs, key=lambda r: r.data.metrics[metric])
            entry.update(
                best_value=round(max(values), 4),
                best_run_name=top.data.tags.get("mlflow.runName"),
                mean_value=round(sum(values) / len(values), 4),
            )
        out.append(entry)
    return {"metric": metric, "experiments": out}


def get_model_registry(model_name: str | None = None) -> dict[str, Any]:
    client = _client()
    if model_name:
        _check(_NAME_RE, model_name, "model_name")
        names = [model_name]
    else:
        names = [m.name for m in client.search_registered_models()]
    models = []
    for name in names:
        versions = client.search_model_versions(f"name='{name}'")
        items = [
            {
                "version": str(v.version),
                "run_id": v.run_id,
                "status": v.status,
                "aliases": list(getattr(v, "aliases", None) or []),
                "stage": getattr(v, "current_stage", None),
            }
            for v in sorted(versions, key=lambda x: int(x.version), reverse=True)
        ]
        models.append({"name": name, "versions": items})
    return {"registered_models": models, "note": "empty list means nothing is registered"}


TOOLS = [
    ToolSpec(
        "mlflow_list_experiments",
        "List MLflow experiments with their number of finished runs.",
        {},
        list_experiments,
    ),
    ToolSpec(
        "mlflow_list_runs",
        "List recent runs (default FINISHED only) with metrics, params and dataset provenance tags.",
        {
            "experiment_name": "optional, defaults to the project experiment",
            "max_results": "optional int 1-50, default 10",
            "status": "optional FINISHED (default) | FAILED | RUNNING | ALL",
        },
        list_runs,
    ),
    ToolSpec(
        "mlflow_get_run_details",
        "Metrics, params, tags, artifacts and registry versions of one run.",
        {"run_id": "run id, 8+ char id prefix, or run name"},
        get_run_details,
    ),
    ToolSpec(
        "mlflow_find_best_run",
        "Rank FINISHED runs by a metric and return the best with caveats (not a deployment decision).",
        {
            "metric": "optional, default f1",
            "experiment_name": "optional",
            "recent_n": "optional int, only consider the N most recent runs",
            "higher_is_better": "optional bool, default true",
        },
        find_best_run,
    ),
    ToolSpec(
        "mlflow_compare_runs",
        "Side-by-side metrics and differing params for 2-5 runs.",
        {"run_ids": "list of 2-5 run ids / prefixes / names"},
        compare_runs,
    ),
    ToolSpec(
        "mlflow_compare_experiments",
        "Compare best and mean metric across 1-5 experiments.",
        {"experiment_names": "list of experiment names", "metric": "optional, default f1"},
        compare_experiments,
    ),
    ToolSpec(
        "mlflow_get_model_registry",
        "Registered models and their versions, aliases and source runs.",
        {"model_name": "optional; omit to list all registered models"},
        get_model_registry,
    ),
]
