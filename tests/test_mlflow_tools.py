from types import SimpleNamespace as NS

import pytest
from mlflow.exceptions import MlflowException

from agent.tools import mlflow_tools as mt
from agent.tools.base import ToolInputError, execute


def run(rid, name, status, f1, version="v1", start=0, dirty="false", dvc="up-to-date"):
    tags = {
        "mlflow.runName": name,
        "dataset.version": version,
        "dataset.dvc_status": dvc,
        "code.git_dirty": dirty,
        "mlflow.user": "x",
    }
    return NS(
        info=NS(run_id=rid, status=status, start_time=start, experiment_id="1"),
        data=NS(metrics={"f1": f1, "accuracy": f1 - 0.01}, params={"model": name}, tags=tags),
    )


class FakeClient:
    def __init__(self, runs, versions=()):
        self.runs, self.versions = runs, list(versions)

    def get_experiment_by_name(self, name):
        return NS(experiment_id="1", name=name, lifecycle_stage="active") if name == "exp" else None

    def search_experiments(self):
        return [NS(experiment_id="1", name="exp", lifecycle_stage="active")]

    def search_runs(self, experiment_ids, filter_string="", order_by=None, max_results=1000):
        runs = [r for r in self.runs if not filter_string or r.info.status in filter_string]
        return sorted(runs, key=lambda r: -r.info.start_time)[:max_results]

    def get_run(self, run_id):
        for r in self.runs:
            if r.info.run_id == run_id:
                return r
        raise MlflowException("nope", error_code=1)

    def list_artifacts(self, run_id):
        return [NS(path="model", is_dir=True)]

    def search_model_versions(self, flt):
        return [v for v in self.versions if v.run_id in flt or v.name in flt]

    def search_registered_models(self):
        return [NS(name=v.name) for v in self.versions[:1]]


A, B, C, D = "a" * 32, "b" * 32, "c" * 32, "d" * 32


@pytest.fixture
def client(monkeypatch):
    runs = [
        run(A, "v1-logreg", "FINISHED", 0.9811, "v1", start=1),
        run(B, "v1-rf", "FINISHED", 0.9515, "v1", start=2),
        run(C, "v2-logreg", "FINISHED", 0.9660, "v2", start=3),
        run(D, "crashed", "FAILED", 0.9999, "v2", start=4),  # failed run with a great metric
    ]
    fake = FakeClient(runs, [NS(name="clf", version=1, run_id=A, status="READY", aliases=[], current_stage="None")])
    monkeypatch.setattr(mt, "_client", lambda: fake)
    return fake


def test_best_run_ignores_failed_runs(client):
    # regression: the FAILED run has the highest f1 but must never be recommended
    res = mt.find_best_run(metric="f1", experiment_name="exp")
    assert res["best"]["run_name"] == "v1-logreg"
    assert all(r["run_name"] != "crashed" for r in res["ranking"])


def test_best_run_adds_honest_caveats(client):
    res = mt.find_best_run(metric="f1", experiment_name="exp")
    text = " ".join(res["caveats"])
    assert "production-ready" in text
    assert "different dataset versions" in text
    assert res["is_deployment_decision"] is False


def test_best_run_flags_close_scores(client):
    client.runs[1] = run(B, "v1-rf", "FINISHED", 0.9805, "v1", start=2)
    res = mt.find_best_run(metric="f1", experiment_name="exp")
    assert any("within noise" in c for c in res["caveats"])


def test_best_run_unknown_metric_and_unknown_experiment_rejected(client):
    with pytest.raises(ToolInputError):
        mt.find_best_run(metric="nope", experiment_name="exp")
    with pytest.raises(ToolInputError):
        mt.find_best_run(experiment_name="missing")


def test_injection_like_names_rejected(client):
    with pytest.raises(ToolInputError):
        mt.list_runs(experiment_name="exp' OR '1'='1")
    with pytest.raises(ToolInputError):
        mt.find_best_run(metric="f1; drop")


def test_resolve_run_by_prefix_name_and_ambiguity(client):
    assert mt.resolve_run(client, "aaaaaaaa").info.run_id == A
    assert mt.resolve_run(client, "v1-rf").info.run_id == B
    client.runs.append(run("a" * 8 + "f" * 24, "dup", "FINISHED", 0.5))
    with pytest.raises(ToolInputError):
        mt.resolve_run(client, "aaaaaaaa")


def test_compare_runs_lists_differences(client):
    res = mt.compare_runs(["v1-logreg", "v1-rf"])
    assert res["differing_params"]["model"] == {"v1-logreg": "v1-logreg", "v1-rf": "v1-rf"}
    assert res["highest_per_metric"]["f1"] == "v1-logreg"
    with pytest.raises(ToolInputError):
        mt.compare_runs(["v1-rf"])


def test_registry_and_run_details(client):
    reg = mt.get_model_registry()
    assert reg["registered_models"][0]["versions"][0]["run_id"] == A
    details = mt.get_run_details("v1-logreg")
    assert details["registered_versions"] == [{"name": "clf", "version": "1"}]


def test_unavailable_server_returns_no_data(monkeypatch):
    def down():
        raise ConnectionError("Max retries exceeded with url: /api/2.0/mlflow")

    monkeypatch.setattr(mt, "_client", down)
    res = execute("mlflow_list_runs", mt.list_runs, request_id="r1")
    assert res.status == "unavailable" and res.data is None
