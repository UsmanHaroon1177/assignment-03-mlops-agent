import pytest
import requests

from agent.tools import airflow_tools as af
from agent.tools.base import execute

BASE = "http://airflow.test"
TOOL = {t.name: t for t in af.TOOLS}
RUN = "manual__2026-11-13T10:00:00+00:00"


class Resp:
    def __init__(self, code=200, data=None, text=""):
        self.status_code, self._data, self.text = code, data, text

    def json(self):
        return self._data


@pytest.fixture(autouse=True)
def env(monkeypatch):
    monkeypatch.setenv("AIRFLOW_BASE_URL", BASE)
    monkeypatch.setenv("AIRFLOW_USERNAME", "admin")
    monkeypatch.setenv("AIRFLOW_PASSWORD", "s3cret-pass-123")
    af._token.update(value=None, expires=0.0)


@pytest.fixture
def api(monkeypatch):
    state = {"routes": {}, "calls": [], "tokens": 0}

    def send(method, url, *, headers=None, params=None, json_body=None):
        if url.endswith("/auth/token"):
            state["tokens"] += 1
            return Resp(200, {"access_token": f"tok{state['tokens']}"})
        path = url.replace(f"{BASE}/api/v2", "")
        state["calls"].append((method, path, json_body))
        route = state["routes"].get((method, path), Resp(404, {"detail": "not found"}))
        return route() if callable(route) else route

    monkeypatch.setattr(af, "_send", send)
    return state


def run(name, **kw):
    return execute(name, TOOL[name].fn, request_id="t1", **kw)


def tasks_payload(states):
    return {"task_instances": [{"task_id": t, "state": s, "try_number": 1, "max_tries": 0} for t, s in states.items()]}


def test_list_dags_marks_approved(api):
    api["routes"][("GET", "/dags")] = Resp(200, {"dags": [
        {"dag_id": "training_pipeline", "is_paused": False, "tags": [{"name": "mlops"}]},
        {"dag_id": "other", "is_paused": True, "tags": []}]})
    res = run("airflow_list_dags")
    assert res.status == "ok"
    approved = {d["dag_id"]: d["approved_for_trigger"] for d in res.data["dags"]}
    assert approved == {"training_pipeline": True, "other": False}


def test_latest_run_status_with_ordered_tasks(api):
    base = "/dags/training_pipeline"
    api["routes"][("GET", f"{base}/dagRuns")] = Resp(200, {"dag_runs": [
        {"dag_run_id": "old", "state": "success", "run_after": "2026-11-12T00:00:00Z"},
        {"dag_run_id": RUN, "state": "failed", "run_after": "2026-11-13T10:00:00Z"}]})
    api["routes"][("GET", f"{base}/dagRuns/{af.quote(RUN, safe='')}/taskInstances")] = Resp(
        200, tasks_payload({"quality_gate": "failed", "validate_dataset": "success"}))
    res = run("airflow_get_run_status")
    assert res.status == "ok" and res.data["run"]["dag_run_id"] == RUN
    assert [t["task_id"] for t in res.data["tasks"]] == ["validate_dataset", "quality_gate"]


def test_failed_tasks_include_log_tail(api):
    base = "/dags/training_pipeline"
    rid = af.quote(RUN, safe="")
    api["routes"][("GET", f"{base}/dagRuns/{rid}")] = Resp(200, {"dag_run_id": RUN, "state": "failed"})
    api["routes"][("GET", f"{base}/dagRuns/{rid}/taskInstances")] = Resp(
        200, tasks_payload({"train_model": "success", "quality_gate": "failed", "extra": "upstream_failed"}))
    api["routes"][("GET", f"{base}/dagRuns/{rid}/taskInstances/quality_gate/logs/1")] = Resp(
        200, {"content": [{"event": "Quality gate FAILED: f1=0.95 < min_f1=0.999"}]})
    res = run("airflow_get_failed_tasks", dag_run_id=RUN)
    assert res.status == "ok"
    ids = {t["task_id"] for t in res.data["failed_tasks"]}
    assert ids == {"quality_gate", "extra"}
    gate = next(t for t in res.data["failed_tasks"] if t["task_id"] == "quality_gate")
    assert "Quality gate FAILED" in gate["log_tail"]


def test_trigger_sends_only_model_and_min_f1(api):
    api["routes"][("POST", "/dags/training_pipeline/dagRuns")] = Resp(200, {"dag_run_id": "r1", "state": "queued"})
    res = run("airflow_trigger_training", model="rf", min_f1=0.92)
    assert res.status == "ok" and res.data["dag_run_id"] == "r1"
    post = [c for c in api["calls"] if c[0] == "POST"][0]
    assert post[2] == {"logical_date": None, "conf": {"model": "rf", "min_f1": 0.92}}


@pytest.mark.parametrize("kw", [{"model": "xgboost"}, {"min_f1": 1.5}, {"min_f1": "high"}, {"min_f1": True}])
def test_trigger_rejects_invalid_params_without_calling_airflow(api, kw):
    res = run("airflow_trigger_training", **kw)
    assert res.status == "rejected"
    assert api["calls"] == []


def test_trigger_rejects_unknown_argument_such_as_dag_id(api):
    res = run("airflow_trigger_training", dag_id="other_dag")
    assert res.status == "rejected" and api["calls"] == []


def test_trigger_is_flagged_mutating_and_reads_are_not():
    assert TOOL["airflow_trigger_training"].mutating is True
    assert not any(t.mutating for t in af.TOOLS if t.name != "airflow_trigger_training")


@pytest.mark.parametrize("kw", [{"dag_run_id": "../../x"}, {"dag_run_id": "a b"}, {"dag_run_id": 5}])
def test_invalid_run_id_rejected(api, kw):
    assert run("airflow_get_run_status", **kw).status == "rejected"
    assert api["calls"] == []


def test_unknown_task_id_rejected(api):
    assert run("airflow_get_task_logs", task_id="rm -rf").status == "rejected"


def test_unreachable_airflow_is_unavailable_not_invented(monkeypatch):
    def boom(*a, **k):
        raise requests.ConnectionError("Max retries exceeded")

    monkeypatch.setattr(af.requests, "request", boom)
    res = run("airflow_list_dag_runs")
    assert res.status == "unavailable" and res.data is None


def test_timeout_is_unavailable(monkeypatch):
    def slow(*a, **k):
        raise requests.Timeout("read timed out")

    monkeypatch.setattr(af.requests, "request", slow)
    assert run("airflow_list_dags").status == "unavailable"


def test_missing_credentials_unavailable_and_secret_not_leaked(monkeypatch, tmp_path):
    monkeypatch.delenv("AIRFLOW_USERNAME")
    monkeypatch.delenv("AIRFLOW_PASSWORD")
    monkeypatch.setattr(af, "ROOT", tmp_path)
    res = run("airflow_list_dags")
    assert res.status == "unavailable" and "credentials" in res.error


def test_expired_token_is_refreshed_once(api):
    seen = {"n": 0}

    def flaky():
        seen["n"] += 1
        return Resp(401, {"detail": "expired"}) if seen["n"] == 1 else Resp(200, {"dags": []})

    api["routes"][("GET", "/dags")] = flaky
    assert run("airflow_list_dags").status == "ok"
    assert api["tokens"] == 2


def test_server_error_is_unavailable(api):
    api["routes"][("GET", "/dags")] = Resp(503, {"detail": "scheduler down"})
    assert run("airflow_list_dags").status == "unavailable"


def test_unknown_run_is_rejected(api):
    res = run("airflow_get_run_status", dag_run_id="does-not-exist")
    assert res.status == "rejected"


def test_failed_task_log_includes_exception_detail(api):
    base = "/dags/training_pipeline"
    rid = af.quote(RUN, safe="")
    api["routes"][("GET", f"{base}/dagRuns/{rid}")] = Resp(200, {"dag_run_id": RUN, "state": "failed"})
    api["routes"][("GET", f"{base}/dagRuns/{rid}/taskInstances")] = Resp(
        200, tasks_payload({"train_model": "failed"}))
    api["routes"][("GET", f"{base}/dagRuns/{rid}/taskInstances/train_model/logs/1")] = Resp(200, {"content": [
        {"event": "Task failed with exception", "level": "error",
         "error_detail": [{"exc_type": "MlflowException", "exc_value": "403 Invalid Host header", "frames": []}]}]})
    res = run("airflow_get_failed_tasks", dag_run_id=RUN)
    assert res.status == "ok"
    tail = res.data["failed_tasks"][0]["log_tail"]
    assert "Task failed with exception" in tail and "MlflowException" in tail and "Invalid Host header" in tail
