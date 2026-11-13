import json
import logging

from agent.fake_llm import FakeLLM
from agent.orchestrator import Orchestrator
from agent.schemas import LLMAction
from agent.tools.base import ToolSpec


def tool(name, fn):
    return ToolSpec(name, name, {}, fn, False)


def calls(*names):
    return {"action": "tools", "calls": [{"tool": n, "args": {}} for n in names]}


def final(**kw):
    base = {"action": "final", "summary": "s", "known_facts": [], "uncertainties": [], "recommended_actions": []}
    base.update(kw)
    return base


def runs(state):
    return lambda **kw: {"runs": [{"dag_run_id": "r1", "state": state}]}


def test_object_shaped_entries_become_plain_text():
    a = LLMAction.model_validate({
        "action": "final",
        "uncertainties": [{"claim": "link unverified"}, {"text": "t"}, 5],
        "recommended_actions": [{"description": "do x"}],
    })
    assert a.uncertainties == ["link unverified", "t", "5"]
    assert a.recommended_actions == ["do x"]


def test_failed_latest_run_without_inspection_is_flagged():
    tools = {"airflow_list_dag_runs": tool("airflow_list_dag_runs", runs("failed"))}
    resp = Orchestrator(FakeLLM([calls("airflow_list_dag_runs"), final()]), tools).run("q")
    assert any("not inspected" in u for u in resp.uncertainties)
    assert "not inspected" in resp.answer


def test_inspected_failure_or_successful_run_is_not_flagged():
    inspect = {"airflow_list_dag_runs": tool("airflow_list_dag_runs", runs("failed")),
               "airflow_get_failed_tasks": tool("airflow_get_failed_tasks", lambda **kw: {"failed_tasks": []})}
    r1 = Orchestrator(FakeLLM([calls("airflow_list_dag_runs", "airflow_get_failed_tasks"), final()]), inspect).run("q")
    assert not any("not inspected" in u for u in r1.uncertainties)
    ok = {"airflow_list_dag_runs": tool("airflow_list_dag_runs", runs("success"))}
    r2 = Orchestrator(FakeLLM([calls("airflow_list_dag_runs"), final()]), ok).run("q")
    assert not any("not inspected" in u for u in r2.uncertainties)


def test_summary_is_labelled_as_not_evidence_checked():
    resp = Orchestrator(FakeLLM([final(summary="All good")]), {}).run("q")
    assert "model-written" in resp.answer and "All good" in resp.answer


def test_tool_timeout_is_reported_as_unavailable_and_never_a_fact():
    def slow(**kw):
        raise TimeoutError("read timed out")
    tools = {"mlflow_list_runs": tool("mlflow_list_runs", slow)}
    facts = [{"claim": "x", "evidence": ["mlflow_list_runs"]}]
    resp = Orchestrator(FakeLLM([calls("mlflow_list_runs"), final(known_facts=facts)]), tools).run("q")
    assert resp.tool_results[0].status == "unavailable" and resp.known_facts == []
    assert any("mlflow_list_runs" in u and "unavailable" in u for u in resp.uncertainties)


def test_logs_carry_request_id_tool_status_duration_and_no_secrets(caplog):
    secret = "AIza" + "z" * 35
    tools = {"a": tool("a", lambda **kw: {"note": f"key {secret}"})}
    with caplog.at_level(logging.INFO, logger="mlops_agent"):
        resp = Orchestrator(FakeLLM([calls("a"), final()]), tools).run("q", request_id="rid-1")
    rows = [r for r in caplog.records if getattr(r, "fields", {}).get("tool") == "a"]
    assert rows and rows[0].fields["request_id"] == "rid-1"
    assert rows[0].fields["status"] == "ok" and "duration_ms" in rows[0].fields
    everything = " ".join(r.getMessage() + json.dumps(getattr(r, "fields", {}), default=str) for r in caplog.records)
    assert secret not in everything and secret not in str(resp.tool_results[0].data)