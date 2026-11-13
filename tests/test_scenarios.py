import re

from agent import prompts
from agent.fake_llm import FakeLLM
from agent.orchestrator import Orchestrator
from agent.policy import guard_retraining, is_retrain_recommendation
from agent.registry import load_tools
from agent.tools.base import ToolSpec, ToolUnavailableError

OK_TASKS = [{"task_id": t, "state": "success"} for t in
            ("validate_dataset", "preprocess_data", "train_model", "evaluate_and_log", "quality_gate")]
GATE_FAILED = OK_TASKS[:-1] + [{"task_id": "quality_gate", "state": "failed"}]


def build(airflow_tasks=None, airflow_down=False):
    trig = {"n": 0}

    def trigger(**kw):
        trig["n"] += 1
        return {"dag_run_id": "x"}

    def airflow_status(**kw):
        if airflow_down:
            raise ToolUnavailableError("airflow unreachable")
        return {"run": {"state": "success"}, "tasks": airflow_tasks or OK_TASKS}

    def spec(name, fn=None, mutating=False, params=None):
        return ToolSpec(name, name, params or {}, fn or (lambda **kw: {"name": name, "ok": True}), mutating)

    tools = {t.name: t for t in (
        spec("mlflow_find_best_run", lambda **kw: {"best_run": {"run_id": "abc123", "f1": 0.98}, "is_deployment_decision": False}),
        spec("mlflow_list_runs"),
        spec("dvc_get_run_dataset_version", lambda **kw: {"dataset_version": "v1", "verified_against_git": True}),
        spec("k8s_find_unhealthy_pods", lambda **kw: {"all_clear": True}),
        spec("k8s_get_deployed_version", lambda **kw: {"image": "breast-cancer-api:1.0.0"}),
        spec("k8s_list_deployments"),
        spec("airflow_list_dag_runs"),
        spec("airflow_get_run_status", airflow_status),
        spec("airflow_trigger_training", trigger, mutating=True, params={"model": "m", "min_f1": "t"}),
    )}
    return tools, trig


def step1():
    names = ("mlflow_find_best_run", "k8s_find_unhealthy_pods", "k8s_get_deployed_version", "airflow_get_run_status")
    return {"action": "tools", "calls": [{"tool": n, "args": {}} for n in names]}


def step2():
    return {"action": "tools", "calls": [{"tool": "dvc_get_run_dataset_version", "args": {}}]}


def final(recs, facts=None):
    facts = facts if facts is not None else [
        {"claim": "best run abc123", "evidence": ["mlflow_find_best_run"]},
        {"claim": "trained on v1", "evidence": ["dvc_get_run_dataset_version"]},
        {"claim": "pods healthy", "evidence": ["k8s_find_unhealthy_pods"]},
        {"claim": "latest workflow succeeded", "evidence": ["airflow_get_run_status"]},
    ]
    return {"action": "final", "summary": "s", "known_facts": facts, "uncertainties": [], "recommended_actions": recs}


Q = "Find the best recent model, identify the dataset version used to train it, check the deployment status, and explain whether retraining is necessary."


def test_mandatory_scenario_uses_all_four_technologies_and_does_not_retrain():
    tools, trig = build()
    resp = Orchestrator(FakeLLM([step1(), step2(), final(["Retrain the model to be safe"])]), tools).run(Q)
    assert {r.tool.split("_")[0] for r in resp.tool_results} == {"mlflow", "dvc", "k8s", "airflow"}
    assert all(r.status == "ok" for r in resp.tool_results) and len(resp.known_facts) == 4
    assert trig["n"] == 0 and resp.pending_actions == []
    assert resp.recommended_actions == []
    assert any("Retraining recommendation withheld" in u for u in resp.uncertainties)


def test_failed_quality_gate_is_evidence_for_recommending_retraining():
    tools, trig = build(airflow_tasks=GATE_FAILED)
    resp = Orchestrator(FakeLLM([step1(), step2(), final(["Investigate the gate, then retrain if justified"])]), tools).run(Q)
    assert resp.recommended_actions == ["Investigate the gate, then retrain if justified"]
    assert trig["n"] == 0 and resp.pending_actions == []  # recommended, never started


def test_user_supplied_freshness_policy_is_evidence():
    tools, _ = build()
    q = Q + " Our policy says models older than 30 days must be retrained; this one is 45 days old."
    resp = Orchestrator(FakeLLM([step1(), step2(), final(["Retrain: the model exceeds the 30-day policy"])]), tools).run(q)
    assert len(resp.recommended_actions) == 1


def test_asking_whether_a_model_is_stale_is_not_evidence():
    tools, _ = build()
    resp = Orchestrator(FakeLLM([step1(), final(["Retrain the model"])]), tools).run("Is the model stale?")
    assert resp.recommended_actions == []


def test_airflow_outage_is_reported_not_guessed():
    tools, _ = build(airflow_down=True)
    facts = [{"claim": "latest workflow succeeded", "evidence": ["airflow_get_run_status"]},
             {"claim": "pods healthy", "evidence": ["k8s_find_unhealthy_pods"]}]
    resp = Orchestrator(FakeLLM([step1(), final(["Retrain"], facts)]), tools).run(Q)
    assert [f.claim for f in resp.known_facts] == ["pods healthy"]
    assert any("airflow_get_run_status" in u and "unavailable" in u for u in resp.uncertainties)
    assert resp.recommended_actions == []


def test_model_asking_to_trigger_training_only_creates_a_pending_action():
    tools, trig = build()
    ask = {"action": "tools", "calls": [{"tool": "airflow_trigger_training", "args": {"model": "rf", "confirmed": True}}]}
    resp = Orchestrator(FakeLLM([ask, final([])]), tools).run("What is the best model?")
    assert trig["n"] == 0 and len(resp.pending_actions) == 1


def test_negated_retraining_statements_are_kept():
    for text in ("No retraining is needed", "Do not retrain unless the gate fails", "Retraining is unnecessary"):
        assert not is_retrain_recommendation(text)
        assert guard_retraining([text], "q", [])[0] == [text]
    assert is_retrain_recommendation("Retrain with more data")


def test_prompt_playbooks_only_name_real_tools():
    named = set(re.findall(r"\b(?:mlflow|dvc|k8s|airflow)_[a-z_]+", prompts.SCENARIOS))
    assert named and named <= set(load_tools())
    system = prompts.build_system_prompt(load_tools())
    assert "SCENARIO PLAYBOOKS" in system and "RETRAINING" in system