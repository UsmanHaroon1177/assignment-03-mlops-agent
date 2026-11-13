import pytest

import agent.llm as llm_mod
from agent.fake_llm import FakeLLM
from agent.llm import GeminiLLM, LLMError
from agent.orchestrator import Orchestrator
from agent.schemas import ToolCall
from agent.tools import kubernetes_tools as k8s
from agent.tools.base import ToolSpec, ToolUnavailableError
from agent.validation import check_call


def tool(name, fn, params=None, mutating=False):
    return ToolSpec(name, f"{name} description", params or {}, fn, mutating)


def calls(*items):
    return {"action": "tools", "calls": [{"tool": t, "args": a} for t, a in items]}


def final(summary="done", facts=None, unc=None, rec=None):
    return {"action": "final", "summary": summary, "known_facts": facts or [],
            "uncertainties": unc or [], "recommended_actions": rec or []}


def fact(claim, *evidence):
    return {"claim": claim, "evidence": list(evidence)}


# ------------------------------------------------------------ tool selection and execution
def test_runs_selected_tool_and_keeps_evidence_backed_fact():
    tools = {"mlflow_find_best_run": tool("mlflow_find_best_run", lambda metric="f1": {"best": "abc123", "f1": 0.97}, {"metric": "m"})}
    llm = FakeLLM([calls(("mlflow_find_best_run", {"metric": "f1"})),
                   final("Best run is abc123", facts=[fact("Best run abc123 has f1 0.97", "mlflow_find_best_run")])])
    resp = Orchestrator(llm, tools).run("best model?", request_id="r1")
    assert resp.request_id == "r1" and resp.tool_results[0].status == "ok"
    assert resp.tool_results[0].request_id == "r1"
    assert [f.claim for f in resp.known_facts] == ["Best run abc123 has f1 0.97"]
    assert "abc123" in resp.answer and "Known facts" in resp.answer
    assert "0.97" in llm.calls[1]["messages"][-1]["text"]  # the model saw the tool result


def test_multi_tool_request_across_steps():
    tools = {"a": tool("a", lambda: {"x": 1}), "b": tool("b", lambda: {"y": 2}), "c": tool("c", lambda: {"z": 3})}
    llm = FakeLLM([calls(("a", {}), ("b", {})), calls(("c", {})),
                   final(facts=[fact("a b c all ok", "a", "b", "c")])])
    resp = Orchestrator(llm, tools).run("check everything")
    assert [r.tool for r in resp.tool_results] == ["a", "b", "c"] and len(resp.known_facts) == 1


def test_identical_call_is_not_repeated():
    counter = {"n": 0}
    def fn():
        counter["n"] += 1
        return {"v": 1}
    llm = FakeLLM([calls(("a", {})), calls(("a", {})), final()])
    Orchestrator(llm, {"a": tool("a", fn)}).run("q")
    assert counter["n"] == 1


# ------------------------------------------------------------ evidence-only answers
def test_unavailable_tool_never_becomes_a_fact():
    def down():
        raise ToolUnavailableError("cluster unreachable")
    tools = {"k8s_list_pods": tool("k8s_list_pods", down)}
    llm = FakeLLM([calls(("k8s_list_pods", {})),
                   final("All pods are healthy", facts=[fact("All pods are healthy", "k8s_list_pods")])])
    resp = Orchestrator(llm, tools).run("is the API healthy?")
    assert resp.tool_results[0].status == "unavailable" and resp.tool_results[0].data is None
    assert resp.known_facts == []
    assert any("withheld" in u for u in resp.uncertainties)
    assert any("k8s_list_pods" in u and "unavailable" in u for u in resp.uncertainties)


def test_fact_without_evidence_or_citing_uncalled_tool_is_withheld():
    tools = {"a": tool("a", lambda: {"x": 1})}
    llm = FakeLLM([calls(("a", {})),
                   final(facts=[fact("invented number 0.99", ), fact("claims about b", "b"), fact("real", "a")])])
    resp = Orchestrator(llm, tools).run("q")
    assert [f.claim for f in resp.known_facts] == ["real"]
    assert sum("withheld" in u for u in resp.uncertainties) == 2


# ------------------------------------------------------------ invalid requests
def test_unknown_tool_and_unknown_args_are_rejected_without_running():
    counter = {"n": 0}
    def fn(metric="f1"):
        counter["n"] += 1
        return {}
    tools = {"a": tool("a", fn, {"metric": "m"})}
    llm = FakeLLM([calls(("rm_rf", {}), ("a", {"bogus": 1})), final()])
    resp = Orchestrator(llm, tools).run("q")
    assert counter["n"] == 0
    assert [r.status for r in resp.tool_results] == ["rejected", "rejected"]
    assert "unknown argument" in resp.tool_results[1].error


def test_empty_and_oversized_questions_do_not_call_the_model():
    llm = FakeLLM([])
    orch = Orchestrator(llm, {})
    assert "Please" in orch.run("   ").answer
    assert "too long" in orch.run("x" * 5000).answer
    assert llm.calls == []


# ------------------------------------------------------------ model failures
def test_invalid_json_gets_one_repair_then_succeeds():
    llm = FakeLLM(["this is not json", final("fine")])
    resp = Orchestrator(llm, {}).run("q")
    assert resp.summary == "fine" and resp.llm_error is None
    assert "invalid" in llm.calls[1]["messages"][-1]["text"]


def test_invalid_json_twice_yields_no_invented_answer():
    resp = Orchestrator(FakeLLM(["nope", "still nope"]), {}).run("q")
    assert resp.known_facts == [] and resp.llm_error and "could not complete" in resp.answer


def test_llm_outage_returns_tool_evidence_but_no_conclusions():
    tools = {"a": tool("a", lambda: {"x": 1})}
    resp = Orchestrator(FakeLLM([calls(("a", {})), LLMError("quota exceeded")]), tools).run("q")
    assert resp.tool_results[0].status == "ok" and resp.known_facts == []
    assert "quota" in resp.llm_error and "Model unavailable" in resp.answer


def test_step_limit_forces_a_final_answer():
    counter = {"n": 0}
    def fn(i=0):
        counter["n"] += 1
        return {"i": i}
    tools = {"a": tool("a", fn, {"i": "n"})}
    llm = FakeLLM([calls(("a", {"i": 1})), calls(("a", {"i": 2})), final("stopped", facts=[fact("ran", "a")])])
    resp = Orchestrator(llm, tools, max_steps=2).run("q")
    assert counter["n"] == 2 and resp.summary == "stopped" and len(resp.known_facts) == 1
    assert "Step limit" in llm.calls[2]["messages"][-1]["text"]


def test_oversized_tool_result_is_truncated_for_the_model():
    tools = {"a": tool("a", lambda: {"blob": "x" * 20000})}
    llm = FakeLLM([calls(("a", {})), final()])
    Orchestrator(llm, tools).run("q")
    sent = llm.calls[1]["messages"][-1]["text"]
    assert "data_truncated" in sent and len(sent) < 7000


# ------------------------------------------------------------ safety: mutating tools
@pytest.fixture
def k8s_patches(monkeypatch):
    patches = []
    monkeypatch.setattr(k8s, "_patch_deployment", lambda body: patches.append(body) or {})
    return patches


def test_model_cannot_self_confirm_a_cluster_change(k8s_patches):
    tools = {t.name: t for t in k8s.TOOLS}
    llm = FakeLLM([calls(("k8s_rollout_restart", {"confirmed": True})), final("asked", rec=["restart"])])
    orch = Orchestrator(llm, tools)
    resp = orch.run("restart the API")
    assert k8s_patches == []  # nothing changed
    assert len(resp.pending_actions) == 1 and "confirmed" not in resp.pending_actions[0].args
    assert "Awaiting your confirmation" in resp.answer

    result = orch.confirm(resp.pending_actions[0].id)  # the USER approves
    assert result.status == "ok" and len(k8s_patches) == 1
    assert orch.confirm(resp.pending_actions[0].id).status == "rejected"  # cannot be replayed
    assert len(k8s_patches) == 1


def test_confirmed_action_outside_approved_namespace_is_still_rejected(k8s_patches):
    tools = {t.name: t for t in k8s.TOOLS}
    llm = FakeLLM([calls(("k8s_rollout_restart", {"namespace": "kube-system"})), final()])
    orch = Orchestrator(llm, tools)
    resp = orch.run("restart coredns")
    result = orch.confirm(resp.pending_actions[0].id)
    assert result.status == "rejected" and "mlops-agent" in result.error
    assert k8s_patches == []


def test_retraining_is_never_triggered_automatically():
    counter = {"n": 0}
    def trigger(model="logreg", min_f1=0.9):
        counter["n"] += 1
        return {"dag_run_id": "x"}
    tools = {"airflow_trigger_training": tool("airflow_trigger_training", trigger, {"model": "m", "min_f1": "t"}, mutating=True)}
    llm = FakeLLM([calls(("airflow_trigger_training", {"model": "rf"})), final(rec=["consider retraining"])])
    orch = Orchestrator(llm, tools)
    resp = orch.run("is retraining needed?")
    assert counter["n"] == 0 and len(resp.pending_actions) == 1
    assert orch.confirm(resp.pending_actions[0].id).status == "ok" and counter["n"] == 1


def test_pending_action_expires_and_can_be_rejected():
    now = [1000.0]
    tools = {"m": tool("m", lambda: {"done": True}, mutating=True)}
    orch = Orchestrator(FakeLLM([calls(("m", {})), final(), calls(("m", {})), final()]), tools, clock=lambda: now[0])
    first = orch.run("do m").pending_actions[0]
    now[0] += 10_000
    assert orch.confirm(first.id).status == "rejected"
    second = orch.run("do m again").pending_actions[0]
    assert orch.reject(second.id) is True and orch.confirm(second.id).status == "rejected"


def test_check_call_strips_confirmed():
    tools = {t.name: t for t in k8s.TOOLS}
    spec, args, stripped = check_call(tools, ToolCall(tool="k8s_rollout_restart", args={"confirmed": True, "deployment": "breast-cancer-api"}))
    assert args == {"deployment": "breast-cancer-api"} and stripped is True and spec.mutating


# ------------------------------------------------------------ Gemini client (mocked HTTP)
class Resp:
    def __init__(self, code=200, data=None, text=""):
        self.status_code, self._data, self.text = code, data, text

    def json(self):
        return self._data


KEY = "AIza" + "x" * 35
OK = Resp(200, {"candidates": [{"content": {"parts": [{"text": '{"action":"final"}'}]}}]})


def test_gemini_request_shape_and_key_only_in_header(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", KEY)
    seen = {}
    def post(url, headers, body, timeout):
        seen.update(url=url, headers=headers, body=body)
        return OK
    monkeypatch.setattr(llm_mod, "_post", post)
    out = GeminiLLM(model="models/my-model").generate("sys", [{"role": "user", "text": "hi"}, {"role": "model", "text": "ok"}])
    assert out == '{"action":"final"}'
    assert seen["url"].endswith("/models/my-model:generateContent") and KEY not in seen["url"]
    assert seen["headers"]["x-goog-api-key"] == KEY
    assert [c["role"] for c in seen["body"]["contents"]] == ["user", "model"]


def test_gemini_retries_rate_limit_then_succeeds(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", KEY)
    queue = [Resp(429, {"error": {"message": "quota"}}), OK]
    monkeypatch.setattr(llm_mod, "_post", lambda *a, **k: queue.pop(0))
    slept = []
    assert GeminiLLM(sleep=slept.append).generate("s", [{"role": "user", "text": "x"}])
    assert slept == [6]


def test_gemini_errors_never_leak_the_key_and_missing_key_is_reported(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", KEY)
    monkeypatch.setattr(llm_mod, "_post", lambda *a, **k: Resp(400, {"error": {"message": f"bad key {KEY} supplied"}}))
    with pytest.raises(LLMError) as exc:
        GeminiLLM().generate("s", [{"role": "user", "text": "x"}])
    assert KEY not in str(exc.value) and "400" in str(exc.value)
    monkeypatch.delenv("GEMINI_API_KEY")
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    with pytest.raises(LLMError, match="not set"):
        GeminiLLM().generate("s", [{"role": "user", "text": "x"}])


def test_gemini_timeout_becomes_llm_error_after_retries(monkeypatch):
    import requests
    monkeypatch.setenv("GEMINI_API_KEY", KEY)
    def boom(*a, **k):
        raise requests.Timeout("slow")
    monkeypatch.setattr(llm_mod, "_post", boom)
    with pytest.raises(LLMError, match="after retries"):
        GeminiLLM(sleep=lambda s: None).generate("s", [{"role": "user", "text": "x"}])


def test_gemini_empty_candidate_is_an_error(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", KEY)
    monkeypatch.setattr(llm_mod, "_post", lambda *a, **k: Resp(200, {"promptFeedback": {"blockReason": "SAFETY"}}))
    with pytest.raises(LLMError, match="SAFETY"):
        GeminiLLM().generate("s", [{"role": "user", "text": "x"}])