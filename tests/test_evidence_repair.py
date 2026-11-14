from agent.fake_llm import FakeLLM
from agent.orchestrator import Orchestrator
from agent.tools.base import ToolSpec

TOOLS = {"a": ToolSpec("a", "a", {}, lambda: {"x": 1}, False)}
CALL = {"action": "tools", "calls": [{"tool": "a", "args": {}}]}


def final(evidence):
    return {"action": "final", "summary": "s", "known_facts": [{"claim": "c", "evidence": evidence}],
            "uncertainties": [], "recommended_actions": []}


def test_empty_evidence_is_repaired_once_and_then_kept():
    llm = FakeLLM([CALL, final([]), final(["a"])])
    r = Orchestrator(llm, TOOLS).run("q")
    assert [f.evidence for f in r.known_facts] == [["a"]]
    assert "empty evidence" in llm.calls[2]["messages"][-1]["text"]


def test_still_empty_after_repair_stays_withheld_not_guessed():
    r = Orchestrator(FakeLLM([CALL, final([]), final([])]), TOOLS).run("q")
    assert r.known_facts == [] and any("withheld" in u for u in r.uncertainties)


def test_no_repair_call_when_evidence_present():
    llm = FakeLLM([CALL, final(["a"])])
    Orchestrator(llm, TOOLS).run("q")
    assert len(llm.calls) == 2