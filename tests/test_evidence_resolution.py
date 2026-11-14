from agent.fake_llm import FakeLLM
from agent.orchestrator import Orchestrator
from agent.tools.base import ToolSpec


def tools():
    return {"a": ToolSpec("a", "a", {}, lambda: {"x": 1}, False),
            "b": ToolSpec("b", "b", {}, lambda: (_ for _ in ()).throw(RuntimeError("Max retries exceeded")), False)}


def run(evidence):
    script = [{"action": "tools", "calls": [{"tool": "a", "args": {}}, {"tool": "b", "args": {}}]},
              {"action": "final", "summary": "s", "known_facts": [{"claim": "c", "evidence": evidence}],
               "uncertainties": [], "recommended_actions": []}]
    return Orchestrator(FakeLLM(script), tools()).run("q")


def test_decorated_tool_name_resolves_and_is_kept():
    r = run(["a output"])
    assert [f.evidence for f in r.known_facts] == [["a"]]


def test_citing_a_failed_tool_or_unknown_name_is_still_withheld():
    assert run(["a", "b"]).known_facts == []
    assert run(["some general knowledge"]).known_facts == []
    assert run([]).known_facts == []