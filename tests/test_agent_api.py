import pytest
from fastapi.testclient import TestClient

from agent.fake_llm import FakeLLM
from agent.main import app, get_orchestrator
from agent.orchestrator import Orchestrator
from agent.tools.base import ToolSpec

ASK_MUTATING = {"action": "tools", "calls": [{"tool": "m", "args": {}}]}
FINAL = {"action": "final", "summary": "done", "known_facts": [], "uncertainties": [], "recommended_actions": []}


@pytest.fixture
def ctx(monkeypatch):
    monkeypatch.delenv("AGENT_API_TOKEN", raising=False)
    counter = {"n": 0}

    def m(**kw):
        counter["n"] += 1
        return {"done": True}

    tools = {"m": ToolSpec("m", "mutating thing", {}, m, True),
             "r": ToolSpec("r", "read thing", {}, lambda **kw: {"x": 1})}
    orch = Orchestrator(FakeLLM([ASK_MUTATING, FINAL, ASK_MUTATING, FINAL]), tools)
    app.dependency_overrides[get_orchestrator] = lambda: orch
    yield TestClient(app), counter
    app.dependency_overrides.clear()


def test_ask_confirm_flow_executes_once(ctx):
    client, counter = ctx
    body = client.post("/ask", json={"question": "do m"}).json()
    assert body["request_id"] and "Awaiting your confirmation" in body["answer"] and counter["n"] == 0
    pid = body["pending_actions"][0]["id"]
    assert client.get("/pending").json()[0]["id"] == pid
    assert client.post(f"/confirm/{pid}").json()["status"] == "ok" and counter["n"] == 1
    assert client.post(f"/confirm/{pid}").status_code == 404 and counter["n"] == 1


def test_reject_discards_pending_action(ctx):
    client, counter = ctx
    pid = client.post("/ask", json={"question": "do m"}).json()["pending_actions"][0]["id"]
    assert client.post(f"/reject/{pid}").json() == {"rejected": True}
    assert client.post(f"/confirm/{pid}").status_code == 404 and counter["n"] == 0


def test_invalid_input_is_rejected_by_validation(ctx):
    client, _ = ctx
    assert client.post("/ask", json={"question": ""}).status_code == 422
    assert client.post("/ask", json={"question": "x" * 2001}).status_code == 422
    assert client.post("/confirm/not-an-id").status_code == 422
    assert client.post("/confirm/deadbeef").status_code == 404


def test_tools_endpoint_flags_mutating_tools(ctx):
    client, _ = ctx
    assert {t["name"]: t["mutating"] for t in client.get("/tools").json()} == {"m": True, "r": False}


def test_token_is_enforced_when_configured(ctx, monkeypatch):
    client, _ = ctx
    monkeypatch.setenv("AGENT_API_TOKEN", "test-token-123")
    assert client.get("/health").status_code == 200
    assert client.post("/ask", json={"question": "hi"}).status_code == 401
    assert client.post("/confirm/deadbeef").status_code == 401
    assert client.get("/pending", headers={"X-Agent-Token": "wrong"}).status_code == 401
    assert client.get("/pending", headers={"X-Agent-Token": "test-token-123"}).status_code == 200