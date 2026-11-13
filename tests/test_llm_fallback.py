import pytest

import agent.llm as llm_mod
from agent.llm import GeminiLLM, LLMError

KEY = "AIza" + "y" * 35
MSG = [{"role": "user", "text": "x"}]


class Resp:
    def __init__(self, code=200, data=None, text=""):
        self.status_code, self._data, self.text = code, data, text

    def json(self):
        return self._data


def ok(text="{}"):
    return Resp(200, {"candidates": [{"content": {"parts": [{"text": text}]}}]})


@pytest.fixture(autouse=True)
def env(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", KEY)
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    monkeypatch.delenv("GEMINI_FALLBACK_MODELS", raising=False)


def record(monkeypatch, responses):
    urls, queue = [], list(responses)
    def post(url, headers, body, timeout):
        urls.append(url)
        return queue.pop(0)
    monkeypatch.setattr(llm_mod, "_post", post)
    return urls


def test_env_configures_primary_and_fallbacks(monkeypatch):
    monkeypatch.setenv("GEMINI_MODEL", "gemini-flash-latest")
    monkeypatch.setenv("GEMINI_FALLBACK_MODELS", "gemini-flash-lite-latest, gemini-flash-latest ,")
    assert GeminiLLM().models == ["gemini-flash-latest", "gemini-flash-lite-latest"]


def test_rate_limited_primary_falls_back(monkeypatch):
    urls = record(monkeypatch, [Resp(429, {"error": {"message": "quota"}})] * 3 + [ok('{"a":1}')])
    llm = GeminiLLM(model="primary", fallback_models=["backup"], sleep=lambda s: None)
    assert llm.generate("s", MSG) == '{"a":1}'
    assert [u.split("/models/")[1].split(":")[0] for u in urls] == ["primary"] * 3 + ["backup"]
    assert llm.last_model == "backup" and all(KEY not in u for u in urls)


def test_unknown_primary_model_falls_back_without_retrying(monkeypatch):
    urls = record(monkeypatch, [Resp(404, {"error": {"message": "not found"}}), ok()])
    llm = GeminiLLM(model="nope", fallback_models=["backup"], sleep=lambda s: None)
    llm.generate("s", MSG)
    assert len(urls) == 2 and llm.last_model == "backup"


def test_bad_request_does_not_fall_back(monkeypatch):
    urls = record(monkeypatch, [Resp(400, {"error": {"message": "bad"}}), ok()])
    with pytest.raises(LLMError, match="400"):
        GeminiLLM(model="primary", fallback_models=["backup"]).generate("s", MSG)
    assert len(urls) == 1


def test_all_models_failing_reports_every_model(monkeypatch):
    record(monkeypatch, [Resp(503, {"error": {"message": "down"}})] * 6)
    with pytest.raises(LLMError, match="after retries") as exc:
        GeminiLLM(model="m1", fallback_models=["m2"], sleep=lambda s: None).generate("s", MSG)
    assert "m1:" in str(exc.value) and "m2:" in str(exc.value) and KEY not in str(exc.value)