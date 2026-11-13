"""LLM access. GeminiLLM talks to the Gemini REST API with `requests` (no SDK).

The API key is read lazily from GEMINI_API_KEY (or GOOGLE_API_KEY) and sent only in the
x-goog-api-key header, so it never appears in URLs, logs or error text.
Models: GEMINI_MODEL (primary) and GEMINI_FALLBACK_MODELS (comma-separated, tried in order when the
primary is rate-limited, failing with 5xx, unreachable or not found). Auth/bad-request errors never fall back.
"""
from __future__ import annotations

import os
import re
import time
from typing import Protocol

import requests

from agent.safety import redact

BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
DEFAULT_MODEL = "gemini-2.5-flash"
RETRY_STATUS = {429, 500, 502, 503, 504}
FALLBACK_STATUS = RETRY_STATUS | {404}
_MODEL_RE = re.compile(r"^[A-Za-z0-9._\-]{1,80}$")


class LLMError(RuntimeError):
    """The model could not be reached or returned nothing usable."""


class ModelUnavailable(LLMError):
    """This model failed in a way another model might not (quota, outage, unknown name)."""


class LLM(Protocol):
    def generate(self, system: str, messages: list[dict[str, str]]) -> str:
        """messages: [{"role": "user"|"model", "text": str}, ...] -> raw model text."""


def _api_key(explicit: str | None = None) -> str:
    key = explicit or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not key:
        raise LLMError("GEMINI_API_KEY is not set")
    return key


def _post(url: str, headers: dict, body: dict, timeout: int):
    return requests.post(url, headers=headers, json=body, timeout=timeout)


def _get(url: str, headers: dict, params: dict, timeout: int):
    return requests.get(url, headers=headers, params=params, timeout=timeout)


def _error_text(resp) -> str:
    try:
        text = str(resp.json()["error"]["message"])
    except Exception:  # noqa: BLE001
        text = str(getattr(resp, "text", ""))
    return redact(text)[:200]


def _clean_model(name: str) -> str:
    name = name.strip().removeprefix("models/")
    if not _MODEL_RE.match(name):
        raise LLMError("invalid model name")
    return name


def _fallbacks_from_env() -> list[str]:
    return [m for m in (os.getenv("GEMINI_FALLBACK_MODELS") or "").split(",") if m.strip()]


class GeminiLLM:
    def __init__(self, model: str | None = None, api_key: str | None = None, timeout: int = 60,
                 max_retries: int = 2, temperature: float = 0.1, sleep=time.sleep,
                 fallback_models: list[str] | None = None):
        primary = _clean_model(model or os.getenv("GEMINI_MODEL") or DEFAULT_MODEL)
        extra = _fallbacks_from_env() if fallback_models is None else fallback_models
        self.models: list[str] = [primary]
        for m in extra:
            m = _clean_model(m)
            if m not in self.models:
                self.models.append(m)
        self.model = primary          # configured primary
        self.last_model = primary     # model that produced the most recent answer
        self._api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self.temperature = temperature
        self._sleep = sleep

    def generate(self, system: str, messages: list[dict[str, str]]) -> str:
        key = _api_key(self._api_key)
        body = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [
                {"role": "model" if m["role"] == "model" else "user", "parts": [{"text": m["text"]}]}
                for m in messages
            ],
            "generationConfig": {"temperature": self.temperature, "responseMimeType": "application/json"},
        }
        failures: list[str] = []
        for name in self.models:
            try:
                text = self._call(name, key, body)
            except ModelUnavailable as exc:
                failures.append(f"{name}: {exc}")
                continue
            self.last_model = name
            return text
        raise LLMError("Gemini request failed after retries: " + " | ".join(failures))

    def _call(self, model: str, key: str, body: dict) -> str:
        url = f"{BASE_URL}/models/{model}:generateContent"
        headers = {"x-goog-api-key": key, "Content-Type": "application/json"}
        last = "unknown error"
        for attempt in range(self.max_retries + 1):
            try:
                resp = _post(url, headers, body, self.timeout)
            except (requests.Timeout, requests.ConnectionError) as exc:
                last = f"network error ({type(exc).__name__})"
            else:
                if resp.status_code == 200:
                    return self._text(resp)
                last = f"HTTP {resp.status_code}: {_error_text(resp)}"
                if resp.status_code not in FALLBACK_STATUS:
                    raise LLMError(f"Gemini request failed: {last}")
                if resp.status_code == 404:
                    break  # unknown model: retrying the same name is pointless
            if attempt < self.max_retries:
                self._sleep(6 * (attempt + 1))
        raise ModelUnavailable(last)

    @staticmethod
    def _text(resp) -> str:
        try:
            data = resp.json()
        except Exception as exc:  # noqa: BLE001
            raise LLMError("Gemini returned a non-JSON response") from exc
        cands = data.get("candidates") or []
        if not cands:
            reason = (data.get("promptFeedback") or {}).get("blockReason")
            raise LLMError(f"Gemini returned no candidate (blockReason={reason})")
        parts = (cands[0].get("content") or {}).get("parts") or []
        text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
        if not text.strip():
            raise LLMError(f"Gemini returned an empty response (finishReason={cands[0].get('finishReason')})")
        return text


def list_models(api_key: str | None = None) -> list[str]:
    """Model names that support generateContent for this key."""
    key = _api_key(api_key)
    try:
        resp = _get(f"{BASE_URL}/models", {"x-goog-api-key": key}, {"pageSize": 200}, 20)
    except (requests.Timeout, requests.ConnectionError) as exc:
        raise LLMError(f"network error ({type(exc).__name__})") from exc
    if resp.status_code != 200:
        raise LLMError(f"HTTP {resp.status_code}: {_error_text(resp)}")
    return sorted(
        m["name"].removeprefix("models/")
        for m in resp.json().get("models", [])
        if "generateContent" in (m.get("supportedGenerationMethods") or [])
    )