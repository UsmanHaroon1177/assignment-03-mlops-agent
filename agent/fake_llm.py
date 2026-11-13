"""Scripted LLM for tests and offline demos: no network, no quota."""
from __future__ import annotations

import json

from agent.llm import LLMError


class FakeLLM:
    """script items: dict/str (returned as text), Exception (raised) or callable(messages)."""

    def __init__(self, script: list):
        self.script = list(script)
        self.calls: list[dict] = []

    def generate(self, system: str, messages: list[dict[str, str]]) -> str:
        self.calls.append({"system": system, "messages": [dict(m) for m in messages]})
        if not self.script:
            raise LLMError("FakeLLM script exhausted")
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        if callable(item):
            item = item(messages)
        return item if isinstance(item, str) else json.dumps(item)