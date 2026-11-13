"""Validation of model output and of the tool calls the model asks for."""
from __future__ import annotations

import json
import re

from pydantic import ValidationError

from agent.schemas import LLMAction, ToolCall
from agent.tools.base import ToolInputError, ToolSpec

MAX_QUESTION_CHARS = 2000
_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.I)
_SYSTEM_ONLY_ARGS = {"confirmed"}  # only the orchestrator may set these


class ActionParseError(ValueError):
    """The model reply was not a valid action object."""


class RequestError(ValueError):
    """The user request itself is unusable (empty or too long)."""


def check_question(question: str | None) -> str:
    q = (question or "").strip()
    if not q:
        raise RequestError("Please type a question or request.")
    if len(q) > MAX_QUESTION_CHARS:
        raise RequestError(f"Request is too long (max {MAX_QUESTION_CHARS} characters).")
    return q


def parse_action(text: str) -> LLMAction:
    s = _FENCE.sub("", (text or "").strip()).strip()
    try:
        obj = json.loads(s)
    except json.JSONDecodeError:
        i, j = s.find("{"), s.rfind("}")
        if i < 0 or j <= i:
            raise ActionParseError("reply is not JSON") from None
        try:
            obj = json.loads(s[i : j + 1])
        except json.JSONDecodeError as exc:
            raise ActionParseError(f"invalid JSON: {exc.msg}") from exc
    if not isinstance(obj, dict):
        raise ActionParseError("top-level JSON must be an object")
    try:
        return LLMAction.model_validate(obj)
    except ValidationError as exc:
        detail = "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors()[:3])
        raise ActionParseError(detail) from exc


def check_call(tools: dict[str, ToolSpec], call: ToolCall) -> tuple[ToolSpec, dict, bool]:
    """Return (spec, cleaned_args, stripped_system_args). Raises ToolInputError."""
    spec = tools.get(call.tool)
    if spec is None:
        raise ToolInputError(f"unknown tool '{call.tool[:60]}'. Available: {', '.join(sorted(tools))}")
    args = dict(call.args)
    stripped = any(k in args for k in _SYSTEM_ONLY_ARGS)
    for k in _SYSTEM_ONLY_ARGS:
        args.pop(k, None)
    allowed = set(spec.params) - _SYSTEM_ONLY_ARGS
    unknown = sorted(set(args) - allowed)
    if unknown:
        raise ToolInputError(f"unknown argument(s) {unknown} for {spec.name}; allowed: {sorted(allowed)}")
    return spec, args, stripped