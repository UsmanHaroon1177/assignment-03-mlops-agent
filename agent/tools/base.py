"""Common tool plumbing: ToolSpec, error classes and execute() (timing, logging, redaction)."""
from __future__ import annotations

import inspect
import time
from dataclasses import dataclass
from typing import Any, Callable

from agent.logging_utils import log_tool_call
from agent.safety import redact, redact_obj
from agent.schemas import ToolResult


class ToolInputError(ValueError):
    """The request is invalid or not allowed (bad argument, unapproved operation)."""


class ToolUnavailableError(RuntimeError):
    """The backing service or command could not be reached or timed out."""


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    params: dict[str, str]  # parameter name -> description (mark optional ones)
    fn: Callable[..., dict[str, Any]]
    mutating: bool = False  # True = changes state; needs explicit confirmation (Phase 5)


_UNAVAILABLE_HINTS = (
    "max retries exceeded",
    "connection refused",
    "connection aborted",
    "failed to establish",
    "timed out",
    "getaddrinfo failed",
    "name or service not known",
    "connection error",
)


def looks_unavailable(exc: BaseException) -> bool:
    seen: BaseException | None = exc
    for _ in range(4):  # walk the cause chain a few levels
        if seen is None:
            break
        if isinstance(seen, (ConnectionError, TimeoutError)):
            return True
        text = str(seen).lower()
        if any(h in text for h in _UNAVAILABLE_HINTS):
            return True
        seen = seen.__cause__ or seen.__context__
    return False


def execute(tool: str, fn: Callable[..., dict[str, Any]], request_id: str = "-", **kwargs: Any) -> ToolResult:
    """Run one tool. Never raises. Output is redacted and the call is logged."""
    start = time.perf_counter()
    status, data, error = "ok", None, None
    try:
        try:
            inspect.signature(fn).bind(**kwargs)
        except TypeError as exc:
            raise ToolInputError(f"invalid arguments: {exc}") from exc
        data = fn(**kwargs)
    except ToolInputError as exc:
        status, error = "rejected", str(exc)
    except ToolUnavailableError as exc:
        status, error = "unavailable", str(exc)
    except Exception as exc:  # noqa: BLE001 - the wrapper must never raise
        status = "unavailable" if looks_unavailable(exc) else "error"
        error = f"{type(exc).__name__}: {exc}"
    duration_ms = int((time.perf_counter() - start) * 1000)
    if error:
        error = redact(error)[:500]
    if data is not None:
        data = redact_obj(data)
    log_tool_call(request_id, tool, status, duration_ms, arg_names=list(kwargs), error=error)
    return ToolResult(
        tool=tool, status=status, data=data, error=error, duration_ms=duration_ms, request_id=request_id
    )
