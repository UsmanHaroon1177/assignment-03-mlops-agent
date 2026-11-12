"""Shared data models for the MLOps agent."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel

ToolStatus = Literal["ok", "error", "unavailable", "rejected"]


class ToolResult(BaseModel):
    """Structured result of one tool call.

    Only status == "ok" carries operational facts. For "error", "unavailable" and
    "rejected" the agent must report that the information could not be obtained.
    """

    tool: str
    status: ToolStatus
    data: dict[str, Any] | None = None
    error: str | None = None
    duration_ms: int = 0
    request_id: str = "-"
