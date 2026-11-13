"""Shared data models for the MLOps agent."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

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


class ToolCall(BaseModel):
    tool: str
    args: dict[str, Any] = Field(default_factory=dict)

    @field_validator("args", mode="before")
    @classmethod
    def _none_to_empty(cls, v: Any) -> Any:
        return {} if v is None else v


class Fact(BaseModel):
    """A claim plus the names of the tools whose OK results support it."""

    claim: str
    evidence: list[str] = Field(default_factory=list)

    @field_validator("evidence", mode="before")
    @classmethod
    def _listify(cls, v: Any) -> Any:
        if v is None:
            return []
        return [v] if isinstance(v, str) else v


class LLMAction(BaseModel):
    """One model turn: either request tool calls or give the final answer."""

    action: Literal["tools", "final"]
    calls: list[ToolCall] = Field(default_factory=list)
    summary: str = ""
    known_facts: list[Fact] = Field(default_factory=list)
    uncertainties: list[str] = Field(default_factory=list)
    recommended_actions: list[str] = Field(default_factory=list)

    @field_validator("known_facts", mode="before")
    @classmethod
    def _facts(cls, v: Any) -> Any:
        if v is None:
            return []
        return [{"claim": x} if isinstance(x, str) else x for x in v]

    @field_validator("uncertainties", "recommended_actions", mode="before")
    @classmethod
    def _strings(cls, v: Any) -> Any:
        return [] if v is None else [x if isinstance(x, str) else str(x) for x in v]

    @model_validator(mode="after")
    def _tools_need_calls(self) -> "LLMAction":
        if self.action == "tools" and not self.calls:
            raise ValueError("action 'tools' requires at least one call")
        return self


class PendingAction(BaseModel):
    """A state-changing tool call waiting for explicit user confirmation."""

    id: str
    tool: str
    args: dict[str, Any] = Field(default_factory=dict)
    description: str
    request_id: str = "-"
    created_at: float
    expires_at: float


class AgentResponse(BaseModel):
    request_id: str
    answer: str
    summary: str = ""
    known_facts: list[Fact] = Field(default_factory=list)
    uncertainties: list[str] = Field(default_factory=list)
    recommended_actions: list[str] = Field(default_factory=list)
    tool_results: list[ToolResult] = Field(default_factory=list)
    pending_actions: list[PendingAction] = Field(default_factory=list)
    llm_error: str | None = None