"""Structured JSON logging: tool name, status, duration and request id. No secrets, no
argument values (argument NAMES only)."""
from __future__ import annotations

import json
import logging
import sys
import time
import uuid

from agent.safety import redact

LOGGER_NAME = "mlops_agent"


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)) + "Z",
            "level": record.levelname,
            "event": record.getMessage(),
        }
        payload.update(getattr(record, "fields", {}))
        return json.dumps(payload, default=str)


def configure(level: str = "INFO", stream=None) -> logging.Logger:
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(level)
    if not any(getattr(h, "_mlops", False) for h in logger.handlers):
        handler = logging.StreamHandler(stream or sys.stderr)
        handler.setFormatter(JsonFormatter())
        handler._mlops = True  # type: ignore[attr-defined]
        logger.addHandler(handler)
    return logger


def new_request_id() -> str:
    return uuid.uuid4().hex[:12]


def log_tool_call(
    request_id: str,
    tool: str,
    status: str,
    duration_ms: int,
    arg_names: list[str] | None = None,
    error: str | None = None,
) -> None:
    fields: dict = {
        "request_id": request_id,
        "tool": tool,
        "status": status,
        "duration_ms": duration_ms,
        "arg_names": sorted(arg_names or []),
    }
    if error:
        fields["error"] = redact(error)[:300]
    level = logging.INFO if status == "ok" else logging.WARNING
    logging.getLogger(LOGGER_NAME).log(level, "tool_call", extra={"fields": fields})
