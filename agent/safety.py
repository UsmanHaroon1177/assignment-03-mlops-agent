"""Redaction helpers. Applied to tool output (data and errors) before the model, logs or
reports can see it. Do NOT apply to the model's own prose (playbook bug #3)."""
from __future__ import annotations

import re
from typing import Any

_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (
        re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?(?:-----END [A-Z ]*PRIVATE KEY-----|$)", re.S),
        "[REDACTED_PRIVATE_KEY]",
    ),
    (re.compile(r"AIza[0-9A-Za-z_\-]{30,}"), "[REDACTED_API_KEY]"),
    (re.compile(r"\bAKIA[0-9A-Z]{16}\b"), "[REDACTED_AWS_KEY]"),
    (re.compile(r"\beyJ[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}"), "[REDACTED_JWT]"),
    (re.compile(r"(?i)\b(bearer)\s+[A-Za-z0-9._\-]{16,}"), r"\1 [REDACTED_TOKEN]"),
    (
        re.compile(
            r"(?i)\b((?:api[_-]?key|secret|password|passwd|token|access[_-]?token)\s*[=:]\s*)"
            r"(?!\[REDACTED)([^\s'\",;]{6,})"
        ),
        r"\1[REDACTED]",
    ),
]

_SECRET_KEY_RE = re.compile(
    r"(?i)(password|passwd|secret|api[_-]?key|access[_-]?token|auth[_-]?token|private[_-]?key)|^token$"
)


def redact(text: str) -> str:
    """Mask credential-looking material inside a string."""
    for pattern, repl in _PATTERNS:
        text = pattern.sub(repl, text)
    return text


def redact_obj(obj: Any) -> Any:
    """Recursively redact strings; mask values whose KEY looks like a secret name."""
    if isinstance(obj, str):
        return redact(obj)
    if isinstance(obj, dict):
        out = {}
        for key, value in obj.items():
            if isinstance(value, str) and value and _SECRET_KEY_RE.search(str(key)):
                out[key] = "[REDACTED]"
            else:
                out[key] = redact_obj(value)
        return out
    if isinstance(obj, (list, tuple)):
        return [redact_obj(v) for v in obj]
    return obj
