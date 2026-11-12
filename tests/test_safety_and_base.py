import logging

import pytest

from agent import safety
from agent.tools.base import ToolInputError, ToolUnavailableError, execute


def test_redacts_credential_shapes():
    key = "AIza" + "x" * 35
    text = f"key={key} password: hunter2abc Authorization: Bearer abcdefghijklmnop1234"
    out = safety.redact(text)
    assert key not in out
    assert "hunter2abc" not in out
    assert "abcdefghijklmnop1234" not in out


def test_redacts_by_key_name_in_objects():
    out = safety.redact_obj({"db_password": "s3cretvalue", "nested": [{"api_key": "abcdef123456"}]})
    assert out["db_password"] == "[REDACTED]"
    assert out["nested"][0]["api_key"] == "[REDACTED]"


def test_does_not_mask_version_specifiers_or_plain_prose():
    # regression for playbook bug #4: a mask must not turn ordinary text into a fake secret
    assert safety.redact('"jsonwebtoken": "^9.0.3"') == '"jsonwebtoken": "^9.0.3"'
    assert safety.redact("max_tokens: 5") == "max_tokens: 5"
    assert safety.redact("Pass the untrusted input along") == "Pass the untrusted input along"


def test_execute_ok_records_duration_and_request_id():
    res = execute("t", lambda: {"a": 1}, request_id="req1")
    assert res.status == "ok" and res.data == {"a": 1} and res.request_id == "req1"


def test_execute_classifies_failures_and_never_returns_data():
    def rejected():
        raise ToolInputError("bad ref")

    def unavailable():
        raise ToolUnavailableError("timed out")

    def conn():
        raise ConnectionError("Max retries exceeded")

    def boom():
        raise ValueError("kaboom")

    assert execute("t", rejected).status == "rejected"
    assert execute("t", unavailable).status == "unavailable"
    assert execute("t", conn).status == "unavailable"
    res = execute("t", boom)
    assert res.status == "error" and res.data is None


def test_execute_rejects_unexpected_arguments():
    res = execute("t", lambda a: {"a": a}, a=1, evil="x")
    assert res.status == "rejected" and "invalid arguments" in res.error


def test_execute_redacts_secrets_in_data_and_errors():
    res = execute("t", lambda: {"msg": "password=hunter2abc"})
    assert "hunter2abc" not in str(res.data)

    def leaky():
        raise RuntimeError("login failed password=hunter2abc")

    assert "hunter2abc" not in execute("t", leaky).error


def test_tool_call_is_logged_without_argument_values(caplog):
    with caplog.at_level(logging.INFO, logger="mlops_agent"):
        execute("demo_tool", lambda token: {"ok": True}, request_id="r9", token="supersecretvalue")
    fields = [r.fields for r in caplog.records if hasattr(r, "fields")][0]
    assert fields["tool"] == "demo_tool" and fields["status"] == "ok" and fields["request_id"] == "r9"
    assert fields["arg_names"] == ["token"]
    assert "supersecretvalue" not in str(fields)


@pytest.mark.parametrize("value", [None, 5])
def test_redact_obj_passes_non_strings(value):
    assert safety.redact_obj(value) == value
