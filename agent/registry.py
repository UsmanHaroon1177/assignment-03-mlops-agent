"""Collects the tools of all four modules into one name -> ToolSpec map."""
from __future__ import annotations

from agent.tools.base import ToolSpec


def load_tools() -> dict[str, ToolSpec]:
    from agent.tools import airflow_tools, dvc_tools, kubernetes_tools, mlflow_tools

    tools: dict[str, ToolSpec] = {}
    for module in (dvc_tools, mlflow_tools, airflow_tools, kubernetes_tools):
        for spec in module.TOOLS:
            if spec.name in tools:
                raise ValueError(f"duplicate tool name: {spec.name}")
            tools[spec.name] = spec
    return tools