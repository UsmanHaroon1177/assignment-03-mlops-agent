"""Retraining policy guard, enforced in code after the model has answered.

A recommendation to retrain survives only with evidence:
  * a tool result (status ok) shows the latest Airflow run failed the quality gate, or
  * the USER supplied a freshness-policy / degradation statement containing a number.
Otherwise the recommendation is removed and reported as an uncertainty. This never triggers anything:
starting a training run still needs explicit confirmation (airflow_trigger_training is mutating).
"""
from __future__ import annotations

import re

from agent.schemas import ToolResult

_RETRAIN = re.compile(r"re-?train", re.I)
_NEGATION = re.compile(r"\b(no|not|never|without|unnecessary|avoid|skip)\b|n't", re.I)
_USER_KEYWORDS = re.compile(r"\b(fresh(ness)?|stale|degrad\w*|drift\w*|policy|sla)\b", re.I)
_DIGIT = re.compile(r"\d")


def retraining_evidence(question: str, results: list[ToolResult]) -> list[str]:
    evidence: list[str] = []
    for r in results:
        if r.status != "ok" or not r.data:
            continue
        if r.tool == "airflow_get_failed_tasks":
            if any(t.get("task_id") == "quality_gate" for t in r.data.get("failed_tasks") or []):
                evidence.append("latest Airflow run failed the quality gate (airflow_get_failed_tasks)")
        elif r.tool == "airflow_get_run_status":
            if any(t.get("task_id") == "quality_gate" and t.get("state") == "failed" for t in r.data.get("tasks") or []):
                evidence.append("latest Airflow run failed the quality gate (airflow_get_run_status)")
    if _USER_KEYWORDS.search(question or "") and _DIGIT.search(question or ""):
        evidence.append("the user supplied a freshness-policy or degradation statement")
    return list(dict.fromkeys(evidence))


def is_retrain_recommendation(text: str) -> bool:
    return bool(_RETRAIN.search(text)) and not _NEGATION.search(text)


def guard_retraining(recommendations: list[str], question: str, results: list[ToolResult]) -> tuple[list[str], list[str]]:
    """Return (kept recommendations, uncertainty notes for the ones withheld)."""
    if retraining_evidence(question, results):
        return list(recommendations), []
    kept: list[str] = []
    notes: list[str] = []
    for rec in recommendations:
        if is_retrain_recommendation(rec):
            notes.append("Retraining recommendation withheld (no freshness policy, failed quality gate or "
                         f"degradation evidence was found): {rec}")
        else:
            kept.append(rec)
    return kept, notes

def evidence_gaps(results: list[ToolResult]) -> list[str]:
    """Deterministic notes about things the answer cannot know from the tools that were called."""
    ok = {r.tool for r in results if r.status == "ok"}
    inspected = {"airflow_get_failed_tasks", "airflow_get_run_status"} & ok
    notes: list[str] = []
    for r in results:
        if r.tool == "airflow_list_dag_runs" and r.status == "ok" and r.data and not inspected:
            runs = r.data.get("runs") or []
            if runs and runs[0].get("state") == "failed":
                notes.append("The latest Airflow run is in state 'failed' but its tasks were not inspected, so the "
                             "cause (quality gate, data, or infrastructure) is unknown.")
    return notes
