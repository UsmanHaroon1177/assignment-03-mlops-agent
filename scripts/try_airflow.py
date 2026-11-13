"""Real-service smoke test for the Airflow tools (Airflow must be running)."""
import json
import sys

from agent.tools import airflow_tools as af
from agent.tools.base import execute

TOOL = {t.name: t for t in af.TOOLS}


def call(name, **kw):
    r = execute(name, TOOL[name].fn, request_id="try", **kw)
    print(f"\n== {name} {kw} -> {r.status} ({r.duration_ms} ms)")
    print(json.dumps(r.data if r.status == "ok" else {"error": r.error}, indent=2)[:1500])
    return r


call("airflow_list_dags")
call("airflow_list_dag_runs", limit=3)
call("airflow_get_run_status")
call("airflow_get_failed_tasks")
if "--trigger" in sys.argv:
    call("airflow_trigger_training", model="logreg", min_f1=0.90)