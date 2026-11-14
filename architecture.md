# Architecture

## Overview
User / engineer
| natural language
v
FastAPI agent interface (agent/main.py) validation, optional X-Agent-Token
v
Orchestrator (agent/orchestrator.py) LLM loop, permissions, evidence gate, retraining policy, logging
| JSON tool calls (validated)
+--> DVC tools git/dvc allowlist, read-only
+--> MLflow tools MlflowClient, FINISHED runs only
+--> Airflow tools Airflow 3 REST API (JWT), DAG training_pipeline only
+--> Kubernetes tools official client, namespace mlops-agent only
v
Evidence-based response: known facts, uncertainties, recommended actions, tool-evidence table


## Request flow

1. The question is length-checked, then sent to Gemini with the system prompt (rules, scenario playbooks, tool catalogue).
2. Gemini replies with JSON: either `{"action":"tools","calls":[...]}` or `{"action":"final",...}`.
3. Each call is validated (known tool, known arguments). The model-supplied `confirmed` argument is stripped.
4. Read-only tools run through `execute()`: timing, error classification, redaction, structured log line.
   Mutating tools are parked as **pending actions**.
5. Results go back to the model as data (explicitly "not instructions"), limited to 6 calls per step, 12 in total, 6 steps.
6. On the final answer the code applies the **evidence gate** and the **retraining policy**, then renders the response.

## Guarantees enforced in code

| Guarantee | Mechanism |
|---|---|
| No invented operational facts | A fact survives only if every cited tool returned `ok`; otherwise it moves to uncertainties |
| Unavailable tool means unknown | Every non-ok tool adds an uncertainty |
| Empty citations are repaired, never guessed | One repair turn, then the fact stays withheld |
| No self-confirmation | `confirm(id)` is human-initiated, one-shot, 10 minute TTL, and sets `confirmed=True` itself |
| No automatic retraining | `airflow_trigger_training` is mutating; the policy removes retrain advice without evidence |
| Prompt injection resistance | Tool results are labelled data; arguments and tool names are validated |
| No secret leakage | `redact()` on all tool output and errors; logs record argument names only |
| Failed latest run is not ignored | `evidence_gaps` flags a failed Airflow run whose tasks were not inspected |

Retraining advice is kept only with evidence: a failed `quality_gate` in an `ok` Airflow result, or a user-supplied freshness/degradation statement containing a number.

## Tool inventory and permissions

| Module | Tools | Access |
|---|---|---|
| DVC | `dvc_list_tracked_datasets`, `dvc_get_dataset_version`, `dvc_check_data_changes`, `dvc_compare_dataset_versions`, `dvc_get_run_dataset_version`, `dvc_explain_reproduce_version` | Read-only. git: log, show, tag, rev-parse, ls-files, status. dvc: status, diff. No shell. |
| MLflow | `mlflow_list_experiments`, `mlflow_list_runs`, `mlflow_get_run_details`, `mlflow_find_best_run`, `mlflow_compare_runs`, `mlflow_compare_experiments`, `mlflow_get_model_registry` | Read-only. 10 s timeout, 1 retry. Best run is never a deployment decision. |
| Airflow | `airflow_list_dags`, `airflow_list_dag_runs`, `airflow_get_run_status`, `airflow_get_failed_tasks`, `airflow_get_task_logs` | Read-only, fixed to `training_pipeline` |
| Airflow | `airflow_trigger_training` | **Mutating.** Only `model` and `min_f1`, validated. Needs confirmation. |
| Kubernetes | `k8s_list_deployments`, `k8s_list_pods`, `k8s_get_pod_status`, `k8s_get_pod_logs`, `k8s_list_services`, `k8s_find_unhealthy_pods`, `k8s_get_events`, `k8s_get_deployed_version` | Read-only, namespace `mlops-agent`, label `app=breast-cancer-api`, no namespace argument |
| Kubernetes | `k8s_rollout_restart` | **Mutating.** Only that deployment, that namespace, `confirmed=True` set by the system |

28 tools in total. Tool status values: `ok`, `error`, `unavailable`, `rejected`. Only `ok` carries facts.

## Data and ML flow

- **DVC**: `data/dataset.csv.dvc` tracks the dataset; versions are git tags `data-v1` / `data-v2`; remote is a local folder.
- **Training** (`training/train.py`): logs params, metrics and provenance tags (dataset version, DVC md5, git commit, dirty flag) to MLflow.
- **Airflow DAG**: validate_dataset, preprocess_data, train_model (2 retries), evaluate_and_log (2 retries), quality_gate. Gate failure tags the MLflow run and stops promotion.
- **Serving**: the model is baked into the image; `/readyz` returns 503 until it loads, `/healthz` is liveness, `/version` reports model provenance.

## Kubernetes deployment

Namespace `mlops-agent`; Deployment `breast-cancer-api` (2 replicas, RollingUpdate, readiness and liveness probes,
requests 50m/128Mi, limits 250m/256Mi, non-root UID 10001); ClusterIP Service port 80 to 8000; ConfigMap with
`APP_VERSION`, `MODEL_PATH`, `LOG_LEVEL`.

## Logging

One JSON line per tool call: `request_id`, `tool`, `status`, `duration_ms`, argument names, redacted error.
The same `request_id` is on the response and on every tool result for that request.

## Testing strategy

Offline pytest suite with a scripted `FakeLLM` and mocked MLflow, Airflow, Kubernetes and Gemini HTTP.
It covers tool selection, timeouts, invalid requests, outages, the evidence gate, confirmation gating,
namespace rejection, redaction and log contents. CI runs ruff and pytest with no secrets.

## Known limitations

See the README. In short: the summary is not evidence-checked, Kubernetes reports declared (not runtime) versions,
and free-tier Gemini latency is high.