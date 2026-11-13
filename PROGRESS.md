# PROGRESS

Assignment 03: Production-Ready MLOps AI Agent (DVC, MLflow, Airflow 3, Kubernetes, FastAPI, Gemini)
Repo: https://github.com/UsmanHaroon1177/assignment-03-mlops-agent
Local: D:\University\Fast Semester 9\MLOPS\A3\assignment-03-mlops-agent
Machine: Windows + PowerShell, 8 GB RAM, Docker Desktop with built-in Kubernetes. Deadline: 2 days.

## Decisions
- Airflow 3 runs in Docker (standalone, SQLite). MLflow runs from the venv (SQLite backend).
- DVC remote is a local folder outside the repo.
- Kubernetes: Docker Desktop cluster, namespace `mlops-agent`, 2 replicas, small limits.
- Agent loop is plain Python; Gemini chooses tools via JSON. Fake LLM for tests (no quota in CI).
- Safety: no shell=True, command allowlists, redaction in tools, only DAG `training_pipeline`,
  K8s mutations need explicit confirmation and are limited to the approved namespace.
- Claims must be backed by tool results; unavailable tool = "unknown", never invented.
- Secrets only in env vars. Never commit keys.

## Phases
- [x] 0  Scaffold, git init, .gitignore, PROGRESS.md, first push
- [x] 1  Environment: venv, deps, .wslconfig, Docker + Kubernetes check
- [x] 2  Dataset + DVC + train.py + MLflow logging (DVC version + git commit on each run)
- [x] 3  mlflow_tools.py + dvc_tools.py
- [x] 4  Airflow 3 container + 5-stage DAG + airflow_tools.py
- [ ] 5  FastAPI inference service + Dockerfile + manifests + kubernetes_tools.py
- [ ] 6  Agent core: schemas, llm, safety, validation, orchestrator, fake LLM
- [ ] 7  Multi-tool scenario (no automatic retraining)
- [ ] 8  pytest suite (8+ tests) + GitHub Actions CI
- [ ] 9  Two injected failures, 5 demo scenarios, screenshots
- [ ] 10 README, architecture.md, REPORT.md, fresh-clone test, secret scan

## Current status
Phases 0-4 done. Next: Phase 5 (FastAPI service + Dockerfile + K8s manifests + kubernetes_tools.py).

## Notes / bugs / lessons
(none yet)

## If starting a new chat
Paste this whole file and say which phase you are on.



## Key interfaces (so a new chat writes code that fits)
- MLflow server (keep running in a 2nd terminal):
  mlflow server --backend-store-uri sqlite:///mlflow.db --artifacts-destination ./mlartifacts --host 127.0.0.1 --port 5000
- Experiment: breast-cancer-classifier. Registered model: breast-cancer-clf.
- Train: python -m training.train --model {logreg|rf|gbm} [--register] [--run-name X]
- Run params: model, seed, test_size, n_rows, n_features + hyperparameters.
- Run metrics: accuracy, precision, recall, f1, roc_auc, train_seconds.
- Run tags (provenance): dataset.path, dataset.dvc_file, dataset.dvc_md5, dataset.version (v1/v2),
  dataset.git_commit, dataset.dvc_status (up-to-date/modified/unknown), code.git_commit, code.git_dirty.
- Run artifacts: provenance/dataset.csv.dvc, model/.
- Dataset versions = git tags data-v1 (400 rows), data-v2 (569 rows); DVC remote = ../dvc-remote.
- Reproduce a version: git checkout data-vN -- data/dataset.csv.dvc; dvc checkout data/dataset.csv.dvc
- 6 seeded runs: v1-logreg, v1-rf, v1-gbm(registered), v2-logreg, v2-rf(registered), v2-gbm.
- Deferred: dvc.yaml goes in Phase 4 (DVC refuses nested dvc calls inside stages; provenance.dvc_status
  must be skipped when env DVC_ROOT is set). Airflow container must reach MLflow via host.docker.internal
  (MLflow 3 may need --allowed-hosts).
- Docker memory: needed .wslconfig (memory=4GB, swap=2GB); verify with docker info before Phase 4.

### Bug log
- MLflow 3.18 saves sklearn models with skops; tree models need skops_trusted_types=["sklearn.tree._tree.Tree"].
  The two failed runs had already logged metrics, so MLflow tools must only consider status=FINISHED runs
  (add a regression test in Phase 8). scripts/delete_failed_runs.py cleans them up.


## Phase 3 interfaces (code that exists now)
- agent/schemas.py: ToolResult(tool, status in ok|error|unavailable|rejected, data, error, duration_ms, request_id).
  Only status=ok carries facts. Anything else => the agent must say the information is unknown.
- agent/tools/base.py: ToolSpec(name, description, params, fn, mutating=False); ToolInputError (-> rejected),
  ToolUnavailableError (-> unavailable); execute(tool, fn, request_id, **kwargs) never raises, validates
  arguments, redacts output, logs JSON (tool, status, duration_ms, request_id, arg NAMES only).
- agent/safety.py: redact(text), redact_obj(obj) - for TOOL output only, never the model's prose.
- agent/logging_utils.py: configure(), new_request_id(), log_tool_call().
- Each tool module exposes TOOLS: list[ToolSpec]. Names are prefixed (mlflow_*, dvc_*).
- MLflow tools: list_experiments, list_runs, get_run_details, find_best_run (FINISHED only, returns caveats,
  is_deployment_decision=false), compare_runs, compare_experiments, get_model_registry. resolve_run accepts
  run id, 8+ char prefix or run name. MLflow calls use a 10 s timeout and 1 retry.
- DVC tools: list_tracked_datasets, get_dataset_version, check_data_changes, compare_dataset_versions,
  get_run_dataset_version (MLflow tags verified against git), explain_reproduce_version (read-only).
  Commands: no shell, allowlist git {log,show,tag,rev-parse,ls-files,status}, dvc {status,diff}.
- Tests: 36 offline tests (mocked MLflow client, temp git repo). scripts/try_tools.py = real-service smoke test.
- scripts/start_mlflow.ps1 starts the tracking server.
- Restart routine: start Docker Desktop, run .\scripts\start_mlflow.ps1, then python scripts\check_env.py.

## Phase 4 interfaces (code that exists now)
- Startup: .\scripts\dev_up.ps1 (checks + starts MLflow), .\scripts\dev_up.ps1 -Airflow (also the container).
  One-time: .\scripts\airflow_setup.ps1 (creates gitignored .env.airflow + airflow\passwords.json).
- Airflow: docker-compose.airflow.yml, container mlops-airflow, UI/API http://localhost:8080 (API under /api/v2,
  JWT from POST /auth/token). Container reaches MLflow at http://host.docker.internal:5000
  (start_mlflow.ps1 binds 0.0.0.0 with --allowed-hosts). Project is mounted at /opt/airflow/project.
- DAG training_pipeline (schedule=None): validate_dataset(no retry) -> preprocess_data -> train_model(2 retries)
  -> evaluate_and_log(2 retries) -> quality_gate(no retry). Params: model (logreg|rf|gbm), min_f1 (default 0.90).
  Gate failure raises AirflowFailException and tags the MLflow run quality_gate=failed; nothing is promoted.
  Runs are named airflow-<model>-<run_id> and tagged airflow.dag_id / airflow.dag_run_id.
- agent/tools/airflow_tools.py (names airflow_*): list_dags, list_dag_runs, get_run_status, get_failed_tasks,
  get_task_logs, trigger_training (mutating=True; only model + min_f1). All fixed to training_pipeline.
  Credentials from env AIRFLOW_USERNAME/AIRFLOW_PASSWORD (fallback .env.airflow); base URL env AIRFLOW_BASE_URL.
  HTTP 400/404/409/422 -> rejected; connection/timeout/401/403/5xx -> unavailable.
- dvc.yaml has a `train` stage (dvc repro logs run dvc-repro-logreg). Inside dvc repro, DVC_ROOT is set and
  training/provenance.dvc_status() returns "unknown" (DVC refuses nested dvc calls).
- Smoke scripts: scripts\airflow_smoke.ps1 [-Model rf -MinF1 0.999], scripts\try_airflow.py [--trigger].
- Screenshots: screenshots/airflow_dag_success.png, airflow_gate_failed.png, dvc_reproduce.png.
- Phase 6 must require explicit confirmation before executing any ToolSpec with mutating=True.
