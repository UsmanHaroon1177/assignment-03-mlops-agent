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
- [x] 5  FastAPI inference service + Dockerfile + manifests + kubernetes_tools.py
- [x] 6  Agent core: schemas, llm, safety, validation, orchestrator, fake LLM
- [x] 7  Multi-tool scenario (no automatic retraining)
- [x] 8  pytest suite (8+ tests) + GitHub Actions CI
- [x] 9  Two injected failures, 5 demo scenarios, screenshots
- [x] 10 README, architecture.md, REPORT.md, fresh-clone test, secret scan

## Current status
Phases 0-8 done. Next: Phase 9 (two injected failures, 5 demo scenarios, screenshots).

## Notes / bugs / lessons
- Gemini sometimes returns objects ({"claim": ...}) in uncertainties/recommended_actions: schemas._text_of coerces them.
- The model wrote conclusions in `summary` for tools it never called (e.g. "no data changes" without dvc_check_data_changes).
  The evidence gate only checks known_facts, so the summary is now labelled "model-written, not evidence-checked" and the
  prompt forbids unqueried conclusions. Report as a known limitation.
- A failed latest Airflow run with no task inspection now adds an uncertainty (policy.evidence_gaps).
- Fake mutating tools in tests must declare the same params as the real tool, or check_call rejects the arguments.
- Record the model that answered (ask.py prints "answered by") for every demo in REPORT.md; -latest aliases can change.
- Ruff: line-length 140, select E,F,I,B, B008 allows fastapi.Depends/Path/Header, E501 ignored in tests/.
  Rewriting pyproject.toml with a regex broke it once; edit it whole.
- Coverage (agent/): 88% total; uncovered lines are mostly the raw cluster/HTTP wrappers that tests mock on purpose.

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

## Phase 5 interfaces (code that exists now)
- Inference service: api/main.py (GET /healthz liveness, /readyz readiness 503 until model loaded, /version, POST /predict
  with 30 features). Config from env (ConfigMap): APP_VERSION, MODEL_PATH, LOG_LEVEL. Model baked into the image:
  scripts/export_model.py writes serving/model/{model.joblib (gitignored), metadata.json} and serving/requirements.txt.
- Image breast-cancer-api:1.0.0 (non-root UID 10001). Manifests in kubernetes/: namespace mlops-agent, configmap
  breast-cancer-api-config, deployment breast-cancer-api (2 replicas, probes, requests/limits, label app=breast-cancer-api,
  app.kubernetes.io/version), ClusterIP service breast-cancer-api (port 80 -> 8000).
  Access: kubectl -n mlops-agent port-forward svc/breast-cancer-api 8001:80; python scripts\try_api.py.
- agent/tools/kubernetes_tools.py (names k8s_*): list_deployments, list_pods, get_pod_status(pod_name), get_pod_logs(pod_name,
  tail_lines<=200, previous), list_services, find_unhealthy_pods (reasons + deployment issues + warning events + all_clear),
  get_events, get_deployed_version (declared label/image tag/ConfigMap + mismatches; does not query the app),
  rollout_restart (mutating=True; namespace/deployment/confirmed). Scope fixed to namespace mlops-agent, app=breast-cancer-api.
- rollout_restart REJECTS unless confirmed is exactly True. Phase 6 orchestrator MUST set confirmed=True itself after explicit
  user approval and strip any model-supplied `confirmed` from arguments (also applies to airflow_trigger_training approval).
- Cluster calls go through raw functions (_list_pods, _read_logs, ...) returning dicts; tests monkeypatch those.
- Injected failures available for Phase 9: wrong MODEL_PATH in the ConfigMap (pods Running 0/1, readiness 503), bad image
  tag (ImagePullBackOff), tiny memory limit (OOMKilled), scale to 0.
- Smoke: python -m scripts.try_k8s (read-only). Screenshots: k8s_deployed.png, api_via_k8s.png, k8s_tools_smoke.png.

## Phase 6 interfaces (code that exists now)
- agent/schemas.py: ToolResult plus ToolCall, Fact(claim, evidence[tool names]), LLMAction (action tools|final, calls,
  summary, known_facts, uncertainties, recommended_actions), PendingAction, AgentResponse(answer, known_facts,
  uncertainties, recommended_actions, tool_results, pending_actions, llm_error, request_id).
- agent/llm.py: GeminiLLM (REST via requests, key only in x-goog-api-key header, env GEMINI_API_KEY, model env
  GEMINI_MODEL default gemini-2.5-flash, retries 429/5xx twice), LLMError, list_models(). agent/fake_llm.py: FakeLLM(script).
- agent/validation.py: parse_action (fence-tolerant JSON), check_call (unknown tool/arg -> ToolInputError, strips
  model-supplied `confirmed`), check_question (<=2000 chars). agent/registry.py: load_tools() -> {name: ToolSpec}.
  agent/prompts.py: build_system_prompt(tools).
- agent/orchestrator.py: Orchestrator(llm, tools=None, max_steps=6).run(question, request_id) -> AgentResponse.
  Evidence gate: a known fact is kept only if ALL cited tools returned status ok; otherwise it moves to uncertainties
  ("withheld"); every non-ok tool adds an uncertainty. Mutating tools never run from model output: they become
  PendingAction; orchestrator.confirm(id) (user-initiated, one-shot, 10 min TTL) sets confirmed=True itself;
  reject(id); pending(). Dedupes identical calls; caps 6 calls/step, 12 total; truncates tool data to 5000 chars for the model.
- scripts/ask.py: python -m scripts.ask [--list-models] [--verbose] "question" (real Gemini + real tools, asks yes/no for pending).
- Tests: tests/test_agent_core.py (FakeLLM, no network).

## Phase 7 interfaces (code that exists now)
- agent/prompts.py: SCENARIOS (playbooks A-D + correlation + retraining rules), included in the system prompt.
  Test asserts every tool name in SCENARIOS exists in load_tools().
- agent/policy.py: guard_retraining(recs, question, results) -> (kept, notes). Retrain recommendations survive only with
  evidence: airflow_get_failed_tasks / airflow_get_run_status (status ok) showing quality_gate failed, or a user statement
  with a freshness/degradation keyword AND a number. Negated mentions ("no retraining needed") are kept. Withheld ones become
  uncertainties. Orchestrator._apply_policy runs after _finalize and re-renders the answer.
- agent/main.py: uvicorn agent.main:app --port 8002. POST /ask, GET /pending, POST /confirm/{id}, POST /reject/{id},
  GET /tools, GET /health. Optional env AGENT_API_TOKEN -> header X-Agent-Token required on all but /health.
  .env is loaded in the app lifespan (not at import), so tests never read the real key.
- Known gap (document in REPORT): the guard filters recommended_actions only, not free text in the summary; Kubernetes
  tools cannot prove which MLflow run the deployed image was built from.
- Tests: tests/test_scenarios.py, tests/test_agent_api.py (need httpx).

## Phase 8 interfaces (code that exists now)
- LLM config in .env (gitignored): GEMINI_API_KEY, GEMINI_MODEL=gemini-flash-latest, GEMINI_FALLBACK_MODELS=gemini-flash-lite-latest.
- agent/policy.py also has evidence_gaps(results); orchestrator._apply_policy adds its notes to uncertainties.
- agent/schemas.py _text_of(): dict -> claim/text/description/message/recommendation text, else JSON.
- Rendered answers start with "Summary (model-written; only the facts below are checked against tool evidence):".
- tests/test_agent_hardening.py. CI: .github/workflows/ci.yml (ruff check + pytest --cov=agent, Python 3.12, no secrets,
  all external services mocked). Dependencies in requirements.txt (kubernetes, httpx, requests, pydantic, pytest, pytest-cov, ruff).

## Phase 8 interfaces (code that exists now)
- .env (gitignored): GEMINI_API_KEY, GEMINI_MODEL=gemini-flash-latest, GEMINI_FALLBACK_MODELS=gemini-flash-lite-latest.
- agent/policy.py: evidence_gaps(results); orchestrator._apply_policy adds its notes to uncertainties.
- agent/schemas.py _text_of(): dict -> claim/text/description/message/recommendation text, else JSON.
- Answers start with "Summary (model-written; only the facts below are checked against tool evidence):".
- CI: .github/workflows/ci.yml (ruff check + pytest --cov=agent, Python 3.12, no secrets, services mocked). Green.

## Phase 9 evidence (files that exist now)
- scripts/inject_failure.ps1 -Mode badpath|oom|restore (operator only). evidence/failure1_badpath.txt (names missing.joblib /
  MODEL_PATH), evidence/failure2_oom.txt (OOMKilled exit 137, 24Mi limit, restarts). Screenshots failure1_*, failure2_*.
- scripts/run_demos.py [1..5] writes evidence/demoN_*.txt (answer, model name, request id, tool evidence).
  demo5_multi_tool_airflow_outage.txt = real Airflow outage handled as uncertainty, retrain suggestion withheld.
- Unexpected failure for defence: Airflow train_model failed 3x because MLflow answered 403 "Invalid Host header" to
  host.docker.internal (server started without --allowed-hosts). Fixed by restarting via start_mlflow.ps1. The DAG
  then took 11 min (container I/O bound, 1.9/2.5 GB) but succeeded. The agent saw only "failed at train_model"
  because stderr sits in the exception, past the log tail.
- Lessons: a shell GEMINI_MODEL overrides .env; Gemini free-tier latency makes one answer take about 2 min; demo prompt
  needs refs data-v1/data-v2 (not v1/v2).