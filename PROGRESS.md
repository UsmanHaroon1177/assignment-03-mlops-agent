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
- [ ] 3  mlflow_tools.py + dvc_tools.py
- [ ] 4  Airflow 3 container + 5-stage DAG + airflow_tools.py
- [ ] 5  FastAPI inference service + Dockerfile + manifests + kubernetes_tools.py
- [ ] 6  Agent core: schemas, llm, safety, validation, orchestrator, fake LLM
- [ ] 7  Multi-tool scenario (no automatic retraining)
- [ ] 8  pytest suite (8+ tests) + GitHub Actions CI
- [ ] 9  Two injected failures, 5 demo scenarios, screenshots
- [ ] 10 README, architecture.md, REPORT.md, fresh-clone test, secret scan

## Current status
Phases 0-2 done. Next: Phase 3.

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
