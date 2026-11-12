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
- [ ] 1  Environment: venv, deps, .wslconfig, Docker + Kubernetes check
- [ ] 2  Dataset + DVC + train.py + MLflow logging (DVC version + git commit on each run)
- [ ] 3  mlflow_tools.py + dvc_tools.py
- [ ] 4  Airflow 3 container + 5-stage DAG + airflow_tools.py
- [ ] 5  FastAPI inference service + Dockerfile + manifests + kubernetes_tools.py
- [ ] 6  Agent core: schemas, llm, safety, validation, orchestrator, fake LLM
- [ ] 7  Multi-tool scenario (no automatic retraining)
- [ ] 8  pytest suite (8+ tests) + GitHub Actions CI
- [ ] 9  Two injected failures, 5 demo scenarios, screenshots
- [ ] 10 README, architecture.md, REPORT.md, fresh-clone test, secret scan

## Current status
Phase 0 done once the first push succeeds. Next: Phase 1.

## Notes / bugs / lessons
(none yet)

## If starting a new chat
Paste this whole file and say which phase you are on.
