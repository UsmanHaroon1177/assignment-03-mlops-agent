# assignment-03-mlops-agent

A conversational MLOps agent that answers questions by calling real tools for **DVC**, **MLflow**,
**Apache Airflow 3** and **Kubernetes**. Gemini chooses the tools; the code enforces the rules.
MLOps & Production AI Engineering, Fall 2026, Assignment 03.

Key properties:
- **Evidence only.** A claim is kept only if every tool it cites returned `status=ok`. Unavailable tools become "unknown".
- **Read-only by default.** The only state-changing tools (`airflow_trigger_training`, `k8s_rollout_restart`) never run from model output. They become pending actions that a human confirms.
- **Narrow scope.** One DAG (`training_pipeline`), one namespace (`mlops-agent`), allowlisted git/dvc commands, no `shell=True`.
- **Secrets only in environment variables.** Tool output is redacted before the model or logs see it.

See `architecture.md` for the design and `REPORT.md` for tested conversations and failure investigations.

## Requirements

Windows + PowerShell (commands below), Python 3.12 or 3.13, Docker Desktop with Kubernetes enabled,
Git, and a Gemini API key. About 8 GB RAM; give Docker 4 GB (`.wslconfig`: `memory=4GB`, `swap=2GB`).

## Install

```powershell
git clone https://github.com/UsmanHaroon1177/assignment-03-mlops-agent.git
cd assignment-03-mlops-agent
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Configure

```powershell
Copy-Item .env.example .env      # then edit .env
```

| Variable | Purpose |
|---|---|
| `GEMINI_API_KEY` | Gemini key (never commit) |
| `GEMINI_MODEL`, `GEMINI_FALLBACK_MODELS` | Primary and comma-separated fallback models |
| `MLFLOW_TRACKING_URI` | Default `http://127.0.0.1:5000` |
| `AIRFLOW_BASE_URL`, `AIRFLOW_USERNAME`, `AIRFLOW_PASSWORD` | Airflow 3 API access |
| `AGENT_API_TOKEN` | Optional; if set, the agent API requires header `X-Agent-Token` |
| `KUBE_CONTEXT` | Optional kubeconfig context (default: current) |

Real environment variables override `.env`. If a model name seems ignored, check `$env:GEMINI_MODEL`.
`python -m scripts.ask --list-models` shows which models your key can use.

## Data and experiments (DVC + MLflow)

Dataset versions are git tags `data-v1` (400 rows) and `data-v2` (569 rows); the DVC remote is a local folder `../dvc-remote`.

```powershell
dvc pull                                   # fetch the tracked dataset
.\scripts\start_mlflow.ps1                 # MLflow on :5000 (own terminal, keep open)
python -m training.train --model rf --register
```

Reproduce a dataset version: `git checkout data-v1 -- data/dataset.csv.dvc` then `dvc checkout data/dataset.csv.dvc`.
Each run records the dataset version, DVC md5 and git commit as MLflow tags.

## Airflow 3

```powershell
.\scripts\airflow_setup.ps1                # one time: creates gitignored .env.airflow
.\scripts\dev_up.ps1 -Airflow              # checks + starts MLflow and the Airflow container
.\scripts\airflow_smoke.ps1                # trigger the DAG and watch it
```

UI/API at http://localhost:8080. The DAG `training_pipeline` has five stages:
validate_dataset, preprocess_data, train_model, evaluate_and_log, quality_gate (retries on train/evaluate; a failed gate stops promotion).
The container reaches MLflow through `host.docker.internal`, so MLflow must be started with `start_mlflow.ps1`.

## Inference service on Kubernetes

```powershell
python scripts\export_model.py             # writes serving/model (model.joblib is gitignored)
docker build -t breast-cancer-api:1.0.0 .
kubectl apply -f kubernetes/namespace.yaml
kubectl apply -f kubernetes/configmap.yaml -f kubernetes/deployment.yaml -f kubernetes/service.yaml
kubectl -n mlops-agent rollout status deployment/breast-cancer-api
kubectl -n mlops-agent port-forward svc/breast-cancer-api 8001:80    # optional
python scripts\try_api.py                  # smoke test through the port-forward
```

Deployment: 2 replicas, readiness (`/readyz`) and liveness (`/healthz`) probes, CPU/memory requests and limits, ClusterIP Service, ConfigMap.

## Run the agent

Command line (asks for `yes` before any state-changing action):

```powershell
python -m scripts.ask "Find the best recent model, identify the dataset version used to train it, check the deployment status, and explain whether retraining is necessary."
```

HTTP API:

```powershell
uvicorn agent.main:app --port 8002
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8002/ask -ContentType "application/json" `
  -Body '{"question":"Is my prediction API healthy on Kubernetes?"}'
```

Routes: `POST /ask`, `GET /pending`, `POST /confirm/{id}`, `POST /reject/{id}`, `GET /tools`, `GET /health`.
Docs at http://127.0.0.1:8002/docs. Answers can take about 2 minutes on the free Gemini tier.

Every answer has: a model-written summary (not evidence-checked), **known facts** (each with its tool names),
**uncertainties**, **recommended actions**, and a tool-evidence table.

## Tests and lint

```powershell
ruff check .
python -m pytest -q --cov=agent --cov-report=term-missing
```

All external services (Gemini, MLflow, Airflow, Kubernetes) are mocked, so tests need no keys or cluster.
GitHub Actions (`.github/workflows/ci.yml`) runs lint and tests on pull requests and pushes to main.

## Demos and failure injection

```powershell
python -m scripts.run_demos            # five scenarios, saved to evidence/
.\scripts\inject_failure.ps1 -Mode badpath   # wrong MODEL_PATH -> pods Running 0/1
.\scripts\inject_failure.ps1 -Mode oom       # 24Mi limit -> OOMKilled / CrashLoopBackOff
.\scripts\inject_failure.ps1 -Mode restore
```

Failure injection is an operator action. The agent never injects or repairs anything.

## Repository layout
agent/ orchestrator, llm, policy, prompts, validation, schemas, main.py (HTTP)
agent/tools/ dvc_tools, mlflow_tools, airflow_tools, kubernetes_tools, base
api/ FastAPI inference service
dags/ training_pipeline DAG
training/ train.py and dataset provenance
kubernetes/ namespace, configmap, deployment, service
docker/airflow/ Airflow image; docker-compose.airflow.yml at the root
scripts/ start/smoke/demo/injection helpers
tests/ offline pytest suite
evidence/ saved agent answers screenshots/ report screenshots



## Known limitations

- The model-written summary is not evidence-checked; only the known-facts list is.
- Kubernetes tools report the *declared* version; nothing links a deployed image to a specific MLflow run.
- The retraining guard filters recommendations, not free text in the summary.
- Free-tier Gemini is slow and rate limited; `-latest` model aliases can change behaviour over time.
- Airflow in Docker on 8 GB RAM is I/O bound (one pipeline run took 11 minutes).