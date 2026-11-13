# Starts the MLflow tracking server (SQLite backend, proxied artifacts).
# 0.0.0.0 + allowed-hosts so the Airflow container can reach it via host.docker.internal.
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$py = Join-Path $root '.venv\Scripts\python.exe'
& $py -m mlflow server `
  --backend-store-uri sqlite:///mlflow.db `
  --artifacts-destination ./mlartifacts `
  --host 0.0.0.0 --port 5000 `
  --allowed-hosts "localhost:*,127.0.0.1:*,host.docker.internal:*"