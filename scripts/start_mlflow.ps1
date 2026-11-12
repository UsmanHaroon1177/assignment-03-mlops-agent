# Starts the MLflow tracking server. Keep this window open.
Set-Location -LiteralPath (Split-Path -Parent $PSScriptRoot)
& .\.venv\Scripts\Activate.ps1
mlflow server --backend-store-uri sqlite:///mlflow.db --artifacts-destination ./mlartifacts --host 127.0.0.1 --port 5000
