# One-time local setup: random Airflow credentials (never committed) + pinned image requirements.
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$utf8 = New-Object Text.UTF8Encoding $false
function New-Secret($n) { -join ((48..57) + (65..90) + (97..122) | Get-Random -Count $n | ForEach-Object { [char]$_ }) }

New-Item -ItemType Directory -Force "$root\airflow" | Out-Null
if (-not (Test-Path "$root\.env.airflow")) {
  $text = "AIRFLOW_USERNAME=admin`nAIRFLOW_PASSWORD=$(New-Secret 24)`nAIRFLOW__API_AUTH__JWT_SECRET=$(New-Secret 48)`n"
  [IO.File]::WriteAllText("$root\.env.airflow", $text, $utf8)
  Write-Host "Created .env.airflow (gitignored)"
} else { Write-Host ".env.airflow already exists - keeping it" }

$pw = ((Get-Content "$root\.env.airflow") | Where-Object { $_ -like 'AIRFLOW_PASSWORD=*' }) -replace '^AIRFLOW_PASSWORD=', ''
[IO.File]::WriteAllText("$root\airflow\passwords.json", "{`"admin`": `"$pw`"}`n", $utf8)
Write-Host "Wrote airflow\passwords.json (gitignored)"

$req = "$root\docker\airflow\requirements.txt"
if (-not (Test-Path $req)) {
  $pins = & "$root\.venv\Scripts\python.exe" -m pip freeze |
    Select-String -Pattern '^(mlflow|scikit-learn|pandas|numpy|scipy|skops|dvc|joblib)==' | ForEach-Object { $_.Line }
  [IO.File]::WriteAllText($req, (($pins -join "`n") + "`n"), $utf8)
  Write-Host "Wrote docker\airflow\requirements.txt:"; Get-Content $req
}