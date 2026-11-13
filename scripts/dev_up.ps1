<#
 Pre-checks and service launcher.
   .\scripts\dev_up.ps1              checks + starts MLflow (new window) if it is down
   .\scripts\dev_up.ps1 -Airflow     ...and starts the Airflow container
   .\scripts\dev_up.ps1 -NoMlflow    checks only, do not start MLflow
#>
param([switch]$Airflow, [switch]$NoMlflow)

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$script:bad = 0
function Ok($m)   { Write-Host "[ OK ]  $m" -ForegroundColor Green }
function Warn($m) { Write-Host "[WARN]  $m" -ForegroundColor Yellow }
function Bad($m)  { Write-Host "[FAIL]  $m" -ForegroundColor Red; $script:bad++ }
function Test-Url($url) {
  try { Invoke-WebRequest -Uri $url -UseBasicParsing -TimeoutSec 3 | Out-Null; return $true } catch { return $false }
}
function Wait-Url($url, $seconds) {
  for ($i = 0; $i -lt $seconds; $i += 3) { if (Test-Url $url) { return $true }; Start-Sleep -Seconds 3 }
  return $false
}

Write-Host "`n== Environment ==" -ForegroundColor Cyan
if ($env:VIRTUAL_ENV) { Ok "venv active: $env:VIRTUAL_ENV" } else { Warn "venv not active (run .\.venv\Scripts\Activate.ps1)" }
if (Test-Path "$root\.venv\Scripts\python.exe") { Ok "venv python found" } else { Bad "no .venv in repo root" }

Write-Host "`n== Docker / Kubernetes ==" -ForegroundColor Cyan
docker info 2>$null | Out-Null
if ($LASTEXITCODE -ne 0) { Bad "Docker daemon not running - start Docker Desktop and retry"; exit 1 }
Ok "Docker daemon running"
$mem = docker info --format '{{.MemTotal}}'
$gb = [math]::Round([double]$mem / 1GB, 1)
if ($gb -ge 3.5) { Ok "Docker memory: $gb GB" } else { Warn "Docker memory only $gb GB (need ~4 GB; check .wslconfig, then wsl --shutdown)" }
$ctx = kubectl config current-context 2>$null
if ($ctx -eq 'docker-desktop') { Ok "kubectl context: $ctx" } else { Warn "kubectl context is '$ctx' (expected docker-desktop)" }
$nodes = kubectl get nodes --no-headers 2>$null
if ($LASTEXITCODE -eq 0 -and $nodes -match 'Ready') { Ok "Kubernetes node Ready" } else { Warn "Kubernetes not ready (Docker Desktop > Settings > Kubernetes)" }

Write-Host "`n== MLflow (http://127.0.0.1:5000) ==" -ForegroundColor Cyan
if (-not (Test-Url 'http://127.0.0.1:5000/health') -and -not $NoMlflow) {
  Write-Host "MLflow is down - starting it in a new window..."
  Start-Process powershell -WorkingDirectory $root -ArgumentList '-NoExit','-ExecutionPolicy','Bypass','-File',"$root\scripts\start_mlflow.ps1"
  Wait-Url 'http://127.0.0.1:5000/health' 45 | Out-Null
}
if (Test-Url 'http://127.0.0.1:5000/health') { Ok "MLflow healthy" } else { Bad "MLflow not reachable" }

if ($Airflow) {
  Write-Host "`n== Airflow (http://localhost:8080) ==" -ForegroundColor Cyan
  if (-not (Test-Path "$root\.env.airflow")) { Bad "run .\scripts\airflow_setup.ps1 first"; exit 1 }
  docker compose -f docker-compose.airflow.yml up -d
  if (Wait-Url 'http://localhost:8080/api/v2/monitor/health' 240) { Ok "Airflow API healthy" } else { Bad "Airflow not healthy after 240 s (docker logs mlops-airflow)" }
  $code = docker exec mlops-airflow curl -s -o /dev/null -w '%{http_code}' http://host.docker.internal:5000/health 2>$null
  if ($code -eq '200') { Ok "Airflow container can reach MLflow" } else { Bad "container -> MLflow returned '$code' (firewall? allowed-hosts?)" }
}

Write-Host "`n== Git ==" -ForegroundColor Cyan
git status -sb | Select-Object -First 5

Write-Host ""
if ($script:bad -eq 0) { Write-Host "All required checks passed." -ForegroundColor Green } else { Write-Host "$($script:bad) check(s) failed." -ForegroundColor Red }