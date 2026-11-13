# Authenticated Airflow 3 API smoke test: token -> trigger -> poll -> task states.
param([string]$Model = 'logreg', [double]$MinF1 = 0.90, [int]$TimeoutSec = 600)
$root = Split-Path -Parent $PSScriptRoot
$cfg = @{}
foreach ($l in Get-Content "$root\.env.airflow") { if ($l -match '^([^#=]+)=(.*)$') { $cfg[$Matches[1]] = $Matches[2] } }
$base = 'http://localhost:8080'
try {
  $tok = Invoke-RestMethod -Method Post -Uri "$base/auth/token" -ContentType 'application/json' `
         -Body (@{ username = $cfg['AIRFLOW_USERNAME']; password = $cfg['AIRFLOW_PASSWORD'] } | ConvertTo-Json)
  $h = @{ Authorization = "Bearer $($tok.access_token)" }
  Write-Host "Token OK. DAGs:" -ForegroundColor Green
  (Invoke-RestMethod -Uri "$base/api/v2/dags" -Headers $h).dags | Select-Object dag_id, is_paused | Format-Table

  $body = @{ logical_date = $null; conf = @{ model = $Model; min_f1 = $MinF1 } } | ConvertTo-Json -Depth 5
  $run = Invoke-RestMethod -Method Post -Uri "$base/api/v2/dags/training_pipeline/dagRuns" -Headers $h -ContentType 'application/json' -Body $body
  $id = $run.dag_run_id
  Write-Host "Triggered $id (model=$Model, min_f1=$MinF1)" -ForegroundColor Cyan

  $deadline = (Get-Date).AddSeconds($TimeoutSec)
  do {
    Start-Sleep -Seconds 5
    $state = (Invoke-RestMethod -Uri "$base/api/v2/dags/training_pipeline/dagRuns/$id" -Headers $h).state
    Write-Host "  run state: $state"
  } while ($state -in @('queued', 'running') -and (Get-Date) -lt $deadline)

  Write-Host "`nFinal state: $state" -ForegroundColor Cyan
  (Invoke-RestMethod -Uri "$base/api/v2/dags/training_pipeline/dagRuns/$id/taskInstances" -Headers $h).task_instances |
    Select-Object task_id, state, try_number | Format-Table
} catch {
  Write-Host "API error: $($_.Exception.Message)" -ForegroundColor Red
  if ($_.ErrorDetails) { Write-Host $_.ErrorDetails.Message }
  exit 1
}