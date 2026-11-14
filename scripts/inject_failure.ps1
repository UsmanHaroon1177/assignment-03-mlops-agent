<#
Operator tool for the failure demos (the agent never does this).
  .\scripts\inject_failure.ps1 -Mode badpath   # wrong MODEL_PATH in the ConfigMap -> pods Running 0/1
  .\scripts\inject_failure.ps1 -Mode oom       # tiny memory limit -> OOMKilled / CrashLoopBackOff
  .\scripts\inject_failure.ps1 -Mode restore   # re-apply the committed manifests and restart
#>
param([Parameter(Mandatory)][ValidateSet('badpath', 'oom', 'restore')][string]$Mode)
$ErrorActionPreference = 'Stop'
$ns = 'mlops-agent'

function Patch-File($Json, [scriptblock]$Run) {
    $tmp = New-TemporaryFile
    [IO.File]::WriteAllText($tmp.FullName, $Json, (New-Object Text.UTF8Encoding $false))
    try { & $Run $tmp.FullName } finally { Remove-Item $tmp.FullName -ErrorAction SilentlyContinue }
}

switch ($Mode) {
    'badpath' {
        Patch-File '{"data":{"MODEL_PATH":"/app/serving/model/missing.joblib"}}' {
            param($f) kubectl -n $ns patch configmap breast-cancer-api-config --type merge --patch-file $f
        }
        kubectl -n $ns rollout restart deployment/breast-cancer-api
        $wait = 45
    }
    'oom' {
        kubectl -n $ns set resources deployment/breast-cancer-api --requests=memory=16Mi --limits=memory=24Mi
        $wait = 90
    }
    'restore' {
        kubectl apply -f kubernetes/configmap.yaml -f kubernetes/deployment.yaml
        kubectl -n $ns rollout restart deployment/breast-cancer-api
        kubectl -n $ns rollout status deployment/breast-cancer-api --timeout=180s
        kubectl -n $ns get deploy,pods
        return
    }
}
Write-Host "waiting $wait s for the failure to develop..."
Start-Sleep $wait
kubectl -n $ns get deploy,pods