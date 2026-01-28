Param(
    [string]$RexRoot = "C:\RexNode",
    [string]$PackageSource = "rex-ai-assistant",
    [string]$Services = "event_bus,workflow_runner,memory_store,credential_manager",
    [int]$Port = 8765,
    [switch]$DryRun
)

Write-Host "Installing Rex lean node to $RexRoot"

if (-not (Test-Path $RexRoot)) {
    if ($DryRun) {
        Write-Host "[DRY RUN] Would create $RexRoot"
    } else {
        New-Item -ItemType Directory -Path $RexRoot | Out-Null
    }
}

$env:REX_SERVICES = $Services
$env:REX_SERVICE_PORT = "$Port"

if ($DryRun) {
    Write-Host "[DRY RUN] python -m venv $RexRoot\venv"
} else {
    python -m venv "$RexRoot\venv"
}

$pip = "$RexRoot\venv\Scripts\pip.exe"
$python = "$RexRoot\venv\Scripts\python.exe"

if ($DryRun) {
    Write-Host "[DRY RUN] $pip install $PackageSource"
    Write-Host "[DRY RUN] $pip install pywin32"
} else {
    & $pip install $PackageSource
    & $pip install pywin32
}

$envFile = "$RexRoot\.env.node"
if (-not (Test-Path $envFile)) {
    if ($DryRun) {
        Write-Host "[DRY RUN] Copying .env.node template to $envFile"
    } else {
        Copy-Item "$PSScriptRoot\.env.node" $envFile
    }
}

if ($DryRun) {
    Write-Host "[DRY RUN] $python -m rex.windows_service install"
    Write-Host "[DRY RUN] $python -m rex.windows_service start"
} else {
    & $python -m rex.windows_service install
    & $python -m rex.windows_service start
}

Write-Host "Register the node with the gateway (stub):"
Write-Host "  Invoke-RestMethod -Method Post -Uri $env:REX_GATEWAY_URL/api/nodes/register -Headers @{Authorization=\"Bearer $env:REX_NODE_TOKEN\"}"
