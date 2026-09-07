Param(
    [string]$RexRoot = "C:\RexNode",
    [string]$PackageSource = "rex-ai-assistant",
    [string]$Services = "event_bus,workflow_runner,memory_store,credential_manager",
    [int]$Port = 8765,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($RexRoot)) {
    throw "RexRoot must not be empty."
}
$RexRoot = [System.IO.Path]::GetFullPath($RexRoot)

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

$venv = Join-Path $RexRoot "venv"
$pip = Join-Path $RexRoot "venv\Scripts\pip.exe"
$python = Join-Path $RexRoot "venv\Scripts\python.exe"

if ($DryRun) {
    Write-Host "[DRY RUN] python -m venv `"$venv`""
} else {
    python -m venv $venv
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to create the Rex lean-node virtual environment at $venv."
    }
    if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
        throw "Rex lean-node Python interpreter was not created at the expected absolute path: $python"
    }
}

if ($DryRun) {
    Write-Host "[DRY RUN] & `"$pip`" install `"$PackageSource`""
    Write-Host "[DRY RUN] & `"$pip`" install pywin32"
} else {
    & $pip install $PackageSource
    & $pip install pywin32
}

$envFile = Join-Path $RexRoot ".env.node"
if (-not (Test-Path $envFile)) {
    if ($DryRun) {
        Write-Host "[DRY RUN] Copying .env.node template to $envFile"
    } else {
        Copy-Item "$PSScriptRoot\.env.node" $envFile
    }
}

if ($DryRun) {
    Write-Host "[DRY RUN] & `"$python`" -m rex.windows_service install"
    Write-Host "[DRY RUN] & `"$python`" -m rex.windows_service start"
} else {
    & $python -m rex.windows_service install
    & $python -m rex.windows_service start
}

Write-Host "Register the node with the gateway (stub):"
Write-Host "  Invoke-RestMethod -Method Post -Uri $env:REX_GATEWAY_URL/api/nodes/register -Headers @{Authorization=\"Bearer $env:REX_NODE_TOKEN\"}"
