param(
    [Parameter(Mandatory = $true)]
    [string]$RepoRoot,

    [Parameter(Mandatory = $true)]
    [string]$PromptFile,

    [Parameter(Mandatory = $true)]
    [string]$LogFile,

    [Parameter(Mandatory = $true)]
    [int]$Iteration
)

$ErrorActionPreference = "Stop"

function Write-Log {
    param([string]$Message)
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Add-Content -Path $LogFile -Value "[$timestamp] $Message"
}

Write-Host "Running Claude Code iteration $Iteration"

New-Item -ItemType Directory -Force -Path (Split-Path $LogFile) | Out-Null

$prompt = Get-Content -Raw $PromptFile

Set-Location $RepoRoot

try {

    $prompt | claude | Tee-Object -FilePath $LogFile

    Write-Log "Claude Code iteration completed"
    exit 0

}
catch {

    Write-Log "Claude Code failed: $_"
    exit 1

}
