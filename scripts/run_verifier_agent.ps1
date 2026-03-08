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

Write-Host "Running Claude verifier iteration $Iteration"

New-Item -ItemType Directory -Force -Path (Split-Path $LogFile) | Out-Null

$basePrompt = Get-Content -Raw $PromptFile

$verifierPrompt = @"
You are the verification pass of the Rex Ralph Circle.

Verify the changes from the previous iteration.

Tasks:

1. Confirm the task selected was correct.
2. Confirm implementation actually solves the issue.
3. Confirm tests pass.
4. Confirm task board updates are truthful.
5. Identify regressions or architectural drift.

If the implementation is incorrect, mark the task back to TODO.

Base prompt:

$basePrompt
"@

Set-Location $RepoRoot

try {

    $verifierPrompt | claude | Tee-Object -FilePath $LogFile

    Write-Log "Verifier completed"
    exit 0

}
catch {

    Write-Log "Verifier failed: $_"
    exit 1

}
