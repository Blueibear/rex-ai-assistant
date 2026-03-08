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
    param(
        [string]$Message
    )
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Add-Content -Path $LogFile -Value "[$timestamp] $Message"
}

function Invoke-ClaudeCode {
    param(
        [string]$RepoRootPath,
        [string]$PromptFilePath
    )

    $promptText = Get-Content -Raw -Path $PromptFilePath

    Set-Location $RepoRootPath

    # =========================================================================
    # IMPORTANT:
    # Replace the placeholder command below with the EXACT Claude Code command
    # you already use successfully in PowerShell.
    #
    # Example shape only, NOT a guaranteed valid Claude command:
    # claude --print --dangerously-skip-permissions $promptText
    #
    # Or if your workflow uses stdin:
    # $promptText | claude --print --dangerously-skip-permissions
    #
    # Do NOT leave the placeholder line in place.
    # =========================================================================

    throw "Edit scripts/run_primary_agent.ps1 and replace the placeholder Claude Code command with the exact working command you use in PowerShell."
}

New-Item -ItemType Directory -Force -Path (Split-Path -Parent $LogFile) | Out-Null
"=== PRIMARY AGENT ITERATION $Iteration START ===" | Set-Content -Path $LogFile
Write-Log "RepoRoot: $RepoRoot"
Write-Log "PromptFile: $PromptFile"

try {
    Invoke-ClaudeCode -RepoRootPath $RepoRoot -PromptFilePath $PromptFile
    Write-Log "Primary agent completed successfully."
    exit 0
}
catch {
    Write-Log "Primary agent failed: $($_.Exception.Message)"
    Write-Error $_
    exit 1
}
