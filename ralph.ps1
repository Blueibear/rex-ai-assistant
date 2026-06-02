param(
    [int]$MaxIterations = 100,
    [int]$SleepSeconds = 2,

    # Which AI agent to use: 'claude' or 'codex'
    [ValidateSet('claude', 'codex')]
    [string]$Agent = 'claude',

    # Target a single PRD file (default: PRD.md for backward compatibility)
    [string]$PrdFile = "PRD.md",

    # Run multiple PRDs in sequence, e.g.:
    #   .\ralph.ps1 -PrdFiles PRD.md,PRD-repo-quality.md,PRD-production-readiness.md
    # When specified, -PrdFile is ignored.
    [string[]]$PrdFiles = @(),

    # Safety override: allow running directly on master/main.
    # Default behavior blocks this because the loop can auto-edit and auto-commit.
    [switch]$AllowMainBranch,

    # Safety override: allow starting with a dirty working tree.
    # Default behavior requires a clean repo before each PRD starts.
    [switch]$AllowDirtyStart
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

function Get-ProgressFile {
    param(
        [Parameter(Mandatory = $true)]
        [string]$PrdPath
    )

    $base = [System.IO.Path]::GetFileNameWithoutExtension($PrdPath)

    if ($base -eq "PRD") {
        return "progress.txt"
    }

    # PRD-repo-quality -> progress-repo-quality.txt
    # PRD-production-readiness -> progress-production-readiness.txt
    $suffix = $base -replace '^PRD', ''
    return "progress$suffix.txt"
}

function Test-GitAvailable {
    try {
        $null = git --version 2>$null
        return ($LASTEXITCODE -eq 0)
    }
    catch {
        return $false
    }
}

function Get-GitBranchName {
    if (-not (Test-GitAvailable)) {
        return ""
    }

    $branch = (git branch --show-current 2>$null | Out-String).Trim()
    if ($LASTEXITCODE -ne 0) {
        return ""
    }

    return $branch
}

function Get-GitStatusShort {
    if (-not (Test-GitAvailable)) {
        return ""
    }

    $status = (git status --short 2>$null | Out-String).Trim()
    if ($LASTEXITCODE -ne 0) {
        return ""
    }

    return $status
}

function Assert-SafeGitState {
    param(
        [bool]$AllowMain,
        [bool]$AllowDirty
    )

    if (-not (Test-GitAvailable)) {
        Write-Warning "Git was not found. Skipping git safety checks."
        return
    }

    $branch = Get-GitBranchName

    if (-not $AllowMain -and ($branch -eq "master" -or $branch -eq "main")) {
        Write-Error "Refusing to run Ralph directly on '$branch'. Create a working branch first, or rerun with -AllowMainBranch."
        exit 1
    }

    $status = Get-GitStatusShort

    if (-not $AllowDirty -and -not [string]::IsNullOrWhiteSpace($status)) {
        Write-Error @"
Refusing to start with a dirty working tree.

Current git status:
$status

Commit, stash, or discard these changes first.
To override intentionally, rerun with -AllowDirtyStart.
"@
        exit 1
    }
}

function Invoke-Agent {
    param(
        [Parameter(Mandatory = $true)]
        [ValidateSet('claude', 'codex')]
        [string]$AgentName,

        [Parameter(Mandatory = $true)]
        [string]$Prompt
    )

    $output = ""
    $exitCode = 0

    switch ($AgentName) {
        'claude' {
            $output = (& claude --dangerously-skip-permissions -p $Prompt 2>&1 | Out-String)
            $exitCode = $LASTEXITCODE
        }

        'codex' {
            $output = ($Prompt | & codex exec --full-auto - 2>&1 | Out-String)
            $exitCode = $LASTEXITCODE
        }
    }

    return [PSCustomObject]@{
        Output = $output
        ExitCode = $exitCode
    }
}

function Invoke-RalphOnPrd {
    param(
        [Parameter(Mandatory = $true)]
        [string]$PrdPath,

        [Parameter(Mandatory = $true)]
        [ValidateSet('claude', 'codex')]
        [string]$AgentName,

        [Parameter(Mandatory = $true)]
        [int]$MaxIter,

        [Parameter(Mandatory = $true)]
        [int]$Sleep
    )

    if (-not (Test-Path $PrdPath)) {
        Write-Error "PRD file not found: $PrdPath"
        return $false
    }

    $progressFile = Get-ProgressFile -PrdPath $PrdPath

    if (-not (Test-Path $progressFile)) {
        New-Item -ItemType File -Path $progressFile | Out-Null
    }

    Write-Host ""
    Write-Host "==========================================="
    Write-Host "  Ralph starting on: $PrdPath"
    Write-Host "  Agent:             $AgentName"
    Write-Host "  Progress log:      $progressFile"
    Write-Host "  Max iterations:    $MaxIter"
    Write-Host "==========================================="
    Write-Host ""

    for ($i = 1; $i -le $MaxIter; $i++) {
        Write-Host "==========================================="
        Write-Host "  Iteration $i of $MaxIter  ($PrdPath)"
        Write-Host "==========================================="

        $prompt = @"
You are Ralph, an autonomous coding agent. Do exactly ONE User Story per iteration.

## COMMIT FORMAT - READ THIS BEFORE ANYTHING ELSE

Every commit MUST use Conventional Commits format or CI will fail with "subject-empty":

  <type>(<optional scope>): <subject>

Valid types: feat, fix, test, docs, refactor, chore, perf, ci

CORRECT:
  feat(voice): add streaming TTS playback
  fix(auth): reject expired session tokens
  test(planner): add unit tests for tool selection
  docs: update README quick start section
  chore(deps): upgrade pip-audit to 2.7.0

WRONG - these will fail CI:
  Fix coverage test gating          <- no type prefix, fails commitlint
  feat:add streaming TTS            <- missing space after colon
  Feat: Add streaming TTS.          <- capital type, capital subject, trailing period

Before running git commit, verify your message matches this pattern:
  ^(feat|fix|test|docs|refactor|chore|perf|ci)(\(.+\))?: [a-z].+[^.]$

If it does not match, rewrite the message before committing.

## Repo Instructions

Before making changes:

1. Read CLAUDE.md if it exists.
2. Read AGENTS.md if it exists.
3. Read ${PrdPath}.
4. Read ${progressFile}, especially the most recent Learnings sections.
5. Follow CLAUDE.md and AGENTS.md unless they conflict with this PRD.
6. If this task changes commands, scripts, file structure, dependencies, config files, environment variables, integrations, or recurring repo rules, update CLAUDE.md in the same commit.

## Task Selection Rule

1. Find the first unfinished User Story section in ${PrdPath} whose acceptance criteria contain any unchecked [ ] box.
2. A task means one full User Story section, not one checkbox.
3. Complete exactly one User Story per iteration.
4. Do not move to the next User Story until the current one is completed, split, or explicitly blocked.
5. If the current User Story is too large to complete safely in one iteration, split it into smaller User Stories in ${PrdPath}, explain the split in ${progressFile}, commit that PRD-only change, and stop the iteration.

## Hard Limits

- Do not start broad refactors unless the active User Story explicitly requires it.
- Do not add new features unless the active User Story explicitly requires it.
- Do not weaken, delete, or skip tests just to make CI green.
- Do not hide dependency vulnerabilities behind broad suppressions.
- Do not delete legacy surfaces unless the active User Story explicitly requires it.
- Do not modify files unrelated to the current User Story except tests/docs directly required for verification.
- Do not mark work complete based only on appearances. Verify it.
- Do not report success unless code, tests, and acceptance criteria prove it.

## Implementation Steps

1. Identify the first unfinished User Story in ${PrdPath}.
2. State the User Story ID/title you are working on.
3. Implement that ONE User Story only.
4. Run the validation commands listed in that User Story.
5. If no validation commands are listed, run the smallest relevant tests/typechecks needed to prove the acceptance criteria.
6. Check git diff before committing.
7. Only update checkboxes that are truthfully complete.

## Critical: Complete Only If Story Acceptance Criteria Are Verified

Use the current User Story's own acceptance criteria and validation commands as the source of truth.

Some stories intentionally require reproducing known failures, audit failures, collection errors, or baseline errors.
For those stories, a nonzero command exit may be expected success if the User Story explicitly says to reproduce, confirm, or record that failure.

Examples:
- If a baseline story says "pytest collection reproduces ModuleNotFoundError", then that nonzero pytest result can satisfy the criterion.
- If a baseline story says "pip-audit reproduces known vulnerabilities", then a failing pip-audit can satisfy the criterion.
- If a later hardening story says "pip-audit exits 0", then a failing pip-audit does NOT satisfy the criterion.

If every acceptance criterion for the current User Story is truthfully satisfied:

1. Update ${PrdPath} first.
   - Mark every satisfied acceptance criterion for the current User Story as [x].
   - If the story is complete, leave no unchecked boxes in that User Story.
   - ${PrdPath} is the authoritative task tracker.

2. Update ${progressFile} second.
   - Add the iteration summary, files changed, commands run, validation results, and learnings.

3. Stage all completed-story files together:
   git add ${PrdPath} ${progressFile} <all code/test/doc files changed for this story>

4. Before committing, run:
   git diff --cached --name-only

5. If ${PrdPath} is not listed in git diff --cached --name-only, do not commit.
   Output exactly:
   <promise>BLOCKED</promise>

6. Commit only after ${PrdPath}, ${progressFile}, implementation files, and test files are staged together.
   Write your commit message, verify it matches the COMMIT FORMAT above, then commit.

7. Do not commit implementation work while leaving the completed User Story unchecked in ${PrdPath}.

- End normally unless the whole PRD is complete.

If any acceptance criterion is not satisfied:

- Do NOT mark the User Story complete.
- Do NOT commit broken code.
- Revert code changes made during this iteration unless they are intentional diagnostic notes requested by the User Story.
- Append what went wrong to ${progressFile}.
- Output exactly: <promise>BLOCKED</promise>

## Progress Notes Format

Append to ${progressFile} using this format:

## Iteration [N] - [User Story ID and Title]
- What was implemented or verified
- Files changed
- Commands run and outcomes
- Acceptance criteria completed
- Acceptance criteria still open, if any
- Learnings for future iterations:
  - Patterns discovered
  - Gotchas encountered
  - Useful context
---

## Update AGENTS.md and CLAUDE.md If Applicable

If you discover a reusable pattern that future work should know about:

- Check if AGENTS.md exists in the project root.
- Check if CLAUDE.md exists in the project root.
- Add patterns like: "This codebase uses X for Y" or "Always do Z when changing W."
- Only add genuinely reusable knowledge, not task-specific details.
- If a previous mistake would have been prevented by a short rule, add that rule.

## End Condition

After completing your User Story, check ${PrdPath}:

- If ALL User Stories are complete, output exactly: <promise>COMPLETE</promise>
- If the current User Story is blocked, output exactly: <promise>BLOCKED</promise>
- If User Stories remain unfinished, just end your response normally.
"@

        $agentResult = Invoke-Agent -AgentName $AgentName -Prompt $prompt
        $result = $agentResult.Output
        $exitCode = $agentResult.ExitCode

        Write-Host $result
        Write-Host ""

        if ($exitCode -ne 0) {
            Write-Warning "$AgentName exited with code $exitCode. Stopping."
            return $false
        }

        if ($result -match "<promise>BLOCKED</promise>") {
            Write-Warning "${PrdPath} blocked on iteration $i. Stopping."
            return $false
        }

        if ($result -match "<promise>COMPLETE</promise>") {
            Write-Host "==========================================="
            Write-Host "  ${PrdPath} complete after $i iterations!"
            Write-Host "==========================================="
            return $true
        }

        Start-Sleep -Seconds $Sleep
    }

    Write-Host "==========================================="
    Write-Host "  Reached max iterations ($MaxIter) on ${PrdPath}"
    Write-Host "==========================================="
    return $false
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

$targets = @(
    if ($PrdFiles.Count -gt 0) {
        $PrdFiles
    }
    else {
        $PrdFile
    }
)

Write-Host "Starting Ralph - Agent: $Agent | $($targets.Count) PRD(s) | $MaxIterations iterations each"

Assert-SafeGitState -AllowMain:$AllowMainBranch.IsPresent -AllowDirty:$AllowDirtyStart.IsPresent

$allComplete = $true

foreach ($target in $targets) {
    if (-not (Test-Path $target)) {
        Write-Error "PRD file not found: $target"
        exit 1
    }

    $ok = Invoke-RalphOnPrd -PrdPath $target -AgentName $Agent -MaxIter $MaxIterations -Sleep $SleepSeconds

    if (-not $ok) {
        $allComplete = $false
        Write-Warning "Did not finish $target within $MaxIterations iterations. Stopping."
        exit 1
    }
}

if ($allComplete) {
    Write-Host ""
    Write-Host "==========================================="
    Write-Host "  All PRDs complete!"
    Write-Host "==========================================="
    exit 0
}
