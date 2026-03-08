# Ralph Circle Rules
Last updated: 2026-03-08

## Source of truth

Read these files before every cycle:
- CODEX_REPO_AUDIT.md
- CODEX_REPO_AUDIT_ISSUES.json
- APP_ROADMAP.md
- RALPH_TASK_BOARD.md
- CLAUDE.md

If docs and code disagree, trust current code plus the audit, not stale documentation.

## Work-selection rules

1. Always choose the highest-priority unresolved task that is safe to execute now.
2. Do not skip a higher-priority task just because a lower one is easier.
3. If the current task depends on a human decision or live external validation, mark it blocked and stop.

## Implementation rules

- Make the smallest real fix that materially advances the chosen task.
- Do not add fake scaffolding and call it complete.
- Do not claim a capability is implemented unless it is executable through the real path that users will hit.
- Do not delete tests to hide failures.
- Do not suppress warnings just to make reports quieter.
- Do not casually expand scope.
- Do not rewrite architecture unless the task truly requires it.

## Validation rules

After every implementation cycle, run the smallest honest validation set that proves the work.

Possible validation includes:
- `python -m rex --help`
- `python scripts/doctor.py`
- `python scripts/security_audit.py`
- `python scripts/validate_deployment.py`
- targeted `pytest` runs
- `python -m ruff check <changed files>`
- `python -m black --check <changed files>`
- `python -m mypy <changed files or modules>`

If validation fails, the cycle is not complete.

## Documentation rules

When commands, structure, dependencies, runtime requirements, environment variables, config files, or integrations change:
- update `CLAUDE.md` in the same change set
- update user-facing docs in the same change set if they are affected

Prefer truthful narrowing over misleading expansion.

## Commit rules

- Use Conventional Commits
- Commit only validated states
- Final cycle reports must include:
  - selected task
  - files changed
  - commands run
  - validation results
  - task-board updates
  - blockers or follow-ups
 
Commit every completed task.

Commit message format:

ralph: <task-id> <short description>

## Verification rules

The verifier must not trust implementation summaries.
It must re-check the relevant code and rerun the listed validations where possible.

## Stop rules

Stop immediately if:
- a blocker is discovered
- a required human decision is missing
- the agent or verifier exits non-zero
- the repo cannot be backed up safely
- the same unresolved task persists without meaningful board progress for the configured stall limit

## Completion rule

The loop is complete only when:
- all task-board items that define the current project scope are marked `[x]`
- no items are marked `[!]`
- the last verifier run succeeds
