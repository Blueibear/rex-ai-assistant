# Ralph Circle Rules
Last updated: 2026-03-08

These rules govern every Ralph Circle iteration.

## Source of truth

Always read these files before making changes:
- `CODEX_REPO_AUDIT.md`
- `CODEX_REPO_AUDIT_ISSUES.json`
- `APP_ROADMAP.md`
- `RALPH_TASK_BOARD.md`
- `CLAUDE.md`

Never assume README claims are correct unless they match the audit or current code.

## Execution order

1. Pick the highest-priority unresolved item from `RALPH_TASK_BOARD.md`
2. Read the relevant code before changing anything
3. Implement the smallest real fix that moves the issue toward done
4. Run validation appropriate to the change
5. Update `RALPH_TASK_BOARD.md`
6. Update docs and `CLAUDE.md` if required
7. Commit only if the repo is in a better and validated state

## Non-negotiable rules

- Do not stub new behavior unless the task explicitly calls for scaffolding
- Do not mark a task complete without evidence
- Do not suppress warnings just to make checks quieter
- Do not delete tests to make failures disappear
- Do not broaden scope unless required to complete the current issue correctly
- Do not introduce new heavy dependencies unless clearly justified
- Do not store secrets in source-controlled config
- Do not rewrite architecture casually
- Do not claim production readiness
- Do not trust stale docs over executable reality

## Validation rules

For every cycle, run only the smallest validation set that truthfully proves the change.

Possible validation includes:
- targeted pytest runs
- `python -m rex --help`
- `python scripts/doctor.py`
- `python scripts/security_audit.py`
- `python scripts/validate_deployment.py`
- `ruff check <changed files>`
- `black --check <changed files>`
- `mypy <changed files or module>`

If a change affects commands, file structure, dependencies, runtime requirements, environment variables, config files, or integrations, update `CLAUDE.md` in the same batch.

## Documentation rules

If docs and code disagree:
- fix the docs unless the task explicitly says to implement the missing behavior

Prefer truthful narrowing over optimistic wording.

Docs must clearly separate:
- implemented and verified
- implemented but not yet verified live
- stubbed or scaffolding
- planned only

## Batch sizing rules

Each batch should be:
- one issue, or
- one tightly related set of issues

Do not combine unrelated risky changes in one batch.

Good examples:
- SEC-001 plus related Docker docs
- DOC-001 plus DOC-002 when both are config-entrypoint truth fixes

Bad examples:
- Docker packaging, voice-loop refactor, and WooCommerce features together

## Stop rules

Stop and report instead of guessing when:
- a fix depends on an unresolved human decision
- a live credential or external service is required to verify behavior
- changing one issue reveals a larger architectural conflict
- the current batch would become a large refactor

When blocked, report:
- what is blocked
- why it is blocked
- the exact next best move
- whether work can continue on another batch safely

## Commit rules

Use Conventional Commits.

Every cycle report must include:
- files changed
- commands run
- validation results
- task board changes
- blockers or follow-ups

Do not commit a broken or unvalidated state unless the task explicitly allows a draft checkpoint and that is clearly labeled.

## Codex and Claude division of labor

Claude Code:
- implements bounded slices
- updates docs and `CLAUDE.md`
- runs validation
- produces a detailed summary

Codex:
- verifies claims skeptically
- reruns validation
- confirms whether the issue is actually resolved
- produces the next audit or remediation findings

## Definition of progress

A cycle counts as progress only if at least one of these is true:
- a priority issue is fully resolved
- a misleading claim is removed
- validation signal becomes more trustworthy
- architecture becomes more internally consistent
- a roadmap cycle is completed and verified
