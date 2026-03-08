# Batch 001 Prompt

Read:
- CODEX_REPO_AUDIT.md
- CODEX_REPO_AUDIT_ISSUES.json
- APP_ROADMAP.md
- RALPH_TASK_BOARD.md
- RALPH_RULES.md
- CLAUDE.md

Work only on:
- SEC-001
- DOC-001
- DOC-002

Objectives:
- Fix Docker packaging so build context does not capture local secrets or runtime state
- Rewrite runtime configuration docs to match JSON runtime config plus secrets-only `.env`
- Rewrite Windows quickstart so it truthfully distinguishes the real runtime entrypoints

Constraints:
- No unrelated feature work
- No broad refactors
- No production-readiness claims
- No issue-priority reshuffling

Required validation:
- inspect and update `.dockerignore`
- inspect and update `Dockerfile`
- run `python -m rex --help`
- run `python scripts/security_audit.py` if packaging files changed
- run targeted tests only if touched code requires them
- include exact commands and results in the final summary

Definition of done:
- SEC-001 is resolved or materially advanced with real file changes
- DOC-001 is resolved
- DOC-002 is resolved
- RALPH_TASK_BOARD.md is updated accurately
- final summary is complete and honest
- DOC-001 is resolved
- DOC-002 is resolved
- task board is updated accurately
- final summary is complete and honest
