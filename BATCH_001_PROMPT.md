# Batch 001 Prompt

Read:
- CODEX_REPO_AUDIT.md
- CODEX_REPO_AUDIT_ISSUES.json
- APP_ROADMAP.md
- RALPH_TASK_BOARD.md
- RALPH_RULES.md
- CLAUDE.md

Work only on this batch:

Batch ID: BATCH-001

Scope:
- SEC-001
- DOC-001
- DOC-002

Objectives:
- lock down Docker packaging so build context does not capture secrets or local state
- make runtime configuration docs match the current JSON runtime config plus secrets-only `.env` model
- fix Windows quickstart so it points to the real current entrypoints and runtime behavior

Constraints:
- do not work on unrelated feature additions
- do not attempt broad refactors
- do not claim production readiness
- do not modify issue priorities
- update CLAUDE.md if commands, runtime requirements, or documentation rules need correction

Required validation:
- review Dockerfile and `.dockerignore`
- run relevant targeted tests if code changes require it
- run `python scripts/security_audit.py` if packaging files change
- run `python -m rex --help`
- include exact commands run and outcomes in the summary

Definition of done:
- SEC-001 is resolved or materially advanced with validated file changes
- DOC-001 is resolved
- DOC-002 is resolved
- task board is updated accurately
- final summary is complete and honest
