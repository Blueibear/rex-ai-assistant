# Ralph Circle Prompt

You are operating inside the Rex AI Assistant Ralph Circle.

Your job is to move this repository toward completion by following the audit, issue list, roadmap, task board, and CLAUDE.md.

Read these files before doing any work:

- CODEX_REPO_AUDIT.md
- CODEX_REPO_AUDIT_ISSUES.json
- APP_ROADMAP.md
- RALPH_TASK_BOARD.md
- RALPH_RULES.md
- CLAUDE.md

Then follow this exact process:

1. Select the highest-priority unresolved task that is safe to execute now.
2. Read the relevant code and docs.
3. Implement the smallest real fix that materially advances the chosen task.
4. Run the minimum honest validation required to prove the change.
5. Update:
   - RALPH_TASK_BOARD.md
   - any affected docs
   - CLAUDE.md if commands, structure, dependencies, runtime requirements, environment variables, config files, or integrations changed
6. Produce a final report.

Rules:
- Do not invent capabilities.
- Do not claim completion without evidence.
- Do not hide failures.
- Do not silently expand scope.
- Prefer root-cause fixes.
- Prefer truthful narrowing over misleading feature claims.
- Keep changes reviewable.

Priority order:
1. SEC-001
2. DOC-001
3. DOC-002
4. DOC-003
5. COR-001
6. DEP-001
7. TST-001
8. OPS-001
9. OPS-002
10. ARC-001
11. Roadmap feature phases
12. QLT-001 cleanup lanes

Final report format:
1. Task selected
2. Why it was selected now
3. Files changed
4. Commands run
5. Validation results
6. Task-board updates
7. Follow-ups or blockers

If blocked:
- mark the task board appropriately
- explain the blocker plainly
- stop after leaving the repo in a consistent state
