# Ralph Circle Prompt

Use the following prompt for Codex or Claude Code when running the Ralph Circle.

---

You are operating inside the Rex AI Assistant Ralph Circle.

Your job is to move this repository toward completion using the audit, issue list, roadmap, and task board as the source of truth.

Before doing any work, read these files fully:

- CODEX_REPO_AUDIT.md
- CODEX_REPO_AUDIT_ISSUES.json
- APP_ROADMAP.md
- RALPH_TASK_BOARD.md
- RALPH_RULES.md
- CLAUDE.md

Then follow this exact process:

1. Identify the single highest-priority unresolved task that is safe to work on now
2. Read the relevant code and docs for that task
3. Implement the smallest real fix needed
4. Run the minimum honest validation needed to prove the change
5. Update:
   - RALPH_TASK_BOARD.md
   - docs affected by the change
   - CLAUDE.md if commands, structure, dependencies, runtime requirements, environment variables, config files, or integrations changed
6. Produce a final summary

Rules:
- Do not invent capabilities
- Do not claim completion without evidence
- Do not stub unless the task explicitly calls for scaffolding
- Do not silently expand scope
- Do not delete tests to hide failures
- Prefer truthful narrowing over misleading feature claims
- Prefer root-cause fixes over symptom suppression
- Use Conventional Commits
- Keep changes small and reviewable

Priority policy:
- Safety and truthfulness before feature growth
- Current priority order:
  1. SEC-001
  2. COR-001
  3. DOC-001
  4. DOC-002
  5. DOC-003
  6. DEP-001
  7. TST-001
  8. OPS-001
  9. OPS-002
  10. ARC-001
  11. INC-002
- Only move to roadmap expansion work after the critical truth and containment items are under control

Output format:
1. Task selected
2. Why this task was chosen now
3. Files changed
4. Commands run
5. Validation results
6. Task board updates
7. Follow-ups or blockers

If blocked:
- do not guess
- state the blocker clearly
- explain the exact next best move
- stop after leaving the repo in a consistent state
