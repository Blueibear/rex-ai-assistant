# Assignment 001 — Independent current-state audit

Act as the independent verification and architecture auditor for AskRex. Read and obey `CLAUDE.md` before inspecting the repository.

Desktop repository: current worktree.
Read-only mobile repository:
`C:\Users\james\rex-ai-test\askrex-mobile\AskRex-lead`

Authoritative requirements:
- `docs/planning/source-of-truth/REX_Unified_Build_Spec_UPDATED.md`
- `docs/planning/source-of-truth/REX_ACTIVE_CHECKLIST.md`
- `docs/planning/TEAM_LEAD_OPERATING_RULES.md`

Audit actual code and tests rather than trusting PRD completion marks. This assignment is analysis only; do not implement product code and do not modify the mobile repository.

Conserve usage: prioritize high-risk and high-value paths, use targeted searches, and run only focused validations needed to establish evidence.

Do not access paid services, purchase credits, rotate credentials, deploy, publish, or make external account changes.

## Required deliverable

Create `docs/planning/CODEX_CURRENT_STATE_AUDIT.md` with evidence-based findings for:

1. Desktop capability inventory and whether each capability is exposed and configurable in Electron.
2. Mobile functionality, transport, authentication, session handling, and desktop-interaction readiness.
3. Desktop/mobile pairing and capability authorization gaps.
4. Identity, per-user isolation, memory, history, device ownership, and credential boundaries.
5. Voice path: wake word, STT, TTS, device routing, barge-in, recovery, and latency.
6. Assistant intelligence: intent/context, model routing/fallback, planning, tool selection, verification, memory retrieval, and failure recovery.
7. OpenClaw boundary and dynamic plugin/skill requirements while preserving Rex core independence.
8. Packaging, installer, private mobile distribution, updates, CI, dependency security, and release gates.
9. Desktop/mobile branding, logo, design-token and navigation parity.
10. Misleading, stubbed, hidden, dead, or undocumented functions and settings.

Every finding must include severity, exact location, observed evidence, impact, root cause, recommended fix, and validation commands. Distinguish verified facts from inference.

End with the ten best implementation batches in dependency order, with clear acceptance gates. Commit only the audit and this assignment file using a Conventional Commit message.