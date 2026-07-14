# Fable Handoff

This planning package reduces the mobile gateway implementation to two Fable sessions.

## Session 1

Implement only **Foundation and Authentication** from `MOBILE_API_IMPLEMENTATION_PLAN.md` on a fresh branch based on current `origin/master`.

Read, in order:

1. root `CLAUDE.md`;
2. this directory's `CLAUDE.md`;
3. `MOBILE_API_MASTER_SPEC.md`;
4. `MOBILE_CLIENT_CONTRACT_AUDIT.md`;
5. `MOBILE_API_ARCHITECTURE.md`;
6. the Session 1 section of `MOBILE_API_IMPLEMENTATION_PLAN.md`;
7. the applicable rows in `MOBILE_API_TEST_MATRIX.md`.

Do not continue into chat, WebSocket, voice, or TTS. Open the backend PR, wait for/fix CI, leave it unmerged, and stop.

## Session 2

Start only after Session 1 merges. Implement **Chat, WebSocket, Voice, TTS, and Client Alignment** from the plan. Coordinate the backend PR with mobile draft PR #3 and draft PR #5. Leave all PRs unmerged after CI and report automated, LAN, and physical-iPhone validation separately.

## Token discipline

- Use targeted searches/reads; do not dump entire files or passing logs into chat.
- Keep long command output in local log files.
- Report only phase transitions, blockers, failing tests/root causes/fixes, PR details, and final validation.
- Run focused tests first and the full suite only after focused gates pass.
- Use these planning documents as project memory rather than repeating reconnaissance.
- Preserve every security, identity, testing, and documentation requirement; token savings must not weaken implementation.
