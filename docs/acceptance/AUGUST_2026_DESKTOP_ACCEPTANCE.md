# August 2026 Desktop Acceptance

## Scope

This record covers PR A: onboarding deferral, canonical per-user profiles,
profile IPC/UI, navigation cleanup, and truthful integration details.

## Automated evidence

- Deferred Home Assistant onboarding persists no URL or credential.
- Profile composition projects live permissions and voice enrollment without
  exposing private paths or secrets.
- Avatar writes are bounded, validated, normalized, atomic, and user-isolated.
- Electron profile IPC rejects cross-user authority fields and malformed data.
- The profile page uses typed IPC, displays avatar or initials, and separates
  private profile data from shared household settings.
- Primary navigation omits SMS and duplicate Settings while preserving direct
  routes and the persistent Settings shortcut.
- Integration cards distinguish configuration from tested authentication and
  show state-specific detail plus the next safe action.

## Verification record

- Targeted PR-A Python matrix: 196 tests passed.
- Full Python suite: 8,456 passed, 85 skipped, 0 failed.
- Coverage-gated Python run: 83% total coverage, above the 75% release floor;
  all 15 coverage-contract tests passed against the generated report.
- Electron chat-stream smoke: passed after clearing stale test-only Electron
  processes left by older local validation runs.
- GUI suite: 19 Vitest files / 99 tests passed; both TypeScript typechecks and
  the production Electron/Vite build passed.
- Fresh `npm ci` passed. `npm audit --audit-level=high` passed after the lockfile
  advanced `js-yaml` to 4.3.1 and Electron to 42.8.1 within existing version
  ranges. Two moderate React Router advisories remain; npm's automated remedy
  requires the breaking React Router 7 migration and is outside this PR.
- `python -m rex doctor --release-gate` passed with one non-blocking warning that
  no API keys are configured in this local worktree.
- `python scripts/security_audit.py --release-gate` passed with zero actionable
  placeholder findings and zero exposed secrets.
- Ruff, Black, the CI-equivalent detect-secrets scan, and `git diff --check`
  passed on the finalized local diff. GitHub Linux pre-commit remains the
  authoritative cross-platform hook confirmation before merge.

## Remaining physical checks

- Launch the Electron app on Windows and confirm the profile button opens the
  active authenticated user's profile.
- Upload and remove a real JPEG and PNG; restart and confirm persistence.
- Confirm James and Cole sessions never display each other's avatar,
  preferences, voice status, or private scope labels.
- Confirm deferred onboarding reaches the main application without Home
  Assistant configuration.
- Confirm integration detail remains readable in narrow and expanded layouts.

## Cost and credential statement

No paid service, new subscription, or new credential was introduced by this
workstream. Existing local and repository tooling was used.
