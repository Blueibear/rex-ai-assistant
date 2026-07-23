# Audit Remediation Ledger — 2026-07-22

Source task: AskRex audit remediation findings A–K. Active planning source: `PRD-production-readiness.md`.

Status meanings follow the task contract: Implemented, Locally verified, CI verified, Externally or hardware verified, and Blocked. A locally verified item remains pending CI until the pull request checks finish.

| Finding | Status | Commit | Files / implementation evidence | Local validation | Remaining external verification |
|---|---|---|---|---|---|
| A — Calendar test isolation | Locally verified | `53fcb0e` | `tests/conftest.py`, calendar backend/CLI tests; deterministic temporary account configuration | Focused calendar tests passed | None |
| B — Electron identity/data isolation | Locally verified | `3dac391` | `gui/src/main/sessionIdentity.ts`, identity bridge, all private-data handlers, ownership migration, two-user isolation tests | Session Identity Vitest plus Python identity/migration/isolation suites passed | User-driven migration review on an existing household data directory |
| C — Home Assistant safety | Locally verified | `e118325` | `rex/ha/mutation_service.py`, Electron mutation bridge, OpenClaw HA adapter, truthful response builder | HA policy, confirmation, replay/expiry/cross-user, state verification, IPC, and response-language tests passed | Live HA locks, alarms, covers, lights, stale state, and timeout behavior |
| D — Tool execution contract | Locally verified | `226741c` | `rex/tools/execution.py` and all dispatch adapters; normalized typed lifecycle and redacted audit | Lifecycle, router, registry, false-success, duplicate, denial, missing-argument, timeout, and unverifiable-result tests passed | Future external adapters must adopt the contract before being enabled |
| E — Self-contained Windows distribution | Locally verified | `41a2ee4` | managed Python builder, pinned Core/Voice requirements, packaged bridge resolver, resource scanner | Wheel/runtime/package tests passed; installed artifact launched without machine Python/Node or checkout | Authenticode signing; GPU/CUDA and XTTS remain optional external profiles |
| F — Windows artifact CI | Locally verified | `55eee02` | blocking reusable Windows workflow, release dependency, install/reinstall/uninstall harness | Workflow contract tests and local installed-artifact harness passed | GitHub-hosted Windows job on this PR |
| G — Hold to Talk | Locally verified | `a10577b` | response TTS, selected output, cancellation, replay, barge-in, device fallback, repeated turns, structured timing | 123 voice tests and 12 GUI tests passed during implementation | Physical microphone, audible selected output, barge-in, and device-loss hardware checks; wake word remains beta |
| H — Integration truth | Locally verified | `310bcb4` | shared state vocabulary, CLI, doctor, API/capability registry, Electron UI, email draft-only UX | 88 focused Python tests, 12 GUI tests, GUI typecheck, and lint with no errors | Email/calendar/SMS/provider writes and external OpenClaw/search reachability |
| I — Diagnostics/security gates | Locally verified | `e55701b` | distinct PASS/WARN/ERROR, doctor release gate, actionable/expiring security suppressions | Doctor, registry, and security-audit tests passed | None |
| J — GUI lint/dependencies | Locally verified | `6bb97e4` | ESLint 9 flat config, upgraded dependencies, locked overrides | lint passes with three existing warnings; typecheck/build/Vitest/audit passed | GitHub dependency and GUI jobs on this PR |
| K — Documentation/planning truth | Locally verified | `fa22da6` | README, INSTALL, RUNNING, surface docs, integration contract, CLAUDE, active/superseded PRDs, this ledger | 65 focused release-contract tests passed; local Markdown links validated in 10 canonical files | Final artifact/signing/CI outcomes must be recorded after completion |

## Packaging evidence

- Managed runtime: Python 3.11.9 embeddable distribution with installed AskRex wheel, pinned runtime dependencies, CPU Whisper/Torch, and bundled FFmpeg.
- Local Voice runtime directory: approximately 880 MB; unpacked app approximately 1.25 GB.
- Final locally validated installer: `gui/dist/AskRex Setup 1.0.0.exe`, 305,471,434 bytes, SHA-256 `AE2750340B38D2C6AA630E02B1D1AE1E6274F64D58B50BC3B9F309E963C13396`.
- Artifact scan found no Flask runtime dependency and no bundled credentials, profiles, memories, logs, or transcripts.
- Clean install, typed IPC, managed bridge startup, deterministic safe chat, read-only memories, reinstall, and uninstall passed with PATH restricted to Windows System32.
- Packaged Voice metadata/imports verified AskRex 1.4.1, CPU Torch 2.12.1, and Whisper 20250625; the build fails if Torch drifts outside that exact supported pin.
- Authenticode status was `NotSigned` for both installer and unpacked executable.

## Final local gate evidence

- CI-equivalent Python gate passed after remediation: 8,122 passed, 84 skipped, 83.05% coverage (75% required).
- The final tool lifecycle preserves one retry for transient read failures, does not retry authentication failures or mutations, and retains the established user-facing timeout/error contract.
- GitHub-hosted checks remain pending; the finding statuses above therefore remain Locally verified.

## Git reconciliation evidence

- Original local `master`: `d1b56ff794f6e2bdddc463414604cb7a0bb78687`.
- Safety branch: `codex/safety-audit-remediation-20260722-d1b56ff`.
- Remote-only commits retained: `abd023f`, `8ece8b1`.
- Five local-only commits were replayed onto `origin/master` as `6e3afd4`, `078fdec`, `c5c5063`, `2e19c00`, and `8db4afc`.
- `git range-diff` showed equivalent patches and no conflict resolution was required.

## Final-gate policy

Do not advance this ledger to CI verified until required GitHub checks pass. Do not describe the artifact as a signed public production release until Authenticode signing is configured and verified. Hardware/service-dependent items remain complete but externally unverified rather than silently passed.
