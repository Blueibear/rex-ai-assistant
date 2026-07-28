# Fable Shipping Progress

- **Baseline master SHA:** `8dccbe3f8bc03e71fa957d60b19f79fb0d812013` (merge of PR #331)
- **Working branch:** `fable/shipping-readiness`
- **Release scope:** Reliable private Windows 11 release (packaged Electron, unsigned private beta until Authenticode is provisioned). Mobile is next priority, beta-classified pending physical iPhone/LAN validation. OpenClaw optional/experimental, disabled by default.
- **Current status:** P0 backlog US-023–US-036 verified/closed; secret-gated Authenticode signing support implemented. Awaiting PR #332 CI (incl. Windows artifact run) before merge.

## Completed milestones

- [x] Initializer: repo/PR/issue/CI reconciliation, `docs/release/FABLE-SHIPPING-CONTEXT.md` written and QC'd. Commit `bf0df76` (2026-07-28).
- [x] PR #326 closed as superseded: independently re-verified that, restricted to the files the PR changed, the only delta vs master was a one-line docstring in `rex/mobile_api/capabilities.py`; its payload landed on master as `6e3afd4`/`c5c5063`/`8db4afc` during the PR #331 reconciliation (audit ledger, "Git reconciliation evidence"). Closed with evidence comment (2026-07-28).
- [x] GUI ESLint wired into CI as a job (`gui-lint`) and Black CI target extended to `scripts/`. Local evidence: `npm run lint` exit 0 (0 errors, 3 pre-existing warnings); `black --check scripts/` exit 0 (31 files unchanged). Commit `36f4cb2`.
- [x] CPU torch stack made resolvable: `requirements-cpu.txt` was ResolutionImpossible (torch 2.12.0 pinned with torch-2.6-era torchvision 0.21.0/torchaudio 2.6.0). Repinned to the packaged-Voice-aligned triple torch 2.12.1 / torchvision 0.27.1 / torchaudio 2.11.0 and widened the stale `ml`/`full` extras caps in `pyproject.toml` to match the existing `torch<2.13.0` ceiling. Evidence: `pip install --dry-run -r requirements-cpu.txt` resolves cleanly incl. TTS 0.22.0. Resolution-verified only; runtime torch 2.12.1 evidence remains the packaged Voice CI. Commit `f23b825`.
- [x] Mobile API gateway added to `SURFACE-CLASSIFICATION.md` (developer-only, counts 50→51) and `INTEGRATIONS_STATUS.md` (beta backend, credential-gated, truthful-capability contract). Commit `09d0a14`.
- [x] Focused validation for the above: `pytest tests/test_us057_ci_pipeline.py tests/test_us141_readme_install.py tests/test_windows_artifact_workflow.py tests/test_us093_dependency_scan.py` — 45 passed. ci.yml YAML-parses with 13 jobs.
- [x] P0 CI/security stories US-023–US-036 verified against current code and closed with evidence (commits `6019dde`–`d4dfd80`): Security Audit Gate CI job (US-034, local exit 0); `scripts/check_no_generated_artifacts.py` + allowlisted `rex/ui/dist/index.html` + 5 tests (US-035); GUI Vitest tree-clean step (US-036); GUI settings secret redaction `gui/src/main/settingsRedaction.ts` wired into `writeGuiSettings` + Vitest coverage (US-027, 16 GUI tests pass); planted-secret detect-secrets fixture test + PR-template checklist line (US-028, 21 tests pass); docs for tool risk gates / HA endpoint auth / Twilio fail-closed (US-024/025/026 — mechanics pre-existed under `ToolRisk`, `rex/ha/mutation_service.py`, `ImportError`/`SMSSendError`; reconciled, not reimplemented); 8 stale open audit-inventory rows resolved with commit evidence (US-029).
- [x] Secret-gated Authenticode signing support (commit `0881b4c`): workflow exports `CSC_LINK` only when `WINDOWS_CSC_LINK` secret exists; `Get-AuthenticodeSignature` verification fails closed when signing configured but artifact not Valid, truthfully reports NotSigned otherwise; RFC 3161 timestamping in `gui/package.json`; docs in `docs/distribution.md`; 8 workflow contract tests pass. No certificate purchased or claimed.

## Current milestone

Land PR #332: await required CI checks (incl. the Windows Electron artifact run triggered by gui/workflow changes), fix anything CI surfaces, merge when green. Fixture values are scanner-clean at every commit: the planted-secret value is derived at runtime and the GUI fixture uses inert placeholders (history rewritten pre-merge so no secret-looking string ever entered the branch).

## Remaining release blockers (software)

- PR #332 merge (awaiting CI).
- Remaining PRD P0 stories beyond US-036 (e.g. US-037 skip budget, Phases 13–17 backlog) need the same verify-then-implement pass; many boxes are stale relative to shipped code.
- `requirements-gpu-cu124.txt` / `requirements-gpu.txt` carry the same broken triple (torch 2.12.0 + cu124 0.21.0/2.6.0 companions) — optional external GPU profile, not release-blocking; fix needs cu124-index resolution verification.
- Issue #253 (transformers 5.x): NOT obsolete — `<5.0` pin and CI suppressions still active (`ci.yml:386-387` pre-change numbering), expiry 2026-08-29. Blocked on validating transformers 5.x against Coqui TTS 0.22.0; deliberate dependency-validation task, not done in passing.
- Authenticode signing support (activate-only-when-secret-present) not yet implemented in release automation.
- Release-please PR #322 (1.5.0) timing decision.
- Dependabot PR #327 (torch 2.13.0) must be evaluated against the exact Voice-pin gate before merge.

## User-only blockers

Physical Windows 11 acceptance matrix (#299); wake-word hardware recordings (#304); physical iPhone/LAN mobile validation (#323); live Home Assistant device control; Authenticode certificate purchase/secret provisioning; external provider credentials.

## PR / CI status

- PR for this branch: opened after commit `09d0a14` (see GitHub); required checks pending at time of writing.
- Master tip CI: green on all release-gate checks at baseline.
- Open PRs: #330/#328/#327/#288/#233 (dependabot), #322 (release-please). #326 closed as superseded.

## Decisions and lessons (not recorded elsewhere)

- `.claude/ralph-loop.local.md`: stale ralph-loop cancelled (user-approved); the file deletion is user-owned local state, kept uncommitted per the PR #331 precedent.
- CI never installs the `ml`/`full` extras, so extras-resolvability breakage is invisible to CI — verify with `pip install --dry-run` when touching torch-family pins.
