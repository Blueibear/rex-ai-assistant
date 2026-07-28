# Fable Shipping Progress

- **Baseline master SHA:** `8dccbe3f8bc03e71fa957d60b19f79fb0d812013` (merge of PR #331)
- **Working branch:** `fable/shipping-readiness`
- **Release target:** Windows-first packaged Electron release (unsigned installer currently); next version candidate is `1.5.0` via open release-please PR #322
- **Current status:** Initializer complete. Repo is clean, branch is at master tip, CI is green on the baseline SHA. No product code has been touched by this pass.

## Completed milestones

- [x] Initializer: repo/PR/issue/CI reconciliation, `docs/release/FABLE-SHIPPING-CONTEXT.md` written and QC'd (2026-07-28).

## Next executable milestone

Per `FABLE-SHIPPING-CONTEXT.md` Section 8, item 1: confirm PR #326 disposition
with the repo owner (its content already exists on master under different
commit SHAs — see Section 6 of the context doc), then close it. Follow with
items 2–5 (GUI lint wired into CI, Black CI coverage for `scripts/`, stale
Torch comment fix, mobile API added to surface/integration-status docs).

## Unresolved blockers (non-user)

- PRD-production-readiness.md P0 security backlog (US-025–US-034) needs
  current-code verification before any new implementation — some mechanics
  may already exist (see context doc Section 3/7).
- Dependabot PR #327 (torch 2.12.0→2.13.0) must be evaluated against the
  exact-pin requirement for the packaged Voice runtime before merging.
- Release-please PR #322 (1.5.0) timing decision pending.

## User-only blockers

See `FABLE-SHIPPING-CONTEXT.md` Section 10 — physical Windows 11 hardware
matrix (#299), wake-word hardware recordings (#304), physical iPhone/LAN
mobile testing (#323), live Home Assistant device control, Authenticode
signing certificate, external provider credentials.

## PR / CI status snapshot (2026-07-28)

- Master tip CI: all release-gate checks green.
- Open PRs: #330, #328, #327 (dependabot); #326 (draft, superseded — see
  context doc); #322 (release-please 1.5.0); #288, #233 (dependabot).
- Open issues tracked: #299, #302, #304, #323, #253 — none closed by this
  pass; all still require the work described in the context doc.
