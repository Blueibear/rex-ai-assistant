# ci-fixes.patch archive note

Commit-style summary:
`ci: harden diff-aware CI checks, add commitlint workflow, and refresh OpenClaw/wakeword/UI test coverage`

Assessment on 2026-04-03:
- The root-level `ci-fixes.patch` artifact no longer applies cleanly with `git apply --check`.
- The reverse check also fails, so the patch is not a clean "already applied" candidate either.
- The current tree already contains the obvious CI workflow outcomes from the patch, including the dedicated `.github/workflows/commitlint.yml` workflow and deleted-file filtering in `.github/workflows/ci.yml`.
- The remaining hunks target files that have substantially diverged since the patch was created, making the patch a stale audit artifact rather than a safe patch to apply.

Resolution:
- Archive the original patch under `docs/archive/housekeeping/`.
- Remove all root-level `*.patch` artifacts.
- Keep `*.patch` in `.gitignore` to prevent future accidental root commits.
