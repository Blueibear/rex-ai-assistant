# AskRex Assistant - Development Workflow

## Overview
Rex uses a two-branch workflow to separate validated changes from ongoing development work.

---

## Branch Structure

### master (protected)
- Represents the primary branch for validated changes.
- Changes merged here should have passed the repository's current validation checks.
- No direct commits or force pushes are allowed.
- All changes must be merged through a **pull request** from `dev`.

### dev (active development)
- Used for all current work: feature additions, experiments, and fixes.
- Commits can be made freely.
- Periodically merged into `master` once validation is complete.

---

## Standard Workflow

1. **Start work**
   ```bash
   git checkout dev
   ```

2. **Make and test changes locally**
   ```bash
   git add .
   git commit -m "Describe your update here"
   git push
   ```

3. **Merge to master**
   - Go to the GitHub repo page.
   - Click **Compare & pull request** next to the dev branch.
   - Review, confirm, and merge.

4. **Sync local copies**
   ```bash
   git checkout main
   git pull origin main
   git checkout dev
   git merge main
   ```

---

## Future Branch Types (optional expansion)
- `feature/*` — for individual new features.
- `hotfix/*` — for urgent fixes applied to `main`.
- `release/*` — for preparing structured version releases.

---

## Tagging Releases
To mark milestones:
```bash
git tag -a v2.0 -m "Advanced local version synced to GitHub"
git push origin v2.0
```

---

## Notes
- All commits should include clear, descriptive messages.
- Avoid committing large binaries or logs; use `.gitignore` properly.
- Keep documentation (like this one) updated as new processes are added.
