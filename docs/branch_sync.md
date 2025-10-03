# Branch Synchronization Helper

The `scripts/sync_branches.sh` utility automates the workflow for keeping every branch aligned with the `master` branch that the remote exposes.

## What the script does

1. Fetches the latest history (with prune) from the selected remote.
2. Iterates through every remote branch (creating local tracking branches when needed).
3. Resets each local branch to match its remote counterpart.
4. Merges the remote `master` branch into every other branch.
5. Pushes all synchronized branches back to the remote.

## Usage

```bash
./scripts/sync_branches.sh [remote-name] [master-branch]
```

- `remote-name` defaults to `origin`.
- `master-branch` defaults to `master`.

> **Note:** The script requires the remote to be configured locally. In this environment no remotes are defined, so running the script will exit with an error until a remote such as `origin` is added.
