#!/bin/bash
set -euo pipefail

usage() {
  cat <<'HELP'
Usage: scripts/sync_branches.sh [--force|--dry-run] [remote] [branch]

Synchronises all local branches with a remote, rebasing them on top of the
specified master branch. By default the script refuses to perform destructive
operations.

  --dry-run   Show the operations that would be performed without mutating anything.
  --force     Proceed with the hard resets and pushes. Required for actual syncs.
  remote      Remote name to sync with (default: origin)
  branch      Name of the primary branch to merge (default: master)
HELP
}

FORCE=0
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --force)
      FORCE=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    -*)
      echo "Unknown option: $1" >&2
      usage
      exit 64
      ;;
    *)
      break
      ;;
  esac
done

REMOTE_NAME=${1:-origin}
MASTER_BRANCH=${2:-master}

if [[ $FORCE -eq 0 && $DRY_RUN -eq 0 ]]; then
  cat <<'MSG' >&2
Refusing to perform hard resets without explicit confirmation.
Re-run with --dry-run to preview or --force to execute the sync.
MSG
  exit 65
fi

if ! git rev-parse --git-dir >/dev/null 2>&1; then
  echo "Error: scripts/sync_branches.sh must be run inside a git repository." >&2
  exit 1
fi

if ! git remote | grep -qx "${REMOTE_NAME}"; then
  echo "Error: remote '${REMOTE_NAME}' does not exist. Configure it before running this script." >&2
  exit 2
fi

CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)

echo "Fetching latest updates from ${REMOTE_NAME}..."
git fetch --prune "${REMOTE_NAME}"

if ! git show-ref --verify --quiet "refs/remotes/${REMOTE_NAME}/${MASTER_BRANCH}"; then
  echo "Error: remote branch '${REMOTE_NAME}/${MASTER_BRANCH}' not found." >&2
  exit 3
fi

mapfile -t REMOTE_BRANCHES < <(git for-each-ref --format='%(refname:strip=2)' "refs/remotes/${REMOTE_NAME}" | sort)

for REMOTE_BRANCH in "${REMOTE_BRANCHES[@]}"; do
  BRANCH_NAME=${REMOTE_BRANCH#${REMOTE_NAME}/}

  if [ "${BRANCH_NAME}" = "HEAD" ]; then
    continue
  fi

  printf '\n=== Processing branch %s ===\n' "${BRANCH_NAME}"

  if [[ $DRY_RUN -eq 1 ]]; then
    if ! git show-ref --verify --quiet "refs/heads/${BRANCH_NAME}"; then
      echo "[dry-run] Would create local branch ${BRANCH_NAME} tracking ${REMOTE_NAME}/${BRANCH_NAME}"
    else
      echo "[dry-run] Would reset ${BRANCH_NAME} to ${REMOTE_NAME}/${BRANCH_NAME}"
    fi
  else
    if ! git show-ref --verify --quiet "refs/heads/${BRANCH_NAME}"; then
      echo "Creating local branch ${BRANCH_NAME} tracking ${REMOTE_NAME}/${BRANCH_NAME}..."
      git checkout -B "${BRANCH_NAME}" "${REMOTE_NAME}/${BRANCH_NAME}"
    else
      git checkout "${BRANCH_NAME}"
      echo "Resetting ${BRANCH_NAME} to ${REMOTE_NAME}/${BRANCH_NAME}..."
      git reset --hard "${REMOTE_NAME}/${BRANCH_NAME}"
    fi
  fi

  if [ "${BRANCH_NAME}" = "${MASTER_BRANCH}" ]; then
    echo "Branch '${BRANCH_NAME}' is the master branch; skipping merge."
    continue
  fi

  if [[ $DRY_RUN -eq 1 ]]; then
    echo "[dry-run] Would merge ${REMOTE_NAME}/${MASTER_BRANCH} into ${BRANCH_NAME}"
  else
    echo "Merging ${REMOTE_NAME}/${MASTER_BRANCH} into ${BRANCH_NAME}..."
    if ! git merge --ff-only "${REMOTE_NAME}/${MASTER_BRANCH}"; then
      echo "Fast-forward merge failed; attempting regular merge." >&2
      if ! git merge "${REMOTE_NAME}/${MASTER_BRANCH}"; then
        echo "Error: merge into '${BRANCH_NAME}' failed. Resolve conflicts manually before re-running." >&2
        exit 4
      fi
    fi
  fi
done

if [[ $DRY_RUN -eq 0 ]]; then
  if [ "${CURRENT_BRANCH}" != "$(git rev-parse --abbrev-ref HEAD)" ]; then
    git checkout "${CURRENT_BRANCH}"
  fi

  printf '\nPushing synchronised branches to %s...\n' "${REMOTE_NAME}"
  git push --all "${REMOTE_NAME}"

  echo "Branch synchronization complete."
else
  echo "[dry-run] Skipping checkout of original branch and remote push."
fi
