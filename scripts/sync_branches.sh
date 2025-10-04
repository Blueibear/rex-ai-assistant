#!/bin/bash
set -euo pipefail

REMOTE_NAME=${1:-origin}
MASTER_BRANCH=${2:-master}

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

  if ! git show-ref --verify --quiet "refs/heads/${BRANCH_NAME}"; then
    echo "Creating local branch ${BRANCH_NAME} tracking ${REMOTE_NAME}/${BRANCH_NAME}..."
    git checkout -B "${BRANCH_NAME}" "${REMOTE_NAME}/${BRANCH_NAME}"
  else
    git checkout "${BRANCH_NAME}"
    echo "Resetting ${BRANCH_NAME} to ${REMOTE_NAME}/${BRANCH_NAME}..."
    git reset --hard "${REMOTE_NAME}/${BRANCH_NAME}"
  fi

  if [ "${BRANCH_NAME}" = "${MASTER_BRANCH}" ]; then
    echo "Branch '${BRANCH_NAME}' is the master branch; skipping merge."
    continue
  fi

  echo "Merging ${REMOTE_NAME}/${MASTER_BRANCH} into ${BRANCH_NAME}..."
  if ! git merge --ff-only "${REMOTE_NAME}/${MASTER_BRANCH}"; then
    echo "Fast-forward merge failed; attempting regular merge." >&2
    if ! git merge "${REMOTE_NAME}/${MASTER_BRANCH}"; then
      echo "Error: merge into '${BRANCH_NAME}' failed. Resolve conflicts manually before re-running." >&2
      exit 4
    fi
  fi

done

if [ "${CURRENT_BRANCH}" != "$(git rev-parse --abbrev-ref HEAD)" ]; then
  git checkout "${CURRENT_BRANCH}"
fi

printf '\nPushing synchronized branches to %s...\n' "${REMOTE_NAME}"
git push --all "${REMOTE_NAME}"

echo "Branch synchronization complete."
