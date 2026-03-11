# GitHub Integration

Rex can connect to the GitHub API to list repositories, manage issues and pull requests, and create commits.

## Authentication

Rex reads the GitHub personal access token from the credential manager under the key `github`.

Set the token in your `.env` file:

```env
GITHUB_TOKEN=ghp_your_token_here
```

The credential manager picks up `GITHUB_TOKEN` automatically on startup.

**Required token scopes:**

| Operation | Required scope |
|-----------|---------------|
| List repos / PRs / issues | `repo` (read) |
| Create issues / PRs / comments | `repo` (write) |
| Create commits via Contents API | `repo` (write) |

## CLI Commands

### `rex gh repos`

List repositories accessible to the authenticated user.

```bash
rex gh repos                # all repositories
rex gh repos --type owner   # only repos you own
rex gh repos --type member  # only repos you are a member of
```

### `rex gh prs`

List pull requests for a repository.

```bash
rex gh prs owner/repo                 # open PRs (default)
rex gh prs owner/repo --state closed  # closed PRs
rex gh prs owner/repo --state all     # all PRs
```

### `rex gh issue-create`

Create an issue in a repository.

```bash
rex gh issue-create owner/repo \
  --title "Bug: something broke" \
  --body "Steps to reproduce..."

rex gh issue-create owner/repo \
  --title "Feature request" \
  --body "Detailed description" \
  --labels "bug,enhancement"
```

Write operations are subject to the policy engine. The default policy requires
`SUGGEST` mode or higher for issue creation.

### `rex gh pr-create`

Create a pull request.

```bash
rex gh pr-create owner/repo \
  --head feature-branch \
  --base main \
  --title "Add new feature" \
  --body "What this PR does..."
```

## Python API

```python
from rex.github_service import get_github_service

svc = get_github_service()

# List repos
repos = svc.list_repos(type_filter="owner")

# List issues
issues = svc.list_issues("owner/repo", state="open")

# Create an issue
issue = svc.create_issue(
    repo="owner/repo",
    title="Bug report",
    body="Description",
    labels=["bug"],
)

# Create a commit via the Contents API
result = svc.create_commit(
    repo="owner/repo",
    path="path/to/file.txt",
    message="chore: update file",
    content="file content here",
    branch="main",
)
```

## Mock Mode

For development and testing without a real GitHub token:

```python
from rex.github_service import GitHubService

svc = GitHubService(mock_mode=True, mock_data_path="data/mock_github.json")
repos = svc.list_repos()
```

Mock data is loaded from the JSON file at `mock_data_path`. Keys follow the
format `"METHOD:/endpoint"`.

## Policy Controls

Write operations (create issue, create PR, create commit, apply patch) pass
through the policy engine before execution. Configure autonomy levels in
`config/autonomy.json`:

- **OFF**: Write operations are blocked
- **SUGGEST**: Write operations require confirmation
- **AUTO**: Write operations execute automatically (not recommended for production)

All operations are logged to the audit log regardless of policy outcome.

## Error Handling

All methods raise `RuntimeError` on API failure. The error message includes
the underlying HTTP error or subprocess output. Transient network errors are
retried up to 3 times with exponential backoff.

```python
try:
    repos = svc.list_repos()
except RuntimeError as e:
    print(f"GitHub API error: {e}")
```
