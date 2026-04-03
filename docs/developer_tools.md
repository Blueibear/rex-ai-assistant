# Developer Tools Integration

AskRex Assistant provides integrated developer tools for working with code, GitHub, and development workflows.

## GitHub Integration

### Overview

The GitHub integration provides programmatic access to GitHub's API for:

- Repository management
- Pull request operations
- Issue tracking
- Code review automation
- Branch and commit operations

### Setup

#### Authentication

Set your GitHub personal access token:

```bash
# Via environment variable
export GITHUB_TOKEN="ghp_your_token_here"

# Or add to config/credentials.json
{
  "github": "ghp_your_token_here"
}
```

Create a token at: https://github.com/settings/tokens

Required scopes:
- `repo` - Full control of private repositories
- `read:org` - Read organization data (if needed)

### CLI Commands

#### List Repositories

```bash
# List all accessible repositories
rex gh repos
```

#### List Pull Requests

```bash
# List open PRs for a repository
rex gh prs owner/repo

# List all PRs (open, closed, merged)
rex gh prs owner/repo --state all

# List closed PRs
rex gh prs owner/repo --state closed
```

#### Create an Issue

```bash
rex gh issue-create owner/repo \
  --title "Bug: Feature X not working" \
  --body "Detailed description of the issue" \
  --labels "bug,priority"
```

#### Create a Pull Request

```bash
rex gh pr-create owner/repo \
  --head feature-branch \
  --base main \
  --title "Add new feature" \
  --body "Description of changes"
```

### Python API

#### List Repositories

```python
from rex.github_service import get_github_service

service = get_github_service()
repos = service.list_repos()

for repo in repos:
    print(f"{repo.full_name}: {repo.description}")
    print(f"  URL: {repo.url}")
    print(f"  Private: {repo.private}")
```

#### List Pull Requests

```python
service = get_github_service()
prs = service.list_prs("owner/repo", state="open")

for pr in prs:
    print(f"#{pr.number}: {pr.title}")
    print(f"  Author: {pr.author}")
    print(f"  Branch: {pr.head_branch} → {pr.base_branch}")
    print(f"  URL: {pr.url}")
```

#### Create an Issue

```python
service = get_github_service()

issue = service.create_issue(
    repo="owner/repo",
    title="Bug Report",
    body="Description of the bug",
    labels=["bug", "needs-triage"]
)

print(f"Created issue #{issue.number}")
print(f"URL: {issue.url}")
```

#### Comment on Issue or PR

```python
service = get_github_service()

service.comment_issue(
    repo="owner/repo",
    issue_number=42,
    comment="This looks good! Approved."
)
```

#### Create a Pull Request

```python
service = get_github_service()

pr = service.create_pr(
    repo="owner/repo",
    head_branch="feature-branch",
    base_branch="main",
    title="Add new feature",
    body="## Summary\n\n- Added feature X\n- Fixed bug Y"
)

print(f"Created PR #{pr.number}")
print(f"URL: {pr.url}")
```

#### Apply a Patch

```python
service = get_github_service()

result = service.apply_patch(
    repo_path="/path/to/local/repo",
    branch="feature-branch",
    patch_file="changes.patch",
    commit_message="Apply patch"
)

print(f"Patch applied: {result['status']}")
```

### Mock Mode for Testing

Use mock mode to test without real API calls:

```python
service = GitHubService(mock_mode=True, mock_data_path="data/mock_github.json")

# All operations use mock data
repos = service.list_repos()
```

### Security Considerations

1. **Token Security**: Never commit tokens to version control
2. **Policy Checks**: Write operations require approval
3. **Audit Logging**: All API calls are logged
4. **Rate Limiting**: Be mindful of GitHub's rate limits

## VS Code Integration

### Overview

The VS Code integration provides file operations, patch application, and test execution:

- File reading and display
- Unified diff patch application
- Test execution with pytest
- Diff generation

### CLI Commands

#### Open and Display a File

```bash
# Display file contents
rex code open path/to/file.py
```

#### Apply a Patch

```bash
# Apply a unified diff patch
rex code patch path/to/file.py --patch-file changes.patch
```

#### Run Tests

```bash
# Run all tests
rex code test

# Run specific test file
rex code test --path tests/test_browser.py

# Run tests matching a pattern
rex code test --pattern test_authentication

# Verbose output
rex code test -v
```

### Python API

#### Open a File

```python
from rex.vscode_service import get_vscode_service

service = get_vscode_service()
result = service.open_file("path/to/file.py")

print(f"File: {result['path']}")
print(f"Lines: {result['lines']}")
print(result['content'])
```

#### Apply a Patch

```python
service = get_vscode_service()

with open("changes.patch", 'r') as f:
    patch_content = f.read()

result = service.apply_patch("path/to/file.py", patch_content)

if result.success:
    print(f"Patch applied successfully!")
    print(f"Hunks applied: {result.hunks_applied}")
else:
    print(f"Patch failed!")
    print(f"Errors: {result.errors}")
```

#### Generate a Diff

```python
service = get_vscode_service()

new_content = """
def hello():
    print("Hello, World!")
"""

diff = service.generate_diff("file.py", new_content)
print(diff)
```

#### Run Tests

```python
service = get_vscode_service()

result = service.run_tests(
    test_path="tests/",
    pattern="test_api",
    verbose=True
)

print(f"Tests: {result.passed}/{result.total} passed")
print(f"Duration: {result.duration_seconds}s")

if not result.success:
    print("Failed tests:")
    print(result.output)
```

#### List Files

```python
service = get_vscode_service()

# List all Python files
files = service.list_files(".", pattern="*.py")

for f in files:
    print(f"{f['name']}: {f['size']} bytes")
```

### Patch Format

Patches should be in unified diff format:

```diff
--- a/file.py
+++ b/file.py
@@ -1,3 +1,3 @@
 def hello():
-    print("Hello")
+    print("Hello, World!")
     return True
```

Generate patches using:

```bash
# Git diff
git diff > changes.patch

# Diff command
diff -u old_file.py new_file.py > changes.patch
```

### Test Execution

Tests are run using pytest. Requirements:

- Tests in `tests/` directory
- Test files match `test_*.py` pattern
- Test functions match `test_*` pattern

Output includes:

- Total tests run
- Passed/failed/errors/skipped counts
- Execution duration
- Full test output (in verbose mode)

### Security Considerations

1. **Path Restrictions**: File operations respect workspace boundaries
2. **Policy Checks**: File modifications require approval
3. **Audit Logging**: All operations are logged
4. **Test Isolation**: Tests run in subprocess with timeout

## Workflow Integration

### Automated Code Review

```python
from rex.github_service import get_github_service
from rex.vscode_service import get_vscode_service

gh = get_github_service()
vs = get_vscode_service()

# Get open PRs
prs = gh.list_prs("owner/repo", state="open")

for pr in prs:
    print(f"Reviewing PR #{pr.number}: {pr.title}")

    # Run tests
    test_result = vs.run_tests()

    # Comment on PR
    if test_result.success:
        gh.comment_issue(
            "owner/repo",
            pr.number,
            f"✅ All tests passed ({test_result.passed}/{test_result.total})"
        )
    else:
        gh.comment_issue(
            "owner/repo",
            pr.number,
            f"❌ Tests failed ({test_result.failed} failures)"
        )
```

### Automated Issue Triage

```python
gh = get_github_service()

# Create issue from error logs
gh.create_issue(
    repo="owner/repo",
    title="Runtime error in module X",
    body=f"Error details:\n\n```\n{error_log}\n```",
    labels=["bug", "automated"]
)
```

### Patch-Based Development

```python
vs = get_vscode_service()
gh = get_github_service()

# Generate patch from changes
diff = vs.generate_diff("file.py", new_content)

# Apply to local repo
gh.apply_patch(
    repo_path="/path/to/repo",
    branch="fix-branch",
    patch_file="fix.patch",
    commit_message="Apply automated fix"
)
```

## Best Practices

### GitHub Integration

1. **Use meaningful commit messages** with session references
2. **Review changes** before creating PRs
3. **Add labels** to issues for organization
4. **Use mock mode** for testing
5. **Monitor rate limits** to avoid throttling

### VS Code Integration

1. **Test patches** on backup files first
2. **Run tests** before and after changes
3. **Use relative paths** within workspace
4. **Review diffs** before applying
5. **Handle test failures** gracefully

### Security

1. **Store credentials** in credential manager, not code
2. **Review policies** before write operations
3. **Check audit logs** regularly
4. **Use mock data** for development
5. **Validate input** before API calls

## Troubleshooting

### GitHub Integration

**Authentication failed:**
- Verify token is set: `echo $GITHUB_TOKEN`
- Check token scopes at GitHub settings
- Regenerate token if expired

**API rate limit exceeded:**
- Check remaining calls: `gh api rate_limit`
- Wait for rate limit reset
- Use conditional requests when possible

**Repository not found:**
- Verify repository name format: `owner/repo`
- Check repository access permissions
- Ensure token has required scopes

### VS Code Integration

**File not found:**
- Use relative paths from workspace root
- Check current working directory
- Verify file exists: `rex code open <path>`

**Patch application failed:**
- Ensure patch format is correct (unified diff)
- Check line numbers match current file
- Verify no conflicting changes

**Tests not found:**
- Check test directory exists: `tests/`
- Verify test file naming: `test_*.py`
- Install pytest: `pip install pytest`

**Tests timeout:**
- Increase timeout in service configuration
- Optimize slow tests
- Run tests in smaller batches

## Examples

### Complete PR Workflow

```python
from rex.github_service import get_github_service
from rex.vscode_service import get_vscode_service

gh = get_github_service()
vs = get_vscode_service()

# 1. Make code changes
new_code = """
def improved_function():
    # Better implementation
    return result
"""

# 2. Generate patch
diff = vs.generate_diff("module.py", new_code)

# 3. Apply patch locally
gh.apply_patch(
    repo_path="/path/to/repo",
    branch="improvement",
    patch_file="improvement.patch",
    commit_message="Improve function implementation"
)

# 4. Run tests
test_result = vs.run_tests()

if test_result.success:
    # 5. Create PR
    pr = gh.create_pr(
        repo="owner/repo",
        head_branch="improvement",
        base_branch="main",
        title="Improve function implementation",
        body=f"## Changes\n\n{diff}\n\n## Tests\n\n✅ {test_result.passed}/{test_result.total} passed"
    )

    print(f"PR created: {pr.url}")
else:
    print("Tests failed, not creating PR")
```

### Automated Code Quality Check

```python
vs = get_vscode_service()

# Run tests
test_result = vs.run_tests(verbose=True)

# Report
print(f"Code Quality Report")
print(f"==================")
print(f"Tests: {test_result.passed}/{test_result.total} passed")
print(f"Duration: {test_result.duration_seconds:.2f}s")

if test_result.failed > 0:
    print(f"\nFailed: {test_result.failed}")
if test_result.errors > 0:
    print(f"Errors: {test_result.errors}")
if test_result.skipped > 0:
    print(f"Skipped: {test_result.skipped}")
```
