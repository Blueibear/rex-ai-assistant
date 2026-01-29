"""
GitHub integration service.

Provides GitHub API integration with:
- Repository listing
- Pull request management
- Issue management
- Branch and commit operations
- Policy-gated write operations
- Credential manager integration
"""

import json
import logging
import subprocess
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Optional

import requests

from rex.audit import LogEntry, get_audit_logger
from rex.credentials import get_credential_manager
from rex.policy_engine import get_policy_engine
from rex.contracts.core import ToolCall
from rex.retry import RetryPolicy, retry_call

logger = logging.getLogger(__name__)


@dataclass
class Repository:
    """GitHub repository information."""
    name: str
    full_name: str
    owner: str
    url: str
    description: Optional[str]
    private: bool
    default_branch: str


@dataclass
class PullRequest:
    """GitHub pull request information."""
    number: int
    title: str
    state: str
    url: str
    author: str
    head_branch: str
    base_branch: str
    created_at: str
    updated_at: str


@dataclass
class Issue:
    """GitHub issue information."""
    number: int
    title: str
    state: str
    url: str
    author: str
    created_at: str
    updated_at: str
    labels: list[str]


class GitHubService:
    """
    Service for GitHub API integration.

    Features:
    - List repositories and pull requests
    - Create issues and pull requests
    - Comment on issues/PRs
    - Apply patches to branches
    - Policy checks for write operations
    - Credential manager integration
    """

    def __init__(
        self,
        credential_name: str = "github",
        api_base: str = "https://api.github.com",
        mock_mode: bool = False,
        mock_data_path: Optional[str] = None,
    ):
        """
        Initialize GitHub service.

        Args:
            credential_name: Name of credential in credential manager
            api_base: GitHub API base URL
            mock_mode: Use mock data instead of real API calls
            mock_data_path: Path to mock data JSON file
        """
        self.credential_name = credential_name
        self.api_base = api_base
        self.mock_mode = mock_mode
        self.mock_data_path = mock_data_path or "data/mock_github.json"

        self._credential_manager = get_credential_manager()
        self._audit_logger = get_audit_logger()
        self._policy_engine = get_policy_engine()

        # Load mock data if in mock mode
        self._mock_data = {}
        if self.mock_mode and Path(self.mock_data_path).exists():
            with open(self.mock_data_path, 'r') as f:
                self._mock_data = json.load(f)

    def _get_token(self) -> str:
        """Get GitHub token from credential manager."""
        token = get_credential_manager().get_token(self.credential_name)
        if token is None or not token.strip():
            raise ValueError("GitHub token not found")
        return token

    def _make_request(
        self,
        method: str,
        endpoint: str,
        data: Optional[dict] = None,
        params: Optional[dict] = None,
    ) -> dict[str, Any]:
        """
        Make a GitHub API request.

        Args:
            method: HTTP method (GET, POST, PUT, PATCH, DELETE)
            endpoint: API endpoint (e.g., "/user/repos")
            data: Request body data
            params: Query parameters

        Returns:
            Response JSON data
        """
        if self.mock_mode:
            # Return mock data
            mock_key = f"{method}:{endpoint}"
            if mock_key in self._mock_data:
                return self._mock_data[mock_key]
            return {"message": "Mock data not found"}

        token = self._get_token()
        url = f"{self.api_base}{endpoint}"

        headers = {
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github.v3+json",
        }

        def _do_request() -> dict[str, Any]:
            response = requests.request(
                method=method,
                url=url,
                headers=headers,
                json=data,
                params=params,
                timeout=30,
            )
            response.raise_for_status()

            if response.status_code == 204:  # No content
                return {"status": "success"}

            return response.json()

        retry_policy = RetryPolicy(
            attempts=3,
            initial_backoff_seconds=1.0,
            backoff_multiplier=2.0,
            max_backoff_seconds=8.0,
            retry_exceptions=(requests.exceptions.RequestException,),
        )

        try:
            return retry_call(
                _do_request,
                policy=retry_policy,
                on_retry=lambda attempt, exc, delay: logger.warning(
                    "GitHub request failed (attempt %d/%d). Retrying in %.1fs: %s",
                    attempt,
                    retry_policy.attempts,
                    delay,
                    exc,
                ),
            )
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"GitHub API request failed: {str(e)}") from e

    def list_repos(self, type_filter: str = "all") -> list[Repository]:
        """
        List repositories accessible to the authenticated user.

        Args:
            type_filter: Repository type filter ("all", "owner", "member")

        Returns:
            List of Repository objects
        """
        action_id = str(uuid.uuid4())
        start_time = datetime.now()

        try:
            params = {"type": type_filter, "per_page": 100}
            data = self._make_request("GET", "/user/repos", params=params)

            repos = []
            for item in data:
                repos.append(Repository(
                    name=item["name"],
                    full_name=item["full_name"],
                    owner=item["owner"]["login"],
                    url=item["html_url"],
                    description=item.get("description"),
                    private=item["private"],
                    default_branch=item["default_branch"],
                ))

            duration_ms = int((datetime.now() - start_time).total_seconds() * 1000)

            # Audit log
            self._audit_logger.log(LogEntry(
                action_id=action_id,
                tool="github_list_repos",
                tool_call_args={"type": type_filter},
                policy_decision="allowed",
                tool_result={"count": len(repos)},
                error=None,
                duration_ms=duration_ms,
            ))

            return repos

        except Exception as e:
            duration_ms = int((datetime.now() - start_time).total_seconds() * 1000)
            error_msg = f"Failed to list repositories: {str(e)}"

            self._audit_logger.log(LogEntry(
                action_id=action_id,
                tool="github_list_repos",
                tool_call_args={"type": type_filter},
                policy_decision="allowed",
                tool_result=None,
                error=error_msg,
                duration_ms=duration_ms,
            ))

            raise RuntimeError(error_msg) from e

    def list_prs(
        self,
        repo: str,
        state: str = "open",
    ) -> list[PullRequest]:
        """
        List pull requests for a repository.

        Args:
            repo: Repository in format "owner/repo"
            state: PR state filter ("open", "closed", "all")

        Returns:
            List of PullRequest objects
        """
        action_id = str(uuid.uuid4())
        start_time = datetime.now()

        try:
            endpoint = f"/repos/{repo}/pulls"
            params = {"state": state, "per_page": 100}
            data = self._make_request("GET", endpoint, params=params)

            prs = []
            for item in data:
                prs.append(PullRequest(
                    number=item["number"],
                    title=item["title"],
                    state=item["state"],
                    url=item["html_url"],
                    author=item["user"]["login"],
                    head_branch=item["head"]["ref"],
                    base_branch=item["base"]["ref"],
                    created_at=item["created_at"],
                    updated_at=item["updated_at"],
                ))

            duration_ms = int((datetime.now() - start_time).total_seconds() * 1000)

            # Audit log
            self._audit_logger.log(LogEntry(
                action_id=action_id,
                tool="github_list_prs",
                tool_call_args={"repo": repo, "state": state},
                policy_decision="allowed",
                tool_result={"count": len(prs)},
                error=None,
                duration_ms=duration_ms,
            ))

            return prs

        except Exception as e:
            duration_ms = int((datetime.now() - start_time).total_seconds() * 1000)
            error_msg = f"Failed to list pull requests: {str(e)}"

            self._audit_logger.log(LogEntry(
                action_id=action_id,
                tool="github_list_prs",
                tool_call_args={"repo": repo, "state": state},
                policy_decision="allowed",
                tool_result=None,
                error=error_msg,
                duration_ms=duration_ms,
            ))

            raise RuntimeError(error_msg) from e

    def create_issue(
        self,
        repo: str,
        title: str,
        body: str,
        labels: Optional[list[str]] = None,
    ) -> Issue:
        """
        Create an issue in a repository.

        Args:
            repo: Repository in format "owner/repo"
            title: Issue title
            body: Issue body
            labels: Optional list of label names

        Returns:
            Created Issue object
        """
        # Policy check for write operation
        tool_call = ToolCall(
            tool="github_create_issue",
            args={"repo": repo, "title": title},
            requested_by="user",
            created_at=datetime.now(),
        )
        decision = self._policy_engine.decide(tool_call, metadata={})

        if not decision.allowed:
            raise PermissionError(f"Policy denied issue creation: {decision.reason}")

        action_id = str(uuid.uuid4())
        start_time = datetime.now()

        try:
            endpoint = f"/repos/{repo}/issues"
            data = {
                "title": title,
                "body": body,
            }
            if labels:
                data["labels"] = labels

            response = self._make_request("POST", endpoint, data=data)

            issue = Issue(
                number=response["number"],
                title=response["title"],
                state=response["state"],
                url=response["html_url"],
                author=response["user"]["login"],
                created_at=response["created_at"],
                updated_at=response["updated_at"],
                labels=[label["name"] for label in response.get("labels", [])],
            )

            duration_ms = int((datetime.now() - start_time).total_seconds() * 1000)

            # Audit log
            self._audit_logger.log(LogEntry(
                action_id=action_id,
                tool="github_create_issue",
                tool_call_args={"repo": repo, "title": title},
                policy_decision="allowed",
                tool_result={"issue_number": issue.number, "url": issue.url},
                error=None,
                duration_ms=duration_ms,
            ))

            return issue

        except Exception as e:
            duration_ms = int((datetime.now() - start_time).total_seconds() * 1000)
            error_msg = f"Failed to create issue: {str(e)}"

            self._audit_logger.log(LogEntry(
                action_id=action_id,
                tool="github_create_issue",
                tool_call_args={"repo": repo, "title": title},
                policy_decision="allowed",
                tool_result=None,
                error=error_msg,
                duration_ms=duration_ms,
            ))

            raise RuntimeError(error_msg) from e

    def comment_issue(
        self,
        repo: str,
        issue_number: int,
        comment: str,
    ) -> dict[str, Any]:
        """
        Comment on an issue or pull request.

        Args:
            repo: Repository in format "owner/repo"
            issue_number: Issue or PR number
            comment: Comment text

        Returns:
            Dictionary with comment information
        """
        # Policy check for write operation
        tool_call = ToolCall(
            tool="github_comment_issue",
            args={"repo": repo, "issue_number": issue_number},
            requested_by="user",
            created_at=datetime.now(),
        )
        decision = self._policy_engine.decide(tool_call, metadata={})

        if not decision.allowed:
            raise PermissionError(f"Policy denied commenting: {decision.reason}")

        action_id = str(uuid.uuid4())
        start_time = datetime.now()

        try:
            endpoint = f"/repos/{repo}/issues/{issue_number}/comments"
            data = {"body": comment}

            response = self._make_request("POST", endpoint, data=data)

            result = {
                "id": response["id"],
                "url": response["html_url"],
                "created_at": response["created_at"],
            }

            duration_ms = int((datetime.now() - start_time).total_seconds() * 1000)

            # Audit log
            self._audit_logger.log(LogEntry(
                action_id=action_id,
                tool="github_comment_issue",
                tool_call_args={"repo": repo, "issue_number": issue_number},
                policy_decision="allowed",
                tool_result=result,
                error=None,
                duration_ms=duration_ms,
            ))

            return result

        except Exception as e:
            duration_ms = int((datetime.now() - start_time).total_seconds() * 1000)
            error_msg = f"Failed to comment: {str(e)}"

            self._audit_logger.log(LogEntry(
                action_id=action_id,
                tool="github_comment_issue",
                tool_call_args={"repo": repo, "issue_number": issue_number},
                policy_decision="allowed",
                tool_result=None,
                error=error_msg,
                duration_ms=duration_ms,
            ))

            raise RuntimeError(error_msg) from e

    def create_pr(
        self,
        repo: str,
        head_branch: str,
        base_branch: str,
        title: str,
        body: str,
    ) -> PullRequest:
        """
        Create a pull request.

        Args:
            repo: Repository in format "owner/repo"
            head_branch: Source branch
            base_branch: Target branch
            title: PR title
            body: PR body

        Returns:
            Created PullRequest object
        """
        # Policy check for write operation
        tool_call = ToolCall(
            tool="github_create_pr",
            args={"repo": repo, "title": title},
            requested_by="user",
            created_at=datetime.now(),
        )
        decision = self._policy_engine.decide(tool_call, metadata={})

        if not decision.allowed:
            raise PermissionError(f"Policy denied PR creation: {decision.reason}")

        action_id = str(uuid.uuid4())
        start_time = datetime.now()

        try:
            endpoint = f"/repos/{repo}/pulls"
            data = {
                "title": title,
                "body": body,
                "head": head_branch,
                "base": base_branch,
            }

            response = self._make_request("POST", endpoint, data=data)

            pr = PullRequest(
                number=response["number"],
                title=response["title"],
                state=response["state"],
                url=response["html_url"],
                author=response["user"]["login"],
                head_branch=response["head"]["ref"],
                base_branch=response["base"]["ref"],
                created_at=response["created_at"],
                updated_at=response["updated_at"],
            )

            duration_ms = int((datetime.now() - start_time).total_seconds() * 1000)

            # Audit log
            self._audit_logger.log(LogEntry(
                action_id=action_id,
                tool="github_create_pr",
                tool_call_args={"repo": repo, "title": title},
                policy_decision="allowed",
                tool_result={"pr_number": pr.number, "url": pr.url},
                error=None,
                duration_ms=duration_ms,
            ))

            return pr

        except Exception as e:
            duration_ms = int((datetime.now() - start_time).total_seconds() * 1000)
            error_msg = f"Failed to create PR: {str(e)}"

            self._audit_logger.log(LogEntry(
                action_id=action_id,
                tool="github_create_pr",
                tool_call_args={"repo": repo, "title": title},
                policy_decision="allowed",
                tool_result=None,
                error=error_msg,
                duration_ms=duration_ms,
            ))

            raise RuntimeError(error_msg) from e

    def apply_patch(
        self,
        repo_path: str,
        branch: str,
        patch_file: str,
        commit_message: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Apply a patch file to a local repository.

        Note: This operates on a local git repository, not via the GitHub API.

        Args:
            repo_path: Path to local git repository
            branch: Branch to apply patch to
            patch_file: Path to patch/diff file
            commit_message: Optional commit message

        Returns:
            Dictionary with apply status
        """
        # Policy check for write operation
        tool_call = ToolCall(
            tool="github_apply_patch",
            args={"repo_path": repo_path, "branch": branch},
            requested_by="user",
            created_at=datetime.now(),
        )
        decision = self._policy_engine.decide(tool_call, metadata={})

        if not decision.allowed:
            raise PermissionError(f"Policy denied patch application: {decision.reason}")

        action_id = str(uuid.uuid4())
        start_time = datetime.now()

        try:
            repo_path_obj = Path(repo_path)
            if not repo_path_obj.exists():
                raise FileNotFoundError(f"Repository not found: {repo_path}")

            patch_file_obj = Path(patch_file)
            if not patch_file_obj.exists():
                raise FileNotFoundError(f"Patch file not found: {patch_file}")

            # Checkout branch
            subprocess.run(
                ["git", "checkout", branch],
                cwd=repo_path,
                check=True,
                capture_output=True,
            )

            # Apply patch
            subprocess.run(
                ["git", "apply", str(patch_file_obj.resolve())],
                cwd=repo_path,
                check=True,
                capture_output=True,
            )

            # Commit if message provided
            if commit_message:
                subprocess.run(
                    ["git", "add", "-A"],
                    cwd=repo_path,
                    check=True,
                    capture_output=True,
                )
                subprocess.run(
                    ["git", "commit", "-m", commit_message],
                    cwd=repo_path,
                    check=True,
                    capture_output=True,
                )

            duration_ms = int((datetime.now() - start_time).total_seconds() * 1000)

            result = {
                "status": "success",
                "repo_path": repo_path,
                "branch": branch,
                "committed": commit_message is not None,
            }

            # Audit log
            self._audit_logger.log(LogEntry(
                action_id=action_id,
                tool="github_apply_patch",
                tool_call_args={"repo_path": repo_path, "branch": branch},
                policy_decision="allowed",
                tool_result=result,
                error=None,
                duration_ms=duration_ms,
            ))

            return result

        except subprocess.CalledProcessError as e:
            duration_ms = int((datetime.now() - start_time).total_seconds() * 1000)
            error_msg = f"Patch application failed: {e.stderr.decode()}"

            self._audit_logger.log(LogEntry(
                action_id=action_id,
                tool="github_apply_patch",
                tool_call_args={"repo_path": repo_path, "branch": branch},
                policy_decision="allowed",
                tool_result=None,
                error=error_msg,
                duration_ms=duration_ms,
            ))

            raise RuntimeError(error_msg) from e

        except Exception as e:
            duration_ms = int((datetime.now() - start_time).total_seconds() * 1000)
            error_msg = f"Patch application failed: {str(e)}"

            self._audit_logger.log(LogEntry(
                action_id=action_id,
                tool="github_apply_patch",
                tool_call_args={"repo_path": repo_path, "branch": branch},
                policy_decision="allowed",
                tool_result=None,
                error=error_msg,
                duration_ms=duration_ms,
            ))

            raise RuntimeError(error_msg) from e


# Singleton instance
_github_service: Optional[GitHubService] = None


def get_github_service() -> GitHubService:
    """Get or create the global GitHub service."""
    global _github_service
    if _github_service is None:
        _github_service = GitHubService()
    return _github_service


def set_github_service(service: GitHubService) -> None:
    """Set the global GitHub service."""
    global _github_service
    _github_service = service


def reset_github_service() -> None:
    """Reset the global GitHub service (for testing)."""
    global _github_service
    _github_service = None
