"""Tests for GitHub service module."""

from unittest.mock import Mock, patch

import pytest

from rex.github_service import (
    GitHubService,
    get_github_service,
    reset_github_service,
)


class TestGitHubService:
    """Tests for GitHubService class."""

    def test_service_init(self):
        """Test service initialization."""
        service = GitHubService(mock_mode=True)
        assert service.credential_name == "github"
        assert service.mock_mode is True

    def test_list_repos_mock_mode(self):
        """Test listing repos in mock mode."""
        mock_data = {
            "GET:/user/repos": [
                {
                    "name": "test-repo",
                    "full_name": "user/test-repo",
                    "owner": {"login": "user"},
                    "html_url": "https://github.com/user/test-repo",
                    "description": "Test repository",
                    "private": False,
                    "default_branch": "main",
                }
            ]
        }

        service = GitHubService(mock_mode=True)
        service._mock_data = mock_data

        repos = service.list_repos()

        assert len(repos) == 1
        assert repos[0].name == "test-repo"
        assert repos[0].full_name == "user/test-repo"
        assert repos[0].private is False

    def test_list_prs_mock_mode(self):
        """Test listing PRs in mock mode."""
        mock_data = {
            "GET:/repos/user/repo/pulls": [
                {
                    "number": 1,
                    "title": "Test PR",
                    "state": "open",
                    "html_url": "https://github.com/user/repo/pull/1",
                    "user": {"login": "author"},
                    "head": {"ref": "feature"},
                    "base": {"ref": "main"},
                    "created_at": "2024-01-01T00:00:00Z",
                    "updated_at": "2024-01-02T00:00:00Z",
                }
            ]
        }

        service = GitHubService(mock_mode=True)
        service._mock_data = mock_data

        prs = service.list_prs("user/repo")

        assert len(prs) == 1
        assert prs[0].number == 1
        assert prs[0].title == "Test PR"
        assert prs[0].state == "open"

    def test_create_issue_mock_mode(self):
        """Test creating an issue in mock mode."""
        mock_data = {
            "POST:/repos/user/repo/issues": {
                "number": 1,
                "title": "Test Issue",
                "state": "open",
                "html_url": "https://github.com/user/repo/issues/1",
                "user": {"login": "user"},
                "created_at": "2024-01-01T00:00:00Z",
                "updated_at": "2024-01-01T00:00:00Z",
                "labels": [],
            }
        }

        service = GitHubService(mock_mode=True)
        service._mock_data = mock_data

        issue = service.create_issue("user/repo", "Test Issue", "Test body")

        assert issue.number == 1
        assert issue.title == "Test Issue"
        assert issue.state == "open"

    def test_create_pr_mock_mode(self):
        """Test creating a PR in mock mode."""
        mock_data = {
            "POST:/repos/user/repo/pulls": {
                "number": 1,
                "title": "Test PR",
                "state": "open",
                "html_url": "https://github.com/user/repo/pull/1",
                "user": {"login": "user"},
                "head": {"ref": "feature"},
                "base": {"ref": "main"},
                "created_at": "2024-01-01T00:00:00Z",
                "updated_at": "2024-01-01T00:00:00Z",
            }
        }

        service = GitHubService(mock_mode=True)
        service._mock_data = mock_data

        pr = service.create_pr("user/repo", "feature", "main", "Test PR", "Test body")

        assert pr.number == 1
        assert pr.title == "Test PR"
        assert pr.head_branch == "feature"
        assert pr.base_branch == "main"

    def test_get_token_missing(self):
        """Test that missing token raises error."""
        service = GitHubService()

        with patch("rex.github_service.get_credential_manager") as mock_cred_manager:
            mock_manager = Mock()
            mock_manager.get_token.return_value = None
            mock_cred_manager.return_value = mock_manager

            with pytest.raises(ValueError, match="GitHub token not found"):
                service._get_token()


class TestGitHubServiceSingleton:
    """Tests for GitHub service singleton."""

    def test_get_github_service(self):
        """Test getting GitHub service singleton."""
        reset_github_service()
        service1 = get_github_service()
        service2 = get_github_service()
        assert service1 is service2

    def test_reset_github_service(self):
        """Test resetting GitHub service."""
        service1 = get_github_service()
        reset_github_service()
        service2 = get_github_service()
        assert service1 is not service2
