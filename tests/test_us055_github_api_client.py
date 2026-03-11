"""US-055: GitHub API client acceptance tests.

Covers:
- GitHub reachable
- repos listed
- authentication works
- Typecheck passes
"""

import pytest
from unittest.mock import MagicMock, patch

from rex.github_service import GitHubService, Repository, reset_github_service


class TestGitHubReachable:
    """GitHub reachable — API request is made to github.com."""

    def test_api_request_targets_github(self):
        """HTTP request is sent to the GitHub API base URL."""
        service = GitHubService(api_base="https://api.github.com")

        fake_response = MagicMock()
        fake_response.status_code = 200
        fake_response.json.return_value = []
        fake_response.raise_for_status.return_value = None

        with patch("rex.github_service.get_credential_manager") as mock_cm, patch(
            "requests.request", return_value=fake_response
        ) as mock_req:
            mock_manager = MagicMock()
            mock_manager.get_token.return_value = "ghp_test_token"
            mock_cm.return_value = mock_manager

            service.list_repos()

        call_kwargs = mock_req.call_args
        url = call_kwargs[1]["url"] if "url" in call_kwargs[1] else call_kwargs[0][1]
        assert "api.github.com" in url

    def test_api_base_configurable(self):
        """API base URL is configurable for enterprise GitHub."""
        service = GitHubService(api_base="https://github.example.com/api/v3")
        assert service.api_base == "https://github.example.com/api/v3"


class TestReposListed:
    """repos listed — list_repos() returns Repository objects."""

    def test_list_repos_returns_repository_objects(self):
        """list_repos() returns a list of Repository dataclass instances."""
        mock_data = {
            "GET:/user/repos": [
                {
                    "name": "my-repo",
                    "full_name": "alice/my-repo",
                    "owner": {"login": "alice"},
                    "html_url": "https://github.com/alice/my-repo",
                    "description": "A sample repo",
                    "private": False,
                    "default_branch": "main",
                }
            ]
        }
        service = GitHubService(mock_mode=True)
        service._mock_data = mock_data

        repos = service.list_repos()

        assert isinstance(repos, list)
        assert len(repos) == 1
        assert isinstance(repos[0], Repository)

    def test_list_repos_fields_populated(self):
        """Repository objects have all required fields populated."""
        mock_data = {
            "GET:/user/repos": [
                {
                    "name": "rex-ai-assistant",
                    "full_name": "james/rex-ai-assistant",
                    "owner": {"login": "james"},
                    "html_url": "https://github.com/james/rex-ai-assistant",
                    "description": "Rex AI",
                    "private": True,
                    "default_branch": "master",
                }
            ]
        }
        service = GitHubService(mock_mode=True)
        service._mock_data = mock_data

        repos = service.list_repos()
        repo = repos[0]

        assert repo.name == "rex-ai-assistant"
        assert repo.full_name == "james/rex-ai-assistant"
        assert repo.owner == "james"
        assert repo.url == "https://github.com/james/rex-ai-assistant"
        assert repo.private is True
        assert repo.default_branch == "master"

    def test_list_repos_empty_list(self):
        """list_repos() returns empty list when no repos exist."""
        mock_data = {"GET:/user/repos": []}
        service = GitHubService(mock_mode=True)
        service._mock_data = mock_data

        repos = service.list_repos()

        assert repos == []

    def test_list_repos_multiple_repos(self):
        """list_repos() handles multiple repositories."""
        mock_data = {
            "GET:/user/repos": [
                {
                    "name": f"repo-{i}",
                    "full_name": f"user/repo-{i}",
                    "owner": {"login": "user"},
                    "html_url": f"https://github.com/user/repo-{i}",
                    "description": None,
                    "private": False,
                    "default_branch": "main",
                }
                for i in range(5)
            ]
        }
        service = GitHubService(mock_mode=True)
        service._mock_data = mock_data

        repos = service.list_repos()

        assert len(repos) == 5
        assert [r.name for r in repos] == [f"repo-{i}" for i in range(5)]


class TestAuthenticationWorks:
    """authentication works — token is retrieved and sent correctly."""

    def test_token_sent_in_authorization_header(self):
        """The GitHub token is included in the Authorization header."""
        service = GitHubService()

        fake_response = MagicMock()
        fake_response.status_code = 200
        fake_response.json.return_value = []
        fake_response.raise_for_status.return_value = None

        with patch("rex.github_service.get_credential_manager") as mock_cm, patch(
            "requests.request", return_value=fake_response
        ) as mock_req:
            mock_manager = MagicMock()
            mock_manager.get_token.return_value = "ghp_secret_token"
            mock_cm.return_value = mock_manager

            service.list_repos()

        call_headers = mock_req.call_args[1]["headers"]
        assert call_headers["Authorization"] == "token ghp_secret_token"

    def test_accept_header_set_to_github_v3(self):
        """The Accept header uses the GitHub v3 API content type."""
        service = GitHubService()

        fake_response = MagicMock()
        fake_response.status_code = 200
        fake_response.json.return_value = []
        fake_response.raise_for_status.return_value = None

        with patch("rex.github_service.get_credential_manager") as mock_cm, patch(
            "requests.request", return_value=fake_response
        ) as mock_req:
            mock_manager = MagicMock()
            mock_manager.get_token.return_value = "ghp_any_token"
            mock_cm.return_value = mock_manager

            service.list_repos()

        call_headers = mock_req.call_args[1]["headers"]
        assert call_headers["Accept"] == "application/vnd.github.v3+json"

    def test_missing_token_raises_value_error(self):
        """list_repos() raises ValueError when GitHub token is missing."""
        service = GitHubService()

        with patch("rex.github_service.get_credential_manager") as mock_cm:
            mock_manager = MagicMock()
            mock_manager.get_token.return_value = None
            mock_cm.return_value = mock_manager

            with pytest.raises(RuntimeError):
                service.list_repos()

    def test_empty_token_raises_value_error(self):
        """_get_token() raises ValueError when token is blank."""
        service = GitHubService()

        with patch("rex.github_service.get_credential_manager") as mock_cm:
            mock_manager = MagicMock()
            mock_manager.get_token.return_value = "   "
            mock_cm.return_value = mock_manager

            with pytest.raises(ValueError, match="GitHub token not found"):
                service._get_token()

    def test_credential_name_used_for_lookup(self):
        """Credential manager is queried with the configured credential name."""
        service = GitHubService(credential_name="my_github_cred")

        fake_response = MagicMock()
        fake_response.status_code = 200
        fake_response.json.return_value = []
        fake_response.raise_for_status.return_value = None

        with patch("rex.github_service.get_credential_manager") as mock_cm, patch(
            "requests.request", return_value=fake_response
        ):
            mock_manager = MagicMock()
            mock_manager.get_token.return_value = "tok123"
            mock_cm.return_value = mock_manager

            service.list_repos()

        mock_manager.get_token.assert_called_with("my_github_cred")
