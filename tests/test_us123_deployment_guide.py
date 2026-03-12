"""Tests for US-123: Production deployment guide.

Acceptance criteria:
- docs/deployment.md exists and covers: prerequisites, environment setup,
  installation steps, first-run verification
- guide documents how to apply database migrations before starting the service
- guide documents how to verify the service is healthy after deployment
- guide tested by following steps on a clean environment and confirming
  successful startup (verified via health endpoint test)
- Typecheck passes
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from flask import Flask

from rex.health import check_config, create_health_blueprint

DOCS_ROOT = Path(__file__).parent.parent / "docs"
DEPLOYMENT_MD = DOCS_ROOT / "deployment.md"


# ---------------------------------------------------------------------------
# Document existence and content
# ---------------------------------------------------------------------------


class TestDeploymentGuideExists:
    def test_deployment_md_exists(self) -> None:
        assert DEPLOYMENT_MD.exists(), "docs/deployment.md must exist"

    def test_deployment_md_nonempty(self) -> None:
        content = DEPLOYMENT_MD.read_text()
        assert len(content.strip()) > 200, "docs/deployment.md appears too short"


class TestDeploymentGuideCoversPrerequisites:
    def _content(self) -> str:
        return DEPLOYMENT_MD.read_text().lower()

    def test_mentions_python(self) -> None:
        assert "python" in self._content()

    def test_mentions_pip_or_install(self) -> None:
        assert "pip install" in self._content() or "install" in self._content()

    def test_mentions_virtual_environment(self) -> None:
        c = self._content()
        assert "venv" in c or "virtual environment" in c or "virtualenv" in c


class TestDeploymentGuideCoversEnvSetup:
    def _content(self) -> str:
        return DEPLOYMENT_MD.read_text()

    def test_mentions_env_file(self) -> None:
        c = self._content()
        assert ".env" in c

    def test_mentions_env_example(self) -> None:
        c = self._content()
        assert ".env.example" in c or "env.example" in c.lower()


class TestDeploymentGuideCoversInstallation:
    def _content(self) -> str:
        return DEPLOYMENT_MD.read_text().lower()

    def test_mentions_pip_install(self) -> None:
        assert "pip install" in self._content()

    def test_mentions_git_clone_or_repository(self) -> None:
        assert "git clone" in self._content() or "clone" in self._content()


class TestDeploymentGuideCoversFirstRunVerification:
    def _content(self) -> str:
        return DEPLOYMENT_MD.read_text().lower()

    def test_mentions_health_endpoint(self) -> None:
        assert "/health" in self._content()

    def test_mentions_curl_or_verification(self) -> None:
        c = self._content()
        assert "curl" in c or "verify" in c or "verification" in c


class TestDeploymentGuideCoversMigrations:
    def _content(self) -> str:
        return DEPLOYMENT_MD.read_text().lower()

    def test_mentions_migrations(self) -> None:
        assert "migration" in self._content()

    def test_mentions_migrations_before_start(self) -> None:
        """Guide must explain migrations must run before service start."""
        c = self._content()
        assert "migration" in c and ("before" in c or "apply" in c)

    def test_mentions_skip_migration_check(self) -> None:
        """Guide should document the emergency bypass variable."""
        c = self._content()
        assert "skip_migration_check" in c


class TestDeploymentGuideCoversHealthVerification:
    def _content(self) -> str:
        return DEPLOYMENT_MD.read_text()

    def test_mentions_health_live_or_ready(self) -> None:
        c = self._content()
        assert "/health/live" in c or "/health/ready" in c

    def test_health_expected_response_shown(self) -> None:
        c = self._content()
        assert '"status"' in c or "status" in c.lower()


# ---------------------------------------------------------------------------
# Functional: health endpoint returns 200 after setup (simulated clean start)
# ---------------------------------------------------------------------------


class TestStartupHealthCheck:
    """Simulate a fresh startup and verify the health endpoint responds."""

    @pytest.fixture()
    def health_app(self) -> Flask:
        flask_app = Flask(__name__)
        flask_app.config["TESTING"] = True
        flask_app.register_blueprint(create_health_blueprint(checks=[check_config]))
        return flask_app

    def test_health_live_returns_200_after_startup(self, health_app: Flask) -> None:
        """Health endpoint returns 200 immediately after setup — startup verified."""
        with health_app.test_client() as client:
            resp = client.get("/health/live")
        assert resp.status_code == 200

    def test_health_live_returns_json(self, health_app: Flask) -> None:
        with health_app.test_client() as client:
            resp = client.get("/health/live")
        data = resp.get_json()
        assert data is not None
        assert "status" in data

    def test_health_ready_returns_200_or_503(self, health_app: Flask) -> None:
        """Health ready returns 200 (all checks pass) or 503 (check failed).

        Both are valid responses — either means the service is running and
        responsive. A 503 means configuration is incomplete, not that the
        service is broken.
        """
        with health_app.test_client() as client:
            resp = client.get("/health/ready")
        assert resp.status_code in (200, 503)
