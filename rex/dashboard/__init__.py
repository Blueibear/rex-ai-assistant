"""Rex Dashboard module.

Provides a responsive web dashboard for Rex AI Assistant with:
- Authentication (token-based sessions)
- Settings management (view/edit configuration)
- Chat interface (text-based conversation)
- Scheduler/reminders UI (CRUD for jobs)

Usage:
    from rex.dashboard import dashboard_bp

    # Register with Flask app
    app.register_blueprint(dashboard_bp)

The dashboard is then available at:
    GET  /dashboard           - Main UI
    GET  /dashboard/assets/*  - Static files (CSS/JS)
    GET  /api/dashboard/status - Health/status endpoint
    POST /api/dashboard/login  - Authentication
    POST /api/dashboard/logout - Logout
    GET  /api/settings         - Get configuration
    PATCH /api/settings        - Update configuration
    POST /api/chat             - Send chat message
    GET  /api/chat/history     - Get chat history
    GET  /api/scheduler/jobs   - List jobs
    POST /api/scheduler/jobs   - Create job
    GET  /api/scheduler/jobs/<id> - Get job
    PATCH /api/scheduler/jobs/<id> - Update job
    DELETE /api/scheduler/jobs/<id> - Delete job
    POST /api/scheduler/jobs/<id>/run - Run job now

Configuration:
    Environment variables:
    - REX_DASHBOARD_PASSWORD: Password for authentication (optional)
    - REX_DASHBOARD_SECRET: Secret key for session tokens (auto-generated)
    - REX_DASHBOARD_SESSION_EXPIRY: Session expiry in seconds (default: 28800)
    - REX_DASHBOARD_ALLOW_LOCAL: Allow local access without auth (default: "1")

    Config file (config/rex_config.json):
    - dashboard.password: Password for authentication
    - dashboard.enabled: Enable/disable dashboard (default: true)
"""

from rex.dashboard.auth import (
    Session,
    SessionManager,
    get_dashboard_password,
    get_session_manager,
    is_password_required,
    verify_password,
)
from rex.dashboard.routes import dashboard_bp

__all__ = [
    "dashboard_bp",
    "Session",
    "SessionManager",
    "get_session_manager",
    "get_dashboard_password",
    "verify_password",
    "is_password_required",
]
