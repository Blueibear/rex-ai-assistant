from __future__ import annotations

from typing import Any

EXPECTED_ROUTE_SNAPSHOT = [
    ("/api/admin/permissions/grant", "_admin_grant_permission", ("POST",), ()),
    ("/api/admin/permissions/revoke", "_admin_revoke_permission", ("POST",), ()),
    ("/api/auth/login", "_auth_login", ("POST",), ()),
    ("/api/auth/logout", "_auth_logout", ("POST",), ()),
    ("/api/auth/register", "_auth_register", ("POST",), ()),
    ("/api/calendar/events", "_calendar_events", ("GET",), ()),
    ("/api/capabilities", "_list_capabilities", ("GET",), ()),
    ("/api/chat/clear", "_chat_clear", ("POST",), ()),
    ("/api/chat/history", "_chat_history", ("GET",), ()),
    ("/api/chat/send", "_chat_send", ("POST",), ()),
    ("/api/dashboard/status", "_dashboard_status_stub", ("GET",), ()),
    ("/api/devices", "_list_devices", ("GET",), ()),
    ("/api/devices/<path:entity_id>/command", "_device_command", ("POST",), ()),
    ("/api/email/inbox", "_email_inbox", ("GET",), ()),
    ("/api/ha/save", "_ha_save_config", ("POST",), ()),
    ("/api/ha/states", "_ha_get_states", ("GET",), ()),
    ("/api/ha/test", "_ha_test_connection", ("POST",), ()),
    ("/api/history", "_command_history", ("GET",), ()),
    ("/api/integrations", "_list_integrations", ("GET",), ()),
    ("/api/logs/download", "_logs_download", ("GET",), ()),
    ("/api/logs/stream", "_logs_stream", ("GET",), ()),
    ("/api/personalities", "_list_personalities", ("GET",), ()),
    ("/api/quick-actions", "_add_quick_action", ("POST",), ()),
    ("/api/quick-actions", "_list_quick_actions", ("GET",), ()),
    ("/api/quick-actions/<action_id>", "_delete_quick_action", ("DELETE",), ()),
    ("/api/quick-actions/<action_id>/run", "_run_quick_action", ("POST",), ()),
    ("/api/setup/complete", "_setup_complete", ("POST",), ()),
    ("/api/setup/status", "_setup_status", ("GET",), ()),
    ("/api/sms/threads", "_sms_threads", ("GET",), ()),
    ("/api/status/current", "_status_current", ("GET",), ()),
    ("/api/status/stream", "_status_stream", ("GET",), ()),
    ("/api/tools", "_list_tools", ("GET",), ()),
    ("/api/usage", "_usage_summary", ("GET",), ()),
    ("/api/user/avatar", "_get_avatar", ("GET",), ()),
    ("/api/user/avatar", "_upload_avatar", ("POST",), ()),
    ("/api/user/permissions", "_get_my_permissions", ("GET",), ()),
    ("/api/user/preferences", "_get_preferences", ("GET",), ()),
    ("/api/user/preferences", "_patch_preferences", ("PATCH",), ()),
    ("/dashboard", "_dashboard_redirect", ("GET",), ()),
    ("/ui/", "_serve_ui", ("GET",), (("filename", "index.html"),)),
    ("/ui/<path:filename>", "_serve_ui", ("GET",), ()),
]


def _route_snapshot(
    app: Any,
) -> list[tuple[str, str, tuple[str, ...], tuple[tuple[str, Any], ...]]]:
    return [
        (
            rule.rule,
            rule.endpoint,
            tuple(sorted(method for method in rule.methods if method not in {"HEAD", "OPTIONS"})),
            tuple(sorted((rule.defaults or {}).items())),
        )
        for rule in sorted(app.url_map.iter_rules(), key=lambda item: (item.rule, item.endpoint))
    ]


def test_gui_route_snapshot_is_unchanged(monkeypatch: Any, tmp_path: Any) -> None:
    monkeypatch.setenv("REX_DATA_DIR", str(tmp_path))
    from rex.gui_app import _create_flask_app

    app = _create_flask_app(ui_enabled=True)
    assert _route_snapshot(app) == EXPECTED_ROUTE_SNAPSHOT
