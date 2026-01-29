# Rex Dashboard

The Rex Dashboard is a responsive web interface for managing your Rex AI Assistant. It provides a single point of tactile interface for settings, chat, and reminders.

## Features

- **Chat Interface**: Text-based conversation with Rex
- **Settings Management**: View and edit configuration (with sensitive values redacted)
- **Reminders/Scheduler**: Create, view, enable/disable, and delete scheduled jobs
- **Status Overview**: System health, uptime, and version information
- **Responsive Design**: Works on desktop and mobile devices

## Quick Start

### Running the Dashboard

1. Start the Rex server:
   ```bash
   python flask_proxy.py
   ```

2. Open your browser and navigate to:
   ```
   http://localhost:5000/dashboard
   ```

3. If authentication is enabled, enter your password to log in.

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `REX_DASHBOARD_PASSWORD` | Password for dashboard authentication | None (local-only access) |
| `REX_DASHBOARD_SECRET` | Secret key for session tokens | Auto-generated |
| `REX_DASHBOARD_SESSION_EXPIRY` | Session expiry in seconds | 28800 (8 hours) |
| `REX_DASHBOARD_ALLOW_LOCAL` | Allow local access without auth | "1" (enabled) |

### Config File (rex_config.json)

```json
{
  "dashboard": {
    "password": "your-secure-password",
    "enabled": true
  }
}
```

## Authentication

### Local Access (Default)

By default, the dashboard allows unauthenticated access from localhost (127.0.0.1). This is controlled by `REX_DASHBOARD_ALLOW_LOCAL=1`.

### Password Authentication

To require password authentication:

1. Set a password via environment variable:
   ```bash
   export REX_DASHBOARD_PASSWORD="your-secure-password"
   ```

   Or in your `.env` file:
   ```
   REX_DASHBOARD_PASSWORD=your-secure-password
   ```

2. Alternatively, set it in `config/rex_config.json`:
   ```json
   {
     "dashboard": {
       "password": "your-secure-password"
     }
   }
   ```

### Session Management

- Sessions are valid for 8 hours by default
- Tokens are stored in HTTP-only cookies for security
- Logging out invalidates the session immediately

## API Endpoints

### Authentication

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/dashboard/status` | GET | Health check and status info (public) |
| `/api/dashboard/login` | POST | Authenticate and get session token |
| `/api/dashboard/logout` | POST | Invalidate current session |

### Settings

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/settings` | GET | Get configuration (redacted) |
| `/api/settings` | PATCH | Update configuration keys |

Notes:
- PATCH requests are validated against known configuration keys and value types.
- Unknown keys or invalid value types are rejected with a 400 response.

### Chat

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/chat` | POST | Send message, get reply |
| `/api/chat/history` | GET | Get chat history |

### Scheduler/Reminders

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/scheduler/jobs` | GET | List all jobs |
| `/api/scheduler/jobs` | POST | Create new job |
| `/api/scheduler/jobs/<id>` | GET | Get specific job |
| `/api/scheduler/jobs/<id>` | PATCH | Update job |
| `/api/scheduler/jobs/<id>` | DELETE | Delete job |
| `/api/scheduler/jobs/<id>/run` | POST | Run job immediately |

## Security Considerations

### Safe Exposure

The dashboard is designed for local use. If you need to expose it externally:

1. **Use a reverse proxy** (nginx, Caddy) with HTTPS
2. **Enable password authentication**
3. **Consider additional auth layers** (Cloudflare Access, VPN)

Example nginx configuration:
```nginx
server {
    listen 443 ssl;
    server_name rex.example.com;

    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;

    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

### Sensitive Data

- API keys and passwords are automatically redacted in settings display
- Sensitive values cannot be viewed, only updated
- Session tokens are stored securely with HTTP-only cookies

## UI Sections

### Chat

The chat interface provides:
- Message input and send button
- Message history display
- "Thinking..." indicator for responses taking >300ms
- Automatic scrolling to newest messages

### Settings

The settings editor shows:
- Grouped configuration options
- Search/filter functionality
- Restart-required indicators
- Save/discard buttons for batch updates

### Reminders

The reminders section allows:
- Viewing all scheduled jobs
- Creating new reminders (interval or daily)
- Enabling/disabling jobs
- Running jobs immediately
- Deleting jobs

### Status

The status page displays:
- System status (OK/Error)
- Uptime
- Version
- Authentication mode
- Scheduled job statistics
- Server time

## Troubleshooting

### "Authentication required" error

1. Check if password is configured correctly
2. Verify you're accessing from localhost if local access is enabled
3. Try logging out and back in

### Settings not saving

1. Check browser console for errors
2. Verify the server is running
3. Ensure you have authentication token

### Chat not responding

1. Check LLM provider configuration
2. Verify API keys if using external providers
3. Check server logs for errors

## Development

### File Structure

```
rex/dashboard/
├── __init__.py       # Module exports
├── auth.py           # Authentication utilities
├── routes.py         # Flask blueprint with all routes
├── static/
│   ├── css/
│   │   └── dashboard.css
│   └── js/
│       └── dashboard.js
└── templates/
    └── index.html
```

### Running Tests

```bash
python -m pytest tests/test_dashboard.py -v
```

### Adding New Features

1. Add API endpoints in `routes.py`
2. Update UI in `templates/index.html`
3. Add JavaScript handlers in `static/js/dashboard.js`
4. Add CSS in `static/css/dashboard.css`
5. Write tests in `tests/test_dashboard.py`
