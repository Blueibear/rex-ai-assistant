# E2E Manual Test: GUI Launch and Backend Connection

**User Story:** US-328
**Purpose:** Verify the Electron GUI launches, connects to the Flask backend, and renders without crash or login loop.

---

## Prerequisites

- Python virtual environment activated (`.venv`)
- Dependencies installed: `pip install .`
- `config/rex_config.json` exists (auto-created by US-301 on first run)
- Node.js installed; GUI built or dev server available

---

## Test Steps

### Step 1 — Start the Rex Flask backend

```bash
rex-gui
```

Expected: Flask starts on `http://127.0.0.1:5000` (or the configured port).
Console output should include: `* Running on http://127.0.0.1:5000`

### Step 2 — Launch the Electron GUI (development mode)

```bash
cd gui
npm run dev
```

Or for a production build:

```bash
cd gui
npm run build
npx electron .
```

### Step 3 — Observe first screen

Expected first screen (within 10 seconds of launch):
- Title bar shows **"AskRex"** (not "Electron" or blank)
- Sidebar navigation is visible on the left
- The main area displays the **Home** page content:
  - "Home" heading
  - "Device Control" card with a "Configure Home Assistant" link
  - "Suggestions" card
  - "Music" card

**Failure indicators:**
- Spinner that never resolves → backend not reachable
- "Rex backend unavailable" error state → Flask is not running
- Blank white screen → renderer crash (check DevTools console)

### Step 4 — Verify backend connection

Open DevTools in the Electron window (`Ctrl+Shift+I` / `Cmd+Option+I`).

In the **Network** tab, confirm these requests succeed (HTTP 200):
- `GET /api/setup/status` → `{"needs_setup": false}` (or `true` on first run)
- `GET /api/status/current` → `{"status": "ready"}` (or similar non-error status)

In the **Console** tab, confirm:
- No errors matching: `bridge exited`, `ENOENT`, `failed to fetch`, `Cannot read properties`
- `[bridgeResolver] All bridge scripts validated successfully.` appears in the main process log (terminal window)

### Step 5 — Setup wizard check (first run only)

If `config/rex_config.json` does not exist or `needs_setup` is `true`:

Expected: Setup wizard page renders with guided configuration steps.
After completing setup, the main app loads **without** returning to the wizard.

**Login loop test:** Reload the app (`Ctrl+R`). Expected: main app loads directly, not the setup wizard again.

### Step 6 — Navigate to Home page

Click **Home** in the sidebar (or confirm it is already active).

Expected:
- Page renders within 2 seconds
- No console errors from Home page load
- "Configure Home Assistant" link is visible and clickable

### Step 7 — Auth flow (if applicable)

This build has **no authentication requirement** — the GUI connects directly to the local Flask server without a login prompt. If a login screen appears unexpectedly:

1. Check `.env` for `REX_AUTH_REQUIRED=true` and remove it
2. Verify Flask is running on the expected port
3. Check `config/rex_config.json` for `auth_required: true` and set to `false`

---

## Pass/Fail Criteria

| Check | Pass | Fail |
|-------|------|------|
| App launches without crash | Window opens within 10s | Process exits or hangs |
| First screen is Home page | Home page content visible | Spinner, error state, or blank screen |
| `/api/setup/status` succeeds | HTTP 200 | Network error or timeout |
| `/api/status/current` succeeds | HTTP 200 with status field | Network error or timeout |
| No bridge errors in console | No ENOENT/bridge errors | Any bridge path error |
| No login loop | Main app loads on reload | Setup wizard reappears after completion |

---

## Troubleshooting

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| "Rex backend unavailable" shown | Flask not running | Run `rex-gui` first |
| Spinner never resolves | Both API calls failing | Check Flask port and CORS config |
| Bridge errors in console | Missing `rex_*_bridge.py` files | Run `pip install .` to regenerate entry points |
| Setup wizard loops | `needs_setup` always returns `true` | Check `config/rex_config.json` write permissions |
| Blank white screen | React render crash | Open DevTools → Console for stack trace |
