/**
 * Rex Dashboard JavaScript
 * Single-page application for Rex AI Assistant management
 */

(function() {
    'use strict';

    // State management
    const state = {
        authenticated: false,
        token: null,
        currentSection: 'overview',
        settings: null,
        pendingSettings: {},
        chatHistory: [],
        notifications: [],
        notifFilters: { unread: false, priority: '', channel: '' },
    };

    // Voice recording state
    const voiceState = {
        mediaRecorder: null,
        chunks: [],
        recording: false,
    };

    // Valid section identifiers for navigation
    const VALID_SECTIONS = ['chat', 'voice', 'schedule', 'overview', 'settings', 'reminders', 'notifications', 'status'];

    // Flag to prevent pushState during popstate-driven navigation
    let _navigatingFromHistory = false;

    // Notification polling timer handle
    let _notifPollTimer = null;
    // SSE EventSource handle for real-time notifications
    let _notifEventSource = null;
    // SSE reconnect state
    const _SSE_MAX_RECONNECT = 5;
    let _sseReconnectCount = 0;
    let _sseReconnectTimer = null;

    // API helper
    async function api(endpoint, options = {}) {
        const headers = {
            'Content-Type': 'application/json',
            ...options.headers,
        };

        if (state.token) {
            headers['Authorization'] = `Bearer ${state.token}`;
        }

        const response = await fetch(endpoint, {
            ...options,
            headers,
            credentials: 'same-origin',
        });

        if ((response.status === 401 || response.status === 403) && state.authenticated) {
            // Session expired or unauthorized for current token
            handleLogout();
            throw new Error(response.status === 401 ? 'Session expired' : 'Access denied');
        }

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || `HTTP ${response.status}`);
        }

        return data;
    }

    // DOM helpers
    function $(selector) {
        return document.querySelector(selector);
    }

    function $$(selector) {
        return document.querySelectorAll(selector);
    }

    function show(element) {
        if (typeof element === 'string') element = $(element);
        element?.classList.remove('hidden');
    }

    function hide(element) {
        if (typeof element === 'string') element = $(element);
        element?.classList.add('hidden');
    }

    // Login handling
    async function handleLogin(event) {
        event.preventDefault();

        const password = $('#password').value;
        const errorEl = $('#login-error');
        const statusEl = $('#login-status');

        errorEl.textContent = '';
        statusEl.textContent = 'Logging in...';

        try {
            const data = await api('/api/dashboard/login', {
                method: 'POST',
                body: JSON.stringify({ password }),
            });

            state.token = data.token;
            state.authenticated = true;
            localStorage.setItem('rex_dashboard_token', data.token);

            showDashboard();
        } catch (error) {
            errorEl.textContent = error.message || 'Login failed';
            statusEl.textContent = '';
        }
    }

    function handleLogout() {
        api('/api/dashboard/logout', { method: 'POST' }).catch(() => {});

        state.token = null;
        state.authenticated = false;
        localStorage.removeItem('rex_dashboard_token');

        stopNotifPolling();
        stopNotifStream();
        stopNotifReconnect();
        showLogin();
    }

    // Screen management
    function showLogin() {
        hide('#dashboard-screen');
        show('#login-screen');
        $('#password').value = '';
        $('#login-error').textContent = '';
    }

    function showDashboard() {
        hide('#login-screen');
        show('#dashboard-screen');

        // Detect section from URL hash or path for deep-linking
        const hashSection = window.location.hash.slice(1);
        const initialSection = (VALID_SECTIONS.includes(hashSection) ? hashSection : null)
            || _sectionFromPath(window.location.pathname)
            || state.currentSection;
        switchSection(initialSection);
    }

    function _sectionFromPath(pathname) {
        const map = {
            '/dashboard/notifications': 'notifications',
            '/dashboard/reminders': 'reminders',
            '/dashboard/settings': 'settings',
            '/dashboard/status': 'status',
            '/dashboard/chat': 'chat',
        };
        return map[pathname] || null;
    }

    // Section navigation
    function switchSection(sectionId) {
        // Stop notification polling/streaming when leaving notifications section
        if (state.currentSection === 'notifications' && sectionId !== 'notifications') {
            stopNotifPolling();
            stopNotifStream();
            stopNotifReconnect();
        }
        // Stop voice state SSE when leaving voice section
        if (state.currentSection === 'voice' && sectionId !== 'voice') {
            stopVoiceStateStream();
        }

        state.currentSection = sectionId;

        // Update nav links
        $$('.nav-link, .mobile-nav-link').forEach(link => {
            link.classList.toggle('active', link.dataset.section === sectionId);
        });

        // Show/hide sections
        $$('.content-section').forEach(section => {
            section.classList.toggle('hidden', section.id !== `${sectionId}-section`);
        });

        // Update browser history so back/forward navigation works
        if (!_navigatingFromHistory && VALID_SECTIONS.includes(sectionId)) {
            history.pushState({section: sectionId}, '', '#' + sectionId);
        }

        // Load section data
        switch (sectionId) {
            case 'overview':
                loadOverview();
                break;
            case 'voice':
                startVoiceStateStream();
                break;
            case 'chat':
                loadChatHistory();
                setTimeout(() => { const inp = $('#chat-input'); if (inp) inp.focus(); }, 0);
                break;
            case 'settings':
                loadSettings();
                break;
            case 'schedule':
                loadScheduleJobs();
                break;
            case 'reminders':
                loadReminders();
                break;
            case 'notifications':
                loadNotifications();
                startNotifRealtime();
                break;
            case 'status':
                loadStatus();
                break;
        }
    }

    // Chat functionality
    function _retryErrorHtml(message, handlerKey) {
        return `<div class="panel-error error-message">
            <span>${escapeHtml(message)}</span>
            <button class="btn btn-secondary btn-sm retry-btn"
                onclick="window.dashboardHandlers['${handlerKey}']()">Retry</button>
        </div>`;
    }

    async function loadOverview() {
        const grid = $('#overview-status-grid');
        if (!grid) return;
        grid.innerHTML = '<div class="loading">Loading overview...</div>';
        try {
            const [status, voiceMode, schedulerData, notifData] = await Promise.all([
                api('/api/dashboard/status'),
                api('/api/voice/mode'),
                api('/api/scheduler/jobs'),
                api('/api/notifications?limit=50'),
            ]);
            renderOverview({ status, voiceMode, schedulerData, notifData });
        } catch (error) {
            grid.innerHTML = _retryErrorHtml(`Failed to load overview: ${error.message}`, 'loadOverview');
        }
    }

    function _overviewIndicator(ok) {
        return `<span class="overview-indicator ${ok ? 'ok' : 'error'}"></span>`;
    }

    function renderOverview(data) {
        const { status, voiceMode, schedulerData, notifData } = data;
        const grid = $('#overview-status-grid');
        if (!grid) return;
        const rexOk = status.status === 'ok';
        const voiceActive = voiceMode.active === true;
        const lmStudioOk = rexOk; // LM Studio reachable if backend is up
        const scheduledCount = schedulerData.total || 0;
        const unreadCount = notifData.unread_count || 0;

        grid.innerHTML = `
            <div class="overview-status-card">
                ${_overviewIndicator(rexOk)}
                <div class="overview-status-label">Rex Status</div>
                <div class="overview-status-value">${rexOk ? 'Online' : 'Offline'}</div>
            </div>
            <div class="overview-status-card">
                ${_overviewIndicator(voiceActive)}
                <div class="overview-status-label">Voice Mode</div>
                <div class="overview-status-value">${voiceActive ? 'Active' : 'Idle'}</div>
            </div>
            <div class="overview-status-card">
                ${_overviewIndicator(lmStudioOk)}
                <div class="overview-status-label">LM Studio</div>
                <div class="overview-status-value">${lmStudioOk ? 'Connected' : 'Unavailable'}</div>
            </div>
            <div class="overview-status-card">
                ${_overviewIndicator(true)}
                <div class="overview-status-label">Scheduled Items</div>
                <div class="overview-status-value">${scheduledCount}</div>
            </div>
            <div class="overview-status-card">
                ${_overviewIndicator(unreadCount === 0)}
                <div class="overview-status-label">Notifications</div>
                <div class="overview-status-value">${unreadCount} unread</div>
            </div>
        `;
    }

    async function loadChatHistory() {
        const container = $('#chat-messages');
        if (container) container.innerHTML = '<div class="loading">Loading chat history...</div>';
        try {
            const data = await api('/api/chat/history?limit=50');
            state.chatHistory = data.history || [];
            renderChatMessages();
        } catch (error) {
            if (container) container.innerHTML = _retryErrorHtml(`Failed to load chat history: ${error.message}`, 'loadChatHistory');
        }
    }

    function renderChatMessages() {
        const container = $('#chat-messages');

        if (state.chatHistory.length === 0) {
            container.innerHTML = '<div class="chat-placeholder" style="text-align: center; color: var(--text-secondary); padding: 2rem;">Start a conversation with Rex</div>';
            return;
        }

        container.innerHTML = state.chatHistory.map(entry => `
            <div class="chat-message user">${escapeHtml(entry.user_message)}</div>
            <div class="chat-message assistant">${escapeHtml(entry.assistant_reply)}</div>
        `).join('');

        container.scrollTop = container.scrollHeight;
    }

    async function handleChatSubmit(event) {
        event.preventDefault();

        const input = $('#chat-input');
        const message = input.value.trim();

        if (!message) return;

        input.value = '';
        input.disabled = true;

        // Add user message immediately
        const container = $('#chat-messages');
        const placeholder = container.querySelector('.chat-placeholder');
        if (placeholder) placeholder.remove();

        container.innerHTML += `<div class="chat-message user">${escapeHtml(message)}</div>`;
        container.innerHTML += `<div class="chat-message thinking" id="thinking-indicator">Thinking...</div>`;
        container.scrollTop = container.scrollHeight;

        let assistantBubble = null;
        let accumulatedReply = '';
        let lastTimestamp = null;

        try {
            const headers = { 'Content-Type': 'application/json' };
            const authToken = localStorage.getItem('rex_auth_token');
            if (authToken) headers['X-Auth-Token'] = authToken;

            const response = await fetch('/api/chat/stream', {
                method: 'POST',
                headers,
                body: JSON.stringify({ message }),
            });

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}`);
            }

            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let buffer = '';

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                buffer += decoder.decode(value, { stream: true });
                const parts = buffer.split('\n\n');
                buffer = parts.pop() || '';

                for (const part of parts) {
                    const lines = part.split('\n');
                    const eventLine = lines.find(l => l.startsWith('event: '));
                    const dataLine = lines.find(l => l.startsWith('data: '));
                    if (!eventLine || !dataLine) continue;

                    const eventName = eventLine.slice(7).trim();
                    let payload;
                    try { payload = JSON.parse(dataLine.slice(6)); } catch (e) { continue; }

                    if (eventName === 'token') {
                        // Replace thinking indicator with streaming bubble on first token
                        const thinking = $('#thinking-indicator');
                        if (thinking) {
                            thinking.remove();
                            const el = document.createElement('div');
                            el.className = 'chat-message assistant';
                            el.id = 'streaming-bubble';
                            container.appendChild(el);
                            assistantBubble = el;
                        }
                        accumulatedReply += payload.token;
                        if (assistantBubble) {
                            assistantBubble.textContent = accumulatedReply;
                        }
                        container.scrollTop = container.scrollHeight;
                    } else if (eventName === 'done') {
                        lastTimestamp = payload.timestamp;
                        if (assistantBubble) assistantBubble.removeAttribute('id');
                    } else if (eventName === 'error') {
                        throw new Error(payload.error || 'Streaming error');
                    }
                }
            }

            // Update local history
            if (accumulatedReply) {
                state.chatHistory.push({
                    user_message: message,
                    assistant_reply: accumulatedReply,
                    timestamp: lastTimestamp || new Date().toISOString(),
                });
            }
        } catch (error) {
            const thinking = $('#thinking-indicator');
            if (thinking) {
                thinking.textContent = `Error: ${error.message}`;
                thinking.classList.remove('thinking');
                thinking.style.color = 'var(--error-color)';
            } else if (assistantBubble) {
                assistantBubble.textContent += ` [Error: ${error.message}]`;
            }
        } finally {
            input.disabled = false;
            input.focus();
        }
    }

    // Settings functionality
    async function loadSettings() {
        const container = $('#settings-container');
        container.innerHTML = '<div class="loading">Loading settings...</div>';

        try {
            const data = await api('/api/settings');
            state.settings = data.settings;
            state.pendingSettings = {};
            renderSettings(data.settings, data.metadata);
            hide('#settings-save-bar');
        } catch (error) {
            container.innerHTML = _retryErrorHtml(`Failed to load settings: ${error.message}`, 'loadSettings');
        }
    }

    function renderSettings(settings, metadata = {}, filter = '') {
        const container = $('#settings-container');
        const groups = {};

        // Flatten settings into groups
        function flatten(obj, prefix = '') {
            for (const [key, value] of Object.entries(obj)) {
                const path = prefix ? `${prefix}.${key}` : key;

                if (value && typeof value === 'object' && !Array.isArray(value)) {
                    flatten(value, path);
                } else {
                    const groupName = prefix || 'general';
                    if (!groups[groupName]) groups[groupName] = [];
                    groups[groupName].push({ key, path, value });
                }
            }
        }

        flatten(settings);

        // Filter settings
        const filterLower = filter.toLowerCase();

        // Render groups
        container.innerHTML = Object.entries(groups)
            .filter(([groupName, items]) => {
                if (!filter) return true;
                return groupName.toLowerCase().includes(filterLower) ||
                    items.some(item => item.key.toLowerCase().includes(filterLower));
            })
            .map(([groupName, items]) => {
                const filteredItems = filter
                    ? items.filter(item =>
                        item.key.toLowerCase().includes(filterLower) ||
                        item.path.toLowerCase().includes(filterLower))
                    : items;

                if (filteredItems.length === 0) return '';

                return `
                    <div class="settings-group">
                        <div class="settings-group-header">${escapeHtml(groupName)}</div>
                        ${filteredItems.map(item => renderSettingItem(item, metadata)).join('')}
                    </div>
                `;
            })
            .filter(Boolean)
            .join('');

        if (!container.innerHTML.trim()) {
            container.innerHTML = '<div class="loading">No settings match your search</div>';
        }
    }

    function renderSettingItem(item, metadata) {
        const meta = metadata[item.path] || {};
        const isRedacted = item.value === '[REDACTED]';
        const pendingValue = state.pendingSettings[item.path];
        const value = pendingValue !== undefined ? pendingValue : item.value;

        return `
            <div class="settings-item" data-path="${escapeHtml(item.path)}">
                <div class="settings-item-info">
                    <div class="settings-item-key">${escapeHtml(item.key)}</div>
                    <div class="settings-item-meta">
                        ${meta.restart_required ? '<span class="settings-badge">Restart required</span>' : ''}
                    </div>
                </div>
                <div class="settings-item-value ${isRedacted ? 'redacted' : ''}">
                    ${isRedacted
                        ? '[REDACTED]'
                        : `<input type="text" value="${escapeHtml(String(value ?? ''))}"
                            data-path="${escapeHtml(item.path)}"
                            data-original="${escapeHtml(String(item.value ?? ''))}"
                            onchange="window.dashboardHandlers.settingChanged(this)">`
                    }
                </div>
            </div>
        `;
    }

    function handleSettingChanged(input) {
        const path = input.dataset.path;
        const original = input.dataset.original;
        const newValue = input.value;

        if (newValue === original) {
            delete state.pendingSettings[path];
        } else {
            // Try to parse as appropriate type
            let parsedValue = newValue;
            if (newValue === 'true') parsedValue = true;
            else if (newValue === 'false') parsedValue = false;
            else if (newValue === 'null') parsedValue = null;
            else if (/^-?\d+$/.test(newValue)) parsedValue = parseInt(newValue, 10);
            else if (/^-?\d+\.\d+$/.test(newValue)) parsedValue = parseFloat(newValue);

            state.pendingSettings[path] = parsedValue;
        }

        updateSaveBar();
    }

    function updateSaveBar() {
        const count = Object.keys(state.pendingSettings).length;
        const saveBar = $('#settings-save-bar');

        if (count > 0) {
            show(saveBar);
            $('#unsaved-count').textContent = `${count} unsaved change${count !== 1 ? 's' : ''}`;
        } else {
            hide(saveBar);
        }
    }

    async function saveSettings() {
        if (Object.keys(state.pendingSettings).length === 0) return;

        try {
            const data = await api('/api/settings', {
                method: 'PATCH',
                body: JSON.stringify(state.pendingSettings),
            });

            state.pendingSettings = {};
            updateSaveBar();

            if (data.restart_required) {
                alert('Settings saved. Some changes require a restart to take effect.');
            }

            // Reload settings
            loadSettings();
        } catch (error) {
            alert(`Failed to save settings: ${error.message}`);
        }
    }

    function discardSettings() {
        state.pendingSettings = {};
        loadSettings();
    }

    // Reminders functionality
    async function loadReminders() {
        const container = $('#reminders-list');
        container.innerHTML = '<div class="loading">Loading reminders...</div>';

        try {
            const data = await api('/api/scheduler/jobs');
            renderReminders(data.jobs);
        } catch (error) {
            container.innerHTML = _retryErrorHtml(`Failed to load reminders: ${error.message}`, 'loadReminders');
        }
    }

    function renderReminders(jobs) {
        const container = $('#reminders-list');

        if (jobs.length === 0) {
            container.innerHTML = '<div class="loading">No reminders scheduled</div>';
            return;
        }

        container.innerHTML = jobs.map(job => `
            <div class="reminder-card" data-id="${escapeHtml(job.job_id)}">
                <div class="reminder-header">
                    <span class="reminder-name">${escapeHtml(job.name)}</span>
                    <span class="reminder-status ${job.enabled ? 'enabled' : 'disabled'}">
                        ${job.enabled ? 'Enabled' : 'Disabled'}
                    </span>
                </div>
                <div class="reminder-details">
                    <p><strong>Schedule:</strong> ${formatSchedule(job.schedule)}</p>
                    <p><strong>Next run:</strong> ${job.next_run ? new Date(job.next_run).toLocaleString() : 'Not scheduled'}</p>
                    <p><strong>Run count:</strong> ${job.run_count}${job.max_runs ? ` / ${job.max_runs}` : ''}</p>
                </div>
                <div class="reminder-actions">
                    <button class="btn btn-secondary btn-sm" onclick="window.dashboardHandlers.runReminder('${escapeHtml(job.job_id)}')">
                        Run Now
                    </button>
                    <button class="btn btn-secondary btn-sm" onclick="window.dashboardHandlers.toggleReminder('${escapeHtml(job.job_id)}', ${!job.enabled})">
                        ${job.enabled ? 'Disable' : 'Enable'}
                    </button>
                    <button class="btn btn-danger btn-sm" onclick="window.dashboardHandlers.deleteReminder('${escapeHtml(job.job_id)}')">
                        Delete
                    </button>
                </div>
            </div>
        `).join('');
    }

    function formatSchedule(schedule) {
        if (schedule.startsWith('interval:')) {
            const seconds = parseInt(schedule.split(':')[1], 10);
            if (seconds < 60) return `Every ${seconds} seconds`;
            if (seconds < 3600) return `Every ${Math.floor(seconds / 60)} minutes`;
            if (seconds < 86400) return `Every ${Math.floor(seconds / 3600)} hours`;
            return `Every ${Math.floor(seconds / 86400)} days`;
        }
        if (schedule.startsWith('at:')) {
            return `Daily at ${schedule.split(':').slice(1).join(':')}`;
        }
        return schedule;
    }

    async function loadScheduleJobs() {
        const container = $('#schedule-list');
        if (!container) return;
        container.innerHTML = '<div class="loading">Loading schedule...</div>';
        try {
            const data = await api('/api/scheduler/jobs');
            const jobs = data.jobs || [];
            renderScheduleJobs(jobs);
            renderComingUp(jobs);
        } catch (error) {
            container.innerHTML = _retryErrorHtml(`Failed to load schedule: ${error.message}`, 'loadScheduleJobs');
        }
    }

    function _timeUntil(ms) {
        const mins = Math.floor(ms / 60000);
        if (mins < 60) return `in ${mins} minute${mins !== 1 ? 's' : ''}`;
        const hours = Math.floor(mins / 60);
        return `in ${hours} hour${hours !== 1 ? 's' : ''}`;
    }

    function renderComingUp(jobs) {
        const container = $('#schedule-coming-up-list');
        if (!container) return;
        const now = Date.now();
        const in24h = now + 24 * 60 * 60 * 1000;
        const upcoming = jobs
            .filter(job => {
                if (!job.next_run) return false;
                const t = new Date(job.next_run).getTime();
                return t >= now && t <= in24h;
            })
            .sort((a, b) => new Date(a.next_run).getTime() - new Date(b.next_run).getTime());
        if (upcoming.length === 0) {
            container.innerHTML = '<div class="schedule-empty">Nothing due in the next 24 hours.</div>';
            return;
        }
        container.innerHTML = upcoming.map(job => {
            const ms = new Date(job.next_run).getTime() - now;
            const timeStr = _timeUntil(ms);
            return `<div class="schedule-coming-up-item">
                <span class="schedule-coming-up-name">${escapeHtml(job.name)}</span>
                <span class="schedule-coming-up-time">${timeStr}</span>
            </div>`;
        }).join('');
    }

    function renderScheduleJobs(jobs) {
        const container = $('#schedule-list');
        if (!container) return;
        if (jobs.length === 0) {
            container.innerHTML = '<div class="schedule-empty">No items scheduled.</div>';
            return;
        }
        container.innerHTML = jobs.map(job => `
            <div class="schedule-item" data-id="${escapeHtml(job.job_id)}">
                <div class="schedule-item-header">
                    <span class="schedule-item-name">${escapeHtml(job.name)}</span>
                    <span class="schedule-item-status ${job.enabled ? 'enabled' : 'disabled'}">
                        ${job.enabled ? 'Enabled' : 'Disabled'}
                    </span>
                </div>
                <div class="schedule-item-details">
                    <p><strong>Schedule:</strong> ${formatSchedule(job.schedule)}</p>
                    <p><strong>Next run:</strong> ${job.next_run ? new Date(job.next_run).toLocaleString() : 'Not scheduled'}</p>
                </div>
                <div class="schedule-item-actions">
                    <button class="btn btn-secondary btn-sm schedule-toggle-btn"
                        data-job-id="${escapeHtml(job.job_id)}"
                        data-enabled="${job.enabled}"
                        onclick="window.dashboardHandlers.toggleScheduleJob('${escapeHtml(job.job_id)}', ${job.enabled})">
                        ${job.enabled ? 'Disable' : 'Enable'}
                    </button>
                </div>
            </div>
        `).join('');
    }

    async function toggleScheduleJob(jobId, currentEnabled) {
        const newEnabled = !currentEnabled;
        // Optimistic UI update
        const item = $(`[data-id="${jobId}"]`);
        const statusEl = item ? item.querySelector('.schedule-item-status') : null;
        const toggleBtn = item ? item.querySelector('.schedule-toggle-btn') : null;
        if (statusEl) {
            statusEl.className = `schedule-item-status ${newEnabled ? 'enabled' : 'disabled'}`;
            statusEl.textContent = newEnabled ? 'Enabled' : 'Disabled';
        }
        if (toggleBtn) {
            toggleBtn.dataset.enabled = String(newEnabled);
            toggleBtn.setAttribute('onclick', `window.dashboardHandlers.toggleScheduleJob('${jobId}', ${newEnabled})`);
            toggleBtn.textContent = newEnabled ? 'Disable' : 'Enable';
        }
        try {
            await api(`/api/scheduler/jobs/${jobId}`, {
                method: 'PATCH',
                body: JSON.stringify({ enabled: newEnabled }),
            });
        } catch (error) {
            // Revert on failure
            if (statusEl) {
                statusEl.className = `schedule-item-status ${currentEnabled ? 'enabled' : 'disabled'}`;
                statusEl.textContent = currentEnabled ? 'Enabled' : 'Disabled';
            }
            if (toggleBtn) {
                toggleBtn.dataset.enabled = String(currentEnabled);
                toggleBtn.setAttribute('onclick', `window.dashboardHandlers.toggleScheduleJob('${jobId}', ${currentEnabled})`);
                toggleBtn.textContent = currentEnabled ? 'Disable' : 'Enable';
            }
            const errorEl = document.createElement('div');
            errorEl.className = 'schedule-toggle-error error-message';
            errorEl.textContent = `Failed to update: ${error.message}`;
            if (item) {
                item.appendChild(errorEl);
                setTimeout(() => errorEl.remove(), 4000);
            }
        }
    }

    async function runReminder(jobId) {
        try {
            await api(`/api/scheduler/jobs/${jobId}/run`, { method: 'POST' });
            loadReminders();
        } catch (error) {
            alert(`Failed to run reminder: ${error.message}`);
        }
    }

    async function toggleReminder(jobId, enabled) {
        try {
            await api(`/api/scheduler/jobs/${jobId}`, {
                method: 'PATCH',
                body: JSON.stringify({ enabled }),
            });
            loadReminders();
        } catch (error) {
            alert(`Failed to update reminder: ${error.message}`);
        }
    }

    async function deleteReminder(jobId) {
        if (!confirm('Are you sure you want to delete this reminder?')) return;

        try {
            await api(`/api/scheduler/jobs/${jobId}`, { method: 'DELETE' });
            loadReminders();
        } catch (error) {
            alert(`Failed to delete reminder: ${error.message}`);
        }
    }

    function showReminderModal() {
        show('#reminder-modal');
        $('#reminder-name').value = '';
        $('#reminder-type').value = 'interval';
        $('#reminder-interval').value = '3600';
        $('#reminder-time').value = '09:00';
        $('#reminder-enabled').checked = true;
        updateReminderFormFields();
    }

    function hideReminderModal() {
        hide('#reminder-modal');
    }

    function updateReminderFormFields() {
        const type = $('#reminder-type').value;
        if (type === 'interval') {
            show('#interval-fields');
            hide('#at-fields');
        } else {
            hide('#interval-fields');
            show('#at-fields');
        }
    }

    async function createReminder(event) {
        event.preventDefault();

        const name = $('#reminder-name').value.trim();
        const type = $('#reminder-type').value;
        const enabled = $('#reminder-enabled').checked;

        let schedule;
        if (type === 'interval') {
            const interval = parseInt($('#reminder-interval').value, 10) || 3600;
            schedule = `interval:${Math.max(60, interval)}`;
        } else {
            const time = $('#reminder-time').value || '09:00';
            schedule = `at:${time}`;
        }

        try {
            await api('/api/scheduler/jobs', {
                method: 'POST',
                body: JSON.stringify({ name, schedule, enabled }),
            });

            hideReminderModal();
            loadReminders();
        } catch (error) {
            alert(`Failed to create reminder: ${error.message}`);
        }
    }

    // --- Notifications functionality ---

    function _buildNotifParams() {
        const params = new URLSearchParams();
        params.set('limit', '200');
        if (state.notifFilters.unread) {
            params.set('unread', 'true');
        }
        if (state.notifFilters.priority) {
            params.set('priority', state.notifFilters.priority);
        }
        return params.toString();
    }

    async function loadNotifications() {
        const container = $('#notif-list');
        if (!container) return;
        container.innerHTML = '<div class="loading">Loading notifications...</div>';

        try {
            const qs = _buildNotifParams();
            const data = await api(`/api/notifications?${qs}`);
            state.notifications = data.notifications || [];
            renderNotifications(state.notifications, data.unread_count);
            updateNotifBadge(data.unread_count);
        } catch (error) {
            container.innerHTML = _retryErrorHtml(`Failed to load notifications: ${error.message}`, 'loadNotifications');
        }
    }

    function renderNotifications(notifications, unreadCount) {
        const container = $('#notif-list');
        const summaryEl = $('#notif-summary');
        if (!container) return;

        // Apply client-side channel filter
        const chanFilter = state.notifFilters.channel;
        const filtered = chanFilter
            ? notifications.filter(n => n.channel === chanFilter)
            : notifications;

        // Update summary
        if (summaryEl) {
            summaryEl.textContent = unreadCount > 0
                ? `${unreadCount} unread notification${unreadCount !== 1 ? 's' : ''}`
                : 'No unread notifications';
        }

        if (filtered.length === 0) {
            container.innerHTML = '<div class="notif-empty">No notifications to display.</div>';
            return;
        }

        container.innerHTML = filtered.map(n => {
            const ts = n.timestamp ? new Date(n.timestamp).toLocaleString() : '';
            const readClass = n.read ? 'read' : 'unread';
            const priorityClass = `notif-priority-${escapeHtml(n.priority || 'normal')}`;
            return `
                <div class="notif-item ${readClass}" data-id="${escapeHtml(n.id)}">
                    <div class="notif-item-header">
                        <span class="notif-priority-badge ${priorityClass}">${escapeHtml(n.priority || 'normal')}</span>
                        <span class="notif-channel">${escapeHtml(n.channel || '')}</span>
                        <span class="notif-timestamp">${escapeHtml(ts)}</span>
                        ${!n.read ? `
                            <button class="btn btn-secondary btn-sm notif-mark-read-btn"
                                onclick="window.dashboardHandlers.markNotifRead('${escapeHtml(n.id)}')">
                                Mark read
                            </button>
                        ` : '<span class="notif-read-label">Read</span>'}
                    </div>
                    <div class="notif-item-title">${escapeHtml(n.title)}</div>
                    <div class="notif-item-body">${escapeHtml(n.body)}</div>
                </div>
            `;
        }).join('');
    }

    function updateNotifBadge(unreadCount) {
        const badges = [$('#notif-badge'), $('#notif-badge-mobile')].filter(Boolean);
        if (badges.length === 0) return;

        badges.forEach((badge) => {
            if (unreadCount > 0) {
                badge.textContent = unreadCount > 99 ? '99+' : String(unreadCount);
                show(badge);
            } else {
                hide(badge);
            }
        });
    }

    async function markNotifRead(notifId) {
        try {
            await api(`/api/notifications/${notifId}/read`, { method: 'POST' });
            loadNotifications();
        } catch (error) {
            alert(`Failed to mark notification as read: ${error.message}`);
        }
    }

    async function markAllNotifsRead() {
        try {
            await api('/api/notifications/read-all', { method: 'POST' });
            loadNotifications();
        } catch (error) {
            alert(`Failed to mark all notifications as read: ${error.message}`);
        }
    }

    // Real-time notifications using SSE, with limited reconnect and polling fallback
    function startNotifRealtime() {
        stopNotifPolling();
        stopNotifStream();
        stopNotifReconnect();
        _sseReconnectCount = 0;

        if (!window.EventSource) {
            startNotifPolling();
            return;
        }

        _connectNotifStream();
    }

    function _connectNotifStream() {
        stopNotifStream();

        // Server supports query token for this endpoint only.
        // If token is missing, the server may still accept cookie auth.
        const tokenQS = state.token ? `?token=${encodeURIComponent(state.token)}` : '';
        const url = `/api/notifications/stream${tokenQS}`;

        try {
            _notifEventSource = new EventSource(url, { withCredentials: true });

            _notifEventSource.addEventListener('init', (event) => {
                // Successful connection — reset reconnect counter
                _sseReconnectCount = 0;
                _setNotifStreamStatus('');
                try {
                    const data = JSON.parse(event.data || '{}');
                    updateNotifBadge(data.unread_count || 0);
                } catch (_) {
                    // ignore malformed init payload
                }
            });

            _notifEventSource.addEventListener('notification', () => {
                if (state.currentSection === 'notifications') {
                    loadNotifications();
                }
            });

            _notifEventSource.onerror = () => {
                stopNotifStream();
                _sseReconnectCount += 1;

                if (_sseReconnectCount <= _SSE_MAX_RECONNECT) {
                    // Exponential backoff: 2, 4, 8, 16, 32 seconds
                    const delayMs = Math.pow(2, _sseReconnectCount) * 1000;
                    _setNotifStreamStatus(
                        `Connection lost. Reconnecting in ${Math.round(delayMs / 1000)}s… (attempt ${_sseReconnectCount}/${_SSE_MAX_RECONNECT})`
                    );
                    _sseReconnectTimer = setTimeout(() => {
                        _sseReconnectTimer = null;
                        _connectNotifStream();
                    }, delayMs);
                } else {
                    // Max reconnect attempts reached — fall back to polling
                    _setNotifStreamStatus('');
                    startNotifPolling();
                }
            };
        } catch (e) {
            stopNotifStream();
            startNotifPolling();
        }
    }

    function _setNotifStreamStatus(message) {
        const el = $('#notif-summary');
        if (!el) return;
        if (message) {
            el.textContent = message;
        }
        // Don't clear the element when message is empty — loadNotifications() will update it
    }

    function startNotifPolling() {
        stopNotifPolling();
        _notifPollTimer = setInterval(() => {
            if (state.currentSection === 'notifications') {
                loadNotifications();
            }
        }, 30000); // poll every 30 seconds
    }

    function stopNotifPolling() {
        if (_notifPollTimer !== null) {
            clearInterval(_notifPollTimer);
            _notifPollTimer = null;
        }
    }

    function stopNotifStream() {
        if (_notifEventSource) {
            _notifEventSource.close();
            _notifEventSource = null;
        }
    }

    function stopNotifReconnect() {
        if (_sseReconnectTimer !== null) {
            clearTimeout(_sseReconnectTimer);
            _sseReconnectTimer = null;
        }
    }

    // Status functionality
    async function loadStatus() {
        const container = $('#status-container');
        container.innerHTML = '<div class="loading">Loading status...</div>';

        try {
            const [status, schedulerData] = await Promise.all([
                api('/api/dashboard/status'),
                api('/api/scheduler/jobs'),
            ]);

            const metrics = schedulerData.metrics || {};

            container.innerHTML = `
                <div class="status-card">
                    <h3>Status</h3>
                    <div class="value ${status.status === 'ok' ? 'ok' : ''}">${status.status.toUpperCase()}</div>
                </div>
                <div class="status-card">
                    <h3>Uptime</h3>
                    <div class="value">${formatUptime(status.uptime_seconds)}</div>
                </div>
                <div class="status-card">
                    <h3>Version</h3>
                    <div class="value">${escapeHtml(status.version)}</div>
                </div>
                <div class="status-card">
                    <h3>Authentication</h3>
                    <div class="value">${status.auth_enabled ? 'Required' : 'Local Only'}</div>
                </div>
                <div class="status-card">
                    <h3>Scheduled Jobs</h3>
                    <div class="value">${metrics.total_jobs || 0} (${metrics.enabled_jobs || 0} active)</div>
                </div>
                <div class="status-card">
                    <h3>Job Runs (Total)</h3>
                    <div class="value">${metrics.total_runs || 0}</div>
                </div>
                <div class="status-card">
                    <h3>Server Time</h3>
                    <div class="value" style="font-size: 1rem;">${new Date(status.server_time).toLocaleString()}</div>
                </div>
            `;
        } catch (error) {
            container.innerHTML = _retryErrorHtml(`Failed to load status: ${error.message}`, 'loadStatus');
        }
    }

    function formatUptime(seconds) {
        const days = Math.floor(seconds / 86400);
        const hours = Math.floor((seconds % 86400) / 3600);
        const minutes = Math.floor((seconds % 3600) / 60);

        if (days > 0) return `${days}d ${hours}h ${minutes}m`;
        if (hours > 0) return `${hours}h ${minutes}m`;
        return `${minutes}m`;
    }

    // Utility functions
    function escapeHtml(str) {
        if (str === null || str === undefined) return '';
        const div = document.createElement('div');
        div.textContent = String(str);
        return div.innerHTML;
    }

    // Global error display
    function showGlobalError(message) {
        const banner = $('#global-error-banner');
        const msg = $('#global-error-msg');
        if (!banner || !msg) return;
        msg.textContent = message;
        banner.classList.remove('hidden');
    }

    function hideGlobalError() {
        const banner = $('#global-error-banner');
        if (banner) banner.classList.add('hidden');
    }

    // Voice mode toggle (Voice panel)

    let _voiceModeEs = null;

    function _clearVoiceTranscript() {
        const inputEl = $('#voice-input-transcript');
        const responseEl = $('#voice-response-transcript');
        if (inputEl) inputEl.textContent = '';
        if (responseEl) responseEl.textContent = '';
    }

    async function _fetchVoiceTranscript() {
        try {
            const data = await api('/api/voice/transcript');
            const inputEl = $('#voice-input-transcript');
            const responseEl = $('#voice-response-transcript');
            if (inputEl) inputEl.textContent = data.input || '';
            if (responseEl) responseEl.textContent = data.response || '';
        } catch (err) {
            console.error('Voice transcript fetch error:', err);
        }
    }

    function _updateVoiceModeUI(stateData) {
        const btn = $('#voice-mode-btn');
        const label = $('#voice-state-label');
        if (!btn || !label) return;
        const active = stateData.active;
        const stateName = stateData.state || (active ? 'Listening' : 'Idle');
        btn.textContent = active ? 'Stop Listening' : 'Start Listening';
        btn.classList.toggle('listening', active);
        label.textContent = stateName;
        const waveform = $('#voice-waveform');
        if (waveform) {
            waveform.classList.toggle('hidden', stateName !== 'Listening');
        }
        if (stateName === 'Listening') {
            _clearVoiceTranscript();
        } else if (stateName === 'Idle' && !active) {
            _fetchVoiceTranscript();
        }
    }

    async function toggleVoiceMode() {
        const btn = $('#voice-mode-btn');
        if (!btn) return;
        const currentlyActive = btn.textContent.trim() === 'Stop Listening';
        try {
            const data = await api('/api/voice/mode', {
                method: 'POST',
                body: JSON.stringify({ active: !currentlyActive }),
            });
            _updateVoiceModeUI(data);
        } catch (err) {
            console.error('Voice mode toggle error:', err);
        }
    }

    function startVoiceStateStream() {
        if (_voiceModeEs) return; // already open
        const tokenParam = localStorage.getItem('rex_auth_token') ? `?token=${encodeURIComponent(localStorage.getItem('rex_auth_token'))}` : '';
        _voiceModeEs = new EventSource('/api/voice/state/stream' + tokenParam);
        _voiceModeEs.addEventListener('state', (e) => {
            try {
                const data = JSON.parse(e.data);
                _updateVoiceModeUI(data);
            } catch (_) {}
        });
        _voiceModeEs.onerror = () => {
            if (_voiceModeEs) { _voiceModeEs.close(); _voiceModeEs = null; }
        };
    }

    function stopVoiceStateStream() {
        if (_voiceModeEs) { _voiceModeEs.close(); _voiceModeEs = null; }
    }

    // Voice interface

    function setVoiceStatus(text, isRecording) {
        const statusEl = $('#voice-status');
        const voiceBtn = $('#voice-btn');
        if (!statusEl || !voiceBtn) return;

        if (text) {
            statusEl.textContent = text;
            statusEl.classList.remove('hidden');
        } else {
            statusEl.classList.add('hidden');
        }

        voiceBtn.textContent = isRecording ? 'Stop' : 'Mic';
        voiceBtn.classList.toggle('recording', isRecording);
    }

    function speakReply(text) {
        if (!text || !window.speechSynthesis) return;
        const utterance = new SpeechSynthesisUtterance(text);
        window.speechSynthesis.speak(utterance);
    }

    async function startVoiceRecording() {
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            alert('Microphone not supported in this browser.');
            return;
        }

        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            voiceState.chunks = [];
            voiceState.mediaRecorder = new MediaRecorder(stream);

            voiceState.mediaRecorder.addEventListener('dataavailable', (e) => {
                if (e.data.size > 0) voiceState.chunks.push(e.data);
            });

            voiceState.mediaRecorder.addEventListener('stop', async () => {
                // Stop all tracks so the mic indicator clears
                stream.getTracks().forEach(t => t.stop());

                const blob = new Blob(voiceState.chunks, { type: voiceState.mediaRecorder.mimeType || 'audio/webm' });
                voiceState.chunks = [];
                voiceState.recording = false;

                setVoiceStatus('Transcribing...', false);

                try {
                    const formData = new FormData();
                    formData.append('audio', blob, 'recording.webm');

                    const headers = {};
                    if (state.token) headers['Authorization'] = `Bearer ${state.token}`;

                    const response = await fetch('/api/voice', {
                        method: 'POST',
                        headers,
                        credentials: 'same-origin',
                        body: formData,
                    });

                    const data = await response.json();

                    if (!response.ok) {
                        setVoiceStatus(`Error: ${data.error || 'Voice request failed'}`, false);
                        return;
                    }

                    setVoiceStatus('', false);

                    // Display in chat
                    const container = $('#chat-messages');
                    const placeholder = container.querySelector('.chat-placeholder');
                    if (placeholder) placeholder.remove();

                    container.innerHTML += `<div class="chat-message user">${escapeHtml(data.transcript)}</div>`;
                    container.innerHTML += `<div class="chat-message assistant">${escapeHtml(data.reply)}</div>`;
                    container.scrollTop = container.scrollHeight;

                    state.chatHistory.push({
                        user_message: data.transcript,
                        assistant_reply: data.reply,
                        timestamp: data.timestamp,
                    });

                    // Speak the reply aloud
                    speakReply(data.reply);

                } catch (err) {
                    setVoiceStatus(`Error: ${err.message}`, false);
                }
            });

            voiceState.mediaRecorder.start();
            voiceState.recording = true;
            setVoiceStatus('Recording... (click Mic to stop)', true);

        } catch (err) {
            setVoiceStatus(`Microphone error: ${err.message}`, false);
        }
    }

    function stopVoiceRecording() {
        if (voiceState.mediaRecorder && voiceState.mediaRecorder.state !== 'inactive') {
            voiceState.mediaRecorder.stop();
        }
    }

    function handleVoiceBtnClick() {
        if (voiceState.recording) {
            stopVoiceRecording();
        } else {
            startVoiceRecording();
        }
    }

    // Initialize
    async function init() {
        // Check for existing session
        const savedToken = localStorage.getItem('rex_dashboard_token');
        if (savedToken) {
            state.token = savedToken;

            try {
                // Verify token is still valid
                await api('/api/dashboard/status');
                state.authenticated = true;
                showDashboard();
            } catch {
                // Token invalid, show login
                showLogin();
            }
        } else {
            // Check if auth is required
            try {
                const status = await fetch('/api/dashboard/status').then(r => r.json());
                if (!status.auth_enabled) {
                    // Try auto-login for local access
                    try {
                        const data = await api('/api/dashboard/login', {
                            method: 'POST',
                            body: JSON.stringify({}),
                        });
                        state.token = data.token;
                        state.authenticated = true;
                        localStorage.setItem('rex_dashboard_token', data.token);
                        showDashboard();
                        return;
                    } catch {
                        // Fall through to login screen
                    }
                }
            } catch {
                // Fall through to login screen
            }
            showLogin();
        }

        // Global error banner close button
        const closeBtn = $('#global-error-close');
        if (closeBtn) closeBtn.addEventListener('click', hideGlobalError);

        // Capture uncaught JS errors and display them in the global banner
        window.addEventListener('error', (event) => {
            const message = (event.error && event.error.message) || event.message || 'Unknown error';
            console.error('Uncaught error:', message);
            showGlobalError(`Error: ${message}`);
        });

        // Capture unhandled promise rejections
        window.addEventListener('unhandledrejection', (event) => {
            const reason = event.reason;
            const message = reason instanceof Error ? reason.message : String(reason || 'Unknown error');
            console.error('Unhandled rejection:', message);
            showGlobalError(`Error: ${message}`);
        });

        // Set up event listeners
        $('#login-form').addEventListener('submit', handleLogin);
        $('#logout-btn').addEventListener('click', handleLogout);

        // Navigation
        $$('.nav-link, .mobile-nav-link').forEach(link => {
            link.addEventListener('click', (e) => {
                e.preventDefault();
                switchSection(link.dataset.section);
            });
        });

        // Back/forward browser navigation
        window.addEventListener('popstate', (e) => {
            const section = (e.state && e.state.section)
                || window.location.hash.slice(1)
                || 'chat';
            if (VALID_SECTIONS.includes(section)) {
                _navigatingFromHistory = true;
                switchSection(section);
                _navigatingFromHistory = false;
            }
        });

        // Chat
        $('#chat-form').addEventListener('submit', handleChatSubmit);
        $('#voice-btn').addEventListener('click', handleVoiceBtnClick);

        // Voice mode toggle (Voice panel)
        const voiceModeBtn = $('#voice-mode-btn');
        if (voiceModeBtn) voiceModeBtn.addEventListener('click', toggleVoiceMode);

        // Settings
        $('#settings-search').addEventListener('input', (e) => {
            if (state.settings) {
                renderSettings(state.settings, {}, e.target.value);
            }
        });
        $('#save-settings-btn').addEventListener('click', saveSettings);
        $('#discard-settings-btn').addEventListener('click', discardSettings);

        // Reminders
        $('#add-reminder-btn').addEventListener('click', showReminderModal);
        $$('.modal-close').forEach(btn => btn.addEventListener('click', hideReminderModal));
        $('#reminder-type').addEventListener('change', updateReminderFormFields);
        $('#reminder-form').addEventListener('submit', createReminder);

        // Close modal on backdrop click
        $('#reminder-modal').addEventListener('click', (e) => {
            if (e.target.id === 'reminder-modal') hideReminderModal();
        });

        // Notifications filters
        $('#notif-filter-unread').addEventListener('change', (e) => {
            state.notifFilters.unread = e.target.checked;
            loadNotifications();
        });
        $('#notif-filter-priority').addEventListener('change', (e) => {
            state.notifFilters.priority = e.target.value;
            loadNotifications();
        });
        $('#notif-filter-channel').addEventListener('change', (e) => {
            state.notifFilters.channel = e.target.value;
            // Channel filter is applied client-side; no API call needed, just re-render
            const unreadCount = state.notifications.filter(n => !n.read).length;
            renderNotifications(state.notifications, unreadCount);
        });
        $('#mark-all-read-btn').addEventListener('click', markAllNotifsRead);
    }

    // Expose handlers for inline onclick
    window.dashboardHandlers = {
        settingChanged: handleSettingChanged,
        runReminder,
        toggleReminder,
        deleteReminder,
        markNotifRead,
        toggleScheduleJob,
        navigateTo: switchSection,
        loadOverview,
        loadChatHistory,
        loadSettings,
        loadReminders,
        loadScheduleJobs,
        loadNotifications,
        loadStatus,
    };

    // Start app
    document.addEventListener('DOMContentLoaded', init);
})();
