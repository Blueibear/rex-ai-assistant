# PRD: Rex GUI, Autonomy Engine Overhaul, and Integration Completion

> **IMPORTANT — Ralph loop instructions:**
> Read this file at the start of every iteration. Find the first unchecked story.
> Implement it completely. Run `npm run typecheck` (GUI stories) or `python -m mypy` (Python stories) before committing.
> Commit using Conventional Commits format: `<type>(<scope>): <subject>` — e.g. `feat(gui): add chat message list component`.
> Mark the story's checkbox as `[x]` in this file and commit that change in the same commit.
> Then stop. Do not start the next story.

---

## Introduction

This PRD covers three major workstreams for Rex:

1. **Full Desktop GUI** — A sleek, modern Electron + React desktop application (TypeScript throughout) that exposes every Rex capability through a polished interface: text chat, voice chat, task scheduling, calendar, reminders, user memory editing, and all settings panels.

2. **Autonomy Engine Overhaul** — Replace the current rule-based planner with a fully LLM-driven engine. Adds dynamic replanning on failure, execution history learning, multi-goal dependency resolution, API cost optimization, and user preference learning.

3. **Beta Integration Completion** — Complete the email/calendar, SMS, and smart notifications subsystems with full data models, credential-ready service layers, and GUI views. Without credentials, all three run on realistic mock/stub data. Supplying real credentials (Gmail, Twilio, etc.) activates live mode transparently.

Story numbering continues from the existing PRD ecosystem (previous PRDs end at US-174).

---

## Goals

- Ship a cross-platform Electron + React GUI that lets users do everything Rex can do without touching the terminal.
- Replace rule-based autonomy with flexible LLM-driven planning that handles ambiguous goals, adapts to failures, and learns over time.
- Complete email/calendar, SMS, and notification integrations with production-ready service interfaces and credential-ready stubs.
- Maintain full type safety throughout; all new code passes typecheck.
- Each story is completable by a single AI agent in one context window (~10 minutes of work).

---

## Non-Goals

- No mobile app (iOS/Android) — desktop only.
- No cloud sync or remote deployment — Rex remains local-first.
- No real email sending or SMS delivery in stub mode — stubs return mock data only.
- No OAuth flow UI for third-party integrations (credentials entered manually in settings).
- No multi-user or multi-profile support.
- No autonomy stories implement external tool integrations beyond what already exists in Rex.

---

## Technical Considerations

- **GUI Stack:** Electron + React 18 + TypeScript + Vite (`electron-vite` CLI). Renderer uses `react-router-dom` in hash mode. State via Zustand. Styling via Tailwind CSS + CSS custom properties for design tokens.
- **IPC Pattern:** All renderer-to-backend calls go through a typed `contextBridge` preload. Main process manages the Python subprocess. Communication is via Electron IPC (not HTTP) for security.
- **Python Backend:** Existing Rex Python backend remains unchanged in structure. New services (autonomy, email, SMS, notifications) are added as Python modules following the existing pattern.
- **Type Safety:** TypeScript strict mode for the GUI; `mypy --strict` for all new Python modules.
- **Testing:** `pytest` for Python; `vitest` for TypeScript logic; Playwright for E2E where noted.
- **Design Language:** Dark-first, minimal chrome. Accent color: electric blue (`#3B82F6`). Font: Inter. Sidebar navigation. Frosted-glass panels where depth is needed.

---

## User Stories

---

### Phase 61 — Electron + React Scaffold

---

### US-175: Bootstrap Electron + React + TypeScript app
**Description:** As a developer, I need a working Electron shell with a React + TypeScript renderer so all GUI development has a runnable foundation.

**Acceptance Criteria:**
- [x] Run `npx electron-vite create rex-gui --template react-ts` (or equivalent) to scaffold the project inside the repo at `gui/`.
- [x] `gui/package.json` includes `electron`, `electron-vite`, `react`, `react-dom`, `typescript`, `tailwindcss` as dependencies.
- [x] `npm run dev` (inside `gui/`) launches the Electron window showing a React root component with text "Rex is starting…".
- [x] `npm run build` produces a distributable in `gui/dist-electron/`.
- [x] `npm run typecheck` passes with zero errors.
- [x] `.gitignore` updated to exclude `gui/dist-electron/`, `gui/node_modules/`, `gui/dist/`.

---

### US-176: Set up typed IPC bridge between renderer and main process
**Description:** As a developer, I need a secure, typed IPC bridge so React components can call Rex backend functions without direct Node access in the renderer.

**Acceptance Criteria:**
- [x] `gui/src/preload/index.ts` defines `contextBridge.exposeInMainWorld('rex', { ... })` with at least: `sendChat`, `getStatus`, `getSettings`, `setSettings`.
- [x] `gui/src/types/ipc.ts` exports TypeScript interfaces for every method exposed on `window.rex`.
- [x] `gui/src/main/index.ts` registers `ipcMain.handle` listeners for each method (stubs returning `{ ok: true }` for now).
- [x] Renderer can call `window.rex.getStatus()` and receive the stub response without errors.
- [x] `npm run typecheck` passes.

---

### US-177: Configure Tailwind CSS and design token CSS variables
**Description:** As a developer, I need Tailwind CSS configured and a set of CSS custom properties for Rex's design language so all components use consistent colors, spacing, and typography.

**Acceptance Criteria:**
- [x] `tailwind.config.ts` configured with `content: ['./src/**/*.{ts,tsx}']` and custom theme extension.
- [x] `gui/src/styles/tokens.css` defines CSS custom properties: `--color-bg`, `--color-surface`, `--color-surface-raised`, `--color-accent`, `--color-text-primary`, `--color-text-secondary`, `--color-border`, `--color-danger`, `--color-success`. Values implement a dark theme (background `#0F1117`, surface `#1A1D27`, accent `#3B82F6`).
- [x] `tokens.css` imported in `gui/src/styles/index.css` which is imported in the app root.
- [x] Inter font loaded (via `@fontsource/inter` or CSS import).
- [x] `npm run typecheck` passes.
- [x] Verify changes work in app.

---

### Phase 62 — Design System Components

---

### US-178: Build core interactive UI components
**Description:** As a developer, I need reusable Button, Input, Textarea, and Badge components that follow Rex's design tokens so every view is visually consistent.

**Acceptance Criteria:**
- [x] `gui/src/components/ui/Button.tsx` — variants: `primary`, `secondary`, `ghost`, `danger`. Sizes: `sm`, `md`, `lg`. Accepts `loading` boolean (shows spinner, disables click).
- [x] `gui/src/components/ui/Input.tsx` — text input with `label`, `error`, `helperText` props. Applies focus ring in accent color.
- [x] `gui/src/components/ui/Textarea.tsx` — auto-grows up to 6 lines, same styling as Input.
- [x] `gui/src/components/ui/Badge.tsx` — variants: `default`, `accent`, `success`, `warning`, `danger`. Small pill shape.
- [x] All components export TypeScript props interfaces.
- [x] `npm run typecheck` passes.
- [x] Verify changes work in app.

---

### US-179: Build layout and container UI components
**Description:** As a developer, I need Card, Modal, Tooltip, and Divider components so content areas have consistent structure and depth.

**Acceptance Criteria:**
- [x] `gui/src/components/ui/Card.tsx` — surface-colored box with `padding`, `hoverable` (subtle scale on hover), and optional `header` slot.
- [x] `gui/src/components/ui/Modal.tsx` — centered overlay with `title`, `children`, `footer` slots. `onClose` prop. Closes on Escape key. Trap focus inside modal.
- [x] `gui/src/components/ui/Tooltip.tsx` — wraps any element, shows text tooltip on hover after 300ms delay. Position: `top | bottom | left | right`.
- [x] `gui/src/components/ui/Divider.tsx` — horizontal rule using `--color-border` with optional label text.
- [x] `npm run typecheck` passes.
- [x] Verify changes work in app.

---

### US-180: Build feedback and status UI components
**Description:** As a developer, I need Spinner, SkeletonLine, EmptyState, and Toast components so the app communicates loading and error states clearly.

**Acceptance Criteria:**
- [x] `gui/src/components/ui/Spinner.tsx` — animated SVG ring. Sizes: `sm`, `md`, `lg`. Color inherits or uses accent.
- [x] `gui/src/components/ui/SkeletonLine.tsx` — animated shimmer placeholder. Accepts `width` and `height` props.
- [x] `gui/src/components/ui/EmptyState.tsx` — centered icon + heading + subtext + optional action button.
- [x] `gui/src/components/ui/Toast.tsx` — slide-in notification at bottom-right. Types: `info`, `success`, `warning`, `error`. Auto-dismisses after 4s. `useToast` hook for triggering from anywhere.
- [x] `npm run typecheck` passes.
- [x] Verify changes work in app.

---

### Phase 63 — App Shell and Navigation

---

### US-181: Scaffold main window layout with sidebar and content area
**Description:** As a user, I want a persistent sidebar navigation and main content area so I can move between Rex's features without losing context.

**Acceptance Criteria:**
- [x] `gui/src/layouts/AppLayout.tsx` renders a fixed left sidebar (240px wide) and a scrollable main content panel.
- [x] Sidebar has a Rex logo/wordmark at the top and a bottom section for user avatar and settings shortcut.
- [x] Main content area has a topbar showing the current section name.
- [x] Layout is responsive: sidebar collapses to icon-only at window width < 900px.
- [x] `npm run typecheck` passes.
- [x] Verify changes work in app.

---

### US-182: Implement client-side routing for all Rex sections
**Description:** As a user, I want navigation between sections to update the content area without full reloads so the app feels instant.

**Acceptance Criteria:**
- [x] `react-router-dom` (v6+) installed and configured with `HashRouter` (required for Electron).
- [x] Routes defined: `/chat`, `/voice`, `/tasks`, `/calendar`, `/reminders`, `/memories`, `/email`, `/sms`, `/notifications`, `/settings`.
- [x] Each route renders a placeholder page component with its section title.
- [x] Default route (`/`) redirects to `/chat`.
- [x] Active route highlighted in sidebar.
- [x] `npm run typecheck` passes.
- [x] Verify changes work in app.

---

### US-183: Build sidebar navigation with icons and notification badges
**Description:** As a user, I want labeled navigation items with icons so I can identify and reach any section at a glance.

**Acceptance Criteria:**
- [x] Sidebar nav items: Chat, Voice, Tasks, Calendar, Reminders, Memories, Email (beta badge), SMS (beta badge), Notifications, Settings.
- [x] Each item has an SVG icon (lucide-react or custom inline SVG).
- [x] Items marked `(beta)` show a small amber "BETA" badge.
- [x] Notifications item shows a numeric unread badge (fed by Zustand store, placeholder value of 0 for now).
- [x] Clicking any item navigates to its route and applies the active highlight style.
- [x] `npm run typecheck` passes.
- [x] Verify changes work in app.

---

### Phase 64 — Text Chat Panel

---

### US-184: Build chat message list component
**Description:** As a user, I want to see a scrolling list of my conversation with Rex so I can follow the dialogue and refer back to earlier messages.

**Acceptance Criteria:**
- [x] `gui/src/components/chat/MessageList.tsx` renders a list of message objects (`{ id, role: 'user'|'rex', content, timestamp }`).
- [x] User messages appear right-aligned with accent bubble; Rex messages left-aligned with surface bubble.
- [x] Timestamps shown in relative format ("2 min ago") below each bubble.
- [x] List auto-scrolls to bottom on new message.
- [x] Supports Markdown rendering in Rex messages (bold, code blocks, lists).
- [x] `npm run typecheck` passes.
- [x] Verify changes work in app.

---

### US-185: Build chat input bar
**Description:** As a user, I want a text input at the bottom of the chat panel so I can type and send messages to Rex.

**Acceptance Criteria:**
- [x] `gui/src/components/chat/ChatInput.tsx` renders the Textarea component with a Send button.
- [x] Send triggered by Enter key (Shift+Enter inserts newline).
- [x] Send button disabled when input is empty or when `sending` prop is true (shows Spinner).
- [x] Input cleared after send.
- [x] Character count shown when message exceeds 200 characters.
- [x] `npm run typecheck` passes.
- [x] Verify changes work in app.

---

### US-186: Wire chat panel to Rex backend via IPC
**Description:** As a user, I want my chat messages sent to Rex's actual AI backend so I get real responses, not stubs.

**Acceptance Criteria:**
- [x] `window.rex.sendChat(message: string): Promise<string>` IPC method implemented end-to-end: renderer calls it → main process forwards to Rex Python backend → returns response string.
- [x] `gui/src/main/handlers/chat.ts` contains the `ipcMain.handle('rex:sendChat', ...)` handler.
- [x] Chat page (`/chat`) uses the handler; sending a message appends the user bubble, calls IPC, appends Rex's response bubble.
- [x] Error from backend shown as an inline error bubble (not a crash).
- [x] `npm run typecheck` passes.
- [x] Verify changes work in app.

---

### US-187: Add streaming response display to chat
**Description:** As a user, I want Rex's response to appear word-by-word as it's generated so the chat feels live and responsive.

**Acceptance Criteria:**
- [x] `window.rex.sendChatStream(message: string, onToken: (token: string) => void): Promise<void>` IPC method added.
- [x] Main process streams tokens from Rex Python backend (via stdout chunks or SSE) and emits each token back to renderer via `webContents.send`.
- [x] Chat panel renders a streaming Rex bubble that grows as tokens arrive; cursor blink animation while streaming.
- [x] Streaming bubble finalized (cursor removed) when stream ends.
- [x] Falls back gracefully to non-streaming if Python backend does not support streaming.
- [x] `npm run typecheck` passes.
- [x] Verify changes work in app.

---

### Phase 65 — Voice Chat Panel

---

### US-188: Build voice toggle and status indicator
**Description:** As a user, I want a clearly visible button to start and stop voice mode so I know when Rex is listening.

**Acceptance Criteria:**
- [x] `gui/src/components/voice/VoiceToggle.tsx` — large circular button. States: `idle` (gray mic icon), `listening` (red pulsing ring), `processing` (spinning accent ring), `speaking` (green waveform icon).
- [x] State label shown below button ("Tap to speak", "Listening…", "Thinking…", "Speaking…").
- [x] Button is keyboard-accessible (Space to toggle).
- [x] `npm run typecheck` passes.
- [x] Verify changes work in app.

---

### US-189: Build audio waveform visualizer
**Description:** As a user, I want to see a live waveform animation while Rex is listening or speaking so I have visual confirmation audio is active.

**Acceptance Criteria:**
- [x] `gui/src/components/voice/WaveformVisualizer.tsx` — canvas-based animated waveform.
- [x] In `listening` state: visualizes microphone input amplitude via `Web Audio API` `AnalyserNode`.
- [x] In `speaking` state: plays a synthetic animated waveform (not real audio analysis — simulated for now).
- [x] In `idle`/`processing` state: renders a flat, subtle line.
- [x] Animation is smooth at 60fps using `requestAnimationFrame`.
- [x] `npm run typecheck` passes.
- [x] Verify changes work in app.

---

### US-190: Wire voice panel to Rex voice backend via IPC
**Description:** As a user, I want the voice button to actually trigger Rex's voice pipeline so I can have a spoken conversation.

**Acceptance Criteria:**
- [x] `window.rex.startVoice(): Promise<void>` and `window.rex.stopVoice(): Promise<void>` IPC methods implemented.
- [x] Main process calls Rex Python voice backend (start/stop listening) and emits state change events back to renderer.
- [x] Voice page (`/voice`) reflects backend state changes in VoiceToggle and WaveformVisualizer.
- [x] On voice session end, the final transcript appended to a session history list below the visualizer.
- [x] Error state handled: if voice backend unavailable, show EmptyState with troubleshoot link.
- [x] `npm run typecheck` passes.
- [x] Verify changes work in app.

---

### US-191: Add live transcript and history display to voice panel
**Description:** As a user, I want to see what Rex heard and what it said in text so I can review the voice conversation.

**Acceptance Criteria:**
- [x] Voice panel shows a scrollable transcript list below the waveform: alternating user/Rex utterances with timestamps.
- [x] Partial (in-progress) transcript shows in italic with blinking cursor during `listening` state.
- [x] "Clear history" button at top-right of transcript area (with confirmation tooltip).
- [x] Transcript persists across voice sessions within the app session (cleared only on explicit action or app restart).
- [x] `npm run typecheck` passes.
- [x] Verify changes work in app.

---

### Phase 66 — Task Scheduler View

---

### US-192: Build task list view
**Description:** As a user, I want to see all my scheduled Rex tasks in a list so I can monitor what's running and when.

**Acceptance Criteria:**
- [x] `gui/src/pages/TasksPage.tsx` renders a list of task cards. Each card shows: task name, schedule expression (e.g., "Every day at 9am"), next run time, status badge (`active`, `paused`, `error`).
- [x] List fetched from `window.rex.getTasks(): Promise<Task[]>` IPC method (stub returns 2-3 hardcoded tasks).
- [x] Empty state shown when no tasks exist.
- [x] `npm run typecheck` passes.
- [x] Verify changes work in app.

---

### US-193: Build task create and edit form
**Description:** As a user, I want to create and edit scheduled tasks from the GUI so I don't need to use the terminal.

**Acceptance Criteria:**
- [x] "New Task" button on TasksPage opens a Modal containing a task form.
- [x] Form fields: Name (text), Prompt/Command (textarea), Schedule (select: every hour, every day at [time], every week on [day], custom cron), Active (toggle).
- [x] Clicking an existing task card opens the same Modal pre-filled with that task's data.
- [x] Form validates: name required, schedule required. Shows inline errors.
- [x] Submit calls `window.rex.saveTask(task): Promise<Task>` IPC stub.
- [x] `npm run typecheck` passes.
- [x] Verify changes work in app.

---

### US-194: Wire tasks page to Rex scheduler backend via IPC
**Description:** As a user, I want task CRUD operations to actually update Rex's scheduler so my changes persist.

**Acceptance Criteria:**
- [x] `window.rex.getTasks`, `window.rex.saveTask`, `window.rex.deleteTask`, `window.rex.setTaskEnabled` IPC methods fully implemented end-to-end (main process calls Rex Python scheduler module).
- [x] TasksPage refetches task list after every create/edit/delete.
- [x] Enable/disable toggle calls `setTaskEnabled` immediately without opening the modal.
- [x] IPC errors surfaced as Toast notifications.
- [x] `npm run typecheck` passes.
- [x] Verify changes work in app.

---

### US-195: Add task run history and last-result indicator to task cards
**Description:** As a user, I want to see whether my last scheduled task run succeeded or failed so I can spot problems without digging through logs.

**Acceptance Criteria:**
- [x] Task card shows "Last run" section: timestamp + result badge (`success`, `failed`, `never`).
- [x] Clicking "Last run" detail expands an inline log output panel (last 20 lines of task output).
- [x] `window.rex.getTaskHistory(taskId): Promise<TaskRun[]>` IPC method added (stub returns 1-2 sample runs).
- [x] Failed tasks show the task card with a left-border in danger color.
- [x] `npm run typecheck` passes.
- [x] Verify changes work in app.

---

### Phase 67 — Calendar View

---

### US-196: Build calendar month and week view components
**Description:** As a user, I want to view my calendar in both month and week layouts so I can plan at different time horizons.

**Acceptance Criteria:**
- [x] `gui/src/components/calendar/CalendarGrid.tsx` renders a month grid (7 columns, rows per week).
- [x] `gui/src/components/calendar/WeekView.tsx` renders a 7-column, time-slotted week view (hourly rows from 6am–10pm).
- [x] Toggle between month and week view via segmented control at top of CalendarPage.
- [x] Today's date is highlighted. Current time shown as a horizontal line in week view.
- [x] Events (passed as props) render as colored chips in the appropriate day/slot.
- [x] `npm run typecheck` passes.
- [x] Verify changes work in app.

---

### US-197: Build event detail panel
**Description:** As a user, I want to click a calendar event to see its full details so I can review what's scheduled.

**Acceptance Criteria:**
- [x] Clicking an event chip opens a slide-in detail panel (not a full modal) on the right side of the calendar.
- [x] Panel shows: title, date/time, duration, location, description, attendees (if available), source (Rex-created vs. synced).
- [x] Panel has "Edit" and "Delete" action buttons (stubs for now, wired in US-198).
- [x] Panel closes on Escape or clicking outside.
- [x] `npm run typecheck` passes.
- [x] Verify changes work in app.

---

### US-198: Wire calendar to Rex calendar backend and create-event flow
**Description:** As a user, I want my calendar to show real Rex-managed events and let me create new ones from the GUI.

**Acceptance Criteria:**
- [x] `window.rex.getCalendarEvents(start: string, end: string): Promise<CalendarEvent[]>` IPC method implemented (calls Rex CalendarService — stub returns mock events when no credentials).
- [x] CalendarPage loads events for the visible date range on mount and on range change.
- [x] "New Event" button opens a create-event Modal: title, date, time, duration, location, description fields.
- [x] Submit calls `window.rex.createCalendarEvent(event): Promise<CalendarEvent>` IPC method.
- [x] Edit/Delete in event detail panel call `updateCalendarEvent` / `deleteCalendarEvent` IPC methods (stubs).
- [x] `npm run typecheck` passes.
- [x] Verify changes work in app.

---

### Phase 68 — Reminders View

---

### US-199: Build reminders list view
**Description:** As a user, I want to see all my upcoming and overdue reminders in one place so nothing falls through the cracks.

**Acceptance Criteria:**
- [x] `gui/src/pages/RemindersPage.tsx` renders three sections: Overdue, Today, Upcoming (grouped by relative time).
- [x] Each reminder card shows: title, due time, priority badge, done checkbox.
- [x] Checking the checkbox marks it done (calls `window.rex.completeReminder(id)` stub).
- [x] Overdue reminders show card with danger-color left border.
- [x] Empty state per section shown when no reminders in that group.
- [x] `npm run typecheck` passes.
- [x] Verify changes work in app.

---

### US-200: Build reminder create and edit form
**Description:** As a user, I want to create and edit reminders from the GUI so I can manage them without the terminal.

**Acceptance Criteria:**
- [x] "New Reminder" button opens a Modal with fields: title (text), notes (textarea), due date (date picker), due time (time picker), priority (select: low/medium/high), repeat (select: none/daily/weekly/custom).
- [x] Editing an existing reminder opens the same Modal pre-filled.
- [x] Inline validation: title required, due date required.
- [x] Submit calls `window.rex.saveReminder(reminder): Promise<Reminder>` IPC stub.
- [x] `npm run typecheck` passes.
- [x] Verify changes work in app.

---

### US-201: Wire reminders page to Rex reminders backend via IPC
**Description:** As a user, I want reminder CRUD operations to persist to Rex's backend so my reminders survive app restarts.

**Acceptance Criteria:**
- [x] `window.rex.getReminders`, `window.rex.saveReminder`, `window.rex.deleteReminder`, `window.rex.completeReminder` IPC methods fully implemented end-to-end.
- [x] RemindersPage refetches after every mutation.
- [x] IPC errors surfaced as Toast notifications.
- [x] Reminders list refreshes automatically every 60 seconds (to catch changes made by Rex autonomously).
- [x] `npm run typecheck` passes.
- [x] Verify changes work in app.

---

### Phase 69 — User Memories Editor

---

### US-202: Build memories list view
**Description:** As a user, I want to browse all the memories Rex has about me so I understand what context it's working with.

**Acceptance Criteria:**
- [x] `gui/src/pages/MemoriesPage.tsx` renders a searchable, paginated list of memory entries.
- [x] Each entry card shows: memory text (truncated to 2 lines), category tag, created/updated timestamp.
- [x] Search input filters entries by text content (client-side, no IPC call).
- [x] Category filter dropdown (All, plus any category names returned by backend).
- [x] Empty state shown when no memories or no search matches.
- [x] `npm run typecheck` passes.
- [x] Verify changes work in app.

---

### US-203: Build inline memory editor
**Description:** As a user, I want to edit a memory entry directly in the list so I can correct Rex's understanding of me.

**Acceptance Criteria:**
- [x] Clicking a memory card expands it into an inline edit form (Textarea + category select + Save/Cancel buttons).
- [x] Only one memory editable at a time (opening another collapses the current one, prompting to save if dirty).
- [x] Save calls `window.rex.updateMemory(id, data): Promise<Memory>` IPC stub.
- [x] Cancel discards changes and collapses the card.
- [x] Optimistic UI: card updates immediately on save, reverts on IPC error.
- [x] `npm run typecheck` passes.
- [x] Verify changes work in app.

---

### US-204: Build memory add form and delete action
**Description:** As a user, I want to manually add new memories and delete incorrect ones so I have full control over Rex's knowledge of me.

**Acceptance Criteria:**
- [x] "Add Memory" button at top of MemoriesPage opens a Modal: text (textarea, required), category (text input with autocomplete from existing categories).
- [x] Submit calls `window.rex.addMemory(data): Promise<Memory>` IPC stub.
- [x] Each memory card has a delete icon (trash). Clicking shows inline confirmation ("Delete this memory?") before calling `window.rex.deleteMemory(id)`.
- [x] Deleted memories removed from list immediately (optimistic).
- [x] `npm run typecheck` passes.
- [x] Verify changes work in app.

---

### US-205: Wire memories page to Rex memory backend via IPC
**Description:** As a user, I want memory operations to actually read from and write to Rex's memory store so changes have real effect on conversations.

**Acceptance Criteria:**
- [x] `window.rex.getMemories`, `window.rex.addMemory`, `window.rex.updateMemory`, `window.rex.deleteMemory` IPC methods fully implemented end-to-end (main process calls Rex Python memory module).
- [x] MemoriesPage loads real memories on mount.
- [x] IPC errors surfaced as Toast notifications.
- [x] `npm run typecheck` passes.
- [x] Verify changes work in app.

---

### Phase 70 — Settings Panel

---

### US-206: Scaffold settings panel with category navigation
**Description:** As a user, I want a settings panel organized into logical sections so I can find any Rex configuration option quickly.

**Acceptance Criteria:**
- [x] `gui/src/pages/SettingsPage.tsx` renders a two-column layout: left column is a category list, right column is the active category's form.
- [x] Categories: General, Voice, AI, Integrations, Notifications, About.
- [x] Clicking a category highlights it and renders the appropriate sub-page in the right column.
- [x] Default active category is General.
- [x] About section shows Rex version, Electron version, Node version (read from `process.versions` in main process via IPC).
- [x] `npm run typecheck` passes.
- [x] Verify changes work in app.

---

### US-207: Build General settings form
**Description:** As a user, I want to configure my name, timezone, and language preferences so Rex addresses me correctly and understands time context.

**Acceptance Criteria:**
- [x] Fields: Display Name (text), Timezone (searchable select from IANA timezone list), Language (select: English, Spanish, French, German, Japanese — extendable), Launch at login (toggle), Start minimized to tray (toggle).
- [x] Form loads current values from `window.rex.getSettings('general')` on mount.
- [x] Changes save on blur of each field (auto-save, no explicit Save button).
- [x] Saved state shown with a "Saved" checkmark that fades after 2s.
- [x] `npm run typecheck` passes.
- [x] Verify changes work in app.

---

### US-208: Build Voice settings form
**Description:** As a user, I want to configure voice input/output devices, TTS engine, and playback parameters so Rex sounds and hears the way I want.

**Acceptance Criteria:**
- [x] Fields: Microphone device (select from `navigator.mediaDevices.enumerateDevices()`), Speaker device (same), TTS engine (select: system, openai, elevenlabs), TTS voice (text, conditionally shown), Speech rate (slider 0.5–2.0), Volume (slider 0–1.0).
- [x] "Test Voice" button plays a short TTS sample using current settings.
- [x] Form loads current values from `window.rex.getSettings('voice')` on mount.
- [x] Auto-saves on field change.
- [x] `npm run typecheck` passes.
- [x] Verify changes work in app.

---

### US-209: Build AI settings form
**Description:** As a user, I want to configure Rex's AI model, temperature, system prompt, and autonomy mode so I can tune its behavior.

**Acceptance Criteria:**
- [x] Fields: AI Model (select: gpt-4o, gpt-4-turbo, claude-opus-4, claude-sonnet-4, gemini-1.5-pro), Temperature (slider 0–1.0 with label "Precise ← → Creative"), Max tokens (number input), System prompt override (textarea, placeholder shows current default), Autonomy mode (select: manual, supervised, full-auto).
- [x] Warning banner shown when Autonomy mode is `full-auto`: "Rex will act without confirmation. Review task history regularly."
- [x] Form loads current values from `window.rex.getSettings('ai')` on mount.
- [x] Auto-saves on field change.
- [x] `npm run typecheck` passes.
- [x] Verify changes work in app.

---

### US-210: Build Integrations settings form
**Description:** As a user, I want to enter and manage credentials for email, calendar, and SMS integrations so Rex can connect to live services.

**Acceptance Criteria:**
- [x] Email section: Provider (select: Gmail, Outlook), OAuth client ID (text), OAuth client secret (password input), Status indicator (connected / not connected).
- [x] Calendar section: same structure as email (shares OAuth in practice, separate UI for clarity).
- [x] SMS section: Twilio Account SID (text), Twilio Auth Token (password input), From phone number (text), Status indicator.
- [x] "Test Connection" button per section calls `window.rex.testIntegration(type): Promise<{ok: boolean, error?: string}>` IPC method.
- [x] All credential fields use `type="password"` with show/hide toggle.
- [x] Values loaded from `window.rex.getSettings('integrations')` (returns masked values for credential fields).
- [x] Auto-saves on field change.
- [x] `npm run typecheck` passes.
- [x] Verify changes work in app.

---

### US-211: Build Notifications settings form
**Description:** As a user, I want to configure quiet hours, digest schedule, and priority thresholds so Rex only interrupts me at the right times and in the right ways.

**Acceptance Criteria:**
- [x] Fields: Quiet hours enabled (toggle), Quiet hours start (time picker), Quiet hours end (time picker), Digest mode enabled (toggle), Digest delivery time (time picker), High-priority threshold (select: critical only / high and critical), Auto-escalation delay (number input, minutes), Desktop notifications enabled (toggle), Sound alerts enabled (toggle).
- [x] Quiet hours start/end pickers disabled when quiet hours toggle is off.
- [x] Digest delivery time disabled when digest mode is off.
- [x] Form loads from `window.rex.getSettings('notifications')`. Auto-saves.
- [x] `npm run typecheck` passes.
- [x] Verify changes work in app.

---

### US-212: Wire all settings forms to Rex config backend via IPC
**Description:** As a developer, I need all settings IPC methods fully implemented so changes in the GUI actually update Rex's configuration files.

**Acceptance Criteria:**
- [x] `window.rex.getSettings(section)` and `window.rex.setSettings(section, values)` IPC methods implemented end-to-end (main process reads/writes Rex's Python config file or settings store).
- [x] Settings changes take effect in Rex without requiring an app restart (Python backend reloads affected config on update).
- [x] `window.rex.testIntegration(type)` IPC method implemented: makes a lightweight API call to verify credentials and returns `{ok, error}`.
- [x] All settings sections confirm with a "Saved" indicator after successful IPC write.
- [x] IPC errors shown as Toast notifications.
- [x] `npm run typecheck` passes.
- [x] Verify changes work in app.

---

### Phase 71 — GUI Polish

---

### US-213: Add page transition animations and hover micro-interactions
**Description:** As a user, I want smooth transitions between pages and subtle hover responses so the app feels polished and alive.

**Acceptance Criteria:**
- [x] Page transitions use a fade + slight upward slide (150ms ease-out). Implemented via CSS classes toggled on route change.
- [x] Sidebar nav items scale slightly (1.02×) on hover with a 100ms transition.
- [x] Button press effect: scale down to 0.97× on `:active`.
- [x] Card hover (for hoverable Cards): subtle `translateY(-2px)` + deeper shadow.
- [x] All transitions respect `prefers-reduced-motion` media query (disabled if true).
- [x] `npm run typecheck` passes.
- [x] Verify changes work in app.

---

### US-214: Add loading skeletons and error boundaries to all pages
**Description:** As a user, I want pages to show skeleton placeholders while loading and friendly error messages on failure so the app never shows a blank screen or crash.

**Acceptance Criteria:**
- [x] Every page that fetches IPC data shows SkeletonLine placeholders during the initial load (not a spinner).
- [x] Each page wrapped in an `ErrorBoundary` component that shows an EmptyState with "Something went wrong" + retry button.
- [x] If an IPC call takes more than 5s, a "Taking longer than expected…" message appears below the skeleton.
- [x] `gui/src/components/ErrorBoundary.tsx` implemented as a React class component (required for error boundaries).
- [x] `npm run typecheck` passes.
- [x] Verify changes work in app.

---

### US-215: Add global keyboard shortcuts with help overlay
**Description:** As a user, I want keyboard shortcuts for common actions so I can use Rex efficiently without reaching for the mouse.

**Acceptance Criteria:**
- [x] Global shortcuts registered: `Ctrl+K` → focus chat input (if on chat page, or navigate to chat and focus), `Ctrl+Shift+V` → toggle voice, `Ctrl+,` → open settings, `Ctrl+N` → new task/reminder/event (context-sensitive), `?` → open help overlay.
- [x] Help overlay (triggered by `?`) shows a modal listing all shortcuts in a two-column table.
- [x] Shortcuts do not fire when focus is inside a text input (except Ctrl+ chords).
- [x] `gui/src/hooks/useGlobalShortcuts.ts` hook manages all registrations via `useEffect`.
- [x] `npm run typecheck` passes.
- [x] Verify changes work in app.

---

### US-216: Add system tray icon and minimize-to-tray behavior
**Description:** As a user, I want Rex to minimize to the system tray so it stays accessible without cluttering my taskbar.

**Acceptance Criteria:**
- [x] `gui/src/main/tray.ts` creates a Tray instance with a Rex icon (16×16 and 32×32 variants in `gui/assets/`).
- [x] Tray context menu: Show Rex, New Chat (opens app + focuses chat), Toggle Voice, Quit Rex.
- [x] Closing the main window hides it (does not quit); app remains in tray.
- [x] Clicking the tray icon shows/restores the main window.
- [x] Quitting via tray menu fully exits the process (including Python subprocess cleanup).
- [x] `npm run typecheck` passes.

---

### Phase 72 — LLM-Based Planning Engine

---

### US-217: Define planning data structures and planner protocol
**Description:** As a developer, I need typed data structures for plans and plan steps so the new LLM planner has a stable interface.

**Acceptance Criteria:**
- [x] `rex/autonomy/models.py` defines (or extends existing models): `PlanStep(id, tool, args, description, status, result, error)`, `Plan(id, goal, steps, status, created_at, completed_at)`, `PlannerProtocol` (abstract base class with `plan(goal: str, context: dict) -> Plan`).
- [x] All models are Pydantic v2 models with full type annotations.
- [x] `mypy --strict rex/autonomy/models.py` passes.
- [x] Unit test in `tests/test_autonomy_models.py` verifies model instantiation and serialization.
- [x] Tests pass.

---

### US-218: Implement LLMPlanner class
**Description:** As a developer, I need an LLMPlanner that uses Rex's AI backend to convert a natural-language goal into an executable plan so ambiguous and complex goals can be handled.

**Acceptance Criteria:**
- [x] `rex/autonomy/llm_planner.py` implements `LLMPlanner(PlannerProtocol)` class.
- [x] `plan(goal, context)` method constructs a structured prompt listing available tools, current context, and the goal; calls Rex's AI backend; parses the response into a `Plan` with typed `PlanStep` list.
- [x] Prompt instructs the LLM to return steps as JSON (tool name, args, description). `LLMPlanner` validates the JSON and raises `PlanningError` on parse failure.
- [x] `LLMPlanner` accepts a `tools: list[ToolDefinition]` argument at init so available tools are explicit.
- [x] `mypy --strict rex/autonomy/llm_planner.py` passes.

---

### US-219: Add unit tests for LLMPlanner with mocked LLM responses
**Description:** As a developer, I need tests for LLMPlanner so I can verify parsing logic without making real API calls.

**Acceptance Criteria:**
- [x] `tests/test_llm_planner.py` tests: valid JSON response → correct Plan with N steps, malformed JSON response → `PlanningError` raised, empty steps response → `PlanningError` raised, unknown tool name in response → step flagged with warning but plan still created.
- [x] All tests use `unittest.mock.patch` to mock the AI backend call.
- [x] Tests pass without network access.
- [x] `mypy --strict tests/test_llm_planner.py` passes.

---

### US-220: Integrate LLMPlanner into autonomy dispatch flow
**Description:** As a developer, I need the autonomy runner to use LLMPlanner instead of the rule-based planner so live executions use the new engine.

**Acceptance Criteria:**
- [x] `rex/autonomy/runner.py` (or equivalent dispatch entry point) imports `LLMPlanner` and uses it by default.
- [x] Old rule-based planner kept in `rex/autonomy/rule_planner.py` (not deleted) with a deprecation comment.
- [x] A `PLANNER` config key in Rex settings (default: `llm`) selects which planner to use (`llm` | `rule`).
- [x] Integration smoke test: calling the autonomy runner with a simple goal (e.g., "Get the weather") produces a non-empty Plan without errors.
- [x] `mypy --strict rex/autonomy/runner.py` passes.

---

### Phase 73 — Dynamic Replanning

---

### US-221: Add step execution result tracking to plan runner
**Description:** As a developer, I need the plan runner to record success/failure for each step so replanning has the context it needs.

**Acceptance Criteria:**
- [x] `PlanStep.status` updated to: `pending | running | success | failed | skipped`.
- [x] `PlanStep.result` (optional string) populated with tool output on success.
- [x] `PlanStep.error` (optional string) populated with error message on failure.
- [x] Plan runner updates step status in real time as each step executes.
- [x] `Plan.status` updated to `completed` when all steps succeed, `failed` when a step fails and no replanning occurs.
- [x] `mypy --strict` passes on changed files. Tests pass.

---

### US-222: Implement replan trigger on step failure
**Description:** As a developer, I need the plan runner to call the LLM for a revised plan when a step fails so the autonomy engine can adapt rather than abort.

**Acceptance Criteria:**
- [x] `rex/autonomy/replanner.py` implements `Replanner` class with `replan(original_plan, failed_step, error_context) -> Plan`.
- [x] `Replanner` prompt includes: original goal, steps completed so far, the failed step + error, and instructions to produce a revised plan for the remaining work.
- [x] Plan runner catches step failure, calls `Replanner.replan()`, and continues with the new plan.
- [x] Max replan attempts per run configurable (default: 2). After max attempts, plan status set to `failed`.
- [x] `mypy --strict rex/autonomy/replanner.py` passes.

---

### US-223: Add retry logic with exponential backoff for transient failures
**Description:** As a developer, I need the plan runner to retry transiently-failing steps automatically so network blips don't abort plans.

**Acceptance Criteria:**
- [x] `rex/autonomy/retry.py` implements `retry_step(step_fn, max_attempts=3, base_delay=1.0)` async function with exponential backoff.
- [x] Retry triggered only for transient errors: `TimeoutError`, `ConnectionError`, HTTP 429/503. Non-transient errors (e.g., `ValueError`, HTTP 4xx) skip retry and go directly to replan.
- [x] Each retry attempt logged at `DEBUG` level: `"Retrying step {id}, attempt {n}/{max}, delay {delay}s"`.
- [x] Plan runner wraps each step execution in `retry_step`.
- [x] Unit tests: transient error retries N times then succeeds; non-transient error fails immediately. Tests pass.

---

### US-224: Implement alternative path exploration
**Description:** As a developer, I need the LLM planner to generate two fallback approaches when replanning so the runner has options if the first revised plan also fails.

**Acceptance Criteria:**
- [x] `LLMPlanner.plan_with_alternatives(goal, context) -> list[Plan]` method added. Prompts the LLM to produce the primary plan plus two alternatives, each as a separate JSON plan.
- [x] `Replanner.replan` uses `plan_with_alternatives` and tries each alternative in order before declaring failure.
- [x] Alternatives logged: `"Trying alternative plan {n}/2 for goal '{goal}'"`.
- [x] Unit tests with mocked LLM: verify three plans returned and runner attempts each on consecutive failures. Tests pass.
- [x] `mypy --strict` passes on changed files.

---

### Phase 74 — Learn from Feedback

---

### US-225: Define ExecutionHistory data model and storage schema
**Description:** As a developer, I need a data model and persistent store for plan execution history so the feedback learning system has data to work with.

**Acceptance Criteria:**
- [x] `rex/autonomy/history.py` defines `ExecutionRecord(id, goal, plan, outcome: 'success'|'partial'|'failed', duration_s, replan_count, error_summary, timestamp)` as a Pydantic model.
- [x] `HistoryStore` class with `append(record)`, `recent(n=20) -> list[ExecutionRecord]`, `by_outcome(outcome) -> list[ExecutionRecord]`.
- [x] Backing store: SQLite via `aiosqlite`. DB file: `~/.rex/execution_history.db`. Migration runs on first access.
- [x] `mypy --strict rex/autonomy/history.py` passes. Unit tests for append + query pass.

---

### US-226: Persist execution history after every plan run
**Description:** As a developer, I need the plan runner to save an ExecutionRecord after each run so history accumulates automatically.

**Acceptance Criteria:**
- [x] Plan runner calls `HistoryStore.append(record)` after every plan completes (success or failure).
- [x] Record includes all `ExecutionRecord` fields populated from the completed `Plan` object.
- [x] History write failure is logged as a warning but does not raise (does not break the plan run).
- [x] Integration test: run a mocked plan end-to-end, assert a record appears in the DB. Test passes.
- [x] `mypy --strict` passes on changed files.

---

### US-227: Implement FeedbackAnalyzer that summarizes history for planner context
**Description:** As a developer, I need a FeedbackAnalyzer that distills execution history into a short summary so the LLM planner can learn from past runs without exceeding context limits.

**Acceptance Criteria:**
- [x] `rex/autonomy/feedback.py` implements `FeedbackAnalyzer` with `summarize(goal: str, history_store: HistoryStore) -> str`.
- [x] Summary retrieves the 10 most recent records. Generates a short (≤ 200 words) text paragraph describing: common failure patterns, which tools tend to succeed/fail, any goals similar to current that succeeded before.
- [x] Summary generated by the LLM (one compact call). If history is empty, returns empty string.
- [x] `mypy --strict rex/autonomy/feedback.py` passes. Unit test with mocked history and mocked LLM verifies non-empty output. Test passes.

---

### US-228: Inject feedback summary into LLMPlanner context on new goals
**Description:** As a developer, I need the LLMPlanner to receive the feedback summary as additional context so it makes better decisions based on what has worked before.

**Acceptance Criteria:**
- [ ] `LLMPlanner.plan()` accepts optional `feedback_summary: str` parameter.
- [ ] If `feedback_summary` is non-empty, it is included in the planner prompt under a `## Past Execution Patterns` section.
- [ ] Plan runner calls `FeedbackAnalyzer.summarize()` before calling `LLMPlanner.plan()` and passes the result.
- [ ] Unit test: planner prompt contains feedback summary when provided; prompt omits section when summary is empty. Tests pass.
- [ ] `mypy --strict` passes on changed files.

---

### Phase 75 — Multi-Goal Planning

---

### US-229: Define GoalGraph data structure
**Description:** As a developer, I need a GoalGraph that models multiple goals and their dependencies so the runner can execute them in the correct order.

**Acceptance Criteria:**
- [ ] `rex/autonomy/goal_graph.py` defines `Goal(id, description, depends_on: list[str], status)` and `GoalGraph(goals: list[Goal])`.
- [ ] `GoalGraph.topological_sort() -> list[Goal]` raises `CyclicDependencyError` if a cycle is detected; otherwise returns goals in valid execution order.
- [ ] `GoalGraph.ready_goals() -> list[Goal]` returns goals whose dependencies are all `completed`.
- [ ] Unit tests: linear dependency sorts correctly; parallel goals with shared dependency; cycle raises error. Tests pass.
- [ ] `mypy --strict rex/autonomy/goal_graph.py` passes.

---

### US-230: Implement GoalParser to extract multiple goals from user input
**Description:** As a developer, I need a GoalParser that uses the LLM to identify multiple goals and their dependencies in a user's message so complex requests can be handled.

**Acceptance Criteria:**
- [ ] `rex/autonomy/goal_parser.py` implements `GoalParser` with `parse(user_input: str) -> GoalGraph`.
- [ ] Parser prompts the LLM to: identify discrete goals in the input, assign each a short ID, identify any that must complete before others (dependencies), flag any that are ambiguous.
- [ ] Returns a `GoalGraph` from the parsed JSON response.
- [ ] If only one goal is found, returns a single-node graph (compatible with existing single-goal flow).
- [ ] `mypy --strict rex/autonomy/goal_parser.py` passes. Unit test with mocked LLM. Test passes.

---

### US-231: Integrate GoalGraph execution into autonomy runner
**Description:** As a developer, I need the autonomy runner to process GoalGraphs so multi-goal inputs execute in the right order with proper dependency handling.

**Acceptance Criteria:**
- [ ] Autonomy runner entry point calls `GoalParser.parse()` on user input, receives `GoalGraph`.
- [ ] Runner executes goals in topological order: for each `Goal`, creates a `Plan` via `LLMPlanner`, runs the plan, marks goal `completed` or `failed`.
- [ ] On goal failure, dependent goals are marked `skipped` (not attempted).
- [ ] Progress logged per goal: `"Executing goal {id}: {description} ({n}/{total})"`.
- [ ] Integration smoke test with two sequential goals. Test passes. `mypy --strict` passes.

---

### US-232: Implement clarification question generator for ambiguous goals
**Description:** As a user, I want Rex to ask me a clarifying question when my request is ambiguous so it doesn't guess wrong and waste time.

**Acceptance Criteria:**
- [ ] `rex/autonomy/clarifier.py` implements `Clarifier` with `needs_clarification(goal: Goal) -> bool` and `generate_question(goal: Goal) -> str`.
- [ ] `needs_clarification` returns True when the goal was flagged ambiguous by `GoalParser`.
- [ ] `generate_question` prompts the LLM to produce a single, specific yes/no or multiple-choice question that resolves the ambiguity.
- [ ] Autonomy runner calls clarifier before planning an ambiguous goal: emits the question to the user via the configured output channel and pauses that goal until answer received.
- [ ] `mypy --strict rex/autonomy/clarifier.py` passes. Unit test with mocked LLM. Test passes.

---

### Phase 76 — Cost Optimization

---

### US-233: Add per-step token and cost tracking to plan runner
**Description:** As a developer, I need the plan runner to record token usage and estimated cost per step so cost data is available for optimization and reporting.

**Acceptance Criteria:**
- [ ] `PlanStep` extended with `tokens_used: int | None` and `cost_usd: float | None`.
- [ ] Plan runner populates these fields from tool call responses where token counts are available.
- [ ] `Plan` has a computed property `total_cost_usd: float` (sum of all step costs).
- [ ] `ExecutionRecord` extended with `total_cost_usd` field and persisted to history DB.
- [ ] `mypy --strict` passes on changed files. Unit tests for cost aggregation pass.

---

### US-234: Implement CostEstimator with pre-execution user prompt
**Description:** As a user, I want Rex to tell me the estimated cost of a plan before running it so I can decide whether to proceed.

**Acceptance Criteria:**
- [ ] `rex/autonomy/cost_estimator.py` implements `CostEstimator` with `estimate(plan: Plan) -> CostEstimate` (calculates upper-bound token estimate from step descriptions and tool signatures).
- [ ] `CostEstimate` has `low_usd`, `high_usd`, `step_count` fields.
- [ ] Autonomy runner calls `CostEstimator.estimate()` before executing, checks against `budget_usd` config value.
- [ ] If `estimate.high_usd > budget_usd`: prompt user for approval ("Estimated cost: $X. Proceed? [y/N]"). Abort if denied.
- [ ] Cost check skipped if `budget_usd` is 0 (unlimited).
- [ ] `mypy --strict rex/autonomy/cost_estimator.py` passes. Unit tests pass.

---

### US-235: Add tool result caching within a plan run
**Description:** As a developer, I need identical tool calls within a single plan run to return cached results so duplicate API calls don't waste money.

**Acceptance Criteria:**
- [ ] `rex/autonomy/tool_cache.py` implements `ToolCache` (in-memory, scoped to one plan run) with `get(tool, args) -> result | None` and `set(tool, args, result)`.
- [ ] Cache key: `(tool_name, frozenset(sorted(args.items())))`.
- [ ] Plan runner passes `ToolCache` instance to each step execution; steps check cache before calling tool.
- [ ] Cache hit logged at DEBUG: `"Tool cache hit: {tool}({args})"`.
- [ ] Cache is NOT shared across plan runs (new instance per run).
- [ ] Unit tests: same call twice → second is a cache hit; different args → both miss. Tests pass.

---

### US-236: Add per-step and global budget configuration to settings
**Description:** As a user, I want to set a maximum cost per step and per plan so Rex doesn't run up large bills autonomously.

**Acceptance Criteria:**
- [ ] `rex/config.py` (or equivalent) adds: `autonomy_budget_per_plan_usd: float = 0.0` (0 = unlimited) and `autonomy_budget_per_step_usd: float = 0.0`.
- [ ] Plan runner enforces per-step budget: if a step's estimated cost exceeds `autonomy_budget_per_step_usd`, skip it and log a warning.
- [ ] AI settings form in GUI (US-209) updated to include these two budget fields (number inputs, labeled in USD).
- [ ] `mypy --strict` passes on changed files. Tests pass.

---

### Phase 77 — User Preference Learning

---

### US-237: Define UserPreferenceProfile data model
**Description:** As a developer, I need a UserPreferenceProfile that captures patterns in how the user invokes Rex so preference learning has a typed target.

**Acceptance Criteria:**
- [ ] `rex/autonomy/preferences.py` defines `UserPreferenceProfile` with: `preferred_autonomy_mode: str`, `preferred_model: str`, `common_goal_patterns: list[str]`, `active_hours: list[int]` (hours of day), `avg_budget_usd: float`, `last_updated: datetime`.
- [ ] `PreferenceStore` class: `load() -> UserPreferenceProfile`, `save(profile)`. Backed by `~/.rex/preferences.json`.
- [ ] `mypy --strict rex/autonomy/preferences.py` passes. Unit tests for load/save round-trip. Tests pass.

---

### US-238: Implement PreferenceLearner that updates profile after each run
**Description:** As a developer, I need a PreferenceLearner that updates the UserPreferenceProfile after each completed workflow so the profile reflects current usage patterns.

**Acceptance Criteria:**
- [ ] `rex/autonomy/preference_learner.py` implements `PreferenceLearner` with `update(record: ExecutionRecord, profile: UserPreferenceProfile) -> UserPreferenceProfile`.
- [ ] Updates: `active_hours` incremented for the current hour; `avg_budget_usd` updated as rolling average; `common_goal_patterns` adds current goal description (deduplicated, max 20 entries, LRU eviction).
- [ ] Plan runner calls `PreferenceLearner.update()` after each successful run and saves updated profile.
- [ ] `mypy --strict rex/autonomy/preference_learner.py` passes. Unit tests pass.

---

### US-239: Surface preference suggestions in AI settings GUI
**Description:** As a user, I want Rex to suggest settings adjustments based on my usage patterns so I can accept optimizations with one click.

**Acceptance Criteria:**
- [ ] AI settings form (US-209) calls `window.rex.getPreferenceSuggestions(): Promise<PreferenceSuggestion[]>` IPC method on mount.
- [ ] `PreferenceSuggestion` has `field`, `current_value`, `suggested_value`, `reason`.
- [ ] If suggestions exist, an info banner appears below the form: "Based on your usage: [suggestion reason]. [Apply] [Dismiss]".
- [ ] "Apply" calls `window.rex.applyPreferenceSuggestion(field, value)` IPC and refreshes the form.
- [ ] Maximum one suggestion shown at a time (most impactful first).
- [ ] `npm run typecheck` passes. `mypy --strict` passes on IPC handler. Verify changes work in app.

---

### US-240: Apply learned preferences as soft defaults in new planning sessions
**Description:** As a developer, I need the plan runner to read the UserPreferenceProfile and use it to pre-populate planning defaults so Rex gets smarter over time without requiring explicit configuration.

**Acceptance Criteria:**
- [ ] Plan runner loads `UserPreferenceProfile` at session start.
- [ ] If `preferred_autonomy_mode` is set and the user hasn't overridden it this session, runner uses the profile value.
- [ ] If `preferred_model` is set, `LLMPlanner` uses it unless overridden in settings.
- [ ] Defaults are soft (user's explicit settings always win).
- [ ] A debug log line emitted when a profile preference is applied: `"Using learned preference: {field}={value}"`.
- [ ] `mypy --strict` passes on changed files. Unit test verifies preference applied when no explicit override. Test passes.

---

### Phase 78 — Email and Calendar Service Layer

---

### US-241: Define EmailMessage and CalendarEvent data models
**Description:** As a developer, I need full, typed data models for email messages and calendar events so all services and the GUI share a consistent structure.

**Acceptance Criteria:**
- [ ] `rex/integrations/models.py` defines: `EmailMessage(id, thread_id, subject, sender, recipients, body_text, body_html, received_at, labels, is_read, priority: 'low'|'medium'|'high'|'critical')`, `CalendarEvent(id, title, start, end, location, description, attendees, source, is_all_day, recurrence)`.
- [ ] All fields typed with Pydantic v2. Optional fields marked `Optional`.
- [ ] `mypy --strict rex/integrations/models.py` passes. Unit test for model instantiation and `.model_dump()` round-trip. Tests pass.

---

### US-242: Implement EmailService with credential-ready stub/live interface
**Description:** As a developer, I need an EmailService that returns mock data without credentials and real Gmail/Outlook data when credentials are supplied so the GUI works in both modes.

**Acceptance Criteria:**
- [ ] `rex/integrations/email_service.py` implements `EmailService` with methods: `list_inbox(limit=20) -> list[EmailMessage]`, `get_thread(thread_id) -> list[EmailMessage]`, `send_draft(to, subject, body) -> EmailMessage`, `archive(id)`, `mark_read(id)`.
- [ ] If no credentials configured: all methods return realistic mock data (3-5 stub messages/events).
- [ ] If Gmail credentials configured: connects via Gmail REST API using stored OAuth tokens.
- [ ] Credential detection: `rex/config.py` `email_provider` field (none / gmail / outlook).
- [ ] `mypy --strict rex/integrations/email_service.py` passes. Unit tests for stub mode. Tests pass.

---

### US-243: Implement CalendarService with credential-ready stub/live interface
**Description:** As a developer, I need a CalendarService matching the same stub/live pattern as EmailService so calendar data flows through the GUI in both modes.

**Acceptance Criteria:**
- [ ] `rex/integrations/calendar_service.py` implements `CalendarService` with: `get_events(start, end) -> list[CalendarEvent]`, `create_event(event_data) -> CalendarEvent`, `update_event(id, event_data) -> CalendarEvent`, `delete_event(id)`.
- [ ] Stub mode: returns 5-8 realistic mock events spanning the next two weeks.
- [ ] Live mode: connects via Google Calendar API when credentials configured.
- [ ] `mypy --strict rex/integrations/calendar_service.py` passes. Unit tests for stub mode. Tests pass.

---

### US-244: Implement EmailTriageEngine for priority scoring and categorization
**Description:** As a user, I want Rex to automatically triage my inbox by priority and category so I can focus on what matters.

**Acceptance Criteria:**
- [ ] `rex/integrations/triage_engine.py` implements `EmailTriageEngine` with `triage(messages: list[EmailMessage]) -> list[EmailMessage]` (returns messages with `priority` field populated).
- [ ] Triage calls the LLM with message subject + sender + snippet to score priority (`low`/`medium`/`high`/`critical`) and assign a category tag (e.g., `action-required`, `newsletter`, `receipt`, `personal`).
- [ ] Results cached in memory per session to avoid re-scoring unchanged messages.
- [ ] `mypy --strict rex/integrations/triage_engine.py` passes. Unit tests with mocked LLM. Tests pass.

---

### US-245: Implement SchedulingEngine for meeting slot suggestions
**Description:** As a user, I want Rex to suggest available meeting times given constraints so I can respond to scheduling requests quickly.

**Acceptance Criteria:**
- [ ] `rex/integrations/scheduling_engine.py` implements `SchedulingEngine` with `find_slots(duration_minutes, participants, earliest, latest, timezone) -> list[TimeSlot]`.
- [ ] `TimeSlot(start, end, confidence: float)` model added to `integrations/models.py`.
- [ ] Engine calls `CalendarService.get_events()` to get existing commitments, then uses LLM to suggest 3 open slots respecting user's `active_hours` preference.
- [ ] In stub mode: returns 3 hardcoded future time slots.
- [ ] `mypy --strict rex/integrations/scheduling_engine.py` passes. Unit tests for stub mode. Tests pass.

---

### Phase 79 — Email and Calendar GUI

---

### US-246: Build email inbox view with triage indicators
**Description:** As a user, I want to see my triaged inbox in the GUI so I can quickly identify and act on high-priority emails.

**Acceptance Criteria:**
- [ ] `gui/src/pages/EmailPage.tsx` renders a list of email rows: sender, subject, snippet, received time, priority badge, read/unread indicator.
- [ ] Rows sorted by priority (critical first) within unread/read groups.
- [ ] Priority badge colors: critical=red, high=orange, medium=blue, low=gray.
- [ ] "Refresh" button at top calls `window.rex.getEmailInbox()` IPC (stub returns mock data).
- [ ] BETA banner at top of page: "Email integration — enter credentials in Settings > Integrations for live data."
- [ ] `npm run typecheck` passes. Verify changes work in app.

---

### US-247: Build email detail view with triage actions
**Description:** As a user, I want to read an email and take triage actions (archive, reply draft) from the GUI so I can process my inbox without leaving Rex.

**Acceptance Criteria:**
- [ ] Clicking an email row opens a detail panel (right-side slide-in, same pattern as calendar event detail).
- [ ] Panel shows: full subject, sender, recipient(s), received time, full body text (HTML rendered in a sandboxed `webview` or sanitized innerHTML).
- [ ] Action buttons: Archive, Mark as Read, Generate Reply Draft (calls `window.rex.generateEmailReply(id)` stub — returns a draft string).
- [ ] "Generate Reply Draft" result shown in a pre-filled compose modal with Send button (stub — logs to console).
- [ ] `npm run typecheck` passes. Verify changes work in app.

---

### US-248: Build calendar integration view in GUI
**Description:** As a user, I want the Calendar page wired to CalendarService so it shows real (or realistic mock) events and lets me create events through Rex.

**Acceptance Criteria:**
- [ ] CalendarPage (Phase 67) `window.rex.getCalendarEvents` IPC handler fully implemented: calls Python `CalendarService.get_events()` and returns results.
- [ ] `window.rex.createCalendarEvent` IPC handler implemented: calls `CalendarService.create_event()`.
- [ ] "Find Meeting Slot" button on CalendarPage opens a modal: duration (select), date range (date pickers), timezone (text). Submit calls `window.rex.findMeetingSlots(params)` IPC → `SchedulingEngine.find_slots()`. Results shown as a list of suggested time slots with "Add to Calendar" button per slot.
- [ ] BETA banner same as EmailPage.
- [ ] `npm run typecheck` passes. `mypy --strict` passes on IPC handler. Verify changes work in app.

---

### Phase 80 — SMS Service Layer

---

### US-249: Define SMSMessage and SMSThread data models
**Description:** As a developer, I need typed data models for SMS messages and threads so all SMS services and the GUI share a consistent structure.

**Acceptance Criteria:**
- [ ] `rex/integrations/models.py` extended with: `SMSMessage(id, thread_id, direction: 'inbound'|'outbound', body, from_number, to_number, sent_at, status: 'sent'|'delivered'|'failed'|'stub')`, `SMSThread(id, contact_name, contact_number, messages, last_message_at, unread_count)`.
- [ ] `mypy --strict rex/integrations/models.py` passes. Unit tests for new models. Tests pass.

---

### US-250: Implement SMSService with Twilio-ready stub/live interface
**Description:** As a developer, I need an SMSService that works in stub mode without Twilio credentials and sends real SMS when credentials are supplied.

**Acceptance Criteria:**
- [ ] `rex/integrations/sms_service.py` implements `SMSService` with: `list_threads() -> list[SMSThread]`, `get_thread(thread_id) -> SMSThread`, `send(to, body) -> SMSMessage`.
- [ ] Stub mode (no Twilio SID/auth): `list_threads()` returns 2 mock threads; `send()` returns an `SMSMessage` with `status='stub'` and logs `"[SMS STUB] Would send to {to}: {body}"`.
- [ ] Live mode: uses `twilio` Python library; `send()` calls `client.messages.create(...)`.
- [ ] `mypy --strict rex/integrations/sms_service.py` passes. Unit tests for stub mode. Tests pass.

---

### US-251: Implement multi-channel message router
**Description:** As a developer, I need a MessageRouter that dispatches outbound messages to the correct channel (SMS or email) based on user intent so Rex can send messages without the user specifying the channel every time.

**Acceptance Criteria:**
- [ ] `rex/integrations/message_router.py` implements `MessageRouter` with `route(contact, body, preferred_channel: str | None = None) -> MessageResult`.
- [ ] If `preferred_channel` is specified, uses it directly.
- [ ] If not: looks up contact in a simple contacts list (`~/.rex/contacts.json`), uses stored preferred channel. Falls back to email if both configured; SMS if only SMS configured; raises `NoChannelError` if neither.
- [ ] `MessageResult(channel, message_id, status)` model added.
- [ ] `mypy --strict rex/integrations/message_router.py` passes. Unit tests for routing logic. Tests pass.

---

### US-252: Add unit tests for SMSService and MessageRouter integration
**Description:** As a developer, I need integration tests that exercise the SMS pipeline end-to-end in stub mode so the wiring is validated before adding the GUI.

**Acceptance Criteria:**
- [ ] `tests/test_sms_integration.py` covers: `SMSService` stub mode send returns correct model; `MessageRouter` routes to SMS when SMS-only configured; routes to email when email-only configured; raises `NoChannelError` when neither configured.
- [ ] All tests use mocked Twilio client (no real API calls).
- [ ] Tests pass. `mypy --strict tests/test_sms_integration.py` passes.

---

### Phase 81 — SMS GUI

---

### US-253: Build SMS thread list view
**Description:** As a user, I want to see my SMS threads in the GUI so I can monitor conversations Rex has had on my behalf.

**Acceptance Criteria:**
- [ ] `gui/src/pages/SMSPage.tsx` renders a list of thread cards: contact name, last message snippet, last message time, unread badge.
- [ ] Threads sorted by `last_message_at` descending.
- [ ] Clicking a thread navigates to or opens the thread detail panel.
- [ ] Thread list fetched from `window.rex.getSMSThreads()` IPC (calls `SMSService.list_threads()`).
- [ ] BETA banner: "SMS integration — enter Twilio credentials in Settings > Integrations to send real messages."
- [ ] `npm run typecheck` passes. Verify changes work in app.

---

### US-254: Build SMS compose and thread detail view
**Description:** As a user, I want to read an SMS thread and compose a reply from the GUI so I can manage text conversations through Rex.

**Acceptance Criteria:**
- [ ] Thread detail panel shows full message list: each message bubble shows body, direction (left=inbound, right=outbound), timestamp.
- [ ] Compose bar at bottom: recipient (pre-filled from thread), message textarea, Send button.
- [ ] Send calls `window.rex.sendSMS(to, body)` IPC → `SMSService.send()`.
- [ ] In stub mode, sent message appears in thread with `status='stub'` and a gray "(stub)" indicator.
- [ ] Outbound message error shown as inline error below compose bar.
- [ ] `npm run typecheck` passes. Verify changes work in app.

---

### US-255: Wire SMS page to Rex SMS backend via IPC and add new conversation flow
**Description:** As a user, I want to start a new SMS conversation from the GUI so I can send a message to any contact without an existing thread.

**Acceptance Criteria:**
- [ ] "New Message" button on SMSPage opens a compose Modal: To field (text, accepts phone number or contact name), message body (textarea), Send button.
- [ ] `window.rex.getSMSThreads`, `window.rex.getSMSThread`, `window.rex.sendSMS` IPC methods all fully implemented end-to-end.
- [ ] After sending a new message, thread list refreshes and new thread appears at top.
- [ ] `npm run typecheck` passes. `mypy --strict` passes on IPC handlers. Verify changes work in app.

---

### Phase 82 — Smart Notifications Service Layer

---

### US-256: Define Notification data model and schema
**Description:** As a developer, I need a typed Notification model with all routing metadata so the notification engine can make correct dispatch decisions.

**Acceptance Criteria:**
- [ ] `rex/notifications/models.py` defines: `Notification(id, title, body, source, priority: 'low'|'medium'|'high'|'critical', channel: 'desktop'|'digest'|'sms'|'email', digest_eligible: bool, quiet_hours_exempt: bool, created_at, delivered_at, read_at, escalation_due_at)`.
- [ ] `NotificationStore` class: backed by SQLite (`~/.rex/notifications.db`). Methods: `add(n)`, `get_unread() -> list[Notification]`, `mark_read(id)`, `update(n)`.
- [ ] `mypy --strict rex/notifications/models.py` passes. Unit tests for store CRUD. Tests pass.

---

### US-257: Implement NotificationRouter with priority-based dispatch
**Description:** As a developer, I need a NotificationRouter that sends high-priority notifications immediately and queues low-priority ones for digest so users aren't interrupted by noise.

**Acceptance Criteria:**
- [ ] `rex/notifications/router.py` implements `NotificationRouter` with `route(notification: Notification)`.
- [ ] `critical` and `high`: dispatch immediately via desktop notification (uses `plyer` or OS notification API); store to DB.
- [ ] `medium`: dispatch via desktop notification unless in quiet hours; store to DB.
- [ ] `low`: mark `digest_eligible=True`, store to DB; do not send desktop notification immediately.
- [ ] `mypy --strict rex/notifications/router.py` passes. Unit tests for each priority level. Tests pass.

---

### US-258: Implement DigestBuilder for batched low-priority notifications
**Description:** As a developer, I need a DigestBuilder that batches low-priority notifications into a single timed digest so users get a clean summary instead of a stream of minor alerts.

**Acceptance Criteria:**
- [ ] `rex/notifications/digest.py` implements `DigestBuilder` with `build_digest() -> str | None`.
- [ ] Collects all unread `digest_eligible` notifications from `NotificationStore`.
- [ ] If any exist: calls LLM to generate a concise paragraph summary ("You have N updates: …").
- [ ] Returns `None` if no digest-eligible notifications.
- [ ] `run_digest()` method: calls `build_digest()`, if non-None dispatches one desktop notification with the summary, marks source notifications as delivered.
- [ ] `mypy --strict rex/notifications/digest.py` passes. Unit tests with mocked LLM. Tests pass.

---

### US-259: Implement QuietHoursGate
**Description:** As a developer, I need a QuietHoursGate that suppresses non-exempt notifications during the user's configured quiet hours so Rex doesn't wake them up.

**Acceptance Criteria:**
- [ ] `rex/notifications/quiet_hours.py` implements `QuietHoursGate` with `is_quiet_now() -> bool` and `should_suppress(notification: Notification) -> bool`.
- [ ] `is_quiet_now()` reads `notifications_quiet_hours_start/end` from Rex config, checks current local time.
- [ ] `should_suppress()` returns True if `is_quiet_now() and not notification.quiet_hours_exempt`.
- [ ] `NotificationRouter.route()` calls `QuietHoursGate.should_suppress()` before dispatching; suppressed notifications stored to DB with `delivered_at=None` and queued for post-quiet-hours delivery.
- [ ] `mypy --strict rex/notifications/quiet_hours.py` passes. Unit tests for boundary conditions (start=23:00, end=07:00 spanning midnight). Tests pass.

---

### US-260: Implement AutoEscalation for unacknowledged notifications
**Description:** As a developer, I need auto-escalation logic that promotes a notification's priority if the user hasn't acknowledged it within a configured threshold so critical items don't get buried.

**Acceptance Criteria:**
- [ ] `rex/notifications/escalation.py` implements `EscalationEngine` with `check_escalations()`.
- [ ] For each unread notification where `escalation_due_at < now`: promote `priority` by one level (low→medium, medium→high, high→critical); re-route via `NotificationRouter.route()` at new priority; update `escalation_due_at` to `now + escalation_delay`.
- [ ] `critical` notifications not escalated further (already max).
- [ ] `escalation_delay` read from Rex config (default: 30 minutes).
- [ ] `check_escalations()` intended to be called on a timer (e.g., every 5 minutes).
- [ ] `mypy --strict rex/notifications/escalation.py` passes. Unit tests. Tests pass.

---

### US-261: Add comprehensive tests for notifications pipeline
**Description:** As a developer, I need an integrated test suite for the notifications pipeline so all components work correctly together.

**Acceptance Criteria:**
- [ ] `tests/test_notifications.py` covers: router dispatches critical immediately; router queues low as digest; quiet hours suppresses medium; quiet-hours-exempt critical not suppressed; escalation promotes priority after delay; digest builder generates summary for N queued items.
- [ ] All tests use mocked desktop notification dispatch and mocked LLM for digest.
- [ ] Tests pass. `mypy --strict tests/test_notifications.py` passes.

---

### Phase 83 — Notifications GUI

---

### US-262: Build notification center panel
**Description:** As a user, I want a notification center in the GUI that shows all my alerts grouped by priority so I can review and dismiss them.

**Acceptance Criteria:**
- [ ] `gui/src/pages/NotificationsPage.tsx` renders notifications grouped: Critical, High, Medium, Low/Digest.
- [ ] Each notification card: title, body (truncated to 2 lines), source badge, received time, read/unread indicator.
- [ ] Clicking a card marks it read (calls `window.rex.markNotificationRead(id)` IPC).
- [ ] "Mark all read" button at top.
- [ ] Unread count in sidebar badge updates in real time via a Zustand store polled every 30 seconds.
- [ ] `npm run typecheck` passes. Verify changes work in app.

---

### US-263: Build notification detail and escalation status view
**Description:** As a user, I want to see full notification details and know whether escalation is pending so I understand urgency.

**Acceptance Criteria:**
- [ ] Clicking a notification card opens a detail panel (right-side slide-in).
- [ ] Panel shows: full title, full body, source, priority badge, channel used, received time, `escalation_due_at` displayed as "Escalates in X minutes" (or "No escalation" if not set).
- [ ] "Dismiss" button: marks read and sets `escalation_due_at = None` (calls `window.rex.dismissNotification(id)` IPC).
- [ ] `npm run typecheck` passes. Verify changes work in app.

---

### US-264: Add notification badge to sidebar and real-time unread count
**Description:** As a user, I want the sidebar to show how many unread notifications I have so I always know when something needs attention.

**Acceptance Criteria:**
- [ ] `gui/src/store/notificationsStore.ts` (Zustand) holds `unreadCount: number` and `fetchUnreadCount()` action.
- [ ] `fetchUnreadCount()` calls `window.rex.getUnreadNotificationCount(): Promise<number>` IPC (calls `NotificationStore.get_unread()` and returns `.length`).
- [ ] App polls `fetchUnreadCount()` every 30 seconds via `useEffect` in `AppLayout`.
- [ ] Sidebar Notifications item displays the count as a red badge if > 0; hidden if 0.
- [ ] `npm run typecheck` passes. Verify changes work in app.

---

### US-265: Wire notification panel to Rex backend and add in-app toast dispatch
**Description:** As a developer, I need all notification IPC methods fully implemented and in-app toasts shown for new high-priority notifications so users see alerts even while using the GUI.

**Acceptance Criteria:**
- [ ] `window.rex.getNotifications`, `window.rex.markNotificationRead`, `window.rex.dismissNotification`, `window.rex.getUnreadNotificationCount` IPC methods fully implemented end-to-end.
- [ ] Main process listens for new notifications emitted by Rex Python backend (via stdout event or IPC push); forwards `critical` and `high` priority ones to renderer via `webContents.send('rex:newNotification', notification)`.
- [ ] Renderer listens for `rex:newNotification` and triggers a Toast (using `useToast` from US-180) with the notification title and body.
- [ ] Toast type matches priority: `critical`→error, `high`→warning, `medium`→info.
- [ ] `npm run typecheck` passes. `mypy --strict` passes on all notification IPC handlers. Verify changes work in app.

---

## Summary

| Phase | Area | Stories | US Range |
|-------|------|---------|----------|
| 61 | Electron Scaffold | 3 | US-175 – US-177 |
| 62 | Design System | 3 | US-178 – US-180 |
| 63 | App Shell | 3 | US-181 – US-183 |
| 64 | Text Chat | 4 | US-184 – US-187 |
| 65 | Voice Chat | 4 | US-188 – US-191 |
| 66 | Task Scheduler | 4 | US-192 – US-195 |
| 67 | Calendar | 3 | US-196 – US-198 |
| 68 | Reminders | 3 | US-199 – US-201 |
| 69 | Memories Editor | 4 | US-202 – US-205 |
| 70 | Settings | 7 | US-206 – US-212 |
| 71 | GUI Polish | 4 | US-213 – US-216 |
| 72 | LLM Planner | 4 | US-217 – US-220 |
| 73 | Dynamic Replanning | 4 | US-221 – US-224 |
| 74 | Feedback Learning | 4 | US-225 – US-228 |
| 75 | Multi-Goal Planning | 4 | US-229 – US-232 |
| 76 | Cost Optimization | 4 | US-233 – US-236 |
| 77 | Preference Learning | 4 | US-237 – US-240 |
| 78 | Email/Calendar Service | 5 | US-241 – US-245 |
| 79 | Email/Calendar GUI | 3 | US-246 – US-248 |
| 80 | SMS Service | 4 | US-249 – US-252 |
| 81 | SMS GUI | 3 | US-253 – US-255 |
| 82 | Notifications Service | 6 | US-256 – US-261 |
| 83 | Notifications GUI | 4 | US-262 – US-265 |
| **Total** | | **91** | **US-175 – US-265** |
