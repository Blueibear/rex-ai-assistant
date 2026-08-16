import React, { useEffect, useState } from 'react'
import type { Settings, SystemSettings } from '../../types/ipc'
import { useToast } from '../../components/ui/Toast'

export function SystemSettingsSection(): React.ReactElement {
  const addToast = useToast()
  const [settings, setSettings] = useState<SystemSettings>({
    toolTimeoutSeconds: 10,
    requireConfirmSystemChanges: true,
    allowedFileRoots: '',
    debugLogging: false
  })
  const [saved, setSaved] = useState(false)
  const [restarting, setRestarting] = useState(false)
  const [showResetConfirm, setShowResetConfirm] = useState(false)
  const [resetting, setResetting] = useState(false)

  useEffect(() => {
    window.rex
      .getSettings('system')
      .then((s: Settings) => {
        setSettings({
          toolTimeoutSeconds: typeof s.toolTimeoutSeconds === 'number' ? s.toolTimeoutSeconds : 10,
          requireConfirmSystemChanges:
            typeof s.requireConfirmSystemChanges === 'boolean' ? s.requireConfirmSystemChanges : true,
          allowedFileRoots: typeof s.allowedFileRoots === 'string' ? s.allowedFileRoots : '',
          debugLogging: typeof s.debugLogging === 'boolean' ? s.debugLogging : false
        })
      })
      .catch(() => {})
  }, [])

  function handleSave(): void {
    window.rex
      .setSettings('system', settings as unknown as Settings)
      .then((result) => {
        if (result.ok) {
          setSaved(true)
          setTimeout(() => setSaved(false), 2000)
        } else {
          addToast(result.error ?? 'Failed to save system settings', 'error')
        }
      })
      .catch(() => addToast('Failed to save system settings', 'error'))
  }

  function handleRestart(): void {
    setRestarting(true)
    window.rex
      .restartRex()
      .catch(() => addToast('Failed to restart Rex', 'error'))
      .finally(() => setRestarting(false))
  }

  function handleResetConfirm(): void {
    setResetting(true)
    window.rex
      .resetToDefaults()
      .then((res) => {
        if (res.ok) {
          addToast('Settings reset to defaults. Restarting…', 'success')
          setTimeout(() => {
            void window.rex.restartRex()
          }, 1500)
        } else {
          addToast(res.error ?? 'Reset failed', 'error')
        }
      })
      .catch(() => addToast('Reset failed', 'error'))
      .finally(() => {
        setResetting(false)
        setShowResetConfirm(false)
      })
  }

  return (
    <div className="p-6 max-w-lg">
      <h2 className="text-lg font-semibold text-text-primary mb-6">System &amp; Advanced</h2>

      {/* Tool timeout */}
      <div className="mb-6">
        <div className="flex items-center justify-between mb-2">
          <label className="text-sm font-medium text-text-primary">
            Tool Timeout
            <span className="ml-2 text-xs text-text-secondary font-normal">
              {settings.toolTimeoutSeconds}s
            </span>
          </label>
        </div>
        <div className="flex items-center gap-2 text-xs text-text-secondary">
          <span>1s</span>
          <input
            type="range"
            min={1}
            max={60}
            step={1}
            value={settings.toolTimeoutSeconds}
            onChange={(e) =>
              setSettings((s) => ({ ...s, toolTimeoutSeconds: parseInt(e.target.value) }))
            }
            className="flex-1 accent-accent"
          />
          <span>60s</span>
        </div>
        <p className="mt-1 text-xs text-text-secondary">
          Maximum time Rex waits for a tool (email, calendar, search) before timing out.
        </p>
      </div>

      {/* Toggles */}
      <div className="mb-6 space-y-4">
        <div className="flex items-center justify-between rounded-xl border border-border bg-surface-raised p-4">
          <div>
            <div className="text-sm font-medium text-text-primary">Require confirmation for system changes</div>
            <div className="text-xs text-text-secondary mt-0.5">
              Ask before Rex modifies volume, brightness, or other system settings.
            </div>
          </div>
          <button
            type="button"
            role="switch"
            aria-checked={settings.requireConfirmSystemChanges}
            onClick={() =>
              setSettings((s) => ({ ...s, requireConfirmSystemChanges: !s.requireConfirmSystemChanges }))
            }
            className={[
              'relative inline-flex h-6 w-11 shrink-0 cursor-pointer rounded-full border-2 border-transparent transition-colors focus:outline-none',
              settings.requireConfirmSystemChanges ? 'bg-accent' : 'bg-border'
            ].join(' ')}
          >
            <span
              className={[
                'pointer-events-none inline-block h-5 w-5 transform rounded-full bg-white shadow transition-transform',
                settings.requireConfirmSystemChanges ? 'translate-x-5' : 'translate-x-0'
              ].join(' ')}
            />
          </button>
        </div>

        <div className="flex items-center justify-between rounded-xl border border-border bg-surface-raised p-4">
          <div>
            <div className="text-sm font-medium text-text-primary">Debug logging</div>
            <div className="text-xs text-text-secondary mt-0.5">
              Write verbose DEBUG-level logs. Useful for diagnosing issues.
            </div>
          </div>
          <button
            type="button"
            role="switch"
            aria-checked={settings.debugLogging}
            onClick={() => setSettings((s) => ({ ...s, debugLogging: !s.debugLogging }))}
            className={[
              'relative inline-flex h-6 w-11 shrink-0 cursor-pointer rounded-full border-2 border-transparent transition-colors focus:outline-none',
              settings.debugLogging ? 'bg-accent' : 'bg-border'
            ].join(' ')}
          >
            <span
              className={[
                'pointer-events-none inline-block h-5 w-5 transform rounded-full bg-white shadow transition-transform',
                settings.debugLogging ? 'translate-x-5' : 'translate-x-0'
              ].join(' ')}
            />
          </button>
        </div>
      </div>

      {/* Allowed file roots */}
      <div className="mb-8">
        <label className="block text-sm font-medium text-text-primary mb-1">
          Allowed File Roots
        </label>
        <p className="text-xs text-text-secondary mb-2">
          Directory paths Rex is allowed to read and write. Defaults to your home directory if left blank.
        </p>
        {/* Folder list */}
        {settings.allowedFileRoots.split(',').map((p) => p.trim()).filter(Boolean).length > 0 && (
          <ul className="mb-2 space-y-1">
            {settings.allowedFileRoots.split(',').map((p) => p.trim()).filter(Boolean).map((folder) => (
              <li key={folder} className="flex items-center justify-between rounded-lg border border-border bg-surface-raised px-3 py-1.5 text-sm text-text-primary">
                <span className="truncate mr-2">{folder}</span>
                <button
                  type="button"
                  onClick={() => {
                    const updated = settings.allowedFileRoots
                      .split(',')
                      .map((p) => p.trim())
                      .filter((p) => p && p !== folder)
                      .join(', ')
                    setSettings((s) => ({ ...s, allowedFileRoots: updated }))
                  }}
                  className="shrink-0 text-xs text-text-secondary hover:text-red-500 focus:outline-none"
                  aria-label={`Remove ${folder}`}
                >
                  ✕
                </button>
              </li>
            ))}
          </ul>
        )}
        {/* Add folder button */}
        <button
          type="button"
          onClick={() => {
            void window.rex.pickFolder().then((res) => {
              if (res.ok && res.path) {
                const existing = settings.allowedFileRoots.split(',').map((p) => p.trim()).filter(Boolean)
                if (!existing.includes(res.path)) {
                  const updated = [...existing, res.path].join(', ')
                  setSettings((s) => ({ ...s, allowedFileRoots: updated }))
                }
              }
            })
          }}
          className="mb-3 rounded-lg border border-border bg-surface-raised px-3 py-1.5 text-sm text-text-primary hover:bg-surface-elevated focus:outline-none"
        >
          + Add Folder
        </button>
        {/* Fallback raw text input */}
        <input
          type="text"
          value={settings.allowedFileRoots}
          onChange={(e) => setSettings((s) => ({ ...s, allowedFileRoots: e.target.value }))}
          placeholder="C:\Users\you, D:\Documents"
          className="w-full rounded-lg border border-border bg-bg px-3 py-2 text-sm text-text-primary placeholder:text-text-secondary focus:border-accent focus:outline-none"
        />
      </div>

      {/* Save */}
      <div className="flex items-center gap-3 mb-8">
        <button
          type="button"
          onClick={handleSave}
          className="rounded-lg bg-accent px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-accent/90 focus:outline-none"
        >
          Save
        </button>
        {saved && (
          <span className="text-xs text-success flex items-center gap-1">
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3">
              <polyline points="20 6 9 17 4 12" />
            </svg>
            Saved
          </span>
        )}
      </div>

      {/* Restart Rex */}
      <div className="border-t border-border pt-6">
        <h3 className="mb-1 text-sm font-semibold text-text-primary">Restart Rex</h3>
        <p className="mb-4 text-xs text-text-secondary">
          Gracefully restarts the Rex application. Use this after changing advanced settings.
        </p>
        <button
          type="button"
          onClick={handleRestart}
          disabled={restarting}
          className="flex items-center gap-2 rounded-lg border border-border bg-bg px-4 py-2 text-sm font-medium text-text-primary transition-colors hover:bg-surface-raised disabled:opacity-50 focus:outline-none"
        >
          {restarting ? (
            <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83" />
            </svg>
          ) : (
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <polyline points="23 4 23 10 17 10" />
              <path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10" />
            </svg>
          )}
          Restart Rex
        </button>
      </div>

      {/* Reset to Defaults */}
      <div className="border-t border-border pt-6 mt-6">
        <h3 className="mb-1 text-sm font-semibold text-text-primary">Reset to Defaults</h3>
        <p className="mb-4 text-xs text-text-secondary">
          Replaces <code className="font-mono">config/rex_config.json</code> with the factory defaults from{' '}
          <code className="font-mono">rex_config.example.json</code>. User profiles, voice samples, and{' '}
              Credentials stored in the Windows credential vault are not affected.
        </p>
        {!showResetConfirm ? (
          <button
            type="button"
            onClick={() => setShowResetConfirm(true)}
            className="flex items-center gap-2 rounded-lg border border-red-400 bg-bg px-4 py-2 text-sm font-medium text-red-500 transition-colors hover:bg-red-50 focus:outline-none"
          >
            Reset to Defaults
          </button>
        ) : (
          <div className="rounded-xl border border-red-300 bg-red-50 p-4">
            <p className="text-sm font-medium text-red-700 mb-1">Are you sure?</p>
            <p className="text-xs text-red-600 mb-4">
              This will overwrite your current runtime configuration with factory defaults and restart Rex.
            User profiles, voice samples, and credentials in the Windows vault will not be changed.
            </p>
            <div className="flex gap-3">
              <button
                type="button"
                onClick={handleResetConfirm}
                disabled={resetting}
                className="rounded-lg bg-red-500 px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-red-600 disabled:opacity-50 focus:outline-none"
              >
                {resetting ? 'Resetting…' : 'Yes, Reset'}
              </button>
              <button
                type="button"
                onClick={() => setShowResetConfirm(false)}
                disabled={resetting}
                className="rounded-lg border border-border bg-bg px-4 py-2 text-sm font-medium text-text-primary transition-colors hover:bg-surface-raised disabled:opacity-50 focus:outline-none"
              >
                Cancel
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
