import React, { useEffect, useState } from 'react'
import { NavLink } from 'react-router-dom'

type TestStatus = 'idle' | 'testing' | 'success' | 'failure'

export function HomeAssistantSettingsPage(): React.ReactElement {
  const [haBaseUrl, setHaBaseUrl] = useState('')
  const [haToken, setHaToken] = useState('')
  const [testStatus, setTestStatus] = useState<TestStatus>('idle')
  const [testError, setTestError] = useState('')
  const [saveStatus, setSaveStatus] = useState<'idle' | 'saving' | 'saved' | 'error'>('idle')
  const [saveError, setSaveError] = useState('')

  // Pre-populate from current config (best-effort).
  useEffect(() => {
    window.rex
      .getSettings('integrations')
      .then((settings) => {
        if (typeof settings.haUrl === 'string') setHaBaseUrl(settings.haUrl)
        if (typeof settings.haToken === 'string') setHaToken(settings.haToken)
      })
      .catch(() => {/* ignore */})
  }, [])

  const handleTest = async (): Promise<void> => {
    setTestStatus('testing')
    setTestError('')
    try {
      const body = await window.rex.testHomeAssistant(haBaseUrl, haToken)
      if (body.ok) {
        setTestStatus('success')
      } else {
        setTestStatus('failure')
        setTestError(body.error ?? 'Connection failed.')
      }
    } catch {
      setTestStatus('failure')
      setTestError('Network error. Is the Rex backend running?')
    }
  }

  const handleSave = async (): Promise<void> => {
    setSaveStatus('saving')
    setSaveError('')
    try {
      const resp = await window.rex.saveHomeAssistant(haBaseUrl, haToken)
      if (resp.ok) {
        setSaveStatus('saved')
      } else {
        setSaveError(resp.error ?? 'Save failed.')
        setSaveStatus('error')
      }
    } catch {
      setSaveError('Network error.')
      setSaveStatus('error')
    }
  }

  const canSave = haBaseUrl.trim().length > 0

  return (
    <div className="p-6 max-w-lg mx-auto space-y-6">
      {/* Breadcrumb */}
      <nav className="text-sm text-text-muted" aria-label="Breadcrumb">
        <NavLink to="/settings" className="hover:text-text-primary transition-colors">
          Settings
        </NavLink>
        <span className="mx-2">→</span>
        <span className="text-text-primary">Home Assistant</span>
      </nav>

      <section>
        <h2 className="text-xl font-semibold text-text-primary mb-1">Home Assistant</h2>
        <p className="text-text-secondary text-sm mb-6">
          Connect Rex to your Home Assistant instance to control devices and automations.
        </p>

        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-text-primary mb-1">
              Home Assistant URL
            </label>
            <input
              type="url"
              value={haBaseUrl}
              onChange={(e) => {
                setHaBaseUrl(e.target.value)
                setTestStatus('idle')
                setSaveStatus('idle')
              }}
              className="w-full px-3 py-2 rounded-lg border border-border bg-surface text-text-primary text-sm focus:outline-none focus:ring-2 focus:ring-accent/50"
              placeholder="http://homeassistant.local:8123"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-text-primary mb-1">
              Long-Lived Access Token
            </label>
            <input
              type="password"
              autoComplete="off"
              value={haToken}
              onChange={(e) => {
                setHaToken(e.target.value)
                setTestStatus('idle')
                setSaveStatus('idle')
              }}
              className="w-full px-3 py-2 rounded-lg border border-border bg-surface text-text-primary text-sm focus:outline-none focus:ring-2 focus:ring-accent/50"
              placeholder="eyJ..."
            />
            <p className="text-text-muted text-xs mt-1">
              Generate in HA → Profile → Long-Lived Access Tokens. Stored in your local .env file.
            </p>
          </div>
        </div>

        {/* Test result */}
        {testStatus === 'success' && (
          <p className="mt-3 text-sm text-green-400">Connection successful!</p>
        )}
        {testStatus === 'failure' && (
          <p className="mt-3 text-sm text-red-400">{testError || 'Connection failed.'}</p>
        )}

        {/* Save result */}
        {saveStatus === 'saved' && (
          <p className="mt-3 text-sm text-green-400">Configuration saved.</p>
        )}
        {saveStatus === 'error' && (
          <p className="mt-3 text-sm text-red-400">{saveError || 'Save failed.'}</p>
        )}

        {/* Action buttons */}
        <div className="flex items-center gap-3 mt-6">
          <button
            type="button"
            onClick={() => void handleTest()}
            disabled={!haBaseUrl.trim() || testStatus === 'testing'}
            className="px-4 py-2 rounded-lg border border-border text-text-secondary text-sm hover:bg-surface-raised transition-colors disabled:opacity-50"
          >
            {testStatus === 'testing' ? 'Testing…' : 'Test Connection'}
          </button>
          <button
            type="button"
            onClick={() => void handleSave()}
            disabled={!canSave || saveStatus === 'saving'}
            className="px-4 py-2 rounded-lg bg-accent text-white text-sm font-medium hover:bg-accent/90 transition-colors disabled:opacity-50"
          >
            {saveStatus === 'saving' ? 'Saving…' : 'Save'}
          </button>
        </div>
      </section>
    </div>
  )
}
