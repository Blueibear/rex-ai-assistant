import React from 'react'
import type { IntegrationsSettings } from '../../../types/ipc'
import { PasswordInput, SavedIndicator } from '../shared'
import { ConnectionBadge, TestConnectionButton } from './IntegrationControls'
import type { TestStatus } from './useIntegrationsSettingsController'

interface OpenClawIntegrationSectionProps {
  form: IntegrationsSettings
  savedField: keyof IntegrationsSettings | null
  status: TestStatus
  error?: string
  inputClass: string
  hasStoredCredential: (form: IntegrationsSettings, field: keyof IntegrationsSettings) => boolean
  setForm: React.Dispatch<React.SetStateAction<IntegrationsSettings>>
  onFieldChange: <K extends keyof IntegrationsSettings>(field: K, value: IntegrationsSettings[K]) => void
  onTest: () => void
}

export function OpenClawIntegrationSection({
  form,
  savedField,
  status,
  error,
  inputClass,
  hasStoredCredential,
  setForm,
  onFieldChange,
  onTest
}: OpenClawIntegrationSectionProps): React.ReactElement {
  const tokenStored = hasStoredCredential(form, 'openclawToken')
  const gatewayConfigured = form.openclawGatewayUrl.trim() !== ''
  const tokenAvailable = tokenStored || form.openclawToken.trim() !== ''
  const controlsDisabled = !gatewayConfigured || !tokenAvailable
  const tokenPlaceholder = tokenStored
    ? 'Stored credential (enter to replace)'
    : 'Enter gateway token'

  return (
    <section className="mb-7">
      <div className="mb-4 flex items-center justify-between gap-3">
        <div>
          <h3 className="text-sm font-semibold text-text-primary">OpenClaw</h3>
          <div className="mt-1 text-xs font-medium text-warning">Experimental - off by default</div>
        </div>
        <ConnectionBadge status={status} hasCredentials={gatewayConfigured && tokenStored} />
      </div>
      <p className="mb-4 text-xs text-text-secondary">
        Optional external capability gateway. The token stays in the credential vault and is never returned to the renderer after saving.
      </p>
      <div className="mb-4">
        <div className="mb-1.5 flex items-center justify-between">
          <label htmlFor="openclawGatewayUrl" className="text-sm font-medium text-text-primary">Gateway URL</label>
          <SavedIndicator visible={savedField === 'openclawGatewayUrl'} />
        </div>
        <input id="openclawGatewayUrl" type="url" value={form.openclawGatewayUrl}
          placeholder="http://127.0.0.1:18789"
          onChange={(event) => setForm((current) => ({ ...current, openclawGatewayUrl: event.target.value }))}
          onBlur={(event) => onFieldChange('openclawGatewayUrl', event.target.value)}
          className={inputClass} />
      </div>
      <div className="mb-4">
        <div className="mb-1.5 flex items-center justify-between">
          <label htmlFor="openclawToken" className="text-sm font-medium text-text-primary">Gateway Token</label>
          <SavedIndicator visible={savedField === 'openclawToken'} />
        </div>
        <PasswordInput id="openclawToken" value={form.openclawToken} placeholder={tokenPlaceholder}
          onChange={(value) => setForm((current) => ({ ...current, openclawToken: value }))}
          onBlur={() => { if (form.openclawToken) onFieldChange('openclawToken', form.openclawToken) }} />
      </div>
      <div className="mb-4 space-y-2">
        <label className="flex items-center gap-2 text-sm text-text-primary">
          <input type="checkbox" checked={form.openclawToolsEnabled} disabled={controlsDisabled}
            onChange={(event) => onFieldChange('openclawToolsEnabled', event.target.checked)} />
          Enable OpenClaw tools
        </label>
        <label className="flex items-center gap-2 text-sm text-text-primary">
          <input type="checkbox" checked={form.openclawVoiceEnabled} disabled={controlsDisabled}
            onChange={(event) => onFieldChange('openclawVoiceEnabled', event.target.checked)} />
          Enable OpenClaw voice backend
        </label>
      </div>
      <TestConnectionButton status={status} error={error} onTest={onTest} />
      {status === 'configured' && !error && (
        <p className="mt-2 text-xs text-text-secondary">
          Gateway reachable; authentication and tool capability are not yet proven.
        </p>
      )}
    </section>
  )
}
