import React, { useEffect, useState } from 'react'
import { NavLink } from 'react-router-dom'
import type { CapabilityInfo, IntegrationInventoryItem } from '../types/ipc'

type Integration = IntegrationInventoryItem
type Capability = CapabilityInfo
type TestableIntegrationKey = 'email' | 'calendar' | 'sms' | 'homeassistant' | 'phone' | 'openclaw'

function isTestableIntegrationKey(key: string): key is TestableIntegrationKey {
  return ['email', 'calendar', 'sms', 'homeassistant', 'phone', 'openclaw'].includes(key)
}

function StatusBadge({ integration }: { integration: Integration }): React.ReactElement {
  const labels: Record<Integration['state'], string> = {
    unavailable: 'Unavailable',
    unconfigured: 'Not configured',
    configured: 'Configured only',
    reachable: 'Reachable',
    authenticated: 'Authenticated',
    degraded: 'Degraded',
    read_only: 'Read-only',
    write_capable: 'Write-capable',
    write_tested: 'Write-tested',
    verified: 'Verified'
  }
  const positive = ['authenticated', 'read_only', 'write_capable', 'write_tested', 'verified']
    .includes(integration.state)
  const negative = integration.state === 'unavailable' || integration.state === 'degraded'
  return (
    <span className={`text-xs font-medium px-2 py-0.5 rounded-full ${
      positive
        ? 'bg-green-500/15 text-green-400'
        : negative
          ? 'bg-red-500/15 text-red-400'
          : integration.state === 'configured' || integration.state === 'reachable'
            ? 'bg-yellow-500/15 text-yellow-400'
            : 'bg-surface-raised text-text-muted'
    }`}>
      {labels[integration.state]}
    </span>
  )
}

export function IntegrationsPage(): React.ReactElement {
  const [integrations, setIntegrations] = useState<Integration[]>([])
  const [capabilities, setCapabilities] = useState<Capability[]>([])
  const [loading, setLoading] = useState(true)
  const [loadError, setLoadError] = useState<string | null>(null)
  const [testingKey, setTestingKey] = useState<string | null>(null)
  const [testErrors, setTestErrors] = useState<Record<string, string>>({})

  useEffect(() => {
    let cancelled = false

    Promise.allSettled([window.rex.getIntegrations(), window.rex.getCapabilities()])
      .then(([integrationResult, capabilityResult]) => {
        if (cancelled) return

        if (integrationResult.status === 'rejected' || !integrationResult.value.ok) {
          const message =
            integrationResult.status === 'rejected'
              ? integrationResult.reason instanceof Error
                ? integrationResult.reason.message
                : String(integrationResult.reason)
              : integrationResult.value.error ?? 'Integration status failed to load.'
          setLoadError(message)
          setIntegrations([])
        } else {
          setLoadError(null)
          setIntegrations(integrationResult.value.integrations)
        }

        if (capabilityResult.status === 'fulfilled' && capabilityResult.value.ok) {
          setCapabilities(capabilityResult.value.capabilities)
        } else {
          setCapabilities([])
        }
      })
      .finally(() => {
        if (!cancelled) setLoading(false)
      })

    return () => {
      cancelled = true
    }
  }, [])

  const handleIntegrationTest = async (integration: Integration): Promise<void> => {
    if (!integration.testable || !isTestableIntegrationKey(integration.key)) return
    setTestingKey(integration.key)
    setTestErrors((current) => ({ ...current, [integration.key]: '' }))
    try {
      const result = await window.rex.testIntegration(integration.key)
      const nextState = result.state ?? (result.ok ? 'reachable' : 'degraded')
      setIntegrations((current) =>
        current.map((item) =>
          item.key === integration.key
            ? {
                ...item,
                state: nextState,
                testedAt: new Date().toISOString(),
                error: result.error
              }
            : item
        )
      )
      if (result.error) {
        setTestErrors((current) => ({ ...current, [integration.key]: result.error ?? '' }))
      }
    } catch {
      const error = 'Connection test failed.'
      setIntegrations((current) =>
        current.map((item) =>
          item.key === integration.key
            ? { ...item, state: 'degraded', testedAt: new Date().toISOString(), error }
            : item
        )
      )
      setTestErrors((current) => ({ ...current, [integration.key]: error }))
    } finally {
      setTestingKey(null)
    }
  }

  const grouped = capabilities.reduce<Record<string, Capability[]>>((acc, cap) => {
    const cat = cap.category ?? 'General'
    if (!acc[cat]) acc[cat] = []
    acc[cat].push(cap)
    return acc
  }, {})
  const configuredCount = integrations.filter((int) => int.configured).length

  return (
    <div className="p-6 max-w-2xl mx-auto space-y-8">
      {/* Integrations */}
      <section>
        <h2 className="text-xl font-semibold text-text-primary mb-4">Integrations</h2>

        {loading ? (
          <p className="text-text-muted text-sm">Loading…</p>
        ) : loadError ? (
          <div className="rounded-lg border border-red-500/30 bg-red-500/10 px-4 py-3">
            <p className="text-sm font-medium text-red-300">Could not load integration status.</p>
            <p className="mt-1 text-xs text-red-200/80">{loadError}</p>
          </div>
        ) : integrations.length === 0 ? (
          <p className="text-text-muted text-sm">No integration inventory is available.</p>
        ) : (
          <>
            {configuredCount === 0 && (
              <p className="text-text-muted text-sm mb-3">
                No integrations are configured yet. Available integrations are listed below.
              </p>
            )}
            <div className="divide-y divide-border border border-border rounded-xl overflow-hidden">
              {integrations.map((int) => (
                <div key={int.key} className="flex items-start justify-between px-4 py-3 bg-surface gap-4">
                  <div className="flex-1 min-w-0">
                    <p className="text-text-primary text-sm font-medium">{int.name}</p>
                    <p className="text-text-muted text-xs mt-1">{int.detail}</p>
                    <p className="text-text-secondary text-xs mt-1">
                      Next action: {int.next_action}
                    </p>
                  </div>
                  <div className="flex flex-col items-end gap-2 flex-shrink-0">
                    <div className="flex items-center gap-3">
                      <StatusBadge integration={int} />
                      {int.testable && isTestableIntegrationKey(int.key) && (
                        <button
                          type="button"
                          onClick={() => void handleIntegrationTest(int)}
                          disabled={testingKey === int.key}
                          className="text-xs text-accent hover:underline disabled:opacity-50"
                        >
                          {testingKey === int.key ? 'Testing…' : 'Test connection'}
                        </button>
                      )}
                      {int.configure_url && (
                        <NavLink
                          to={int.configure_url}
                          className="text-xs text-accent hover:underline"
                        >
                          Configure →
                        </NavLink>
                      )}
                    </div>
                    {testErrors[int.key] && (
                      <p className="max-w-xs text-right text-xs text-red-300">{testErrors[int.key]}</p>
                    )}
                  </div>
                </div>
              ))}
            </div>
          </>
        )}
      </section>

      {/* Capabilities — only rendered when the registry returns entries */}
      {!loading && Object.keys(grouped).length > 0 && (
        <section>
          <h2 className="text-xl font-semibold text-text-primary mb-2">Capabilities</h2>
          <p className="text-text-secondary text-sm mb-4">
            All tools Rex can use. Only enabled capabilities are available during interactions.
          </p>

          <div className="space-y-4">
            {Object.entries(grouped).map(([category, caps]) => (
              <div key={category}>
                <h3 className="text-xs font-semibold uppercase tracking-wider text-text-muted mb-2">
                  {category}
                </h3>
                <div className="divide-y divide-border border border-border rounded-xl overflow-hidden">
                  {caps.map((cap) => (
                    <div key={cap.name} className="flex items-start justify-between px-4 py-3 bg-surface gap-4">
                      <div className="min-w-0">
                        <p className="text-text-primary text-sm font-medium">{cap.name}</p>
                        {cap.description && (
                          <p className="text-text-muted text-xs mt-0.5 truncate">{cap.description}</p>
                        )}
                        {cap.state && (
                          <p className="text-text-muted text-xs mt-0.5">
                            Evidence: {cap.state.replace(/_/g, ' ')}
                          </p>
                        )}
                      </div>
                      <span
                        className={`flex-shrink-0 text-xs font-medium px-2 py-0.5 rounded-full ${
                          cap.enabled
                            ? 'bg-green-500/15 text-green-400'
                            : 'bg-surface-raised text-text-muted'
                        }`}
                      >
                        {cap.enabled ? 'Enabled' : 'Disabled'}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </section>
      )}
    </div>
  )
}
