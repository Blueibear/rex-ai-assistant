import React, { useEffect, useState } from 'react'
import { NavLink } from 'react-router-dom'
import type { CapabilityInfo, IntegrationInventoryItem } from '../types/ipc'

type Integration = IntegrationInventoryItem
type Capability = CapabilityInfo

function StatusBadge({ integration }: { integration: Integration }): React.ReactElement {
  if (integration.status === 'connected') {
    return (
      <span className="text-xs font-medium px-2 py-0.5 rounded-full bg-green-500/15 text-green-400">
        Connected
      </span>
    )
  }
  if (integration.status === 'error') {
    return (
      <span className="text-xs font-medium px-2 py-0.5 rounded-full bg-red-500/15 text-red-400">
        Connection error
      </span>
    )
  }
  if (integration.configured && integration.testable) {
    return (
      <span className="text-xs font-medium px-2 py-0.5 rounded-full bg-yellow-500/15 text-yellow-400">
        Untested
      </span>
    )
  }
  if (integration.configured) {
    return (
      <span className="text-xs font-medium px-2 py-0.5 rounded-full bg-green-500/15 text-green-400">
        Configured
      </span>
    )
  }
  return (
    <span className="text-xs font-medium px-2 py-0.5 rounded-full bg-surface-raised text-text-muted">
      Not configured
    </span>
  )
}

export function IntegrationsPage(): React.ReactElement {
  const [integrations, setIntegrations] = useState<Integration[]>([])
  const [capabilities, setCapabilities] = useState<Capability[]>([])
  const [loading, setLoading] = useState(true)
  const [loadError, setLoadError] = useState<string | null>(null)

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
                <div key={int.key} className="flex items-center justify-between px-4 py-3 bg-surface gap-3">
                  <span className="text-text-primary text-sm font-medium flex-1 min-w-0 truncate">
                    {int.name}
                  </span>
                  <div className="flex items-center gap-3 flex-shrink-0">
                    <StatusBadge integration={int} />
                    {int.configure_url && (
                      <NavLink
                        to={int.configure_url}
                        className="text-xs text-accent hover:underline"
                      >
                        Configure →
                      </NavLink>
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
