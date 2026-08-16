import React from 'react'

export type ModelDiscoveryStatus = 'idle' | 'loading' | 'success' | 'error'

interface ModelDiscoveryResultsProps {
  providerLabel: string
  status: ModelDiscoveryStatus
  models: string[]
  selectedModel: string
  error: string
  onSelect: (model: string) => void
}

export function ModelDiscoveryResults({
  providerLabel,
  status,
  models,
  selectedModel,
  error,
  onSelect
}: ModelDiscoveryResultsProps): React.ReactElement | null {
  if (status === 'idle') return null

  if (status === 'loading') {
    return <p className="mt-2 text-xs text-text-secondary">Discovering models…</p>
  }

  if (status === 'error') {
    return (
      <p role="alert" className="mt-2 text-xs text-danger">
        {error || `${providerLabel} model discovery failed.`}
      </p>
    )
  }

  if (models.length === 0) {
    return (
      <p className="mt-2 text-xs text-text-secondary">
        No models reported by the configured {providerLabel} endpoint.
      </p>
    )
  }

  const value = models.includes(selectedModel) ? selectedModel : ''
  return (
    <div className="mt-3">
      <label className="mb-1.5 block text-sm font-medium text-text-primary">
        Discovered models
      </label>
      <select
        value={value}
        onChange={(event) => {
          if (event.target.value) onSelect(event.target.value)
        }}
        className="w-full rounded-lg border border-border bg-surface-raised px-3 py-2 text-sm text-text-primary focus:outline-none focus:ring-2 focus:ring-accent"
      >
        <option value="">Select a discovered model</option>
        {models.map((model) => (
          <option key={model} value={model}>{model}</option>
        ))}
      </select>
    </div>
  )
}
