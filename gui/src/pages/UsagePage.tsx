import React, { useEffect, useState } from 'react'
import type { UsageSummary, UsageBucket, UsagePeriodSplit } from '../types/ipc'

function fmt(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}K`
  return String(n)
}

function PercentBar({ local, cloud }: { local: number; cloud: number }): React.ReactElement {
  const total = local + cloud
  const localPct = total === 0 ? 50 : Math.round((local / total) * 100)
  const cloudPct = 100 - localPct

  return (
    <div className="mt-2">
      <div className="flex rounded-full overflow-hidden h-3 bg-surface-raised">
        <div
          className="bg-green-500 transition-all duration-500"
          style={{ width: `${localPct}%` }}
          title={`Local ${localPct}%`}
        />
        <div
          className="bg-blue-500 transition-all duration-500"
          style={{ width: `${cloudPct}%` }}
          title={`Cloud ${cloudPct}%`}
        />
      </div>
      <div className="flex justify-between text-xs text-text-secondary mt-1">
        <span className="flex items-center gap-1">
          <span className="inline-block w-2 h-2 rounded-full bg-green-500" />
          Local {localPct}%
        </span>
        <span className="flex items-center gap-1">
          Cloud {cloudPct}%
          <span className="inline-block w-2 h-2 rounded-full bg-blue-500" />
        </span>
      </div>
    </div>
  )
}

function BucketCard({
  label,
  bucket,
  color
}: {
  label: string
  bucket: UsageBucket
  color: string
}): React.ReactElement {
  return (
    <div className={`flex flex-col gap-1 p-4 rounded-xl border border-border bg-surface`}>
      <div className={`text-xs font-semibold uppercase tracking-wide ${color}`}>{label}</div>
      <div className="text-2xl font-bold text-text-primary">{fmt(bucket.requests)}</div>
      <div className="text-xs text-text-secondary">requests</div>
      <div className="text-lg font-semibold text-text-primary mt-1">{fmt(bucket.tokens)}</div>
      <div className="text-xs text-text-secondary">tokens</div>
    </div>
  )
}

function PeriodRow({
  label,
  split
}: {
  label: string
  split: UsagePeriodSplit
}): React.ReactElement {
  const totalReqs = split.local.requests + split.cloud.requests
  const totalTokens = split.local.tokens + split.cloud.tokens

  return (
    <div className="p-4 rounded-xl border border-border bg-surface">
      <div className="flex items-center justify-between mb-2">
        <span className="text-sm font-semibold text-text-primary">{label}</span>
        <span className="text-xs text-text-secondary">
          {fmt(totalReqs)} req · {fmt(totalTokens)} tokens
        </span>
      </div>
      <PercentBar local={split.local.tokens} cloud={split.cloud.tokens} />
      <div className="flex justify-between mt-2 text-xs text-text-secondary">
        <span>Local: {fmt(split.local.requests)} req · {fmt(split.local.tokens)} tok</span>
        <span>Cloud: {fmt(split.cloud.requests)} req · {fmt(split.cloud.tokens)} tok</span>
      </div>
    </div>
  )
}

export function UsagePage(): React.ReactElement {
  const [usage, setUsage] = useState<UsageSummary | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    setLoading(true)
    window.rex
      .getUsage()
      .then((res) => {
        if (res.ok) {
          setUsage(res)
        } else {
          setError(res.error ?? 'Failed to load usage data')
        }
      })
      .catch((err: unknown) => setError(String(err)))
      .finally(() => setLoading(false))
  }, [])

  if (loading) {
    return (
      <div className="flex items-center justify-center h-48 text-text-secondary text-sm">
        Loading usage data…
      </div>
    )
  }

  if (error) {
    return (
      <div className="p-6 text-red-400 text-sm">{error}</div>
    )
  }

  if (!usage) return <></>

  const totalReqs = usage.local.requests + usage.cloud.requests
  const totalTokens = usage.local.tokens + usage.cloud.tokens

  return (
    <div className="p-6 flex flex-col gap-6 max-w-3xl">
      {/* All-time totals */}
      <section>
        <h2 className="text-sm font-semibold text-text-secondary uppercase tracking-wide mb-3">
          All-time totals
        </h2>
        <div className="grid grid-cols-2 gap-4">
          <BucketCard label="Local" bucket={usage.local} color="text-green-400" />
          <BucketCard label="Cloud" bucket={usage.cloud} color="text-blue-400" />
        </div>
        <div className="mt-4 p-4 rounded-xl border border-border bg-surface">
          <div className="flex justify-between text-sm text-text-secondary mb-2">
            <span>Total: {fmt(totalReqs)} requests · {fmt(totalTokens)} tokens</span>
          </div>
          <PercentBar local={usage.local.tokens} cloud={usage.cloud.tokens} />
        </div>
      </section>

      {/* By period */}
      <section>
        <h2 className="text-sm font-semibold text-text-secondary uppercase tracking-wide mb-3">
          Usage by period
        </h2>
        <div className="flex flex-col gap-3">
          <PeriodRow label="Today" split={usage.by_period.today} />
          <PeriodRow label="This week" split={usage.by_period.week} />
          <PeriodRow label="This month" split={usage.by_period.month} />
        </div>
      </section>
    </div>
  )
}
