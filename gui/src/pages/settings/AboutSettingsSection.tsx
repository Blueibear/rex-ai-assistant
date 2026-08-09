import React, { useEffect, useState } from 'react'
import type { VersionInfo } from '../../types/ipc'
import { SkeletonLine } from '../../components/ui/SkeletonLine'

export function AboutSettingsSection(): React.ReactElement {
  const [info, setInfo] = useState<VersionInfo | null>(null)
  const [loading, setLoading] = useState(true)
  const [slow, setSlow] = useState(false)

  useEffect(() => {
    window.rex
      .getVersionInfo()
      .then((v) => {
        setInfo(v)
      })
      .catch(() => {
        setInfo({ rex: 'unknown', electron: 'unknown', node: 'unknown' })
      })
      .finally(() => {
        setLoading(false)
      })
  }, [])

  useEffect(() => {
    if (!loading) return
    const id = setTimeout(() => setSlow(true), 5000)
    return () => clearTimeout(id)
  }, [loading])

  return (
    <div className="p-6">
      <div className="flex items-center gap-3 mb-6">
        <div className="w-12 h-12 rounded-xl bg-accent/20 flex items-center justify-center">
          <span className="text-accent font-bold text-lg">R</span>
        </div>
        <div>
          <h2 className="text-lg font-semibold text-text-primary">AskRex Assistant</h2>
          <p className="text-sm text-text-secondary">Local-first AI companion</p>
        </div>
      </div>

      {loading ? (
        <div className="space-y-3">
          <SkeletonLine width="100%" height="1.25rem" />
          <SkeletonLine width="80%" height="1.25rem" />
          <SkeletonLine width="60%" height="1.25rem" />
          {slow && (
            <p className="text-xs text-text-secondary mt-1 animate-pulse">
              Taking longer than expected…
            </p>
          )}
        </div>
      ) : (
        <div className="space-y-3">
          <VersionRow label="Rex version" value={info?.rex ?? 'unknown'} />
          <VersionRow label="Electron version" value={info?.electron ?? 'unknown'} />
          <VersionRow label="Node version" value={info?.node ?? 'unknown'} />
        </div>
      )}

      <div className="mt-8 pt-6 border-t border-border">
        <p className="text-xs text-text-secondary">
          AskRex Assistant is a local-first, voice-activated AI companion. All data is stored on
          your device.
        </p>
      </div>
    </div>
  )
}

function VersionRow({ label, value }: { label: string; value: string }): React.ReactElement {
  return (
    <div className="flex items-center justify-between py-2 border-b border-border/50">
      <span className="text-sm text-text-secondary">{label}</span>
      <span className="text-sm font-mono text-text-primary">{value}</span>
    </div>
  )
}
