import React, { useEffect, useState } from 'react'
import type { AppStatus } from '../types/ipc'

export function AboutPage(): React.ReactElement {
  const [info, setInfo] = useState<AppStatus | null>(null)

  useEffect(() => {
    window.rex
      .getAppStatus()
      .then((data) => setInfo(data))
      .catch(() => {/* ignore */})
  }, [])

  return (
    <div className="p-6 max-w-2xl mx-auto space-y-8">
      <section>
        <h2 className="text-xl font-semibold text-text-primary mb-4">About Rex</h2>
        <p className="text-text-secondary text-sm mb-6">
          Rex is a local-first, voice-activated AI companion. It runs entirely on your device,
          keeping your data private and accessible even offline.
        </p>

        <div className="border border-border rounded-xl overflow-hidden divide-y divide-border">
          <div className="flex items-center justify-between px-4 py-3 bg-surface">
            <span className="text-text-secondary text-sm">Application</span>
            <span className="text-text-primary text-sm font-medium">AskRex Assistant</span>
          </div>
          <div className="flex items-center justify-between px-4 py-3 bg-surface">
            <span className="text-text-secondary text-sm">Version</span>
            <span className="text-text-primary text-sm font-medium font-mono">
              {info ? info.version : '—'}
            </span>
          </div>
          <div className="flex items-center justify-between px-4 py-3 bg-surface">
            <span className="text-text-secondary text-sm">Python</span>
            <span className="text-text-primary text-sm font-medium font-mono">
              {info ? info.python_version : '—'}
            </span>
          </div>
          <div className="flex items-center justify-between px-4 py-3 bg-surface">
            <span className="text-text-secondary text-sm">Platform</span>
            <span className="text-text-primary text-sm font-medium font-mono">
              {info ? info.platform : '—'}
            </span>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-xl font-semibold text-text-primary mb-4">Privacy</h2>
        <p className="text-text-secondary text-sm">
          Rex processes all voice and text locally. No data is sent to external servers unless you
          explicitly configure a cloud LLM provider or integration. Your conversations and memories
          are stored only on this device.
        </p>
      </section>
    </div>
  )
}
