import React, { useEffect, useRef, useState } from 'react'
import type { Settings, SmartSpeaker, VoiceSettings } from '../../types/ipc'
import { useToast } from '../../components/ui/Toast'

interface AudioDevice {
  deviceId: string
  label: string
}

export function AudioOutputSettingsSection(): React.ReactElement {
  const addToast = useToast()
  const [speakers, setSpeakers] = useState<AudioDevice[]>([])
  const [selectedDeviceId, setSelectedDeviceId] = useState('')
  const [volume, setVolume] = useState(1.0)
  const [testing, setTesting] = useState<string | null>(null) // deviceId being tested
  const [savedVolume, setSavedVolume] = useState(false)
  const volumeTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const [smartSpeakers, setSmartSpeakers] = useState<SmartSpeaker[]>([])
  const [loadingSmartSpeakers, setLoadingSmartSpeakers] = useState(false)

  function loadSmartSpeakers(): void {
    setLoadingSmartSpeakers(true)
    window.rex
      .getSmartSpeakers()
      .then((res) => {
        setSmartSpeakers(res.speakers ?? [])
        if (!res.ok && res.error) addToast(`Smart speaker discovery: ${res.error}`, 'error')
      })
      .catch(() => {
        setSmartSpeakers([])
      })
      .finally(() => setLoadingSmartSpeakers(false))
  }

  useEffect(() => {
    // Load saved voice settings to get current speaker + volume
    window.rex
      .getSettings('voice')
      .then((s: Settings) => {
        if (typeof s.speakerDeviceId === 'string') setSelectedDeviceId(s.speakerDeviceId)
        if (typeof s.volume === 'number') setVolume(s.volume)
      })
      .catch(() => {
        // Non-fatal
      })

    // Enumerate output devices
    if (navigator.mediaDevices?.enumerateDevices) {
      navigator.mediaDevices
        .enumerateDevices()
        .then((devices) => {
          const list = devices
            .filter((d) => d.kind === 'audiooutput')
            .map((d, i) => ({ deviceId: d.deviceId, label: d.label || `Speaker ${i + 1}` }))
          setSpeakers(list)
        })
        .catch(() => {
          setSpeakers([])
        })
    }

    // Discover smart speakers
    loadSmartSpeakers()
  }, [])

  function handleSelectDevice(deviceId: string): void {
    setSelectedDeviceId(deviceId)
    window.rex
      .getSettings('voice')
      .then((s: Settings) => {
        return window.rex.setSettings('voice', { ...s, speakerDeviceId: deviceId } as Settings)
      })
      .then((result) => {
        if (!result.ok) {
          addToast(result.error ?? 'Failed to save speaker selection', 'error')
        }
      })
      .catch(() => {
        addToast('Failed to save speaker selection', 'error')
      })
  }

  function handleVolumeChange(v: number): void {
    setVolume(v)
    if (volumeTimerRef.current) clearTimeout(volumeTimerRef.current)
    volumeTimerRef.current = setTimeout(() => {
      window.rex
        .getSettings('voice')
        .then((s: Settings) => window.rex.setSettings('voice', { ...s, volume: v } as Settings))
        .then((result) => {
          if (result.ok) {
            setSavedVolume(true)
            setTimeout(() => setSavedVolume(false), 2000)
          } else {
            addToast(result.error ?? 'Failed to save volume', 'error')
          }
        })
        .catch(() => {
          addToast('Failed to save volume', 'error')
        })
    }, 400)
  }

  function handleTestDevice(deviceId: string): void {
    setTesting(deviceId)
    if (!deviceId.startsWith('smart:')) {
      // Play a brief 440 Hz sine tone through the selected system output device
      const ctx = new AudioContext()
      const doPlay = async (): Promise<void> => {
        try {
          const ctxExt = ctx as AudioContext & { setSinkId?: (id: string) => Promise<void> }
          if (deviceId && typeof ctxExt.setSinkId === 'function') {
            await ctxExt.setSinkId(deviceId)
          }
          const osc = ctx.createOscillator()
          const gain = ctx.createGain()
          osc.connect(gain)
          gain.connect(ctx.destination)
          osc.frequency.setValueAtTime(440, ctx.currentTime)
          gain.gain.setValueAtTime(0.3, ctx.currentTime)
          gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.5)
          osc.start(ctx.currentTime)
          osc.stop(ctx.currentTime + 0.6)
          await new Promise<void>((resolve) => {
            osc.onended = () => resolve()
          })
        } catch (err) {
          addToast(`Test failed: ${String(err)}`, 'error')
        } finally {
          void ctx.close()
          setTesting(null)
        }
      }
      void doPlay()
    } else {
      // Smart speaker: delegate to IPC
      window.rex
        .testVoice({ speakerDeviceId: deviceId } as unknown as VoiceSettings)
        .then((res) => {
          if (!res.ok) addToast(res.error ?? 'Test failed', 'error')
        })
        .catch(() => {
          addToast('Test failed', 'error')
        })
        .finally(() => setTesting(null))
    }
  }

  const displayDevices: AudioDevice[] =
    speakers.length > 0
      ? speakers
      : [{ deviceId: '', label: 'System Default' }]

  return (
    <div className="p-6 max-w-lg">
      <h2 className="text-lg font-semibold text-text-primary mb-6">Audio Output</h2>

      {/* System output devices */}
      <div className="mb-6">
        <h3 className="mb-4 text-sm font-semibold text-text-primary">System Speakers</h3>
        <div className="space-y-3">
          {displayDevices.map((device) => {
            const isSelected = device.deviceId === selectedDeviceId || (selectedDeviceId === '' && device.deviceId === '')
            return (
              <div
                key={device.deviceId || 'default'}
                className={[
                  'flex items-center gap-3 rounded-xl border p-4 transition-colors',
                  isSelected ? 'border-accent bg-accent/5' : 'border-border bg-surface-raised'
                ].join(' ')}
              >
                <button
                  type="button"
                  onClick={() => handleSelectDevice(device.deviceId)}
                  className={[
                    'h-4 w-4 shrink-0 rounded-full border-2 transition-colors focus:outline-none',
                    isSelected ? 'border-accent bg-accent' : 'border-border bg-bg'
                  ].join(' ')}
                  aria-label={`Select ${device.label}`}
                />
                <div className="flex-1 min-w-0">
                  <div className="text-sm font-medium text-text-primary truncate">{device.label}</div>
                  {isSelected && (
                    <div className="text-xs text-accent mt-0.5">Default TTS output</div>
                  )}
                </div>
                <button
                  type="button"
                  onClick={() => handleTestDevice(device.deviceId)}
                  disabled={testing === device.deviceId}
                  className="flex items-center gap-1.5 rounded-lg border border-border bg-bg px-3 py-1.5 text-xs font-medium text-text-primary transition-colors hover:bg-surface-raised disabled:opacity-50 focus:outline-none"
                >
                  {testing === device.deviceId ? (
                    <svg className="animate-spin h-3 w-3" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                      <path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83" />
                    </svg>
                  ) : (
                    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                      <polygon points="5 3 19 12 5 21 5 3" />
                    </svg>
                  )}
                  Test
                </button>
              </div>
            )
          })}
        </div>
      </div>

      {/* Volume */}
      <div className="mb-8">
        <div className="flex items-center justify-between mb-2">
          <label className="text-sm font-medium text-text-primary">
            TTS Volume
            <span className="ml-2 text-xs text-text-secondary font-normal">
              {Math.round(volume * 100)}%
            </span>
          </label>
          {savedVolume && (
            <span className="text-xs text-success flex items-center gap-1">
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3">
                <polyline points="20 6 9 17 4 12" />
              </svg>
              Saved
            </span>
          )}
        </div>
        <div className="flex items-center gap-2 text-xs text-text-secondary">
          <span>0%</span>
          <input
            type="range"
            min={0}
            max={1.0}
            step={0.05}
            value={volume}
            onChange={(e) => handleVolumeChange(parseFloat(e.target.value))}
            className="flex-1 accent-accent"
          />
          <span>100%</span>
        </div>
      </div>

      {/* Smart speakers */}
      <div className="border-t border-border pt-6">
        <div className="flex items-center justify-between mb-1">
          <h3 className="text-sm font-semibold text-text-primary">Smart Speakers</h3>
          <button
            type="button"
            onClick={loadSmartSpeakers}
            disabled={loadingSmartSpeakers}
            className="flex items-center gap-1 rounded-lg border border-border bg-bg px-2.5 py-1 text-xs font-medium text-text-primary transition-colors hover:bg-surface-raised disabled:opacity-50 focus:outline-none"
          >
            {loadingSmartSpeakers ? (
              <svg className="animate-spin h-3 w-3" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83" />
              </svg>
            ) : (
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <polyline points="23 4 23 10 17 10" />
                <path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10" />
              </svg>
            )}
            Refresh
          </button>
        </div>
        <p className="mb-4 text-xs text-text-secondary">
          Sonos and Bose devices discovered on the local network.
        </p>
        {smartSpeakers.length === 0 ? (
          <div className="rounded-xl border border-dashed border-border bg-surface-raised/40 px-4 py-6 text-center">
            <p className="text-sm text-text-secondary">No smart speakers discovered on the network.</p>
            <p className="mt-1 text-xs text-text-secondary">
              Ensure your Sonos or Bose device is on the same network and click Refresh.
            </p>
          </div>
        ) : (
          <div className="space-y-3">
            {smartSpeakers.map((ss) => (
              <div
                key={`${ss.provider}-${ss.ip}`}
                className="flex items-center gap-3 rounded-xl border border-border bg-surface-raised p-4"
              >
                <div className="flex-1 min-w-0">
                  <div className="text-sm font-medium text-text-primary truncate">{ss.name}</div>
                  <div className="text-xs text-text-secondary mt-0.5 capitalize">
                    {ss.provider} · {ss.model} · {ss.ip}
                  </div>
                </div>
                <button
                  type="button"
                  onClick={() => handleTestDevice(`smart:${ss.provider}:${ss.ip}`)}
                  disabled={testing === `smart:${ss.provider}:${ss.ip}`}
                  className="flex items-center gap-1.5 rounded-lg border border-border bg-bg px-3 py-1.5 text-xs font-medium text-text-primary transition-colors hover:bg-surface-raised disabled:opacity-50 focus:outline-none"
                >
                  {testing === `smart:${ss.provider}:${ss.ip}` ? (
                    <svg className="animate-spin h-3 w-3" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                      <path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83" />
                    </svg>
                  ) : (
                    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                      <polygon points="5 3 19 12 5 21 5 3" />
                    </svg>
                  )}
                  Test
                </button>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}
