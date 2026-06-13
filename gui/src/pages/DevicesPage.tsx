import React, { useEffect, useState } from 'react'
import type { Device } from '../types/ipc'

async function sendCommand(
  entityId: string,
  command: string,
  value?: number
): Promise<{ ok: boolean; error?: string }> {
  const token = localStorage.getItem('rex_token') ?? ''
  const body: Record<string, unknown> = { command }
  if (value !== undefined) body.value = value
  try {
    const resp = await fetch(`/api/devices/${entityId}/command`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${token}`
      },
      body: JSON.stringify(body)
    })
    return (await resp.json()) as { ok: boolean; error?: string }
  } catch {
    return { ok: false, error: 'Network error' }
  }
}

interface LightControlProps {
  device: Device
}

function LightControl({ device }: LightControlProps): React.ReactElement {
  const [on, setOn] = useState(false)
  const [brightness, setBrightness] = useState(255)
  const [busy, setBusy] = useState(false)

  const toggle = async (): Promise<void> => {
    setBusy(true)
    const cmd = on ? 'turn_off' : 'turn_on'
    const result = await sendCommand(device.entity_id, cmd)
    if (result.ok) setOn(!on)
    setBusy(false)
  }

  const handleBrightness = async (e: React.ChangeEvent<HTMLInputElement>): Promise<void> => {
    const val = Number(e.target.value)
    setBrightness(val)
    await sendCommand(device.entity_id, 'set_brightness', val)
  }

  return (
    <div className="flex items-center justify-between px-4 py-3 bg-surface gap-4">
      <div className="min-w-0">
        <p className="text-text-primary text-sm font-medium">{device.name}</p>
        <p className="text-text-muted text-xs mt-0.5">Light</p>
      </div>
      <div className="flex items-center gap-3 flex-shrink-0">
        {on && (
          <input
            type="range"
            min={0}
            max={255}
            value={brightness}
            onChange={(e) => void handleBrightness(e)}
            className="w-24 accent-accent"
            aria-label={`${device.name} brightness`}
          />
        )}
        <button
          type="button"
          onClick={() => void toggle()}
          disabled={busy}
          aria-pressed={on}
          className={`w-10 h-6 rounded-full transition-colors flex-shrink-0 relative ${
            on ? 'bg-accent' : 'bg-surface-raised'
          } disabled:opacity-50`}
          aria-label={`Toggle ${device.name}`}
        >
          <span
            className={`absolute top-0.5 w-5 h-5 rounded-full bg-white shadow transition-all ${
              on ? 'left-4' : 'left-0.5'
            }`}
          />
        </button>
      </div>
    </div>
  )
}

interface SwitchControlProps {
  device: Device
}

function SwitchControl({ device }: SwitchControlProps): React.ReactElement {
  const [on, setOn] = useState(false)
  const [busy, setBusy] = useState(false)

  const toggle = async (): Promise<void> => {
    setBusy(true)
    const cmd = on ? 'turn_off' : 'turn_on'
    const result = await sendCommand(device.entity_id, cmd)
    if (result.ok) setOn(!on)
    setBusy(false)
  }

  return (
    <div className="flex items-center justify-between px-4 py-3 bg-surface gap-4">
      <div className="min-w-0">
        <p className="text-text-primary text-sm font-medium">{device.name}</p>
        <p className="text-text-muted text-xs mt-0.5">Switch</p>
      </div>
      <button
        type="button"
        onClick={() => void toggle()}
        disabled={busy}
        aria-pressed={on}
        className={`w-10 h-6 rounded-full transition-colors flex-shrink-0 relative ${
          on ? 'bg-accent' : 'bg-surface-raised'
        } disabled:opacity-50`}
        aria-label={`Toggle ${device.name}`}
      >
        <span
          className={`absolute top-0.5 w-5 h-5 rounded-full bg-white shadow transition-all ${
            on ? 'left-4' : 'left-0.5'
          }`}
        />
      </button>
    </div>
  )
}

interface MediaPlayerControlProps {
  device: Device
}

function MediaPlayerControl({ device }: MediaPlayerControlProps): React.ReactElement {
  const [playing, setPlaying] = useState(false)
  const [volume, setVolume] = useState(0.5)
  const [busy, setBusy] = useState(false)

  const handlePlayPause = async (): Promise<void> => {
    setBusy(true)
    const cmd = playing ? 'media_pause' : 'media_play'
    const result = await sendCommand(device.entity_id, cmd)
    if (result.ok) setPlaying(!playing)
    setBusy(false)
  }

  const handleNext = async (): Promise<void> => {
    await sendCommand(device.entity_id, 'media_next_track')
  }

  const handleVolume = async (e: React.ChangeEvent<HTMLInputElement>): Promise<void> => {
    const val = Number(e.target.value)
    setVolume(val)
    await sendCommand(device.entity_id, 'volume_set', val)
  }

  return (
    <div className="flex items-center justify-between px-4 py-3 bg-surface gap-4">
      <div className="min-w-0">
        <p className="text-text-primary text-sm font-medium">{device.name}</p>
        <p className="text-text-muted text-xs mt-0.5">Media Player</p>
      </div>
      <div className="flex items-center gap-2 flex-shrink-0">
        <input
          type="range"
          min={0}
          max={1}
          step={0.05}
          value={volume}
          onChange={(e) => void handleVolume(e)}
          className="w-20 accent-accent"
          aria-label={`${device.name} volume`}
        />
        <button
          type="button"
          onClick={() => void handlePlayPause()}
          disabled={busy}
          className="w-8 h-8 flex items-center justify-center rounded-full bg-accent text-white hover:bg-accent/90 transition-colors disabled:opacity-50"
          aria-label={playing ? 'Pause' : 'Play'}
        >
          {playing ? (
            <svg width="12" height="12" viewBox="0 0 12 12" fill="currentColor" aria-hidden="true">
              <rect x="2" y="1" width="3" height="10" />
              <rect x="7" y="1" width="3" height="10" />
            </svg>
          ) : (
            <svg width="12" height="12" viewBox="0 0 12 12" fill="currentColor" aria-hidden="true">
              <path d="M2 1l9 5-9 5V1z" />
            </svg>
          )}
        </button>
        <button
          type="button"
          onClick={() => void handleNext()}
          className="w-8 h-8 flex items-center justify-center rounded-full bg-surface-raised text-text-secondary hover:text-text-primary transition-colors"
          aria-label="Next track"
        >
          <svg width="12" height="12" viewBox="0 0 12 12" fill="currentColor" aria-hidden="true">
            <path d="M2 2l6 4-6 4V2z" />
            <rect x="9" y="2" width="1.5" height="8" />
          </svg>
        </button>
      </div>
    </div>
  )
}

function DeviceRow({ device }: { device: Device }): React.ReactElement {
  if (device.type === 'light') return <LightControl device={device} />
  if (device.type === 'switch') return <SwitchControl device={device} />
  if (device.type === 'media_player') return <MediaPlayerControl device={device} />
  return (
    <div className="flex items-center justify-between px-4 py-3 bg-surface gap-4">
      <p className="text-text-primary text-sm font-medium">{device.name}</p>
      <span className="text-text-muted text-xs">{device.type}</span>
    </div>
  )
}

export function DevicesPage(): React.ReactElement {
  const [devices, setDevices] = useState<Device[]>([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    window.rex
      .getDevices()
      .then((res) => {
        setDevices(res.devices ?? [])
      })
      .catch(() => {/* ignore */})
      .finally(() => setLoading(false))
  }, [])

  return (
    <div className="p-6 max-w-2xl mx-auto space-y-6">
      <section>
        <h2 className="text-xl font-semibold text-text-primary mb-4">Devices</h2>
        <p className="text-text-secondary text-sm mb-4">
          Control your approved Home Assistant devices. Add devices in{' '}
          <code className="text-xs font-mono bg-surface-raised px-1 py-0.5 rounded">
            config/device_aliases.json
          </code>
          .
        </p>

        {loading ? (
          <p className="text-text-muted text-sm">Loading…</p>
        ) : devices.length === 0 ? (
          <p className="text-text-muted text-sm">
            No devices configured. Add devices to{' '}
            <code className="text-xs font-mono bg-surface-raised px-1 py-0.5 rounded">
              config/device_aliases.json
            </code>{' '}
            to see them here.
          </p>
        ) : (
          <div className="divide-y divide-border border border-border rounded-xl overflow-hidden">
            {devices.map((device) => (
              <DeviceRow key={device.entity_id} device={device} />
            ))}
          </div>
        )}
      </section>
    </div>
  )
}
