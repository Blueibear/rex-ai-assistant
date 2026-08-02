import React, { useCallback, useEffect, useMemo, useState } from 'react'
import { Button } from '../components/ui/Button'
import { Card } from '../components/ui/Card'
import type { PairingChallenge, PairedDevice, PendingPairingRequest } from '../types/ipc'

const DEFAULT_SCOPES = ['chat.send', 'chat.history.read', 'voice.use']

export function PairingPage(): React.ReactElement {
  const [challenge, setChallenge] = useState<PairingChallenge | null>(null)
  const [pending, setPending] = useState<PendingPairingRequest[]>([])
  const [devices, setDevices] = useState<PairedDevice[]>([])
  const [desktopId, setDesktopId] = useState('')
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)

  const refresh = useCallback(async (): Promise<void> => {
    const [pendingResult, devicesResult] = await Promise.all([
      window.rex.listPendingPairings(),
      window.rex.listPairedDevices()
    ])
    if (pendingResult.ok) setPending(pendingResult.requests ?? [])
    if (devicesResult.ok) {
      setDevices(devicesResult.devices ?? [])
      setDesktopId(devicesResult.desktop_id ?? '')
    }
  }, [])

  useEffect(() => {
    void refresh()
    const id = window.setInterval(() => void refresh(), 3000)
    return () => window.clearInterval(id)
  }, [refresh])

  const qrText = useMemo(
    () => (challenge ? JSON.stringify(challenge) : ''),
    [challenge]
  )

  const createChallenge = async (): Promise<void> => {
    setBusy(true)
    setError('')
    try {
      const result = await window.rex.createPairingChallenge(DEFAULT_SCOPES)
      if (!result.ok || !result.challenge) {
        setError(result.error ?? 'Could not create a pairing challenge.')
        return
      }
      setChallenge(result.challenge)
    } finally {
      setBusy(false)
    }
  }

  const decide = async (requestId: string, approve: boolean): Promise<void> => {
    setBusy(true)
    setError('')
    try {
      const result = approve
        ? await window.rex.approvePairing(requestId)
        : await window.rex.denyPairing(requestId)
      if (!result.ok) setError(result.error ?? 'Pairing decision failed.')
      await refresh()
    } finally {
      setBusy(false)
    }
  }

  const revoke = async (deviceId: string): Promise<void> => {
    if (!window.confirm('Revoke this device and every active grant?')) return
    const result = await window.rex.revokePairedDevice(deviceId)
    if (!result.ok) setError(result.error ?? 'Device revocation failed.')
    await refresh()
  }

  return (
    <div className="p-6 max-w-4xl mx-auto space-y-6">
      <div>
        <h2 className="text-xl font-semibold text-text-primary">Mobile device pairing</h2>
        <p className="text-sm text-text-secondary mt-1">
          Pairing is approved on this desktop. A password login alone never grants mobile action access.
        </p>
      </div>

      {error && <div className="rounded-lg border border-danger/40 bg-danger/10 p-3 text-sm text-danger">{error}</div>}

      <Card header="Create a one-time enrollment">
        <div className="space-y-3">
          <p className="text-sm text-text-secondary">
            The code expires after 120 seconds and can be submitted only once.
          </p>
          <Button loading={busy} onClick={() => void createChallenge()}>
            Generate pairing code
          </Button>
          {challenge && (
            <div className="rounded-lg bg-bg border border-border p-4 space-y-3">
              <div className="text-3xl tracking-[0.25em] font-mono text-accent">{challenge.code}</div>
              <div className="text-xs text-text-muted">Expires {new Date(challenge.expires_at).toLocaleTimeString()}</div>
              <label className="block text-xs font-medium text-text-secondary">QR payload</label>
              <textarea
                readOnly
                value={qrText}
                rows={5}
                className="w-full rounded-md border border-border bg-surface-raised p-2 font-mono text-xs text-text-secondary"
              />
            </div>
          )}
        </div>
      </Card>

      <Card header={`Pending approvals (${pending.length})`}>
        {pending.length === 0 ? (
          <p className="text-sm text-text-muted">No mobile device is waiting for approval.</p>
        ) : (
          <div className="space-y-3">
            {pending.map((request) => (
              <div key={request.request_id} className="rounded-lg border border-border p-3 flex items-center gap-3">
                <div className="flex-1 min-w-0">
                  <div className="font-medium text-text-primary">{request.device_name || 'Unnamed device'}</div>
                  <div className="text-xs text-text-muted">{request.platform} · {request.user_id}</div>
                  <div className="text-xs text-text-muted truncate">Key {request.key_thumbprint}</div>
                  <div className="text-xs text-text-secondary mt-1">{request.scopes.join(', ')}</div>
                </div>
                <Button size="sm" onClick={() => void decide(request.request_id, true)}>Approve</Button>
                <Button size="sm" variant="danger" onClick={() => void decide(request.request_id, false)}>Deny</Button>
              </div>
            ))}
          </div>
        )}
      </Card>

      <Card header={`Paired devices (${devices.length})`}>
        <p className="text-xs text-text-muted mb-3">Desktop authority: {desktopId || 'not initialized'}</p>
        {devices.length === 0 ? (
          <p className="text-sm text-text-muted">No devices have been approved.</p>
        ) : (
          <div className="space-y-3">
            {devices.map((device) => (
              <div key={device.device_id} className="rounded-lg border border-border p-3 flex items-center gap-3">
                <div className="flex-1">
                  <div className="font-medium text-text-primary">{device.device_name || device.device_id}</div>
                  <div className="text-xs text-text-muted">{device.platform} · grant v{device.grant_version ?? '—'}</div>
                  <div className="text-xs text-text-secondary mt-1">{device.scopes.join(', ') || 'No active scopes'}</div>
                  <div className="text-xs mt-1">{device.revoked_at ? <span className="text-danger">Revoked</span> : <span className="text-success">Approved</span>}</div>
                </div>
                {!device.revoked_at && <Button size="sm" variant="danger" onClick={() => void revoke(device.device_id)}>Revoke</Button>}
              </div>
            ))}
          </div>
        )}
      </Card>
    </div>
  )
}
