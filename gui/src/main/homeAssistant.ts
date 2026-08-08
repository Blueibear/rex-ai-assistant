import type { Settings } from '../types/ipc'
import { readGuiSettings, readRexConfig, readRexConfigStrict, writeGuiSettings, writeRexConfig } from './configStore'
import { mirrorToRexConfig } from './settingsMirror'
import { vaultDeleteSecret, vaultGetSecret, vaultSetSecret, type VaultContext } from './credentialVault'
import { getVaultReference, putVaultReference } from './credentialReferences'
import { randomUUID } from 'crypto'
import { spawn } from 'child_process'
import { bridgeSpawnOptions, resolveBridgePath, resolvePythonCommand } from './bridgeResolver'
import { privateSessionPayload, type ElectronSessionIdentity } from './sessionIdentity'

export interface HaTestResult {
  ok: boolean
  error?: string
}

export interface HaState {
  entity_id: string
  state: string
  friendly_name: string
  last_updated: string
}

export interface HaStatesResult extends HaTestResult {
  states?: HaState[]
  not_configured?: boolean
}

export function normalizeHaUrl(value: unknown): string {
  return typeof value === 'string' ? value.trim().replace(/\/+$/, '') : ''
}

/**
 * Resolve saved Home Assistant credentials. The token is read from the
 * credential vault (S4) only. A vault failure (unavailable, locked, or
 * corrupted) rejects the operation; callers must report that stored state
 * could not be verified and must never consult plaintext `.env`.
 */
const HA_CONTEXT: VaultContext = {
  scope: 'household', integration: 'home_assistant', account: null, slot: 'token'
}

export async function readSavedHomeAssistantCredentials(
  session: ElectronSessionIdentity
): Promise<{ baseUrl: string; token: string; ref: string | null }> {
  const stored = readGuiSettings()
  const integrations = (stored['integrations'] ?? {}) as Record<string, unknown>
  const rexConfig = readRexConfig()
  const haConfig = ((rexConfig.home_assistant ?? {}) as Record<string, unknown>)

  const record = getVaultReference(readRexConfigStrict(), 'HA_TOKEN', HA_CONTEXT, session.userId)
  const ref = record?.ref ?? null
  const token = record ? (await vaultGetSecret(session, record.ref, HA_CONTEXT)) ?? '' : ''

  return {
    baseUrl: normalizeHaUrl(integrations.haUrl) || normalizeHaUrl(haConfig.base_url),
    token,
    ref
  }
}

/**
 * Persist Home Assistant credentials. The token is written through to the
 * credential vault (S4), never to plaintext `.env`. Rejects (does not
 * silently swallow) on a vault failure - callers must surface this to the
 * user rather than reporting a false "Saved" state.
 */
export async function saveHomeAssistantCredentials(
  session: ElectronSessionIdentity,
  baseUrl: string,
  token: string
): Promise<void> {
  const stored = readGuiSettings()
  const originalStored = JSON.parse(JSON.stringify(stored)) as Record<string, Settings>
  const nextStored = JSON.parse(JSON.stringify(stored)) as Record<string, Settings>
  const originalConfig = readRexConfigStrict()
  const nextConfig = JSON.parse(JSON.stringify(originalConfig)) as Record<string, unknown>
  const oldRecord = getVaultReference(originalConfig, 'HA_TOKEN', HA_CONTEXT, session.userId)
  let newRef: string | null = null
  let guiWritten = false
  let configWritten = false
  const integrations = { ...((stored['integrations'] ?? {}) as Record<string, unknown>) }
  integrations.haUrl = baseUrl
  // haToken is NOT stored in gui_settings - canonical secret store is the vault
  nextStored['integrations'] = integrations as Settings
  try {
    if (token.trim()) {
      newRef = await vaultSetSecret(session, token.trim(), HA_CONTEXT)
      putVaultReference(nextConfig, 'HA_TOKEN', newRef, HA_CONTEXT, session.userId)
    }
    writeGuiSettings(nextStored)
    guiWritten = true
    writeRexConfig(nextConfig)
    configWritten = true
    const mirrorResult = mirrorToRexConfig('integrations', integrations as Settings)
    if (!mirrorResult.ok) throw new Error(mirrorResult.error ?? 'Home Assistant config mirror failed')
    if (newRef) {
      const readback = getVaultReference(readRexConfigStrict(), 'HA_TOKEN', HA_CONTEXT, session.userId)
      if (readback?.ref !== newRef) throw new Error('Credential reference readback failed')
      if (oldRecord) await vaultDeleteSecret(session, oldRecord.ref, HA_CONTEXT).catch(() => false)
    }
  } catch (error) {
    let restored = true
    if (guiWritten) {
      try { writeGuiSettings(originalStored) } catch { restored = false }
    }
    if (configWritten) {
      try { writeRexConfig(originalConfig) } catch { restored = false }
    }
    if (newRef && restored) {
      await vaultDeleteSecret(session, newRef, HA_CONTEXT).catch(() => false)
    }
    throw error
  }
}

export type DeviceCommandStatus =
  | 'verified'
  | 'attempted_unverified'
  | 'confirmation_required'
  | 'denied'
  | 'failed'

export interface DeviceCommandResponse {
  status: DeviceCommandStatus
  detail?: string
  expected?: { state: string; attributes: Record<string, unknown> } | null
  actual?: Record<string, unknown> | null
  latencyMs?: number
  confirmationToken?: string
  requestId?: string
}

function describeHaError(error: unknown): string {
  if (error instanceof Error) {
    return error.name === 'AbortError' ? 'Connection timed out.' : error.message
  }
  return String(error)
}

async function requestHomeAssistant(
  baseUrl: string,
  token: string,
  path: string,
  timeoutMs = 5000
): Promise<Response> {
  const controller = new AbortController()
  const timeout = setTimeout(() => controller.abort(), timeoutMs)
  try {
    return await fetch(`${baseUrl}${path}`, {
      headers: token ? { Authorization: `Bearer ${token}` } : {},
      signal: controller.signal
    })
  } finally {
    clearTimeout(timeout)
  }
}

export function callDeviceCommand(
  session: ElectronSessionIdentity,
  entityId: string,
  command: string,
  payload?: { value?: number },
  confirmationToken?: string,
  existingRequestId?: string
): Promise<DeviceCommandResponse> {
  const domain = entityId.split('.')[0]
  let service = command
  const parameters: Record<string, unknown> = {}

  if (command === 'set_brightness' && payload?.value !== undefined) {
    service = 'turn_on'
    parameters.brightness = payload.value
  } else if (command === 'volume_set' && payload?.value !== undefined) {
    parameters.volume_level = payload.value
  }
  const requestId = existingRequestId ?? randomUUID()
  return new Promise((resolve) => {
    const py = spawn(resolvePythonCommand(), [resolveBridgePath('rex_ha_mutation_bridge.py')], {
      ...bridgeSpawnOptions(),
      stdio: ['pipe', 'pipe', 'pipe']
    })
    let stdout = ''
    let _stderr = ''
    py.stdout.on('data', (chunk: Buffer) => { stdout += chunk.toString() })
    py.stderr.on('data', (chunk: Buffer) => { _stderr += chunk.toString() })
    py.on('close', (code) => {
      try {
        const result = JSON.parse(stdout.trim()) as {
          ok: boolean
          status?: DeviceCommandStatus
          detail?: string
          expected?: { state: string; attributes: Record<string, unknown> } | null
          actual?: Record<string, unknown> | null
          latency_ms?: number
          confirmation_token?: string
          request_id?: string
          error?: string
        }
        if (code === 0 && result.ok && result.status) {
          resolve({
            status: result.status,
            detail: result.detail,
            expected: result.expected,
            actual: result.actual,
            latencyMs: result.latency_ms,
            confirmationToken: result.confirmation_token,
            requestId: result.request_id
          })
          return
        }
        resolve({ status: 'failed', detail: 'Home Assistant command failed.' })
      } catch {
        resolve({ status: 'failed', detail: 'Home Assistant service returned an invalid response.' })
      }
    })
    py.on('error', () => {
      resolve({ status: 'failed', detail: 'Home Assistant service could not be started.' })
    })
    py.stdin.write(JSON.stringify(privateSessionPayload(session, {
      entity_id: entityId,
      domain,
      service,
      parameters,
      request_id: requestId,
      confirmation_token: confirmationToken
    })))
    py.stdin.end()
  })
}

export async function testHomeAssistantConnection(
  baseUrl: string,
  token: string
): Promise<HaTestResult> {
  const normalizedUrl = normalizeHaUrl(baseUrl)
  if (!normalizedUrl) return { ok: false, error: 'Home Assistant URL is required.' }
  try {
    const resp = await requestHomeAssistant(normalizedUrl, token.trim(), '/api/')
    if (!resp.ok) return { ok: false, error: `HA returned HTTP ${resp.status}` }
    return { ok: true }
  } catch (err) {
    return { ok: false, error: describeHaError(err) }
  }
}

export async function getHomeAssistantStates(session: ElectronSessionIdentity): Promise<HaStatesResult> {
  const { baseUrl, token } = await readSavedHomeAssistantCredentials(session)
  if (!baseUrl || !token) {
    return {
      ok: false,
      not_configured: true,
      error: 'Home Assistant is not configured.'
    }
  }
  try {
    const resp = await requestHomeAssistant(baseUrl, token, '/api/states', 10000)
    if (!resp.ok) return { ok: false, error: `HA returned HTTP ${resp.status}` }
    const rawStates = (await resp.json()) as Array<Record<string, unknown>>
    const states = rawStates.filter((s) => typeof s === 'object' && s !== null).map((s) => {
      const attrs = s.attributes && typeof s.attributes === 'object'
        ? (s.attributes as Record<string, unknown>)
        : {}
      const entityId = typeof s.entity_id === 'string' ? s.entity_id : ''
      return {
        entity_id: entityId,
        state: typeof s.state === 'string' ? s.state : 'unknown',
        friendly_name: typeof attrs.friendly_name === 'string' ? attrs.friendly_name : entityId,
        last_updated: typeof s.last_updated === 'string' ? s.last_updated : ''
      }
    })
    return { ok: true, states }
  } catch (err) {
    return { ok: false, error: describeHaError(err) }
  }
}
