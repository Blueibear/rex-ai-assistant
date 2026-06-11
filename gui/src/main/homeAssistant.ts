import type { Settings } from '../types/ipc'
import { readEnvFile, readGuiSettings, readRexConfig, writeEnvKey, writeGuiSettings } from './configStore'
import { mirrorToRexConfig } from './settingsMirror'

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

export function readSavedHomeAssistantCredentials(): { baseUrl: string; token: string } {
  const stored = readGuiSettings()
  const integrations = (stored['integrations'] ?? {}) as Record<string, unknown>
  const rexConfig = readRexConfig()
  const haConfig = ((rexConfig.home_assistant ?? {}) as Record<string, unknown>)
  const env = readEnvFile()

  return {
    baseUrl: normalizeHaUrl(integrations.haUrl) || normalizeHaUrl(haConfig.base_url),
    // HA token is stored in .env only - never in gui_settings (canonical secret store)
    token: (typeof env.HA_TOKEN === 'string' && env.HA_TOKEN.trim()) || ''
  }
}

export function saveHomeAssistantCredentials(baseUrl: string, token: string): void {
  const stored = readGuiSettings()
  const integrations = { ...((stored['integrations'] ?? {}) as Record<string, unknown>) }
  integrations.haUrl = baseUrl
  // haToken is NOT stored in gui_settings - canonical secret store is .env only
  stored['integrations'] = integrations as Settings
  writeGuiSettings(stored)
  mirrorToRexConfig('integrations', integrations as Settings)
  if (token) {
    writeEnvKey('HA_TOKEN', token)
  }
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

export async function getHomeAssistantStates(): Promise<HaStatesResult> {
  const { baseUrl, token } = readSavedHomeAssistantCredentials()
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
