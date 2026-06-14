import { ipcMain } from 'electron'
import { existsSync, readFileSync } from 'fs'
import { join } from 'path'
import { getConfigDir } from '../configStore'
import { readSavedHomeAssistantCredentials, normalizeHaUrl } from '../homeAssistant'

interface DeviceEntry {
  entity_id: string
  name: string
  type: string
}

interface DeviceAliasesFile {
  devices?: DeviceEntry[]
}

interface DeviceCommandResult {
  status: 'attempted' | 'completed' | 'verified' | 'failed'
  detail?: string
}

function loadDevices(): DeviceEntry[] {
  const aliasesPath = join(getConfigDir(), 'device_aliases.json')
  if (!existsSync(aliasesPath)) return []
  try {
    const raw = JSON.parse(readFileSync(aliasesPath, 'utf8')) as DeviceAliasesFile
    const devices = Array.isArray(raw.devices) ? raw.devices : []
    return devices.filter(
      (d): d is DeviceEntry =>
        d !== null &&
        typeof d === 'object' &&
        typeof d.entity_id === 'string' &&
        typeof d.name === 'string' &&
        typeof d.type === 'string'
    )
  } catch {
    return []
  }
}

function domainOf(entityId: string): string {
  const dot = entityId.indexOf('.')
  return dot !== -1 ? entityId.slice(0, dot) : entityId
}

function mapCommandToService(
  command: string,
  value?: number
): { service: string; extraBody: Record<string, unknown> } {
  if (command === 'set_brightness') {
    return { service: 'turn_on', extraBody: { brightness: value ?? 255 } }
  }
  if (command === 'volume_set') {
    return { service: 'volume_set', extraBody: { volume_level: value ?? 0.5 } }
  }
  return { service: command, extraBody: {} }
}

async function callHaService(
  baseUrl: string,
  token: string,
  domain: string,
  service: string,
  body: Record<string, unknown>
): Promise<{ ok: boolean; status: number }> {
  const controller = new AbortController()
  const timeoutId = setTimeout(() => controller.abort(), 8000)
  try {
    const resp = await fetch(`${baseUrl}/api/services/${domain}/${service}`, {
      method: 'POST',
      headers: {
        Authorization: `Bearer ${token}`,
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(body),
      signal: controller.signal
    })
    return { ok: resp.ok, status: resp.status }
  } finally {
    clearTimeout(timeoutId)
  }
}

export function registerDeviceHandlers(): void {
  ipcMain.handle(
    'rex:getDevices',
    (): { ok: boolean; devices: DeviceEntry[]; error?: string } => {
      try {
        return { ok: true, devices: loadDevices() }
      } catch (err) {
        return { ok: false, devices: [], error: String(err) }
      }
    }
  )

  ipcMain.handle(
    'rex:sendDeviceCommand',
    async (
      _event: unknown,
      entityId: string,
      command: string,
      value?: number
    ): Promise<DeviceCommandResult> => {
      const { baseUrl, token } = readSavedHomeAssistantCredentials()
      const normalizedUrl = normalizeHaUrl(baseUrl)
      if (!normalizedUrl || !token) {
        return { status: 'failed', detail: 'Home Assistant is not configured.' }
      }
      const domain = domainOf(entityId)
      const { service, extraBody } = mapCommandToService(command, value)
      const serviceBody: Record<string, unknown> = { entity_id: entityId, ...extraBody }
      try {
        const result = await callHaService(normalizedUrl, token, domain, service, serviceBody)
        if (!result.ok) {
          return { status: 'failed', detail: `HA returned HTTP ${result.status}` }
        }
        return { status: 'attempted', detail: `${domain}.${service} sent` }
      } catch (err) {
        const msg =
          err instanceof Error
            ? err.name === 'AbortError'
              ? 'Connection timed out.'
              : err.message
            : String(err)
        return { status: 'failed', detail: msg }
      }
    }
  )
}
