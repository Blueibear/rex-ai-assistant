import { ipcMain } from 'electron'
import { existsSync, readFileSync } from 'fs'
import { join } from 'path'
import { getConfigDir } from '../configStore'

interface DeviceEntry {
  entity_id: string
  name: string
  type: string
}

interface DevicesResponse {
  ok: boolean
  devices: DeviceEntry[]
  error?: string
}

export function registerDevicesHandlers(): void {
  ipcMain.handle('rex:getDevices', (): DevicesResponse => {
    try {
      const aliasesPath = join(getConfigDir(), 'device_aliases.json')
      if (!existsSync(aliasesPath)) {
        return { ok: true, devices: [] }
      }
      const raw = JSON.parse(readFileSync(aliasesPath, 'utf8')) as Record<string, unknown>
      const devices = Array.isArray(raw.devices) ? (raw.devices as DeviceEntry[]) : []
      return { ok: true, devices }
    } catch (err) {
      return { ok: false, devices: [], error: String(err) }
    }
  })
}
