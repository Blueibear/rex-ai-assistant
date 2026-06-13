import { ipcMain } from 'electron'
import { existsSync, readFileSync } from 'fs'
import { join } from 'path'
import { getConfigDir } from '../configStore'

interface DeviceEntry {
  entity_id: string
  name: string
  type: string
}

interface DeviceAliasesFile {
  devices?: DeviceEntry[]
}

function loadDevices(): DeviceEntry[] {
  const aliasesPath = join(getConfigDir(), 'device_aliases.json')
  if (!existsSync(aliasesPath)) return []
  try {
    const raw = JSON.parse(readFileSync(aliasesPath, 'utf8')) as DeviceAliasesFile
    const devices = Array.isArray(raw.devices) ? raw.devices : []
    return devices.filter(
      (d) => d && typeof d.entity_id === 'string' && typeof d.name === 'string'
    )
  } catch {
    return []
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
}
