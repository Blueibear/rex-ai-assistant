import { ipcMain } from 'electron'
import { spawnSync } from 'child_process'
import { bridgeSpawnOptions, resolveBridgePath, resolvePythonCommand } from '../bridgeResolver'
import { privateSessionPayload, type ElectronSessionIdentity } from '../sessionIdentity'

interface HistoryEntry {
  id: number
  timestamp: string
  command: string
  result: string
  success: boolean
}

interface HistoryResponse {
  ok: boolean
  history: HistoryEntry[]
  error?: string
}

export function registerHistoryHandlers(session: ElectronSessionIdentity): void {
  ipcMain.handle('rex:getCommandHistory', (_event, limit: number = 50): { ok: boolean; history: HistoryEntry[]; error?: string } => {
    try {
      const scriptPath = resolveBridgePath('rex_history_bridge.py')
      const result = spawnSync(resolvePythonCommand(), [scriptPath], {
        ...bridgeSpawnOptions(),
        input: JSON.stringify(
          privateSessionPayload(session, {
            command: 'list',
            limit: Math.max(1, Math.min(limit, 500))
          })
        ),
        encoding: 'utf8',
        timeout: 5000
      })
      if (result.status !== 0) {
        const err = (result.stderr || '').trim().slice(0, 300)
        return {
          ok: false,
          history: [],
          error: err || 'Bridge exited non-zero'
        }
      }
      const parsed = JSON.parse(result.stdout.trim()) as HistoryResponse
      return {
        ok: parsed.ok,
        history: parsed.history ?? [],
        error: parsed.error
      }
    } catch (err) {
      return { ok: false, history: [], error: String(err) }
    }
  })
}
