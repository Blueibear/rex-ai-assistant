import { spawn } from 'child_process'
import { ipcMain } from 'electron'
import type { OutputRoutingPolicy, OutputRoutingResponse } from '../../types/outputRouting'
import { bridgeSpawnOptions, resolveBridgePath, resolvePythonCommand } from '../bridgeResolver'
import { privateSessionPayload, type ElectronSessionIdentity } from '../sessionIdentity'

export function callOutputRoutingBridge(
  session: ElectronSessionIdentity,
  payload: Record<string, unknown>
): Promise<OutputRoutingResponse> {
  return new Promise((resolve) => {
    const scriptPath = resolveBridgePath('rex_output_routing_bridge.py')
    const py = spawn(resolvePythonCommand(), [scriptPath], {
      ...bridgeSpawnOptions(),
      stdio: ['pipe', 'pipe', 'pipe']
    })

    let stdout = ''
    py.stdout.on('data', (chunk: Buffer) => {
      stdout += chunk.toString()
    })

    py.on('close', (code) => {
      if (code !== 0) {
        resolve({ ok: false, error: 'Output routing service could not complete the request.' })
        return
      }
      try {
        resolve(JSON.parse(stdout.trim()) as OutputRoutingResponse)
      } catch {
        resolve({ ok: false, error: 'Output routing service returned an invalid response.' })
      }
    })

    py.on('error', () => {
      resolve({ ok: false, error: 'Output routing service is unavailable.' })
    })

    py.stdin.write(JSON.stringify(privateSessionPayload(session, payload)))
    py.stdin.end()
  })
}

export function registerOutputRoutingHandlers(session: ElectronSessionIdentity): void {
  ipcMain.handle('rex:getOutputRoutingPolicy', () =>
    callOutputRoutingBridge(session, { command: 'get_policy' })
  )
  ipcMain.handle('rex:updateOutputRoutingPolicy', (_event, policy: OutputRoutingPolicy) =>
    callOutputRoutingBridge(session, { command: 'update_policy', policy })
  )
  ipcMain.handle('rex:listMediaAccounts', () =>
    callOutputRoutingBridge(session, { command: 'list_media_accounts' })
  )
  ipcMain.handle('rex:testOutputRoutingTarget', (_event, targetId: string) =>
    callOutputRoutingBridge(session, { command: 'test_playback', target_id: targetId })
  )
}
