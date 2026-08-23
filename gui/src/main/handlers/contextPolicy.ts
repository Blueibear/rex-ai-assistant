import { spawn } from 'child_process'
import { ipcMain } from 'electron'
import type { ContextPrivacyResponse } from '../../types/ipc'
import { bridgeSpawnOptions, resolveBridgePath, resolvePythonCommand } from '../bridgeResolver'
import { privateSessionPayload, type ElectronSessionIdentity } from '../sessionIdentity'

export function callContextPolicyBridge(
  session: ElectronSessionIdentity,
  payload: Record<string, unknown>
): Promise<ContextPrivacyResponse> {
  return new Promise((resolve) => {
    const scriptPath = resolveBridgePath('rex_context_policy_bridge.py')
    const py = spawn(resolvePythonCommand(), [scriptPath], {
      ...bridgeSpawnOptions(),
      stdio: ['pipe', 'pipe', 'pipe']
    })

    let stdout = ''
    py.stdout.on('data', (chunk: Buffer) => {
      stdout += chunk.toString()
    })

    py.on('close', (code) => {
      try {
        const parsed = JSON.parse(stdout.trim()) as ContextPrivacyResponse
        resolve(code === 0 ? parsed : { ok: false, error: parsed.error ?? 'Privacy request failed' })
      } catch {
        resolve({ ok: false, error: 'Privacy service returned an invalid response.' })
      }
    })

    py.on('error', () => {
      resolve({ ok: false, error: 'Privacy service is unavailable.' })
    })

    py.stdin.write(JSON.stringify(privateSessionPayload(session, payload)))
    py.stdin.end()
  })
}

export function registerContextPolicyHandlers(session: ElectronSessionIdentity): void {
  ipcMain.handle('rex:getContextPrivacy', () =>
    callContextPolicyBridge(session, { command: 'get_state' })
  )
  ipcMain.handle(
    'rex:updateContextPrivacy',
    (_event, command: string, payload: Record<string, unknown>) =>
      callContextPolicyBridge(session, { command, ...payload })
  )
}
