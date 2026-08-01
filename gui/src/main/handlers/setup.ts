import { ipcMain } from 'electron'
import { spawn } from 'child_process'
import type { SetupCompletePayload, SetupCompleteResponse, SetupStatusResponse } from '../../types/ipc'
import { bridgeSpawnOptions, resolveBridgePath, resolvePythonCommand } from '../bridgeResolver'

function callSetupBridge(payload: Record<string, unknown>): Promise<Record<string, unknown>> {
  return new Promise((resolve) => {
    const scriptPath = resolveBridgePath('rex_setup_bridge.py')

    const py = spawn(resolvePythonCommand(), [scriptPath], {
      ...bridgeSpawnOptions(),
      stdio: ['pipe', 'pipe', 'pipe']
    })

    let stdout = ''
    let stderr = ''

    py.stdout.on('data', (chunk: Buffer) => {
      stdout += chunk.toString()
    })

    py.stderr.on('data', (chunk: Buffer) => {
      stderr += chunk.toString()
    })

    py.on('close', (code) => {
      if (code !== 0) {
        resolve({ ok: false, error: `bridge exited ${code}: ${stderr.slice(0, 200)}` })
        return
      }
      try {
        resolve(JSON.parse(stdout.trim()) as Record<string, unknown>)
      } catch {
        resolve({ ok: false, error: `parse error: ${stdout.slice(0, 100)}` })
      }
    })

    py.on('error', (err) => {
      resolve({ ok: false, error: `spawn error: ${err.message}` })
    })

    py.stdin.write(JSON.stringify(payload))
    py.stdin.end()
  })
}

export function registerSetupHandlers(): void {
  ipcMain.handle(
    'rex:getSetupStatus',
    (): Promise<SetupStatusResponse> =>
      callSetupBridge({ command: 'status' }) as unknown as Promise<SetupStatusResponse>
  )

  ipcMain.handle(
    'rex:completeSetup',
    (_event, payload: SetupCompletePayload): Promise<SetupCompleteResponse> =>
      callSetupBridge({ command: 'complete', ...payload }) as unknown as Promise<SetupCompleteResponse>
  )
}
