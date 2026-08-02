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
    let _stderr = ''

    py.stdout.on('data', (chunk: Buffer) => {
      stdout += chunk.toString()
    })

    py.stderr.on('data', (chunk: Buffer) => {
      _stderr += chunk.toString()
    })

    py.on('close', (code) => {
      if (code !== 0) {
        resolve({ ok: false, error: "Setup service is unavailable." })
        return
      }
      try {
        resolve(JSON.parse(stdout.trim()) as Record<string, unknown>)
      } catch {
        resolve({ ok: false, error: "Setup service returned an invalid response." })
      }
    })

    py.on('error', (_err) => {
      resolve({ ok: false, error: "Setup service could not be started." })
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
