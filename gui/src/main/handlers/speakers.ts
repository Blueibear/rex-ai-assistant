import { ipcMain, app } from 'electron'
import { spawn } from 'child_process'
import { join } from 'path'
import { existsSync } from 'fs'
import type { SmartSpeaker } from '../../types/ipc'

function resolvePythonCommand(): string {
  const bundledVenvPython = join(app.getAppPath(), '..', '.venv', 'Scripts', 'python.exe')
  return existsSync(bundledVenvPython) ? bundledVenvPython : 'python'
}

function callSpeakerBridge(
  payload: Record<string, unknown>
): Promise<{ ok: boolean; speakers: SmartSpeaker[]; error?: string }> {
  return new Promise((resolve) => {
    const scriptPath = join(__dirname, '../../../../rex_speaker_bridge.py')

    const py = spawn(resolvePythonCommand(), [scriptPath], {
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
        resolve({
          ok: false,
          speakers: [],
          error: `Speaker bridge exited with code ${code}: ${stderr.slice(0, 200)}`
        })
        return
      }
      try {
        const result = JSON.parse(stdout.trim()) as {
          ok: boolean
          speakers: SmartSpeaker[]
          error?: string
        }
        resolve(result)
      } catch {
        resolve({ ok: false, speakers: [], error: `Failed to parse response: ${stdout.slice(0, 100)}` })
      }
    })

    py.on('error', (err) => {
      resolve({ ok: false, speakers: [], error: `Failed to spawn speaker bridge: ${err.message}` })
    })

    py.stdin.write(JSON.stringify(payload))
    py.stdin.end()
  })
}

export function registerSpeakerHandlers(): void {
  ipcMain.handle(
    'rex:getSmartSpeakers',
    (): Promise<{ ok: boolean; speakers: SmartSpeaker[]; error?: string }> => {
      return callSpeakerBridge({ command: 'list' })
    }
  )
}
