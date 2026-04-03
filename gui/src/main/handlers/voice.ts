import { app, ipcMain } from 'electron'
import { spawn } from 'child_process'
import { join } from 'path'
import { existsSync } from 'fs'
import type { ChildProcess } from 'child_process'

let voiceProcess: ChildProcess | null = null

type BridgeResult<T> = T & { ok: boolean; error?: string }

function resolvePythonCommand(): string {
  const bundledVenvPython = join(app.getAppPath(), '..', '.venv', 'Scripts', 'python.exe')
  return existsSync(bundledVenvPython) ? bundledVenvPython : 'python'
}

function resolveBridgeScript(scriptName: string): string {
  return join(app.getAppPath(), '..', scriptName)
}

function killVoiceProcess(): void {
  const py = voiceProcess
  voiceProcess = null
  if (py) {
    try {
      py.stdin?.write(JSON.stringify({ command: 'stop' }) + '\n')
      py.stdin?.end()
    } catch {
      py.kill()
    }
  }
}

export function registerVoiceHandlers(): void {
  ipcMain.handle('rex:startVoice', async (event): Promise<{ ok: boolean; error?: string }> => {
    // Kill any existing session first.
    if (voiceProcess) {
      killVoiceProcess()
    }

    const scriptPath = resolveBridgeScript('rex_voice_bridge.py')

    const py = spawn(resolvePythonCommand(), [scriptPath], {
      stdio: ['pipe', 'pipe', 'pipe']
    })

    voiceProcess = py

    function sendIfAlive(channel: string, data: unknown): void {
      if (!event.sender.isDestroyed()) {
        event.sender.send(channel, data)
      }
    }

    let lineBuffer = ''

    py.stdout.on('data', (chunk: Buffer) => {
      lineBuffer += chunk.toString()
      const lines = lineBuffer.split('\n')
      lineBuffer = lines.pop() ?? ''
      for (const line of lines) {
        const trimmed = line.trim()
        if (!trimmed) continue
        try {
          const obj = JSON.parse(trimmed) as {
            type: string
            state?: string
            text?: string
            role?: string
            timestamp?: number
            error?: string
          }
          if (obj.type === 'state' && obj.state) {
            sendIfAlive('rex:voiceState', { state: obj.state })
          } else if (obj.type === 'transcript') {
            sendIfAlive('rex:voiceTranscript', {
              text: obj.text ?? '',
              role: obj.role ?? 'rex',
              timestamp: obj.timestamp ?? Date.now()
            })
          } else if (obj.type === 'error') {
            sendIfAlive('rex:voiceError', { error: obj.error ?? 'Unknown voice error' })
          }
        } catch {
          // skip malformed NDJSON lines
        }
      }
    })

    py.on('close', () => {
      if (voiceProcess === py) {
        voiceProcess = null
      }
      sendIfAlive('rex:voiceState', { state: 'idle' })
    })

    py.on('error', (err) => {
      if (voiceProcess === py) {
        voiceProcess = null
      }
      sendIfAlive('rex:voiceError', { error: `Failed to start voice bridge: ${err.message}` })
    })

    return { ok: true }
  })

  ipcMain.handle('rex:stopVoice', async (): Promise<{ ok: boolean }> => {
    killVoiceProcess()
    return { ok: true }
  })

  ipcMain.handle(
    'rex:listVoices',
    async (
      _event,
      provider: string
    ): Promise<{ ok: boolean; voices: unknown[]; error?: string }> => {
      const scriptPath = resolveBridgeScript('rex_voices_bridge.py')
      return new Promise((resolve) => {
        const py = spawn(resolvePythonCommand(), [scriptPath], { stdio: ['pipe', 'pipe', 'pipe'] })
        let stdout = ''
        let stderr = ''
        py.stdout.on('data', (chunk: Buffer) => { stdout += chunk.toString() })
        py.stderr.on('data', (chunk: Buffer) => { stderr += chunk.toString() })
        py.on('close', (code) => {
          if (code !== 0 && stdout.trim() === '') {
            resolve({ ok: false, voices: [], error: stderr || `Bridge exited with code ${code}` })
            return
          }
          try {
            const result = JSON.parse(stdout.trim()) as { ok: boolean; voices?: unknown[]; error?: string }
            resolve({ ok: result.ok, voices: result.voices ?? [], error: result.error })
          } catch {
            resolve({ ok: false, voices: [], error: stderr || 'Failed to parse response' })
          }
        })
        py.on('error', (err) => {
          resolve({ ok: false, voices: [], error: `Failed to start bridge: ${err.message}` })
        })
        py.stdin?.write(JSON.stringify({ provider }) + '\n')
        py.stdin?.end()
      })
    }
  )

  ipcMain.handle(
    'rex:previewVoice',
    async (
      _event,
      provider: string,
      voiceId: string
    ): Promise<{ ok: boolean; audio_base64?: string; error?: string }> => {
      const scriptPath = resolveBridgeScript('rex_voice_sample_bridge.py')
      return new Promise((resolve) => {
        const py = spawn(resolvePythonCommand(), [scriptPath], { stdio: ['pipe', 'pipe', 'pipe'] })
        let stdout = ''
        let stderr = ''
        py.stdout.on('data', (chunk: Buffer) => { stdout += chunk.toString() })
        py.stderr.on('data', (chunk: Buffer) => { stderr += chunk.toString() })
        py.on('close', (code) => {
          if (code !== 0 && stdout.trim() === '') {
            resolve({ ok: false, error: stderr || `Bridge exited with code ${code}` })
            return
          }
          try {
            const result = JSON.parse(stdout.trim()) as { ok: boolean; audio_base64?: string; error?: string }
            resolve(result)
          } catch {
            resolve({ ok: false, error: stderr || 'Failed to parse response' })
          }
        })
        py.on('error', (err) => {
          resolve({ ok: false, error: `Failed to start bridge: ${err.message}` })
        })
        py.stdin?.write(JSON.stringify({ provider, voice_id: voiceId }) + '\n')
        py.stdin?.end()
      })
    }
  )

  ipcMain.handle(
    'rex:uploadCustomVoice',
    async (
      _event,
      filePath: string,
      voiceName: string
    ): Promise<{ ok: boolean; voice_id?: string; voice_name?: string; duration?: number; error?: string }> => {
      const scriptPath = resolveBridgeScript('rex_voice_upload_bridge.py')
      return new Promise((resolve) => {
        const py = spawn(resolvePythonCommand(), [scriptPath], { stdio: ['pipe', 'pipe', 'pipe'] })
        let stdout = ''
        let stderr = ''
        py.stdout.on('data', (chunk: Buffer) => { stdout += chunk.toString() })
        py.stderr.on('data', (chunk: Buffer) => { stderr += chunk.toString() })
        py.on('close', (code) => {
          if (code !== 0 && stdout.trim() === '') {
            resolve({ ok: false, error: stderr || `Bridge exited with code ${code}` })
            return
          }
          try {
            const result = JSON.parse(stdout.trim()) as {
              ok: boolean
              voice_id?: string
              voice_name?: string
              duration?: number
              error?: string
            }
            resolve(result)
          } catch {
            resolve({ ok: false, error: stderr || 'Failed to parse response' })
          }
        })
        py.on('error', (err) => {
          resolve({ ok: false, error: `Failed to start bridge: ${err.message}` })
        })
        py.stdin?.write(JSON.stringify({ file_path: filePath, voice_name: voiceName }) + '\n')
        py.stdin?.end()
      })
    }
  )

  ipcMain.handle(
    'rex:getVoiceEnrollments',
    async (): Promise<BridgeResult<{ active_user_id: string; enrollments: unknown[] }>> => {
      return callEnrollmentBridge({
        action: 'list'
      }) as Promise<BridgeResult<{ active_user_id: string; enrollments: unknown[] }>>
    }
  )

  ipcMain.handle(
    'rex:enrollVoice',
    async (
      _event,
      userId: string,
      samples: number[][]
    ): Promise<BridgeResult<{ enrollment?: unknown }>> => {
      return callEnrollmentBridge({ action: 'enroll', user_id: userId, samples })
    }
  )

  ipcMain.handle(
    'rex:deleteVoiceEnrollment',
    async (_event, userId: string): Promise<BridgeResult<{ deleted?: boolean }>> => {
      return callEnrollmentBridge({ action: 'delete', user_id: userId })
    }
  )
}

function callEnrollmentBridge(payload: Record<string, unknown>): Promise<BridgeResult<Record<string, unknown>>> {
  const scriptPath = resolveBridgeScript('rex_voice_enrollment_bridge.py')
  return new Promise((resolve) => {
    const py = spawn(resolvePythonCommand(), [scriptPath], { stdio: ['pipe', 'pipe', 'pipe'] })
    let stdout = ''
    let stderr = ''

    py.stdout.on('data', (chunk: Buffer) => {
      stdout += chunk.toString()
    })
    py.stderr.on('data', (chunk: Buffer) => {
      stderr += chunk.toString()
    })
    py.on('close', (code) => {
      if (code !== 0 && stdout.trim() === '') {
        resolve({ ok: false, error: stderr || `Bridge exited with code ${code}` })
        return
      }
      try {
        resolve(JSON.parse(stdout.trim()) as BridgeResult<Record<string, unknown>>)
      } catch {
        resolve({ ok: false, error: stderr || 'Failed to parse response' })
      }
    })
    py.on('error', (err) => {
      resolve({ ok: false, error: `Failed to start bridge: ${err.message}` })
    })
    py.stdin?.write(JSON.stringify(payload) + '\n')
    py.stdin?.end()
  })
}
