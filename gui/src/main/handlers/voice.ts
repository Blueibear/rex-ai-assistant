import { ipcMain } from 'electron'
import { spawn } from 'child_process'
import type { ChildProcess } from 'child_process'
import { resolveBridgePath, resolvePythonCommand } from '../bridgeResolver'

let voiceProcess: ChildProcess | null = null
let currentVoiceState = 'idle'

type BridgeResult<T> = T & { ok: boolean; error?: string }

function resolveBridgeScript(scriptName: string): string {
  return resolveBridgePath(scriptName)
}

function normalizeBridgeVoiceState(state: string): string {
  if (state === 'thinking' || state === 'executing') return 'processing'
  if (state === 'done') return 'idle'
  return state
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

    return new Promise((resolve) => {
      let startupSettled = false
      let stderr = ''
      let startupStatus = 'starting bridge process'

      function settleStartup(result: { ok: boolean; error?: string }): void {
        if (startupSettled) return
        startupSettled = true
        clearTimeout(startupTimer)
        resolve(result)
      }

      function failStartup(error: string): void {
        if (!startupSettled) {
          if (voiceProcess === py) {
            voiceProcess = null
          }
          try {
            py.kill()
          } catch {
            // Process may already be gone.
          }
          settleStartup({ ok: false, error })
        } else {
          sendIfAlive('rex:voiceError', { error })
        }
      }

      function sendIfAlive(channel: string, data: unknown): void {
        if (!event.sender.isDestroyed()) {
          event.sender.send(channel, data)
        }
      }

      const startupTimer = setTimeout(() => {
        const detail = stderr.trim() ? ` Last stderr: ${stderr.trim().slice(-500)}` : ''
        failStartup(`Voice bridge did not become ready within 45 seconds while ${startupStatus}.${detail}`)
      }, 45_000)

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
              status?: string
              traceback?: string
            }
            if (obj.type === 'ready') {
              settleStartup({ ok: true })
            } else if (obj.type === 'status' && obj.status) {
              startupStatus = obj.status.replace(/_/g, ' ')
            } else if (obj.type === 'state' && obj.state) {
              const normalizedState = normalizeBridgeVoiceState(obj.state)
              currentVoiceState = normalizedState
              sendIfAlive('rex:voiceState', { state: normalizedState })
            } else if (obj.type === 'transcript') {
              sendIfAlive('rex:voiceTranscript', {
                text: obj.text ?? '',
                role: obj.role ?? 'rex',
                timestamp: obj.timestamp ?? Date.now()
              })
            } else if (obj.type === 'error') {
              const error = obj.error ?? 'Unknown voice error'
              failStartup(error)
            }
          } catch {
            // skip malformed NDJSON lines
          }
        }
      })

      py.stderr.on('data', (chunk: Buffer) => {
        stderr += chunk.toString()
      })

      py.on('close', (code) => {
        if (voiceProcess === py) {
          voiceProcess = null
        }
        currentVoiceState = 'idle'
        sendIfAlive('rex:voiceState', { state: 'idle' })
        if (!startupSettled) {
          const detail = stderr.trim() ? `: ${stderr.trim().slice(-500)}` : ''
          settleStartup({
            ok: false,
            error: `Voice bridge exited before wake-word mode was ready (code ${code ?? 'unknown'})${detail}`
          })
        }
      })

      py.on('error', (err) => {
        if (voiceProcess === py) {
          voiceProcess = null
        }
        currentVoiceState = 'error'
        failStartup(`Failed to start voice bridge: ${err.message}`)
      })
    })
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
    'rex:listWakeWords',
    async (): Promise<{ ok: boolean; wake_words: unknown[]; error?: string; warning?: string }> => {
      const scriptPath = resolveBridgeScript('rex_wakeword_list_bridge.py')
      return new Promise((resolve) => {
        const py = spawn(resolvePythonCommand(), [scriptPath], { stdio: ['pipe', 'pipe', 'pipe'] })
        let stdout = ''
        let stderr = ''
        py.stdout.on('data', (chunk: Buffer) => { stdout += chunk.toString() })
        py.stderr.on('data', (chunk: Buffer) => { stderr += chunk.toString() })
        py.on('close', (code) => {
          if (code !== 0 && stdout.trim() === '') {
            resolve({ ok: false, wake_words: [], error: stderr || `Bridge exited with code ${code}` })
            return
          }
          try {
            const result = JSON.parse(stdout.trim()) as {
              ok: boolean
              wake_words?: unknown[]
              error?: string
              warning?: string
            }
            resolve({ ok: result.ok, wake_words: result.wake_words ?? [], error: result.error, warning: result.warning })
          } catch {
            resolve({ ok: false, wake_words: [], error: stderr || 'Failed to parse response' })
          }
        })
        py.on('error', (err) => {
          resolve({ ok: false, wake_words: [], error: `Failed to start bridge: ${err.message}` })
        })
        py.stdin?.write('{}')
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
    'rex:previewWakeWordSample',
    async (
      _event,
      wakeWordId: string
    ): Promise<{ ok: boolean; audio_base64?: string; has_sample?: boolean; error?: string }> => {
      const scriptPath = resolveBridgeScript('rex_wakeword_sample_bridge.py')
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
              audio_base64?: string
              has_sample?: boolean
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
        py.stdin?.write(JSON.stringify({ wake_word_id: wakeWordId }) + '\n')
        py.stdin?.end()
      })
    }
  )

  ipcMain.handle(
    'rex:trainWakeWord',
    async (
      _event,
      phrase: string,
      positiveSamples: number[][],
      negativeSamples: number[][]
    ): Promise<{ ok: boolean; model_path?: string; phrase?: string; error?: string }> => {
      const scriptPath = resolveBridgeScript('rex_wakeword_train_bridge.py')
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
              model_path?: string
              phrase?: string
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
        py.stdin?.write(JSON.stringify({ phrase, positive_samples: positiveSamples, negative_samples: negativeSamples }) + '\n')
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

export function getCurrentVoiceState(): string {
  return currentVoiceState
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
