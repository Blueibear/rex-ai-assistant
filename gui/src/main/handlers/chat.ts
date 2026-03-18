import { ipcMain } from 'electron'
import { spawn, ChildProcess } from 'child_process'
import { join } from 'path'

/**
 * Spawn the rex_chat_bridge.py script, pass the message via stdin,
 * and resolve with the reply string from stdout.
 */
function callRexBackend(message: string): Promise<string> {
  return new Promise((resolve, reject) => {
    // Bridge script lives at repo root (two levels up from gui/src/main/)
    const scriptPath = join(__dirname, '../../../../rex_chat_bridge.py')

    const py = spawn('python', [scriptPath], {
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
        reject(new Error(`Rex exited with code ${code}: ${stderr.slice(0, 300)}`))
        return
      }
      try {
        const result = JSON.parse(stdout.trim()) as { ok: boolean; reply?: string; error?: string }
        if (result.ok) {
          resolve(result.reply ?? '')
        } else {
          reject(new Error(result.error ?? 'Unknown error from Rex backend'))
        }
      } catch {
        reject(new Error(`Failed to parse Rex response: ${stdout.slice(0, 200)}`))
      }
    })

    py.on('error', (err) => {
      reject(new Error(`Failed to spawn Python: ${err.message}`))
    })

    py.stdin.write(JSON.stringify({ message }))
    py.stdin.end()
  })
}

// ---------------------------------------------------------------------------
// Persistent STT bridge — keeps Whisper loaded between requests so each
// transcription does not pay the 2-10 s model-load cost.
// ---------------------------------------------------------------------------

let sttProcess: ChildProcess | null = null
let sttReady = false
let sttLineBuffer = ''
const sttReadyCallbacks: Array<(err?: Error) => void> = []
const sttPendingRequests = new Map<
  string,
  { resolve: (t: string) => void; reject: (e: Error) => void }
>()

/**
 * Ensure the persistent STT bridge process is running and ready.
 * If already ready, resolves immediately.
 * If starting, queues until the "ready" event arrives.
 */
function ensureSTTProcess(): Promise<void> {
  if (sttProcess !== null && sttReady) {
    return Promise.resolve()
  }

  if (sttProcess !== null && !sttReady) {
    // Bridge is starting — queue until ready.
    return new Promise((resolve, reject) => {
      sttReadyCallbacks.push((err) => (err ? reject(err) : resolve()))
    })
  }

  // Spawn a new bridge process.
  return new Promise((resolve, reject) => {
    const scriptPath = join(__dirname, '../../../../rex_stt_bridge.py')
    sttProcess = spawn('python', [scriptPath], { stdio: ['pipe', 'pipe', 'pipe'] })
    sttReady = false
    sttLineBuffer = ''

    sttProcess.stdout!.on('data', (chunk: Buffer) => {
      sttLineBuffer += chunk.toString()
      const lines = sttLineBuffer.split('\n')
      sttLineBuffer = lines.pop() ?? ''

      for (const line of lines) {
        const trimmed = line.trim()
        if (!trimmed) continue
        try {
          const obj = JSON.parse(trimmed) as {
            type?: string
            ok?: boolean
            transcript?: string
            error?: string
            request_id?: string
            model?: string
          }

          if (obj.type === 'ready') {
            sttReady = true
            resolve()
            const cbs = sttReadyCallbacks.splice(0)
            cbs.forEach((cb) => cb())
          } else if (obj.type === 'error') {
            const err = new Error(obj.error ?? 'STT bridge startup error')
            reject(err)
            const cbs = sttReadyCallbacks.splice(0)
            cbs.forEach((cb) => cb(err))
            sttProcess = null
            sttReady = false
          } else if (typeof obj.request_id === 'string') {
            const pending = sttPendingRequests.get(obj.request_id)
            if (pending) {
              sttPendingRequests.delete(obj.request_id)
              if (obj.ok) {
                pending.resolve(obj.transcript ?? '')
              } else {
                pending.reject(new Error(obj.error ?? 'Transcription failed'))
              }
            }
          }
        } catch {
          // skip malformed NDJSON lines
        }
      }
    })

    sttProcess.on('close', () => {
      sttProcess = null
      sttReady = false
      const err = new Error('STT bridge process closed unexpectedly')
      sttPendingRequests.forEach((p) => p.reject(err))
      sttPendingRequests.clear()
    })

    sttProcess.on('error', (err) => {
      const startErr = new Error(`Failed to spawn STT bridge: ${err.message}`)
      reject(startErr)
      sttProcess = null
      sttReady = false
    })
  })
}

async function transcribeAudio(audioBase64: string): Promise<string> {
  await ensureSTTProcess()

  if (!sttProcess) {
    throw new Error('STT process is not available')
  }

  return new Promise((resolve, reject) => {
    const requestId = `${Date.now()}-${Math.random()}`
    sttPendingRequests.set(requestId, { resolve, reject })
    sttProcess!.stdin!.write(
      JSON.stringify({ audio_base64: audioBase64, request_id: requestId }) + '\n'
    )
  })
}

export function registerChatHandlers(): void {
  ipcMain.handle('rex:sendChat', async (_event, message: string): Promise<string> => {
    return callRexBackend(message)
  })

  ipcMain.handle(
    'rex:startChatStream',
    async (
      event,
      { message, streamId }: { message: string; streamId: string }
    ): Promise<{ ok: boolean }> => {
      const scriptPath = join(__dirname, '../../../../rex_chat_stream_bridge.py')

      const py = spawn('python', [scriptPath], {
        stdio: ['pipe', 'pipe', 'pipe']
      })

      let sentFinal = false

      function sendToken(token: string): void {
        if (!event.sender.isDestroyed()) {
          event.sender.send('rex:chatToken', { streamId, token })
        }
      }

      function sendDone(): void {
        if (!sentFinal && !event.sender.isDestroyed()) {
          sentFinal = true
          event.sender.send('rex:chatDone', { streamId })
        }
      }

      function sendError(error: string): void {
        if (!sentFinal && !event.sender.isDestroyed()) {
          sentFinal = true
          event.sender.send('rex:chatError', { streamId, error })
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
            const obj = JSON.parse(trimmed) as { type: string; token?: string; error?: string }
            if (obj.type === 'token' && obj.token !== undefined) {
              sendToken(obj.token)
            } else if (obj.type === 'done') {
              sendDone()
            } else if (obj.type === 'error') {
              sendError(obj.error ?? 'Unknown streaming error')
            }
          } catch {
            // skip malformed NDJSON lines
          }
        }
      })

      py.on('close', (code) => {
        // Flush any remaining buffered line
        if (lineBuffer.trim()) {
          try {
            const obj = JSON.parse(lineBuffer.trim()) as {
              type: string
              token?: string
              error?: string
            }
            if (obj.type === 'token' && obj.token !== undefined) sendToken(obj.token)
            else if (obj.type === 'done') sendDone()
            else if (obj.type === 'error') sendError(obj.error ?? 'Streaming error')
          } catch {
            // ignore
          }
        }
        if (code !== 0) {
          sendError(`Streaming bridge exited with code ${code}`)
        } else {
          sendDone()
        }
      })

      py.on('error', (_err) => {
        // Streaming bridge not available — fall back to non-streaming
        callRexBackend(message)
          .then((reply) => {
            sendToken(reply)
            sendDone()
          })
          .catch((e: Error) => {
            sendError(e.message)
          })
      })

      py.stdin.write(JSON.stringify({ message }))
      py.stdin.end()

      return { ok: true }
    }
  )

  ipcMain.handle(
    'rex:sendChatAudio',
    async (
      _event,
      audioBase64: string
    ): Promise<{ ok: boolean; transcript?: string; error?: string }> => {
      try {
        const transcript = await transcribeAudio(audioBase64)
        return { ok: true, transcript }
      } catch (err) {
        return {
          ok: false,
          error: err instanceof Error ? err.message : String(err)
        }
      }
    }
  )
}
