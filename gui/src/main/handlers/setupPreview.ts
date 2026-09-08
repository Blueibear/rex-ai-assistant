import { ipcMain } from 'electron'
import { spawn } from 'child_process'
import type { Settings, WakeWordStatus } from '../../types/ipc'
import { bridgeSpawnOptions, resolveBridgePath, resolvePythonCommand } from '../bridgeResolver'
import { readGuiSettings } from '../configStore'
import { buildWakeWordStatus } from '../voiceSettings'

const SETUP_PREVIEW_CHANNELS = [
  'rex:listVoices',
  'rex:previewVoice',
  'rex:listWakeWords',
  'rex:previewWakeWordSample',
  'rex:getWakeWordStatus'
] as const

function callJsonBridge(
  scriptName: string,
  payload: Record<string, unknown>
): Promise<Record<string, unknown>> {
  return new Promise((resolve) => {
    const py = spawn(resolvePythonCommand(), [resolveBridgePath(scriptName)], {
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
      if (code !== 0 && stdout.trim() === '') {
        resolve({ ok: false, error: stderr || `Bridge exited with code ${code}` })
        return
      }
      try {
        resolve(JSON.parse(stdout.trim()) as Record<string, unknown>)
      } catch {
        resolve({ ok: false, error: stderr || 'Failed to parse response' })
      }
    })
    py.on('error', (error) => {
      resolve({ ok: false, error: `Failed to start bridge: ${error.message}` })
    })
    py.stdin?.write(JSON.stringify(payload) + '\n')
    py.stdin?.end()
  })
}

export function registerSetupPreviewHandlers(): void {
  ipcMain.handle('rex:listVoices', async (_event, provider: string) => {
    const result = await callJsonBridge('rex_voices_bridge.py', { provider })
    return {
      ok: result.ok === true,
      voices: Array.isArray(result.voices) ? result.voices : [],
      error: typeof result.error === 'string' ? result.error : undefined
    }
  })

  ipcMain.handle('rex:previewVoice', async (_event, provider: string, voiceId: string) =>
    callJsonBridge('rex_voice_sample_bridge.py', { provider, voice_id: voiceId })
  )

  ipcMain.handle('rex:listWakeWords', async () => {
    const result = await callJsonBridge('rex_wakeword_list_bridge.py', {})
    return {
      ok: result.ok === true,
      wake_words: Array.isArray(result.wake_words) ? result.wake_words : [],
      error: typeof result.error === 'string' ? result.error : undefined,
      warning: typeof result.warning === 'string' ? result.warning : undefined
    }
  })

  ipcMain.handle('rex:previewWakeWordSample', async (_event, wakeWordId: string) =>
    callJsonBridge('rex_wakeword_sample_bridge.py', { wake_word_id: wakeWordId })
  )

  ipcMain.handle('rex:getWakeWordStatus', (_event, values?: Settings): WakeWordStatus => {
    const stored = readGuiSettings()
    const source = values ?? ((stored.voice ?? {}) as Settings)
    return buildWakeWordStatus(source)
  })
}

export function unregisterSetupPreviewHandlers(): void {
  for (const channel of SETUP_PREVIEW_CHANNELS) {
    ipcMain.removeHandler(channel)
  }
}
