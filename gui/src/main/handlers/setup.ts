import { ipcMain } from 'electron'
import { spawn } from 'child_process'
import type {
  SetupAudioDevicesResponse,
  SetupAudioTestResponse,
  SetupCompletePayload,
  SetupCompleteResponse,
  SetupStatusResponse
} from '../../types/ipc'
import { bridgeSpawnOptions, resolveBridgePath, resolvePythonCommand } from '../bridgeResolver'

export interface ElectronSetupStatus extends SetupStatusResponse {
  background_voice_enabled: boolean
}

type SetupCompletedCallback = () => void | Promise<void>

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
        resolve({ ok: false, error: 'Setup service is unavailable.' })
        return
      }
      try {
        resolve(JSON.parse(stdout.trim()) as Record<string, unknown>)
      } catch {
        resolve({ ok: false, error: 'Setup service returned an invalid response.' })
      }
    })

    py.on('error', () => {
      resolve({ ok: false, error: 'Setup service could not be started.' })
    })

    py.stdin.write(JSON.stringify(payload))
    py.stdin.end()
  })
}

export async function readSetupStatus(): Promise<ElectronSetupStatus> {
  const result = await callSetupBridge({ command: 'status' })
  if (typeof result.needs_setup !== 'boolean') {
    throw new Error(
      typeof result.error === 'string' ? result.error : 'Setup status is unavailable.'
    )
  }
  return {
    needs_setup: result.needs_setup,
    background_voice_enabled: result.background_voice_enabled === true
  }
}

export function registerSetupHandlers(onSetupCompleted?: SetupCompletedCallback): void {
  ipcMain.handle('rex:getSetupStatus', (): Promise<SetupStatusResponse> => readSetupStatus())

  ipcMain.handle(
    'rex:getSetupAudioDevices',
    (): Promise<SetupAudioDevicesResponse> =>
      callSetupBridge({ command: 'audio_devices' }) as unknown as Promise<SetupAudioDevicesResponse>
  )

  ipcMain.handle(
    'rex:testSetupAudioDevice',
    (_event, kind: 'microphone' | 'speaker', deviceIndex: number): Promise<SetupAudioTestResponse> =>
      callSetupBridge({
        command: 'test_audio_device',
        kind,
        device_index: deviceIndex
      }) as unknown as Promise<SetupAudioTestResponse>
  )

  ipcMain.handle(
    'rex:completeSetup',
    async (_event, payload: SetupCompletePayload): Promise<SetupCompleteResponse> => {
      const result = (await callSetupBridge({
        command: 'complete',
        ...payload
      })) as unknown as SetupCompleteResponse
      if (!result.ok) {
        return result
      }

      if (!onSetupCompleted) {
        return { ...result, setup_saved: true, runtime_ready: true }
      }

      try {
        await onSetupCompleted()
        return { ...result, setup_saved: true, runtime_ready: true }
      } catch {
        return {
          ...result,
          setup_saved: true,
          runtime_ready: false,
          warning:
            'Setup was saved, but Rex could not finish starting. Close and reopen AskRex to continue.'
        }
      }
    }
  )
}
