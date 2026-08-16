import { ipcMain } from 'electron'
import { spawn } from 'child_process'
import type {
  AudioTargetsResponse,
  SpeakerGroupMutationResponse,
  SpeakerGroupsResponse
} from '../../types/ipc'
import { bridgeSpawnOptions, resolveBridgePath, resolvePythonCommand } from '../bridgeResolver'
import {
  privateSessionPayload,
  type ElectronSessionIdentity
} from '../sessionIdentity'

type SpeakerBridgeResponse =
  | AudioTargetsResponse
  | SpeakerGroupsResponse
  | SpeakerGroupMutationResponse
  | { ok: boolean; deleted?: boolean; error?: string }

function callSpeakerBridge(
  session: ElectronSessionIdentity,
  payload: Record<string, unknown>
): Promise<SpeakerBridgeResponse> {
  return new Promise((resolve) => {
    const scriptPath = resolveBridgePath('rex_speaker_bridge.py')
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
        resolve({ ok: false, error: 'Speaker service could not complete the request.' })
        return
      }
      try {
        resolve(JSON.parse(stdout.trim()) as SpeakerBridgeResponse)
      } catch {
        resolve({ ok: false, error: 'Speaker service returned an invalid response.' })
      }
    })

    py.on('error', () => {
      resolve({ ok: false, error: 'Speaker service is unavailable.' })
    })

    py.stdin.write(JSON.stringify(privateSessionPayload(session, payload)))
    py.stdin.end()
  })
}

export function registerSpeakerHandlers(session: ElectronSessionIdentity): void {
  ipcMain.handle('rex:getAudioTargets', () =>
    callSpeakerBridge(session, { command: 'list_targets' })
  )
  ipcMain.handle('rex:refreshAudioTargets', () =>
    callSpeakerBridge(session, { command: 'refresh_targets' })
  )
  ipcMain.handle('rex:listSpeakerGroups', () =>
    callSpeakerBridge(session, { command: 'list_groups' })
  )
  ipcMain.handle('rex:createSpeakerGroup', (_event, name: string, memberIds: string[]) =>
    callSpeakerBridge(session, { command: 'create_group', name, member_ids: memberIds })
  )
  ipcMain.handle('rex:renameSpeakerGroup', (_event, groupId: string, name: string) =>
    callSpeakerBridge(session, { command: 'rename_group', group_id: groupId, name })
  )
  ipcMain.handle('rex:setSpeakerGroupMembers', (_event, groupId: string, memberIds: string[]) =>
    callSpeakerBridge(session, {
      command: 'set_group_members',
      group_id: groupId,
      member_ids: memberIds
    })
  )
  ipcMain.handle('rex:deleteSpeakerGroup', (_event, groupId: string) =>
    callSpeakerBridge(session, { command: 'delete_group', group_id: groupId })
  )
}
