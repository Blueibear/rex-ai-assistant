import { ipcMain } from 'electron'
import { spawnSync } from 'child_process'
import { bridgeSpawnOptions, resolveBridgePath, resolvePythonCommand } from '../bridgeResolver'
import { privateSessionPayload, type ElectronSessionIdentity } from '../sessionIdentity'

interface ProfileBridgeResponse {
  ok: boolean
  error?: string
  profile?: Record<string, unknown>
}

interface PreferencesPayload {
  [key: string]: unknown
}

function callProfileBridge(
  session: ElectronSessionIdentity,
  payload: Record<string, unknown>
): ProfileBridgeResponse {
  try {
    const result = spawnSync(
      resolvePythonCommand(),
      [resolveBridgePath('rex_profile_bridge.py')],
      {
        ...bridgeSpawnOptions(),
        input: JSON.stringify(
          privateSessionPayload(session, payload)
        ),
        encoding: 'utf8',
        timeout: 15_000,
        windowsHide: true
      }
    )
    if (result.status !== 0) {
      return { ok: false, error: 'Profile service could not complete the request.' }
    }
    const parsed = JSON.parse((result.stdout || '').trim()) as ProfileBridgeResponse
    if (!parsed.ok) return { ok: false, error: 'Profile operation failed.' }
    if (!('profile' in parsed) && payload.action !== 'update_preferences' && payload.action !== 'remove_avatar' && payload.action !== 'set_avatar') {
      return { ok: false, error: 'Profile service returned an invalid response.' }
    }
    return parsed
  } catch {
    return { ok: false, error: 'Profile service is unavailable.' }
  }
}

export function registerProfileHandlers(session: ElectronSessionIdentity): void {
  ipcMain.handle('rex:getProfile', () =>
    callProfileBridge(session, { action: 'get' })
  )

  ipcMain.handle('rex:updateProfilePreferences', (_event, preferences: unknown) => {
    // Validate arguments before spawning
    if (!preferences || typeof preferences !== 'object' || Array.isArray(preferences)) {
      return { ok: false, error: 'Preferences must be a JSON object.' }
    }

    // Check reasonable serialized size before spawning
    try {
      const serialized = JSON.stringify(preferences)
      if (serialized.length > 32 * 1024) {
        return { ok: false, error: 'Preferences are too large.' }
      }
    } catch {
      return { ok: false, error: 'Preferences are not JSON-serializable.' }
    }

    return callProfileBridge(session, {
      action: 'update_preferences',
      preferences: preferences as PreferencesPayload
    })
  })

  ipcMain.handle('rex:setProfileAvatar', (_event, mimeType: unknown, avatarBase64: unknown) => {
    // Validate arguments before spawning
    if (typeof mimeType !== 'string' || !mimeType.trim()) {
      return { ok: false, error: 'MIME type is required.' }
    }

    if (typeof avatarBase64 !== 'string' || !avatarBase64.trim()) {
      return { ok: false, error: 'Avatar data is required.' }
    }

    // Check size before spawning (strict 2.9 MiB limit on encoded data)
    if (avatarBase64.length > 2_900_000) {
      return { ok: false, error: 'Avatar data is too large.' }
    }

    return callProfileBridge(session, {
      action: 'set_avatar',
      mime_type: mimeType.trim(),
      avatar_base64: avatarBase64.trim()
    })
  })

  ipcMain.handle('rex:removeProfileAvatar', () =>
    callProfileBridge(session, { action: 'remove_avatar' })
  )
}
