import { ipcMain } from 'electron'
import { spawnSync } from 'child_process'
import { bridgeSpawnOptions, resolveBridgePath, resolvePythonCommand } from '../bridgeResolver'
import { privateSessionPayload, type ElectronSessionIdentity } from '../sessionIdentity'
import type { ProfileOperationResponse, UserProfile } from '../../types/ipc'

const MAX_ENCODED_AVATAR = 2_900_000
const MAX_PREFERENCES_BYTES = 32 * 1024

function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === 'object' && !Array.isArray(value)
}

function isStringArray(value: unknown): value is string[] {
  return Array.isArray(value) && value.every((item) => typeof item === 'string')
}

function isNullableString(value: unknown): value is string | null {
  return value === null || typeof value === 'string'
}

function isProfile(value: unknown, expectedUserId: string): value is UserProfile {
  if (!isRecord(value) || value.user_id !== expectedUserId) return false
  return typeof value.name === 'string'
    && typeof value.initials === 'string'
    && typeof value.role === 'string'
    && isStringArray(value.permissions)
    && isRecord(value.preferences)
    && typeof value.voice_enrolled === 'boolean'
    && isNullableString(value.voice_model_id)
    && typeof value.voice_sample_count === 'number'
    && isNullableString(value.voice_updated_at)
    && typeof value.avatar_present === 'boolean'
    && isNullableString(value.avatar_mime_type)
    && isNullableString(value.avatar_data)
    && isRecord(value.scope_labels)
}

function isJsonValue(value: unknown, depth = 0, seen = new WeakSet<object>()): boolean {
  if (depth > 4) return false
  if (value === null || typeof value === 'string' || typeof value === 'boolean') return true
  if (typeof value === 'number') return Number.isFinite(value)
  if (typeof value !== 'object') return false
  if (seen.has(value)) return false
  seen.add(value)
  if (Array.isArray(value)) {
    return value.every((item) => isJsonValue(item, depth + 1, seen))
  }
  const prototype = Object.getPrototypeOf(value)
  if (prototype !== Object.prototype && prototype !== null) return false
  return Object.values(value).every((item) => isJsonValue(item, depth + 1, seen))
}

function isStrictBase64(value: string): boolean {
  if (!value || value.length % 4 !== 0) return false
  return /^(?:[A-Za-z0-9+/]{4})*(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?$/.test(value)
}

function callProfileBridge(
  session: ElectronSessionIdentity,
  payload: Record<string, unknown>
): ProfileOperationResponse {
  try {
    const result = spawnSync(resolvePythonCommand(), [resolveBridgePath('rex_profile_bridge.py')], {
      ...bridgeSpawnOptions(),
      input: JSON.stringify(privateSessionPayload(session, payload)),
      encoding: 'utf8',
      timeout: 15_000,
      windowsHide: true
    })
    if (result.status !== 0) {
      return { ok: false, error: 'Profile service could not complete the request.' }
    }
    const parsed: unknown = JSON.parse((result.stdout || '').trim())
    if (!isRecord(parsed) || parsed.ok !== true || !isProfile(parsed.profile, session.userId)) {
      return { ok: false, error: 'Profile service returned an invalid response.' }
    }
    return { ok: true, profile: parsed.profile }
  } catch {
    return { ok: false, error: 'Profile service is unavailable.' }
  }
}

export function registerProfileHandlers(session: ElectronSessionIdentity): void {
  ipcMain.handle('rex:getProfile', () => callProfileBridge(session, { action: 'get' }))

  ipcMain.handle('rex:updateProfilePreferences', (_event, preferences: unknown) => {
    if (!isRecord(preferences) || !isJsonValue(preferences)) {
      return { ok: false, error: 'Preferences must be a finite JSON object.' }
    }
    const serialized = JSON.stringify(preferences)
    if (Buffer.byteLength(serialized, 'utf8') > MAX_PREFERENCES_BYTES) {
      return { ok: false, error: 'Preferences are too large.' }
    }
    return callProfileBridge(session, { action: 'update_preferences', preferences })
  })

  ipcMain.handle('rex:setProfileAvatar', (_event, mimeType: unknown, avatarBase64: unknown) => {
    if (mimeType !== 'image/jpeg' && mimeType !== 'image/png') {
      return { ok: false, error: 'Avatar must be a JPEG or PNG image.' }
    }
    if (typeof avatarBase64 !== 'string') {
      return { ok: false, error: 'Avatar data is required.' }
    }
    const encoded = avatarBase64.trim()
    if (encoded.length > MAX_ENCODED_AVATAR) {
      return { ok: false, error: 'Avatar data is too large.' }
    }
    if (!isStrictBase64(encoded)) {
      return { ok: false, error: 'Avatar data is not valid base64.' }
    }
    return callProfileBridge(session, {
      action: 'set_avatar',
      mime_type: mimeType,
      avatar_base64: encoded
    })
  })

  ipcMain.handle('rex:removeProfileAvatar', () =>
    callProfileBridge(session, { action: 'remove_avatar' })
  )
}
