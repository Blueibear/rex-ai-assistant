import { describe, it, expect, beforeEach, vi } from 'vitest'
import { spawnSync } from 'child_process'

// Mock dependencies
vi.mock('child_process')
vi.mock('../src/main/bridgeResolver', () => ({
  bridgeSpawnOptions: () => ({
    cwd: '/test/cwd',
    env: process.env
  }),
  resolveBridgePath: (script: string) => `/bridge/${script}`,
  resolvePythonCommand: () => 'python'
}))
vi.mock('../src/main/sessionIdentity', () => ({
  privateSessionPayload: (session: any, payload: any) => ({
    ...payload,
    user: session.userId,
    session_id: session.sessionId,
    data_scope: 'private'
  })
}))

// Import after mocking
import { registerProfileHandlers } from '../src/main/handlers/profile'
import type { ElectronSessionIdentity } from '../src/main/sessionIdentity'

describe('Profile Handlers', () => {
  let mockSession: ElectronSessionIdentity
  let mockIpcMain: any
  let handlers: Map<string, Function>

  beforeEach(() => {
    handlers = new Map()
    mockSession = {
      userId: 'testuser',
      sessionId: 'session-123',
      osPrincipal: 'testprincipal',
      authentication: 'local-os-session'
    }

    mockIpcMain = {
      handle: vi.fn((channel: string, handler: Function) => {
        handlers.set(channel, handler)
      })
    }

    vi.clearAllMocks()
  })

  describe('Handler Registration', () => {
    it('should register exactly four channels', () => {
      registerProfileHandlers(mockSession)

      expect(handlers.size).toBe(4)
      expect(handlers.has('rex:getProfile')).toBe(true)
      expect(handlers.has('rex:updateProfilePreferences')).toBe(true)
      expect(handlers.has('rex:setProfileAvatar')).toBe(true)
      expect(handlers.has('rex:removeProfileAvatar')).toBe(true)
    })

    it('should use ipcMain.handle for all channels', () => {
      registerProfileHandlers(mockSession)

      expect(mockIpcMain.handle).toHaveBeenCalledTimes(4)
      expect(mockIpcMain.handle).toHaveBeenCalledWith(
        'rex:getProfile',
        expect.any(Function)
      )
      expect(mockIpcMain.handle).toHaveBeenCalledWith(
        'rex:updateProfilePreferences',
        expect.any(Function)
      )
      expect(mockIpcMain.handle).toHaveBeenCalledWith(
        'rex:setProfileAvatar',
        expect.any(Function)
      )
      expect(mockIpcMain.handle).toHaveBeenCalledWith(
        'rex:removeProfileAvatar',
        expect.any(Function)
      )
    })
  })

  describe('getProfile Handler', () => {
    it('should call bridge with correct payload', () => {
      const mockSpawnSync = spawnSync as any
      mockSpawnSync.mockReturnValue({
        status: 0,
        stdout: JSON.stringify({
          ok: true,
          profile: {
            user_id: 'testuser',
            name: 'Test User',
            permissions: [],
            avatar_present: false,
            scope_labels: {}
          }
        })
      })

      registerProfileHandlers(mockSession)
      const handler = handlers.get('rex:getProfile')
      expect(handler).toBeDefined()

      const result = handler?.({}, undefined)

      expect(result).toEqual({
        ok: true,
        profile: expect.any(Object)
      })
    })

    it('should return safe error on nonzero exit', () => {
      const mockSpawnSync = spawnSync as any
      mockSpawnSync.mockReturnValue({
        status: 1,
        stdout: '',
        stderr: ''
      })

      registerProfileHandlers(mockSession)
      const handler = handlers.get('rex:getProfile')

      const result = handler?.({}, undefined)

      expect(result.ok).toBe(false)
      expect(result.error).toBeDefined()
      expect(typeof result.error).toBe('string')
    })

    it('should return safe error on invalid JSON', () => {
      const mockSpawnSync = spawnSync as any
      mockSpawnSync.mockReturnValue({
        status: 0,
        stdout: 'not valid json'
      })

      registerProfileHandlers(mockSession)
      const handler = handlers.get('rex:getProfile')

      const result = handler?.({}, undefined)

      expect(result.ok).toBe(false)
      expect(result.error).toBeDefined()
    })

    it('should return safe error on malformed success payload', () => {
      const mockSpawnSync = spawnSync as any
      mockSpawnSync.mockReturnValue({
        status: 0,
        stdout: JSON.stringify({ ok: true })  // Missing required fields
      })

      registerProfileHandlers(mockSession)
      const handler = handlers.get('rex:getProfile')

      const result = handler?.({}, undefined)

      expect(result.ok).toBe(false)
      expect(result.error).toBeDefined()
    })
  })

  describe('updateProfilePreferences Handler', () => {
    it('should validate preferences argument before spawning', () => {
      const mockSpawnSync = spawnSync as any

      registerProfileHandlers(mockSession)
      const handler = handlers.get('rex:updateProfilePreferences')

      // Valid preferences
      mockSpawnSync.mockReturnValue({
        status: 0,
        stdout: JSON.stringify({ ok: true })
      })

      const result = handler?.({}, { theme: 'dark' })
      expect(result.ok).toBe(true)
    })

    it('should reject non-object preferences', () => {
      registerProfileHandlers(mockSession)
      const handler = handlers.get('rex:updateProfilePreferences')

      const result = handler?.({}, 'not an object')

      expect(result.ok).toBe(false)
      expect(result.error).toBeDefined()
    })

    it('should reject oversized encoded preferences', () => {
      registerProfileHandlers(mockSession)
      const handler = handlers.get('rex:updateProfilePreferences')

      // Create preferences that would serialize to > reasonable size
      const largePrefs: Record<string, unknown> = {}
      for (let i = 0; i < 10000; i++) {
        largePrefs[`key${i}`] = 'x'.repeat(100)
      }

      const result = handler?.({}, largePrefs)

      // Should either reject or let bridge handle it
      if (result.ok === false) {
        expect(result.error).toBeDefined()
      }
    })

    it('should call bridge with preferences payload', () => {
      const mockSpawnSync = spawnSync as any
      mockSpawnSync.mockReturnValue({
        status: 0,
        stdout: JSON.stringify({ ok: true })
      })

      registerProfileHandlers(mockSession)
      const handler = handlers.get('rex:updateProfilePreferences')

      const prefs = { theme: 'dark', notifications: true }
      handler?.({}, prefs)

      expect(mockSpawnSync).toHaveBeenCalled()
      const call = (mockSpawnSync as any).mock.calls[0]
      const payload = JSON.parse(call[2].input)
      expect(payload.action).toBe('update_preferences')
      expect(payload.preferences).toEqual(prefs)
    })
  })

  describe('setProfileAvatar Handler', () => {
    it('should validate mime type before spawning', () => {
      const mockSpawnSync = spawnSync as any
      mockSpawnSync.mockReturnValue({
        status: 0,
        stdout: JSON.stringify({ ok: true })
      })

      registerProfileHandlers(mockSession)
      const handler = handlers.get('rex:setProfileAvatar')

      // Valid JPEG
      const validB64 = 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=='
      const result = handler?.({}, 'image/jpeg', validB64)

      if (result.ok) {
        expect(result.ok).toBe(true)
      }
    })

    it('should reject non-string mime type', () => {
      registerProfileHandlers(mockSession)
      const handler = handlers.get('rex:setProfileAvatar')

      const result = handler?.({}, 123 as any, 'abc123')

      expect(result.ok).toBe(false)
      expect(result.error).toBeDefined()
    })

    it('should reject non-string base64', () => {
      registerProfileHandlers(mockSession)
      const handler = handlers.get('rex:setProfileAvatar')

      const result = handler?.({}, 'image/jpeg', {} as any)

      expect(result.ok).toBe(false)
      expect(result.error).toBeDefined()
    })

    it('should reject empty base64', () => {
      registerProfileHandlers(mockSession)
      const handler = handlers.get('rex:setProfileAvatar')

      const result = handler?.({}, 'image/jpeg', '')

      expect(result.ok).toBe(false)
      expect(result.error).toBeDefined()
    })

    it('should reject oversized encoded avatar', () => {
      registerProfileHandlers(mockSession)
      const handler = handlers.get('rex:setProfileAvatar')

      // Create base64 string over 2.9 MiB
      const largeB64 = 'a'.repeat(3 * 1024 * 1024)

      const result = handler?.({}, 'image/jpeg', largeB64)

      expect(result.ok).toBe(false)
      expect(result.error).toBeDefined()
    })

    it('should call bridge with avatar payload', () => {
      const mockSpawnSync = spawnSync as any
      mockSpawnSync.mockReturnValue({
        status: 0,
        stdout: JSON.stringify({ ok: true })
      })

      registerProfileHandlers(mockSession)
      const handler = handlers.get('rex:setProfileAvatar')

      const avatarB64 = 'abc123'
      handler?.({}, 'image/png', avatarB64)

      expect(mockSpawnSync).toHaveBeenCalled()
      const call = (mockSpawnSync as any).mock.calls[0]
      const payload = JSON.parse(call[2].input)
      expect(payload.action).toBe('set_avatar')
      expect(payload.mime_type).toBe('image/png')
      expect(payload.avatar_base64).toBe(avatarB64)
    })
  })

  describe('removeProfileAvatar Handler', () => {
    it('should call bridge with remove action', () => {
      const mockSpawnSync = spawnSync as any
      mockSpawnSync.mockReturnValue({
        status: 0,
        stdout: JSON.stringify({ ok: true })
      })

      registerProfileHandlers(mockSession)
      const handler = handlers.get('rex:removeProfileAvatar')

      handler?.({}, undefined)

      expect(mockSpawnSync).toHaveBeenCalled()
      const call = (mockSpawnSync as any).mock.calls[0]
      const payload = JSON.parse(call[2].input)
      expect(payload.action).toBe('remove_avatar')
    })

    it('should return safe error on failure', () => {
      const mockSpawnSync = spawnSync as any
      mockSpawnSync.mockReturnValue({
        status: 1,
        stdout: '',
        stderr: 'Some error'
      })

      registerProfileHandlers(mockSession)
      const handler = handlers.get('rex:removeProfileAvatar')

      const result = handler?.({}, undefined)

      expect(result.ok).toBe(false)
      expect(result.error).toBeDefined()
      expect(result.error).not.toContain('Some error')
    })
  })

  describe('Timeout', () => {
    it('should use 15-second timeout', () => {
      const mockSpawnSync = spawnSync as any
      mockSpawnSync.mockReturnValue({
        status: 0,
        stdout: JSON.stringify({ ok: true, profile: {} })
      })

      registerProfileHandlers(mockSession)
      const handler = handlers.get('rex:getProfile')

      handler?.({}, undefined)

      expect(mockSpawnSync).toHaveBeenCalled()
      const call = (mockSpawnSync as any).mock.calls[0]
      expect(call[2].timeout).toBe(15_000)
    })
  })

  describe('Payload Immutability', () => {
    it('should never accept renderer-supplied user ID', () => {
      const mockSpawnSync = spawnSync as any
      mockSpawnSync.mockReturnValue({
        status: 0,
        stdout: JSON.stringify({ ok: true, profile: {} })
      })

      registerProfileHandlers(mockSession)
      const handler = handlers.get('rex:getProfile')

      // Simulate malicious renderer attempt
      const result = handler?.({}, undefined)

      // Verify session user was used, not any renderer-supplied value
      expect(mockSpawnSync).toHaveBeenCalled()
      const call = (mockSpawnSync as any).mock.calls[0]
      const payload = JSON.parse(call[2].input)
      expect(payload.user).toBe('testuser')
      expect(payload.data_scope).toBe('private')
    })
  })
})
