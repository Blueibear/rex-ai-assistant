import { beforeEach, describe, expect, it, vi } from 'vitest'

const { mockApp } = vi.hoisted(() => ({
  mockApp: {
    isPackaged: false,
    getAppPath: vi.fn().mockReturnValue('/fake/app'),
    getPath: vi.fn().mockReturnValue('/fake/user-data')
  }
}))

vi.mock('electron', () => ({ app: mockApp }))

const { mockReadRexConfig, mockWriteRexConfig } = vi.hoisted(() => ({
  mockReadRexConfig: vi.fn(),
  mockWriteRexConfig: vi.fn()
}))

vi.mock('../src/main/configStore', () => ({
  readRexConfig: mockReadRexConfig,
  writeRexConfig: mockWriteRexConfig
}))

import { mirrorToRexConfig } from '../src/main/settingsMirror'

describe('mirrorToRexConfig truthful failures (S4)', () => {
  beforeEach(() => {
    mockReadRexConfig.mockReset().mockReturnValue({})
    mockWriteRexConfig.mockReset()
  })

  it('returns {ok:true} on a successful mirror', () => {
    const result = mirrorToRexConfig('ai', { temperature: 0.5 } as never)
    expect(result).toEqual({ ok: true })
    expect(mockWriteRexConfig).toHaveBeenCalledTimes(1)
  })

  it('returns {ok:false, error} instead of swallowing a write failure', () => {
    mockWriteRexConfig.mockImplementation(() => {
      throw new Error('disk full')
    })
    const result = mirrorToRexConfig('ai', { temperature: 0.5 } as never)
    expect(result.ok).toBe(false)
    expect(result.error).toContain('disk full')
  })

  it('returns {ok:false, error} when reading the existing config throws', () => {
    mockReadRexConfig.mockImplementation(() => {
      throw new Error('corrupt json')
    })
    const result = mirrorToRexConfig('ai', { temperature: 0.5 } as never)
    expect(result.ok).toBe(false)
    expect(result.error).toContain('corrupt json')
  })

  it('returns {ok:true} for a section with no mirror mapping (no-op)', () => {
    const result = mirrorToRexConfig('unmapped-section', {} as never)
    expect(result).toEqual({ ok: true })
  })
})
