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

  it('mirrors OpenRouter model and endpoint separately from OpenAI', () => {
    mockReadRexConfig.mockReturnValue({ openai: { model: 'gpt-4o' } })
    const result = mirrorToRexConfig('ai', {
      provider: 'openrouter',
      openrouterModel: 'anthropic/claude-sonnet-4',
      openrouterBaseUrl: 'https://malicious.example.test/v1'
    } as never)
    expect(result).toEqual({ ok: true })
    expect(mockWriteRexConfig).toHaveBeenCalledWith(expect.objectContaining({
      models: expect.objectContaining({ llm_provider: 'openrouter' }),
      openai: { model: 'gpt-4o' },
      openrouter: {
        model: 'anthropic/claude-sonnet-4',
        base_url: 'https://openrouter.ai/api/v1'
      }
    }))
    const written = mockWriteRexConfig.mock.calls[0][0] as Record<string, unknown>
    expect((written.llm as Record<string, unknown>).provider).toBeUndefined()
  })

  it('rejects an empty OpenRouter model instead of persisting an unusable provider', () => {
    const result = mirrorToRexConfig('ai', {
      provider: 'openrouter', openrouterModel: '', openrouterBaseUrl: ''
    } as never)
    expect(result).toEqual({ ok: false, error: 'OpenRouter model is required' })
    expect(mockWriteRexConfig).not.toHaveBeenCalled()
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
