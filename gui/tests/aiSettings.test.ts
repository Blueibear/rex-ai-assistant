import { beforeEach, describe, expect, it, vi } from 'vitest'

const { mockReadRexConfig } = vi.hoisted(() => ({ mockReadRexConfig: vi.fn() }))
vi.mock('../src/main/configStore', () => ({ readRexConfig: mockReadRexConfig }))

import {
  OPENAI_DEFAULT_MODEL,
  OPENROUTER_DEFAULT_BASE_URL,
  OPENROUTER_DEFAULT_MODEL,
  buildAiSettings,
  normalizeGuiAiProvider,
  toRuntimeAiProvider
} from '../src/main/aiSettings'

describe('AI provider settings', () => {
  beforeEach(() => mockReadRexConfig.mockReset().mockReturnValue({}))

  it('uses a provider-compatible OpenAI default instead of a Claude model', () => {
    const settings = buildAiSettings({ provider: 'openai' })
    expect(settings.provider).toBe('openai')
    expect(settings.model).toBe(OPENAI_DEFAULT_MODEL)
  })

  it('loads OpenRouter model and endpoint independently from OpenAI', () => {
    mockReadRexConfig.mockReturnValue({
      models: { llm_provider: 'openrouter' },
      openai: { model: 'gpt-openai-only' },
      openrouter: {
        model: 'anthropic/claude-sonnet-4',
        base_url: 'https://router.example.test/v1'
      }
    })
    const settings = buildAiSettings({})
    expect(settings.provider).toBe('openrouter')
    expect(settings.model).toBe('gpt-openai-only')
    expect(settings.openrouterModel).toBe('anthropic/claude-sonnet-4')
    expect(settings.openrouterBaseUrl).toBe('https://router.example.test/v1')
  })

  it('provides safe OpenRouter defaults and preserves runtime mapping', () => {
    expect(normalizeGuiAiProvider('openrouter')).toBe('openrouter')
    expect(toRuntimeAiProvider('openrouter')).toBe('openrouter')
    const settings = buildAiSettings({ provider: 'openrouter' })
    expect(settings.openrouterModel).toBe(OPENROUTER_DEFAULT_MODEL)
    expect(settings.openrouterBaseUrl).toBe(OPENROUTER_DEFAULT_BASE_URL)
  })
})
