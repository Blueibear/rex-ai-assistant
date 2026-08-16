import { beforeEach, describe, expect, it, vi } from 'vitest'

const {
  mockReadGuiSettings,
  mockReadRexConfigStrict,
  mockWriteGuiSettings,
  mockWriteRexConfig
} = vi.hoisted(() => ({
  mockReadGuiSettings: vi.fn(),
  mockReadRexConfigStrict: vi.fn(),
  mockWriteGuiSettings: vi.fn(),
  mockWriteRexConfig: vi.fn()
}))

vi.mock('../src/main/configStore', () => ({
  readGuiSettings: mockReadGuiSettings,
  readRexConfigStrict: mockReadRexConfigStrict,
  writeGuiSettings: mockWriteGuiSettings,
  writeRexConfig: mockWriteRexConfig
}))

import {
  migrateLegacyAutonomySettings,
  resolveAutonomyMode,
  stripLegacyAutonomyMode
} from '../src/main/autonomySettings'

describe('autonomy settings consolidation (US-073)', () => {
  beforeEach(() => {
    mockReadGuiSettings.mockReset().mockReturnValue({})
    mockReadRexConfigStrict.mockReset().mockReturnValue({})
    mockWriteGuiSettings.mockReset()
    mockWriteRexConfig.mockReset()
  })

  it('uses the canonical runtime autonomy mode when it is valid', () => {
    expect(resolveAutonomyMode('supervised', 'manual', 'full-auto')).toBe('supervised')
  })

  it('chooses the more restrictive legacy mode when duplicate GUI values conflict', () => {
    expect(resolveAutonomyMode(undefined, 'full-auto', 'manual')).toBe('manual')
    expect(resolveAutonomyMode(undefined, 'full-auto', 'supervised')).toBe('supervised')
  })

  it('defaults to manual when no valid autonomy value exists', () => {
    expect(resolveAutonomyMode('invalid', 'also-invalid', undefined)).toBe('manual')
  })

  it('keeps AI settings readable and defers writes when runtime config is malformed', () => {
    mockReadGuiSettings.mockReturnValue({
      ai: { autonomyMode: 'full-auto', model: 'gpt-4o' },
      system: { autonomyMode: 'manual' }
    })
    mockReadRexConfigStrict.mockImplementation(() => { throw new Error('malformed json') })

    const migrated = migrateLegacyAutonomySettings()

    expect(migrated.ai).toEqual({ autonomyMode: 'manual', model: 'gpt-4o' })
    expect(migrated.system).toEqual({ autonomyMode: 'manual' })
    expect(mockWriteRexConfig).not.toHaveBeenCalled()
    expect(mockWriteGuiSettings).not.toHaveBeenCalled()
  })

  it('does not create runtime config just to persist an unsaved default', () => {
    const migrated = migrateLegacyAutonomySettings()

    expect(migrated).toEqual({})
    expect(mockWriteRexConfig).not.toHaveBeenCalled()
    expect(mockWriteGuiSettings).not.toHaveBeenCalled()
  })

  it('migrates duplicate GUI values into the canonical runtime field and removes both copies', () => {
    mockReadGuiSettings.mockReturnValue({
      ai: { autonomyMode: 'full-auto', model: 'gpt-4o' },
      system: { autonomyMode: 'manual', toolTimeoutSeconds: 10 }
    })
    mockReadRexConfigStrict.mockReturnValue({ models: { llm_provider: 'openai' } })

    const migrated = migrateLegacyAutonomySettings()

    expect(mockWriteRexConfig).toHaveBeenCalledWith({
      models: { llm_provider: 'openai', autonomy_mode: 'manual' }
    })
    expect(mockWriteGuiSettings).toHaveBeenCalledTimes(1)
    expect(migrated.ai).toEqual({ model: 'gpt-4o' })
    expect(migrated.system).toEqual({ toolTimeoutSeconds: 10 })
  })

  it('preserves an existing canonical runtime value while removing stale GUI duplicates', () => {
    mockReadGuiSettings.mockReturnValue({
      ai: { autonomyMode: 'full-auto' },
      system: { autonomyMode: 'manual' }
    })
    mockReadRexConfigStrict.mockReturnValue({ models: { autonomy_mode: 'supervised' } })

    migrateLegacyAutonomySettings()

    expect(mockWriteRexConfig).not.toHaveBeenCalled()
    expect(mockWriteGuiSettings).toHaveBeenCalledWith({ ai: {}, system: {} })
  })

  it('does not retain autonomyMode in the GUI AI settings store', () => {
    expect(stripLegacyAutonomyMode({ autonomyMode: 'full-auto', model: 'gpt-4o' })).toEqual({
      model: 'gpt-4o'
    })
  })
})
