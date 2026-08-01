/**
 * US-018: Vitest tests for bridgeResolver.ts
 *
 * Asserts that resolveBridgePath() returns the correct absolute path in both
 * dev mode (app.isPackaged === false) and packaged mode (app.isPackaged === true).
 */

import { join } from 'path'
import { beforeEach, describe, expect, it, vi } from 'vitest'

// vi.hoisted lifts this block before all imports so mockApp is available
// inside the vi.mock factory below.
const { mockApp } = vi.hoisted(() => ({
  mockApp: {
    isPackaged: false,
    getAppPath: vi.fn().mockReturnValue('/fake/app'),
    getPath: vi.fn().mockReturnValue('/fake/user-data'),
  },
}))

const { mockExistsSync } = vi.hoisted(() => ({
  mockExistsSync: vi.fn<(path: string) => boolean>().mockReturnValue(false),
}))

// Replace the 'electron' module with our controllable mock.
vi.mock('electron', () => ({ app: mockApp }))
vi.mock('fs', () => ({ existsSync: mockExistsSync }))

// Import after mocks are registered.
import {
  bridgeSpawnOptions,
  resolveBridgePath,
  resolvePythonCommand,
  resolveRuntimeRoot
} from '../src/main/bridgeResolver'

describe('resolveBridgePath', () => {
  beforeEach(() => {
    mockApp.isPackaged = false
    mockApp.getAppPath.mockReturnValue('/fake/app')
    mockApp.getPath.mockReturnValue('/fake/user-data')
    // process.resourcesPath is an Electron global absent in plain Node.
    Object.defineProperty(process, 'resourcesPath', {
      value: '/fake/resources',
      writable: true,
      configurable: true,
    })
    mockExistsSync.mockReset().mockReturnValue(false)
  })

  it('dev mode: resolves to <appPath>/../bridge/<script>', () => {
    const result = resolveBridgePath('rex_tasks_bridge.py')
    expect(result).toBe(join('/fake/app', '..', 'bridge', 'rex_tasks_bridge.py'))
  })

  it('packaged mode: resolves to <resourcesPath>/bridge/<script>', () => {
    mockApp.isPackaged = true
    const result = resolveBridgePath('rex_tasks_bridge.py')
    expect(result).toBe(join('/fake/resources', 'bridge', 'rex_tasks_bridge.py'))
  })

  it('packaged mode: resolves only the managed Python runtime', () => {
    mockApp.isPackaged = true
    mockExistsSync.mockImplementation((path) => path === join('/fake/resources', 'python', 'python.exe'))
    expect(resolvePythonCommand()).toBe(join('/fake/resources', 'python', 'python.exe'))
  })

  it('packaged mode: fails closed when the managed runtime is missing', () => {
    mockApp.isPackaged = true
    expect(() => resolvePythonCommand()).toThrow('managed Python runtime is missing')
  })

  it('dev mode: prefers the repository virtual environment', () => {
    const venvPython = join('/fake/app', '..', '.venv', 'Scripts', 'python.exe')
    mockExistsSync.mockImplementation((path) => path === venvPython)
    expect(resolvePythonCommand()).toBe(venvPython)
  })
})


describe('runtime path isolation', () => {
  beforeEach(() => {
    mockApp.isPackaged = false
    mockApp.getAppPath.mockReturnValue('/fake/app')
    mockApp.getPath.mockReturnValue('/fake/user-data')
  })

  it('uses the repository root for development bridges', () => {
    expect(resolveRuntimeRoot()).toBe(join('/fake/app', '..'))
    const options = bridgeSpawnOptions()
    expect(options.cwd).toBe(join('/fake/app', '..'))
    expect(options.env.ASKREX_CONFIG_PATH).toBe(
      join('/fake/app', '..', 'config', 'rex_config.json')
    )
    expect(options.env.ASKREX_PROFILES_DIR).toBe(join('/fake/app', '..', 'profiles'))
  })

  it('uses Electron userData for packaged writable state', () => {
    mockApp.isPackaged = true
    expect(resolveRuntimeRoot()).toBe('/fake/user-data')
    const options = bridgeSpawnOptions()
    expect(options.cwd).toBe('/fake/user-data')
    expect(options.env.REX_DATA_DIR).toBe(join('/fake/user-data', 'data'))
    expect(options.env.ASKREX_MEMORY_DIR).toBe(join('/fake/user-data', 'Memory'))
  })
})
