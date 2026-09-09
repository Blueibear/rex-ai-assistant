import { beforeEach, describe, expect, it, vi } from 'vitest'

const writeFileSync = vi.fn()
const quit = vi.fn()
const appendElectronLog = vi.fn()

vi.mock('electron', () => ({ app: { quit } }))
vi.mock('fs', () => ({ writeFileSync }))
vi.mock('../src/main/handlers/logs', () => ({ appendElectronLog }))

type SmokeApi = Record<string, (...args: unknown[]) => unknown>

/**
 * Build a fake preload surface for the first-run smoke. Every method the
 * captured renderer script touches is present. The generic
 * `getWakeWordStatus` is present but rejects, mirroring real packaged
 * behaviour where its main-process handler is only registered after the
 * authenticated runtime bootstrap.
 */
function makeFirstRunApi(calls: string[], overrides: SmokeApi = {}): SmokeApi {
  let setupDone = false
  const api: SmokeApi = {
    getSetupAudioDevices: () => Promise.resolve({ ok: true, devices: [] }),
    testSetupAudioDevice: () => Promise.resolve({ ok: true }),
    listVoices: () => Promise.resolve({ ok: true, voices: [] }),
    previewVoice: () => Promise.resolve({ ok: true }),
    previewWakeWordSample: () => Promise.resolve({ ok: true, has_sample: true }),
    getWakeWordStatus: () => {
      calls.push('getWakeWordStatus')
      return Promise.reject(new Error("No handler registered for 'rex:getWakeWordStatus'"))
    },
    listWakeWords: () => {
      calls.push('listWakeWords')
      return Promise.resolve({
        ok: true,
        wake_words: [{ id: 'alexa', name: 'Alexa', engine: 'openwakeword' }]
      })
    },
    getSetupWakeWordStatus: (wakeWordId: unknown) => {
      calls.push('getSetupWakeWordStatus:' + String(wakeWordId))
      return Promise.resolve({
        requestedBackend: 'openwakeword',
        configuredPhrase: 'alexa',
        fallbackEnabled: true,
        fallbackKeyword: 'alexa',
        assetKind: 'builtin',
        assetPath: '',
        assetExists: true,
        fallbackActive: false,
        status: 'built_in',
        statusLabel: 'Built-in wake word',
        detail: ''
      })
    },
    getSetupStatus: () => {
      calls.push('getSetupStatus')
      return Promise.resolve({ needs_setup: !setupDone })
    },
    completeSetup: (payload: unknown) => {
      const wakeWordId = (payload as Record<string, unknown>)?.wake_word_id
      calls.push('completeSetup:' + String(wakeWordId))
      setupDone = true
      return Promise.resolve({ ok: true })
    },
    getStatus: () => {
      calls.push('getStatus')
      return Promise.resolve({ status: 'ready' })
    }
  }
  return { ...api, ...overrides }
}

async function runFirstRunSmoke(
  api: SmokeApi,
  bodyText = 'Set up Account'
): Promise<Record<string, unknown>> {
  process.env.ASKREX_ARTIFACT_SMOKE_OUTPUT = 'first-run-smoke.json'
  let loaded: (() => void) | undefined
  const executeJavaScript = vi.fn((script: string) => {
    const windowStub = { rex: api, location: { hash: '' } }
    const documentStub = { body: { innerText: bodyText } }
    const runner = new Function('window', 'document', 'setTimeout', 'Date', `return ${script}`)
    return runner(windowStub, documentStub, setTimeout, Date)
  })
  const window = {
    webContents: {
      once: vi.fn((_event: string, cb: () => void) => {
        loaded = cb
      }),
      executeJavaScript
    }
  }
  const { runInstalledFirstRunSmoke } = await import('../src/main/artifactSmoke')
  expect(runInstalledFirstRunSmoke(window as never)).toBe(true)
  loaded?.()
  await vi.waitFor(() => expect(writeFileSync).toHaveBeenCalledOnce())
  return JSON.parse(writeFileSync.mock.calls[0][1] as string) as Record<string, unknown>
}

describe('installed artifact smoke', () => {
  beforeEach(() => {
    vi.resetModules()
    vi.clearAllMocks()
    delete process.env.ASKREX_ARTIFACT_SMOKE_OUTPUT
  })

  it('is disabled during normal application startup', async () => {
    const { runInstalledArtifactSmoke } = await import('../src/main/artifactSmoke')
    const window = { webContents: { once: vi.fn() } }
    expect(runInstalledArtifactSmoke(window as never)).toBe(false)
    expect(window.webContents.once).not.toHaveBeenCalled()
  })

  it('records a successful preload and bridge result then quits', async () => {
    process.env.ASKREX_ARTIFACT_SMOKE_OUTPUT = 'smoke-result.json'
    let loaded: (() => void) | undefined
    const executeJavaScript = vi.fn().mockResolvedValue({
      ok: true,
      typed_ipc: true,
      chat: 'AskRex installed artifact chat verified',
      memories_count: 0,
      openclaw_settings: true,
      openclaw_settings_read_write: true,
      settings_sections: true
    })
    const window = {
      webContents: {
        once: vi.fn((_event: string, callback: () => void) => {
          loaded = callback
        }),
        executeJavaScript
      }
    }
    const { runInstalledArtifactSmoke } = await import('../src/main/artifactSmoke')
    expect(runInstalledArtifactSmoke(window as never)).toBe(true)
    loaded?.()
    await vi.waitFor(() => expect(writeFileSync).toHaveBeenCalledOnce())
    expect(executeJavaScript).toHaveBeenCalledOnce()
    const smokeScript = executeJavaScript.mock.calls[0][0] as string
    expect(smokeScript).toContain('const waitFor = async')
    expect(smokeScript).toContain("window.location.hash = '#/settings'")
    expect(smokeScript).toContain("node.textContent?.trim() === 'Integrations'")
    expect(smokeScript).toContain('await waitFor(() =>')
    expect(smokeScript).toContain("openclawGatewayUrl")
    expect(smokeScript).toContain("Enable OpenClaw tools")
    expect(smokeScript).toContain("setSettings('integrations'")
    expect(smokeScript).toContain("['General', 'General']")
    expect(smokeScript).toContain("['Voice', 'Voice']")
    expect(smokeScript).toContain("['AI', 'AI']")
    expect(smokeScript).toContain("['Integrations', 'Integrations']")
    expect(smokeScript).toContain("['Notifications', 'Notifications']")
    expect(smokeScript).toContain("['Users', 'Users']")
    expect(smokeScript).toContain("['Audio Output', 'Audio Output']")
    expect(smokeScript).toContain("['System', 'System & Advanced']")
    expect(smokeScript).toContain("['About', 'AskRex Assistant']")
    expect(smokeScript).toContain('const waitFor = async')
    expect(smokeScript).toContain('await waitFor(() =>')
    expect(writeFileSync).toHaveBeenCalledWith(
      'smoke-result.json',
      expect.stringContaining('"typed_ipc": true'),
      'utf8'
    )
    expect(quit).toHaveBeenCalledOnce()
  })
})

describe('installed first-run smoke', () => {
  beforeEach(() => {
    vi.resetModules()
    vi.clearAllMocks()
    delete process.env.ASKREX_ARTIFACT_SMOKE_OUTPUT
  })

  it('is disabled during normal application startup', async () => {
    const { runInstalledFirstRunSmoke } = await import('../src/main/artifactSmoke')
    const window = { webContents: { once: vi.fn() } }
    expect(runInstalledFirstRunSmoke(window as never)).toBe(false)
    expect(window.webContents.once).not.toHaveBeenCalled()
  })

  it('enumerates wake words, checks readiness by the enumerated id, and persists that same id', async () => {
    const calls: string[] = []
    const result = await runFirstRunSmoke(makeFirstRunApi(calls))

    expect(result).toMatchObject({
      ok: true,
      setup_ui: true,
      preauth_ipc: true,
      setup_completed: true,
      authenticated_ipc: true,
      background_voice_enabled: false
    })

    const iList = calls.indexOf('listWakeWords')
    const iReady = calls.findIndex((c) => c.startsWith('getSetupWakeWordStatus:'))
    const iComplete = calls.findIndex((c) => c.startsWith('completeSetup:'))
    const iStatus = calls.indexOf('getStatus')
    expect(iList).toBeGreaterThanOrEqual(0)
    expect(iList).toBeLessThan(iReady)
    expect(iReady).toBeLessThan(iComplete)
    expect(iComplete).toBeLessThan(iStatus)

    // Readiness + persistence use the id returned by enumeration, never a
    // hard-coded unenumerated value.
    expect(calls).toContain('getSetupWakeWordStatus:alexa')
    expect(calls).toContain('completeSetup:alexa')
    expect(calls).not.toContain('completeSetup:hey_rex')

    // Generic pre-auth wake readiness is never invoked in this flow.
    expect(calls).not.toContain('getWakeWordStatus')
  })

  it('fails without completing setup when the wake-word inventory bridge fails', async () => {
    const calls: string[] = []
    const api = makeFirstRunApi(calls, {
      listWakeWords: () => {
        calls.push('listWakeWords')
        return Promise.resolve({ ok: false, wake_words: [], error: 'bridge unavailable' })
      }
    })
    const result = await runFirstRunSmoke(api)

    expect(result.ok).toBe(false)
    expect(result.setup_completed).toBe(false)
    expect(calls.some((c) => c.startsWith('getSetupWakeWordStatus:'))).toBe(false)
    expect(calls.some((c) => c.startsWith('completeSetup:'))).toBe(false)
  })

  it('fails without completing setup when the wake-word inventory is empty', async () => {
    const calls: string[] = []
    const api = makeFirstRunApi(calls, {
      listWakeWords: () => {
        calls.push('listWakeWords')
        return Promise.resolve({ ok: true, wake_words: [] })
      }
    })
    const result = await runFirstRunSmoke(api)

    expect(result.ok).toBe(false)
    expect(calls.some((c) => c.startsWith('getSetupWakeWordStatus:'))).toBe(false)
    expect(calls.some((c) => c.startsWith('completeSetup:'))).toBe(false)
  })

  it('fails without completing setup when inventory entries carry no usable id', async () => {
    const calls: string[] = []
    const api = makeFirstRunApi(calls, {
      listWakeWords: () => {
        calls.push('listWakeWords')
        return Promise.resolve({
          ok: true,
          wake_words: [{ name: 'Nameless', engine: 'openwakeword' }]
        })
      }
    })
    const result = await runFirstRunSmoke(api)

    expect(result.ok).toBe(false)
    expect(calls.some((c) => c.startsWith('getSetupWakeWordStatus:'))).toBe(false)
    expect(calls.some((c) => c.startsWith('completeSetup:'))).toBe(false)
  })

  it('fails without completing setup when enumerated readiness rejects', async () => {
    const calls: string[] = []
    const api = makeFirstRunApi(calls, {
      getSetupWakeWordStatus: (wakeWordId: unknown) => {
        calls.push('getSetupWakeWordStatus:' + String(wakeWordId))
        return Promise.reject(new Error('wake-word readiness bridge unavailable'))
      }
    })
    const result = await runFirstRunSmoke(api)

    expect(result.ok).toBe(false)
    expect(result.setup_completed).toBe(false)
    expect(calls).toContain('getSetupWakeWordStatus:alexa')
    expect(calls.some((c) => c.startsWith('completeSetup:'))).toBe(false)
  })
})
