import { beforeEach, describe, expect, it, vi } from 'vitest'

const writeFileSync = vi.fn()
const quit = vi.fn()
const appendElectronLog = vi.fn()

vi.mock('electron', () => ({ app: { quit } }))
vi.mock('fs', () => ({ writeFileSync }))
vi.mock('../src/main/handlers/logs', () => ({ appendElectronLog }))

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
      openclaw_settings_read_write: true
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
    expect(smokeScript).toContain("#/settings?section=integrations")
    expect(smokeScript).toContain("openclawGatewayUrl")
    expect(smokeScript).toContain("Enable OpenClaw tools")
    expect(smokeScript).toContain("setSettings('integrations'")
    expect(writeFileSync).toHaveBeenCalledWith(
      'smoke-result.json',
      expect.stringContaining('"typed_ipc": true'),
      'utf8'
    )
    expect(quit).toHaveBeenCalledOnce()
  })
})
