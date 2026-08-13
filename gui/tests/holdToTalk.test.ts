import { beforeEach, describe, expect, it, vi } from 'vitest'
import { readFileSync } from 'fs'
import { join } from 'path'
import { voiceProvider } from '../src/types/voiceProvider'

describe('Hold-to-Talk production path', () => {
  it('normalizes configured UI engines to packaged providers', () => {
    expect(voiceProvider('system')).toBe('pyttsx3')
    expect(voiceProvider('openai')).toBe('edge-tts')
    expect(voiceProvider('xtts')).toBe('xtts')
  })

  it('keeps cancellation, playback, output routing, replay, and timing wired', () => {
    const source = readFileSync(join(__dirname, '../src/pages/VoicePage.tsx'), 'utf8')
    for (const contract of [
      'AbortController',
      'speakerDeviceId',
      'setSinkId',
      'synthesizeSpeech',
      'Cancel response',
      'Replay',
      'logVoiceTiming',
      'devicechange',
    ]) {
      expect(source).toContain(contract)
    }
    const chatHandler = readFileSync(join(__dirname, '../src/main/handlers/chat.ts'), 'utf8')
    const preload = readFileSync(join(__dirname, '../src/preload/index.ts'), 'utf8')
    expect(chatHandler).toContain("ipcMain.handle('rex:cancelChatStream'")
    expect(preload).toContain("ipcRenderer.invoke('rex:cancelChatStream'")
    expect(chatHandler).toContain("obj.type === 'status'")
    expect(chatHandler).toContain("'rex:chatStatus'")
    const chatPage = readFileSync(join(__dirname, '../src/pages/ChatPage.tsx'), 'utf8')
    expect(chatPage).toContain("status: 'model_failure'")
  })
})

// Regression coverage for the real-hardware-acceptance defect (issue #299):
// every Hold-to-Talk turn crashed with "signal?.addEventListener is not a
// function" because AbortSignal instances don't survive Electron's
// contextBridge isolation boundary with their EventTarget prototype methods
// intact. Unlike the string-matching test above, this actually invokes the
// real preload-exposed `sendChatStream` (captured from a mocked
// `contextBridge.exposeInMainWorld`) so it exercises the real code path
// rather than just checking that certain identifiers appear in the source.
type CancelHandle = {
  isAborted: () => boolean
  onAbort: (cb: () => void) => void
  offAbort: (cb: () => void) => void
}
type TurnStatusUpdate = {
  turnId: string
  sequence: number
  status: string
  terminal: boolean
}
type ExposedRexAPI = {
  sendChatStream: (
    message: string,
    onToken: (token: string) => void,
    cancel?: unknown,
    onStatus?: (status: string) => void
  ) => Promise<void>
  onTurnStatus: (cb: (update: TurnStatusUpdate) => void) => () => void
}

const invoke = vi.fn()
const on = vi.fn()
const removeListener = vi.fn()
const listeners: Record<string, ((...args: unknown[]) => void) | undefined> = {}
let exposed: Record<string, unknown> = {}

vi.mock('electron', () => ({
  contextBridge: {
    exposeInMainWorld: (name: string, api: unknown) => {
      exposed[name] = api
    },
  },
  ipcRenderer: { invoke, on, removeListener },
}))
vi.mock('@electron-toolkit/preload', () => ({ electronAPI: {} }))

function buildCancelHandle(controller: AbortController): CancelHandle {
  return {
    isAborted: () => controller.signal.aborted,
    onAbort: (cb) => controller.signal.addEventListener('abort', cb, { once: true }),
    offAbort: (cb) => controller.signal.removeEventListener('abort', cb),
  }
}

describe('sendChatStream cancel handle (contextBridge-safe)', () => {
  beforeEach(() => {
    vi.resetModules()
    vi.clearAllMocks()
    exposed = {}
    for (const key of Object.keys(listeners)) delete listeners[key]
    // Preload only reaches the contextBridge.exposeInMainWorld branch when
    // contextIsolated is true (matches real packaged/dev Electron).
    ;(process as unknown as { contextIsolated: boolean }).contextIsolated = true
    on.mockImplementation((event: string, cb: (...args: unknown[]) => void) => {
      listeners[event] = cb
    })
    invoke.mockImplementation((channel: string, payload?: { streamId: string }) => {
      if (channel === 'rex:startChatStream' && payload) {
        queueMicrotask(() => listeners['rex:chatDone']?.(null, { streamId: payload.streamId }))
      }
      return Promise.resolve()
    })
  })

  it('resolves normally with a real, renderer-built cancel handle (proves the fix works end-to-end)', async () => {
    await import('../src/preload/index')
    const { sendChatStream } = exposed.rex as ExposedRexAPI
    const controller = new AbortController()
    await expect(
      sendChatStream('hello', () => {}, buildCancelHandle(controller))
    ).resolves.toBeUndefined()
  })

  it('forwards model failure status without turning it into a transport error', async () => {
    await import('../src/preload/index')
    const { sendChatStream } = exposed.rex as ExposedRexAPI
    const statuses: string[] = []
    invoke.mockImplementation((channel: string, payload?: { streamId: string }) => {
      if (channel === 'rex:startChatStream' && payload) {
        queueMicrotask(() => {
          listeners['rex:chatStatus']?.(null, { streamId: payload.streamId, status: 'model_failure' })
          listeners['rex:chatDone']?.(null, { streamId: payload.streamId })
        })
      }
      return Promise.resolve()
    })

    const promise = sendChatStream('hello', () => {}, undefined, (status) => statuses.push(status))
    const statusHandler = listeners['rex:chatStatus']
    expect(statusHandler).toEqual(expect.any(Function))
    await promise

    expect(statuses).toEqual(['model_failure'])
    expect(removeListener).toHaveBeenCalledWith('rex:chatStatus', statusHandler)
  })

  it('normalizes canonical turn status and removes the exact registered listener', async () => {
    await import('../src/preload/index')
    const { onTurnStatus } = exposed.rex as ExposedRexAPI
    const updates: TurnStatusUpdate[] = []

    const cleanup = onTurnStatus((update) => updates.push(update))
    const handler = listeners['rex:turnStatus']
    expect(handler).toEqual(expect.any(Function))

    handler?.(null, {
      turn_id: 'turn-1',
      sequence: 3,
      status: 'verifying',
      terminal: false,
    })

    expect(updates).toEqual([
      { turnId: 'turn-1', sequence: 3, status: 'verifying', terminal: false },
    ])
    cleanup()
    expect(removeListener).toHaveBeenCalledWith('rex:turnStatus', handler)
  })

  it('does not throw when the cancel argument has lost its methods crossing the bridge (regression for the original crash)', async () => {
    await import('../src/preload/index')
    const { sendChatStream } = exposed.rex as ExposedRexAPI
    // This is exactly what an AbortSignal degrades to after crossing
    // Electron's contextBridge: plain data survives (`aborted`), the
    // EventTarget prototype methods do not. Before the fix, the preload code
    // called `signal.addEventListener(...)` unconditionally and threw
    // "signal?.addEventListener is not a function" here.
    const bridgeStrippedSignal = { aborted: false }
    await expect(
      sendChatStream('hello', () => {}, bridgeStrippedSignal)
    ).resolves.toBeUndefined()
  })

  it('still cancels the stream when the real signal aborts (barge-in is not silently broken)', async () => {
    await import('../src/preload/index')
    const { sendChatStream } = exposed.rex as ExposedRexAPI
    const controller = new AbortController()
    // Don't let the fake startChatStream auto-complete for this one - abort first.
    invoke.mockImplementation((channel: string) => {
      if (channel === 'rex:startChatStream') return new Promise(() => {})
      return Promise.resolve()
    })
    const promise = sendChatStream('hello', () => {}, buildCancelHandle(controller))
    controller.abort()
    await expect(promise).rejects.toThrow('Chat response cancelled')
    expect(invoke).toHaveBeenCalledWith('rex:cancelChatStream', expect.any(String))
  })
})
