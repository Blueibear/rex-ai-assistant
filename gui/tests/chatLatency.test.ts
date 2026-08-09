import { readFileSync } from 'node:fs'

import { describe, expect, it, vi } from 'vitest'

import { logChatLatency } from '../src/main/chatLatency'

describe('chat latency logging', () => {
  it('logs only safe transport timing metadata', () => {
    const spy = vi.spyOn(console, 'info').mockImplementation(() => undefined)
    try {
      logChatLatency('chat_ipc_complete', 'stream', 12.345, 'ok')
      expect(spy).toHaveBeenCalledTimes(1)
      const [, fields] = spy.mock.calls[0]
      expect(fields).toEqual({
        event: 'chat_ipc_complete',
        mode: 'stream',
        durationMs: 12.345,
        outcome: 'ok'
      })
      expect(fields).not.toHaveProperty('transcript')
      expect(fields).not.toHaveProperty('prompt')
      expect(fields).not.toHaveProperty('userId')
    } finally {
      spy.mockRestore()
    }
  })

  it('is wired into single and streaming chat handlers', () => {
    const source = readFileSync(new URL('../src/main/handlers/chat.ts', import.meta.url), 'utf8')
    expect(source).toContain("logChatLatency('chat_ipc_complete', 'single'")
    expect(source).toContain("logChatLatency('chat_first_token', 'stream'")
    expect(source).toContain("logChatLatency('chat_ipc_complete', 'stream'")
  })
})
