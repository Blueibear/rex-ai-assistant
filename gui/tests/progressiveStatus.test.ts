import { describe, expect, it } from 'vitest'
import { readFileSync } from 'fs'
import { join } from 'path'
import {
  applyTurnStatusUpdate,
  createTurnStatusState,
} from '../src/types/turnStatusState'

function source(path: string): string {
  return readFileSync(join(__dirname, '..', path), 'utf8')
}

describe('canonical progressive status wiring', () => {
  it('uses one typed global turn-status contract across Electron surfaces', () => {
    const ipc = source('src/types/ipc.ts')
    const chat = source('src/main/handlers/chat.ts')
    const voice = source('src/main/handlers/voice.ts')
    const layout = source('src/layouts/AppLayout.tsx')

    expect(ipc).toContain('TurnStatusValue')
    expect(ipc).toContain('TurnStatusUpdate')
    expect(ipc).toContain('onTurnStatus')
    expect(chat).toContain("'rex:turnStatus'")
    expect(chat).toContain("data.status === 'model_failure'")
    expect(ipc).toContain("export type ChatStreamStatus = 'model_failure'")
    expect(voice).toContain('turn_status')
    expect(voice).toContain("'rex:turnStatus'")
    expect(layout).toContain('onTurnStatus')
  })
  it('tracks overlapping turns, returns idle, and rejects every post-terminal update', () => {
    const state = createTurnStatusState()

    expect(applyTurnStatusUpdate(state, {
      turnId: 'chat-turn', sequence: 1, status: 'thinking', terminal: false,
    })).toBe('thinking')
    expect(applyTurnStatusUpdate(state, {
      turnId: 'voice-turn', sequence: 1, status: 'acting', terminal: false,
    })).toBe('acting')

    expect(applyTurnStatusUpdate(state, {
      turnId: 'chat-turn', sequence: 2, status: 'done', terminal: true,
    })).toBe('acting')
    expect(state.active.has('voice-turn')).toBe(true)
    expect(state.active.has('chat-turn')).toBe(false)

    expect(applyTurnStatusUpdate(state, {
      turnId: 'voice-turn', sequence: 2, status: 'done', terminal: true,
    })).toBe('idle')
    expect(state.active.size).toBe(0)

    expect(applyTurnStatusUpdate(state, {
      turnId: 'voice-turn', sequence: 1, status: 'thinking', terminal: false,
    })).toBe('idle')
    expect(state.active.size).toBe(0)

    expect(applyTurnStatusUpdate(state, {
      turnId: 'voice-turn', sequence: 3, status: 'thinking', terminal: false,
    })).toBe('idle')
    expect(state.active.size).toBe(0)
  })

  it('exposes the presentation labels required by canonical status', () => {
    const layout = source('src/layouts/AppLayout.tsx')
    expect(layout).toContain('Checking')
    expect(layout).toContain('Acting')
    expect(layout).toContain('Verifying')
    expect(layout).toContain('Cancelled')
  })
})
