import type { TurnStatusUpdate, TurnStatusValue } from './ipc'

export interface TurnStatusState {
  active: Map<string, { status: TurnStatusValue; sequence: number }>
  terminalSequence: Map<string, number>
}

export type TurnStatusPresentation = TurnStatusValue | 'idle'

export function createTurnStatusState(): TurnStatusState {
  return {
    active: new Map(),
    terminalSequence: new Map(),
  }
}

export function hasActiveTurn(state: TurnStatusState): boolean {
  return state.active.size > 0
}

function latestActiveStatus(state: TurnStatusState): TurnStatusPresentation {
  return Array.from(state.active.values()).at(-1)?.status ?? 'idle'
}

/** Apply one ordered canonical turn status without reviving stale/terminal turns. */
export function applyTurnStatusUpdate(
  state: TurnStatusState,
  update: TurnStatusUpdate,
): TurnStatusPresentation {
  if (state.terminalSequence.has(update.turnId)) {
    return latestActiveStatus(state)
  }

  const prior = state.active.get(update.turnId)
  if (prior !== undefined && update.sequence <= prior.sequence) {
    return latestActiveStatus(state)
  }

  state.active.delete(update.turnId)
  if (update.terminal) {
    state.terminalSequence.set(update.turnId, update.sequence)
    return latestActiveStatus(state)
  }

  state.active.set(update.turnId, { status: update.status, sequence: update.sequence })
  return update.status
}
