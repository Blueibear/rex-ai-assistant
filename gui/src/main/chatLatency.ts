export type ChatLatencyOutcome = 'ok' | 'error' | 'cancelled'

export function logChatLatency(
  event: string,
  mode: 'single' | 'stream',
  durationMs: number,
  outcome: ChatLatencyOutcome
): void {
  console.info('[latency]', {
    event,
    mode,
    durationMs: Number(durationMs.toFixed(3)),
    outcome
  })
}
