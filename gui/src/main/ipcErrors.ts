// Secret-free IPC error responses for credential/settings/Home Assistant/
// integration save paths (S4).
//
// A raw `err.message` returned to the renderer can carry vault refs,
// filesystem paths, or other internal context surfaced by lower layers
// (Node fs errors, the credential-vault bridge, config mirror failures).
// `safeIpcErrorMessage` never forwards that text to the renderer - only a
// fixed, pre-written fallback message. It also never writes `err.message`
// to the on-disk Electron log; only the JS error's constructor name is
// logged, since the message itself may be secret-derived.

import { appendElectronLog } from './handlers/logs'

/**
 * An error whose message was authored in-repo (a static or user-submitted
 * business-value string with no vault ref, filesystem path, or other
 * internal context) and is therefore safe to return to the renderer
 * verbatim. Use this for deliberate validation/diagnostic throws; never for
 * errors that wrap output from the credential vault, filesystem writes, or
 * other layers whose message text is not fully controlled by this file.
 */
export class SafeValidationError extends Error {
  constructor(message: string) {
    super(message)
    this.name = 'SafeValidationError'
  }
}

export function safeIpcErrorMessage(err: unknown, fallback: string): string {
  if (err instanceof SafeValidationError) return err.message
  const errorName = err instanceof Error ? err.name : typeof err
  try {
    appendElectronLog('ERROR', fallback, { error_type: errorName })
  } catch {
    // Error reporting must remain fail-safe when the log path is unavailable.
  }
  return fallback
}
