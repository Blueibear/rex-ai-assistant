import { readFileSync } from 'fs'
import { fileURLToPath } from 'url'
import { describe, expect, it } from 'vitest'

function source(relativePath: string): string {
  return readFileSync(fileURLToPath(new URL(relativePath, import.meta.url)), 'utf8')
}

const handlerSource = source('../src/main/handlers/setup.ts')
const wizardSource = source('../src/pages/SetupWizardPage.tsx')
const ipcSource = source('../src/types/ipc.ts')

describe('US-125 setup persistence versus runtime readiness truth', () => {
  it('models saved setup separately from post-save runtime readiness', () => {
    expect(ipcSource).toContain('setup_saved?: boolean')
    expect(ipcSource).toContain('runtime_ready?: boolean')
    expect(ipcSource).toContain('warning?: string')
  })

  it('returns saved-but-not-ready state when authenticated bootstrap fails after persistence', () => {
    expect(handlerSource).toContain('setup_saved: true')
    expect(handlerSource).toContain('runtime_ready: true')
    expect(handlerSource).toContain('runtime_ready: false')
    expect(handlerSource).toContain(
      'Setup was saved, but Rex could not finish starting. Close and reopen AskRex to continue.'
    )
  })

  it('does not start voice verification or open the dashboard when runtime bootstrap is not ready', () => {
    expect(wizardSource).toContain('result.runtime_ready === false')
    expect(wizardSource).toContain('setSetupRuntimeWarning(')
    expect(wizardSource).toContain('setStep(DONE_STEP)')
    expect(wizardSource).toContain('setupRuntimeWarning ?')
    expect(wizardSource).toContain('Close and reopen AskRex to continue.')
  })
})
