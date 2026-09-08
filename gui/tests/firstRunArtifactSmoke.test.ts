import { readFileSync } from 'fs'
import { fileURLToPath } from 'url'
import { describe, expect, it } from 'vitest'

const artifactSmokeSource = readFileSync(
  fileURLToPath(new URL('../src/main/artifactSmoke.ts', import.meta.url)),
  'utf8'
)
const mainSource = readFileSync(
  fileURLToPath(new URL('../src/main/index.ts', import.meta.url)),
  'utf8'
)
const workflowSource = readFileSync(
  fileURLToPath(new URL('../../.github/workflows/windows-electron-artifact.yml', import.meta.url)),
  'utf8'
)

describe('US-125 installed first-run smoke', () => {
  it('has a dedicated fresh-install smoke that proves setup UI and setup-to-auth transition', () => {
    expect(artifactSmokeSource).toContain('runInstalledFirstRunSmoke')
    expect(artifactSmokeSource).toContain('Set up Account')
    expect(artifactSmokeSource).toContain('api.completeSetup')
    expect(artifactSmokeSource).toContain('background_voice_enabled: false')
    expect(artifactSmokeSource).toContain('await api.getStatus()')
  })

  it('routes only the explicit first-run artifact mode through the fresh-install smoke', () => {
    expect(mainSource).toContain('ASKREX_ARTIFACT_SMOKE_FIRST_RUN')
    expect(mainSource).toContain('runInstalledFirstRunSmoke')
  })

  it('runs the fresh-install smoke before the existing authenticated artifact smoke', () => {
    expect(workflowSource).toContain('test_installed_electron_first_run.ps1')
    expect(workflowSource.indexOf('test_installed_electron_first_run.ps1')).toBeLessThan(
      workflowSource.indexOf('test_installed_electron_artifact.ps1')
    )
  })
})
