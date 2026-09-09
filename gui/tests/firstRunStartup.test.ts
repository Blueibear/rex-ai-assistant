import { describe, expect, it } from 'vitest'
import { planElectronStartup } from '../src/main/firstRunStartup'

describe('planElectronStartup', () => {
  it('opens setup without identity or background bootstrap on a clean install', () => {
    expect(planElectronStartup({ needsSetup: true, backgroundVoiceEnabled: false })).toEqual({
      mode: 'setup',
      requireIdentity: false,
      bootstrapBackground: false
    })
  })

  it('never bootstraps background voice before first-run privacy opt-in', () => {
    expect(planElectronStartup({ needsSetup: true, backgroundVoiceEnabled: true })).toEqual({
      mode: 'setup',
      requireIdentity: false,
      bootstrapBackground: false
    })
  })

  it('requires identity after setup and honors the persisted background choice', () => {
    expect(planElectronStartup({ needsSetup: false, backgroundVoiceEnabled: false })).toEqual({
      mode: 'authenticated',
      requireIdentity: true,
      bootstrapBackground: false
    })
    expect(planElectronStartup({ needsSetup: false, backgroundVoiceEnabled: true })).toEqual({
      mode: 'authenticated',
      requireIdentity: true,
      bootstrapBackground: true
    })
  })
})
