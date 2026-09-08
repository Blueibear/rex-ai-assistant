export interface ElectronStartupInput {
  needsSetup: boolean
  backgroundVoiceEnabled: boolean
}

export interface ElectronStartupPlan {
  mode: 'setup' | 'authenticated'
  requireIdentity: boolean
  bootstrapBackground: boolean
}

export function planElectronStartup(input: ElectronStartupInput): ElectronStartupPlan {
  if (input.needsSetup) {
    return {
      mode: 'setup',
      requireIdentity: false,
      bootstrapBackground: false
    }
  }

  return {
    mode: 'authenticated',
    requireIdentity: true,
    bootstrapBackground: input.backgroundVoiceEnabled
  }
}
