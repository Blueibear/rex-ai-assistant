import type { VoiceSettings } from './ipc'

export function voiceProvider(engine: VoiceSettings['ttsEngine']): string {
  if (engine === 'openai') return 'edge-tts'
  if (engine === 'elevenlabs' || engine === 'system') return 'pyttsx3'
  return engine
}
