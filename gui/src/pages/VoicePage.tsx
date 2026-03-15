import React, { useState, useCallback } from 'react'
import { VoiceToggle } from '../components/voice/VoiceToggle'
import type { VoiceState } from '../components/voice/VoiceToggle'
import { WaveformVisualizer } from '../components/voice/WaveformVisualizer'

export function VoicePage(): React.ReactElement {
  const [voiceState, setVoiceState] = useState<VoiceState>('idle')

  const handleToggle = useCallback(() => {
    setVoiceState((prev) => {
      if (prev === 'idle') return 'listening'
      return 'idle'
    })
  }, [])

  return (
    <div className="flex flex-col items-center justify-center h-full gap-8">
      <WaveformVisualizer state={voiceState} width={320} height={80} />
      <VoiceToggle state={voiceState} onToggle={handleToggle} />
    </div>
  )
}
