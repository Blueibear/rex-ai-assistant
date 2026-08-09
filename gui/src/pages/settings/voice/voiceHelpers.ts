import type { VoiceSettings, WakeWordInfo } from '../../../types/ipc'

export interface MediaDeviceOption {
  deviceId: string
  label: string
}

export const ENROLLMENT_SAMPLE_TARGET = 3
export const ENROLLMENT_SAMPLE_DURATION_MS = 1600
export const ENROLLMENT_COUNTDOWN_SECONDS = 3
export const WW_POSITIVE_TARGET = 5
export const WW_NEGATIVE_TARGET = 3
export const ENROLLMENT_SAMPLE_RATE = 16000
export const ENROLLMENT_PROMPT_PHRASE = 'Hey Rex, the quick brown fox jumps over the lazy dog.'
export const ENROLLMENT_MIN_RMS = 0.02 // below this → sample is too quiet
export const FALLBACK_BUILTIN_WAKE_WORDS: WakeWordInfo[] = [
  { id: 'hey_jarvis', name: 'Hey Jarvis', engine: 'openwakeword', has_sample: false },
  { id: 'hey_mycroft', name: 'Hey Mycroft', engine: 'openwakeword', has_sample: false },
  { id: 'hey_rhasspy', name: 'Hey Rhasspy', engine: 'openwakeword', has_sample: false },
  { id: 'ok_nabu', name: 'OK Nabu', engine: 'openwakeword', has_sample: false },
  { id: 'alexa', name: 'Alexa', engine: 'openwakeword', has_sample: false }
]

export function slugifyWakeWordPhrase(phrase: string): string {
  const trimmed = phrase.trim().toLowerCase()
  if (!trimmed) return 'hey_rex'
  const slug = trimmed
    .replace(/[^\w\s-]/g, '')
    .replace(/[\s-]+/g, '_')
    .replace(/^_+|_+$/g, '')
  return slug || 'hey_rex'
}

export function defaultCustomWakeWordAssetPath(
  backend: Extract<VoiceSettings['wakeWordBackend'], 'custom_onnx' | 'custom_embedding'>,
  phraseOrId: string
): string {
  const slug = slugifyWakeWordPhrase(phraseOrId)
  return backend === 'custom_onnx'
    ? `config\\wake_words\\${slug}\\model.onnx`
    : `config\\wake_words\\${slug}\\embedding.pt`
}

export function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => {
    window.setTimeout(resolve, ms)
  })
}

export function mergeFloat32Arrays(chunks: Float32Array[]): Float32Array {
  const totalLength = chunks.reduce((sum, chunk) => sum + chunk.length, 0)
  const merged = new Float32Array(totalLength)
  let offset = 0
  for (const chunk of chunks) {
    merged.set(chunk, offset)
    offset += chunk.length
  }
  return merged
}

export function downsampleFloat32(
  samples: Float32Array,
  inputSampleRate: number,
  outputSampleRate: number
): Float32Array {
  if (inputSampleRate === outputSampleRate) {
    return samples
  }
  if (inputSampleRate < outputSampleRate) {
    throw new Error('Microphone sample rate is lower than the enrollment target rate')
  }

  const ratio = inputSampleRate / outputSampleRate
  const outputLength = Math.max(1, Math.round(samples.length / ratio))
  const output = new Float32Array(outputLength)

  for (let i = 0; i < outputLength; i += 1) {
    const start = Math.floor(i * ratio)
    const end = Math.min(samples.length, Math.floor((i + 1) * ratio))
    let sum = 0
    let count = 0
    for (let j = start; j < end; j += 1) {
      sum += samples[j]
      count += 1
    }
    output[i] = count > 0 ? sum / count : samples[Math.min(start, samples.length - 1)]
  }

  return output
}

export function computeRms(samples: number[]): number {
  if (samples.length === 0) return 0
  const sumOfSquares = samples.reduce((acc, s) => acc + s * s, 0)
  return Math.sqrt(sumOfSquares / samples.length)
}

export async function captureEnrollmentSample(stream: MediaStream): Promise<number[]> {
  const audioContext = new AudioContext()
  const source = audioContext.createMediaStreamSource(stream)
  const processor = audioContext.createScriptProcessor(4096, 1, 1)
  const sink = audioContext.createGain()
  const chunks: Float32Array[] = []

  sink.gain.value = 0

  return new Promise<number[]>((resolve, reject) => {
    let settled = false

    const cleanup = (): void => {
      if (!settled) {
        settled = true
        processor.disconnect()
        sink.disconnect()
        source.disconnect()
        void audioContext.close()
      }
    }

    processor.onaudioprocess = (event) => {
      const channel = event.inputBuffer.getChannelData(0)
      chunks.push(new Float32Array(channel))
    }

    source.connect(processor)
    processor.connect(sink)
    sink.connect(audioContext.destination)

    window.setTimeout(() => {
      try {
        if (chunks.length === 0) {
          throw new Error('No microphone audio was captured')
        }
        const merged = mergeFloat32Arrays(chunks)
        const downsampled = downsampleFloat32(
          merged,
          audioContext.sampleRate,
          ENROLLMENT_SAMPLE_RATE
        )
        resolve(Array.from(downsampled))
      } catch (error) {
        reject(error)
      } finally {
        cleanup()
      }
    }, ENROLLMENT_SAMPLE_DURATION_MS)
  })
}

export async function runEnrollmentCountdown(
  onTick: (remaining: number) => void
): Promise<void> {
  for (let remaining = ENROLLMENT_COUNTDOWN_SECONDS; remaining > 0; remaining -= 1) {
    onTick(remaining)
    await sleep(1000)
  }
  onTick(0)
}
