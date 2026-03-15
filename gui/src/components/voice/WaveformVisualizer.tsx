import React, { useRef, useEffect, useCallback } from 'react'
import type { VoiceState } from './VoiceToggle'

export interface WaveformVisualizerProps {
  state: VoiceState
  width?: number
  height?: number
}

const BAR_COUNT = 32
const BAR_GAP = 2

function drawRoundedBar(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  w: number,
  h: number,
): void {
  const r = Math.min(w / 2, 3)
  ctx.beginPath()
  ctx.moveTo(x + r, y)
  ctx.lineTo(x + w - r, y)
  ctx.arcTo(x + w, y, x + w, y + r, r)
  ctx.lineTo(x + w, y + h - r)
  ctx.arcTo(x + w, y + h, x + w - r, y + h, r)
  ctx.lineTo(x + r, y + h)
  ctx.arcTo(x, y + h, x, y + h - r, r)
  ctx.lineTo(x, y + r)
  ctx.arcTo(x, y, x + r, y, r)
  ctx.closePath()
  ctx.fill()
}

export const WaveformVisualizer: React.FC<WaveformVisualizerProps> = ({
  state,
  width = 300,
  height = 80,
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const rafRef = useRef<number>(0)
  const audioContextRef = useRef<AudioContext | null>(null)
  const analyserRef = useRef<AnalyserNode | null>(null)
  const streamRef = useRef<MediaStream | null>(null)

  const stopAudio = useCallback(() => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop())
      streamRef.current = null
    }
    if (audioContextRef.current) {
      void audioContextRef.current.close()
      audioContextRef.current = null
      analyserRef.current = null
    }
  }, [])

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    cancelAnimationFrame(rafRef.current)

    const barW = (width - BAR_GAP * (BAR_COUNT - 1)) / BAR_COUNT
    const cy = height / 2

    // ── idle / processing: static flat line ──────────────────────────────────
    if (state === 'idle' || state === 'processing') {
      ctx.clearRect(0, 0, width, height)
      ctx.strokeStyle = 'rgba(148,163,184,0.15)'
      ctx.lineWidth = 1.5
      ctx.beginPath()
      ctx.moveTo(0, cy)
      ctx.lineTo(width, cy)
      ctx.stroke()
      return
    }

    // ── speaking: synthetic animated bars ────────────────────────────────────
    if (state === 'speaking') {
      let startTs: number | null = null
      const draw = (ts: number): void => {
        if (startTs === null) startTs = ts
        const elapsed = (ts - startTs) / 1000
        ctx.clearRect(0, 0, width, height)
        ctx.fillStyle = '#22c55e'
        for (let i = 0; i < BAR_COUNT; i++) {
          const phase = (i / BAR_COUNT) * Math.PI * 4
          const amp = 0.25 + 0.75 * Math.abs(Math.sin(elapsed * 2.8 + phase))
          const barH = Math.max(4, amp * height * 0.7)
          drawRoundedBar(ctx, i * (barW + BAR_GAP), cy - barH / 2, barW, barH)
        }
        rafRef.current = requestAnimationFrame(draw)
      }
      rafRef.current = requestAnimationFrame(draw)
      return () => cancelAnimationFrame(rafRef.current)
    }

    // ── listening: real mic input via Web Audio API ───────────────────────────
    if (state === 'listening') {
      const startListening = async (): Promise<void> => {
        try {
          const stream = await navigator.mediaDevices.getUserMedia({
            audio: true,
            video: false,
          })
          streamRef.current = stream
          const ac = new AudioContext()
          audioContextRef.current = ac
          const analyser = ac.createAnalyser()
          analyser.fftSize = 64
          analyserRef.current = analyser
          const src = ac.createMediaStreamSource(stream)
          src.connect(analyser)
          const data = new Uint8Array(analyser.frequencyBinCount)
          const binStep = Math.max(1, Math.floor(data.length / BAR_COUNT))

          const draw = (): void => {
            if (!analyserRef.current) return
            analyserRef.current.getByteFrequencyData(data)
            ctx.clearRect(0, 0, width, height)
            ctx.fillStyle = '#ef4444'
            for (let i = 0; i < BAR_COUNT; i++) {
              const bin = data[Math.min(i * binStep, data.length - 1)] ?? 0
              const amp = bin / 255
              const barH = Math.max(4, amp * height * 0.85)
              drawRoundedBar(ctx, i * (barW + BAR_GAP), cy - barH / 2, barW, barH)
            }
            rafRef.current = requestAnimationFrame(draw)
          }
          rafRef.current = requestAnimationFrame(draw)
        } catch {
          // Mic unavailable — synthetic fallback animation
          let startTs: number | null = null
          const draw = (ts: number): void => {
            if (startTs === null) startTs = ts
            const elapsed = (ts - startTs) / 1000
            ctx.clearRect(0, 0, width, height)
            ctx.fillStyle = '#ef4444'
            for (let i = 0; i < BAR_COUNT; i++) {
              const phase = (i / BAR_COUNT) * Math.PI * 4
              const amp = 0.15 + 0.35 * Math.abs(Math.sin(elapsed * 3 + phase))
              const barH = Math.max(4, amp * height * 0.65)
              drawRoundedBar(ctx, i * (barW + BAR_GAP), cy - barH / 2, barW, barH)
            }
            rafRef.current = requestAnimationFrame(draw)
          }
          rafRef.current = requestAnimationFrame(draw)
        }
      }

      void startListening()
      return () => {
        cancelAnimationFrame(rafRef.current)
        stopAudio()
      }
    }

    return () => cancelAnimationFrame(rafRef.current)
  }, [state, width, height, stopAudio])

  // Final cleanup on unmount
  useEffect(() => {
    return () => {
      cancelAnimationFrame(rafRef.current)
      stopAudio()
    }
  }, [stopAudio])

  return (
    <canvas
      ref={canvasRef}
      width={width}
      height={height}
      className="rounded-lg"
      role="img"
      aria-label="Audio waveform visualizer"
    />
  )
}

export default WaveformVisualizer
