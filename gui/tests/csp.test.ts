import { describe, expect, it } from 'vitest'
import { readFileSync } from 'fs'
import { join } from 'path'

// Regression coverage for the real-hardware-acceptance defect (issue #299):
// TTS playback in the Voice page was silently blocked with no audio and no
// console-visible reason other than the generic HTMLMediaElement error
// "Speech playback failed on the selected output device". Root cause: the
// renderer's CSP declared no `media-src` directive, so per the CSP fallback
// spec it inherited `default-src 'self'` - which does not include `data:`.
// speakResponse() in VoicePage.tsx always plays synthesized speech via
// `new Audio('data:audio/...;base64,...')`, so every TTS provider
// (xtts/edge-tts/pyttsx3) was broken by this, not just one provider.
//
// This test parses the actual CSP string from index.html and replicates
// CSP's fallback semantics (a directive with no explicit value falls back
// to default-src) rather than doing a naive substring match, so it fails
// against the pre-fix markup (media-src absent, default-src has no data:)
// and passes once media-src explicitly allows data:.
function parseCsp(csp: string): Record<string, string[]> {
  const directives: Record<string, string[]> = {}
  for (const part of csp.split(';')) {
    const tokens = part.trim().split(/\s+/).filter(Boolean)
    if (tokens.length === 0) continue
    const [name, ...values] = tokens
    directives[name] = values
  }
  return directives
}

function effectiveSources(directives: Record<string, string[]>, directive: string): string[] {
  return directives[directive] ?? directives['default-src'] ?? []
}

describe('renderer Content-Security-Policy', () => {
  const html = readFileSync(join(__dirname, '../src/renderer/index.html'), 'utf8')
  const match = html.match(/http-equiv="Content-Security-Policy"\s+content="([^"]+)"/)

  it('declares a CSP meta tag', () => {
    expect(match).not.toBeNull()
  })

  it('allows data: URIs for media playback (TTS audio), with correct fallback semantics', () => {
    const csp = match?.[1] ?? ''
    const directives = parseCsp(csp)
    const mediaSrc = effectiveSources(directives, 'media-src')
    expect(mediaSrc).toContain('data:')
  })

  it('still restricts everything else to self by default (no accidental over-broadening)', () => {
    const csp = match?.[1] ?? ''
    const directives = parseCsp(csp)
    expect(directives['default-src']).toEqual(['\'self\''])
    expect(directives['script-src']).toEqual(['\'self\''])
  })
})
