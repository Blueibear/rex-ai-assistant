import React from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it, vi } from 'vitest'
import { ModelDiscoveryResults } from './ModelDiscoveryResults'

function render(props: Partial<React.ComponentProps<typeof ModelDiscoveryResults>> = {}): string {
  return renderToStaticMarkup(
    <ModelDiscoveryResults
      providerLabel="Ollama"
      status="idle"
      models={[]}
      selectedModel=""
      error=""
      onSelect={vi.fn()}
      {...props}
    />
  )
}

describe('ModelDiscoveryResults (US-072)', () => {
  it('renders nothing before the user initiates discovery', () => {
    expect(render()).toBe('')
  })

  it('shows an explicit loading state', () => {
    expect(render({ status: 'loading' })).toContain('Discovering models…')
  })

  it('shows a provider error without inventing model availability', () => {
    const html = render({
      status: 'error',
      error: 'Unable to reach configured Ollama endpoint'
    })
    expect(html).toContain('role="alert"')
    expect(html).toContain('Unable to reach configured Ollama endpoint')
    expect(html).not.toContain('<option')
  })

  it('distinguishes a successful empty list from an error', () => {
    const html = render({ status: 'success', models: [] })
    expect(html).toContain('No models reported by the configured Ollama endpoint.')
    expect(html).not.toContain('role="alert"')
  })

  it('renders only models actually returned by discovery', () => {
    const html = render({
      status: 'success',
      models: ['llama3.2:3b', 'qwen2.5:7b'],
      selectedModel: 'hardcoded-example-that-was-not-discovered'
    })
    expect(html).toContain('llama3.2:3b')
    expect(html).toContain('qwen2.5:7b')
    expect(html).not.toContain('hardcoded-example-that-was-not-discovered')
    expect(html).toContain('Select a discovered model')
  })

  it('marks a currently selected discovered model as selected', () => {
    const html = render({
      status: 'success',
      models: ['llama3.2:3b', 'qwen2.5:7b'],
      selectedModel: 'qwen2.5:7b'
    })
    expect(html).toContain('<option value="qwen2.5:7b" selected="">qwen2.5:7b</option>')
  })
})
