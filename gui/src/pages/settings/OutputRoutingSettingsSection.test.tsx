import fs from 'node:fs'
import path from 'node:path'
import React from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'
import { ToastProvider } from '../../components/ui/Toast'
import { OutputRoutingSettingsSection } from './OutputRoutingSettingsSection'

describe('OutputRoutingSettingsSection', () => {
  it('renders a truthful loading state before backend policy arrives', () => {
    const html = renderToStaticMarkup(
      <ToastProvider>
        <OutputRoutingSettingsSection />
      </ToastProvider>
    )
    expect(html).toContain('Loading routing policy…')
  })

  it('exposes every US-122 policy control through Rex-owned preload APIs', () => {
    const source = fs.readFileSync(
      path.resolve(process.cwd(), 'src/pages/settings/OutputRoutingSettingsSection.tsx'),
      'utf8'
    )

    for (const label of [
      'Spoken responses',
      'Timers',
      'Alarms',
      'Media',
      'Prefer the endpoint that received an interactive media request',
      'Default media account',
      'Quiet hours for optional spoken/media output',
      'Conditional routing',
      'Targets and test playback',
      'Speaker groups'
    ]) {
      expect(source).toContain(label)
    }

    for (const apiUse of [
      "window.rex.getSettings('outputRouting')",
      "window.rex.getSettings('outputRoutingAccounts')",
      "window.rex.setSettings('outputRouting'",
      "window.rex.setSettings('outputRoutingTest'",
      'window.rex.getAudioTargets()',
      'window.rex.listSpeakerGroups()',
      'window.rex.createSpeakerGroup(',
      'window.rex.setSpeakerGroupMembers(',
      'window.rex.deleteSpeakerGroup('
    ]) {
      expect(source).toContain(apiUse)
    }

    expect(source).not.toContain('ipcRenderer')
    expect(source).not.toContain('window.electron')
  })
})
