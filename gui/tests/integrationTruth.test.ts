import { readFileSync } from 'fs'
import { join } from 'path'
import { describe, expect, it } from 'vitest'

const root = join(__dirname, '..')
const inventory = readFileSync(join(root, 'src/main/integrationInventory.ts'), 'utf8')
const page = readFileSync(join(root, 'src/pages/IntegrationsPage.tsx'), 'utf8')
const types = readFileSync(join(root, 'src/types/ipc.ts'), 'utf8')
const status = readFileSync(join(root, 'src/main/integrationStatus.ts'), 'utf8')

describe('integration truth contract', () => {
  it('requires non-secret detail and next-action text on every item', () => {
    expect(types).toContain('detail: string')
    expect(types).toContain('next_action: string')
    expect(inventory).toContain('...statusCopy(state, status.error)')
  })

  it('does not treat stored configuration as authenticated evidence', () => {
    expect(inventory).toContain("configured: ['Configuration is stored, but provider access has not been tested.'")
    expect(inventory).toContain("authenticated: ['Provider authentication was tested successfully.'")
    expect(inventory).toContain("reachable: ['The service endpoint responded, but authentication is not established.'")
  })

  it('renders detail and a next safe action without replacing Configure', () => {
    expect(page).toContain('{int.detail}')
    expect(page).toContain('Next action: {int.next_action}')
    expect(page).toContain('Configure →')
  })

  it('preserves explicit Outlook-unavailable and SMS inventory truth', () => {
    expect(status).toContain('OUTLOOK_EMAIL_UNSUPPORTED')
    expect(status).toContain('OUTLOOK_CALENDAR_UNSUPPORTED')
    expect(inventory).toContain("name: 'SMS (Twilio)'")
    expect(inventory).toContain("key: 'sms'")
  })

  it('keeps verified actions distinct from attempted capability', () => {
    expect(inventory).toContain("write_capable: ['The provider reports write capability, but no write was verified.'")
    expect(inventory).toContain("verified: ['The configured capability has current verified evidence.'")
  })
})
