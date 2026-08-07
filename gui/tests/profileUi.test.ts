import { readFileSync } from 'fs'
import { join } from 'path'
import { describe, expect, it } from 'vitest'

const root = join(__dirname, '..')
const app = readFileSync(join(root, 'src/renderer/src/App.tsx'), 'utf8')
const layout = readFileSync(join(root, 'src/layouts/AppLayout.tsx'), 'utf8')
const profile = readFileSync(join(root, 'src/pages/ProfilePage.tsx'), 'utf8')

describe('profile UI contract', () => {
  it('registers the profile route and a real accessible profile button', () => {
    expect(app).toContain('path="/profile"')
    expect(app).toContain('<ProfilePage />')
    expect(layout).toContain('aria-label="Open user profile"')
    expect(layout).toContain("navigate('/profile')")
  })

  it('loads avatar or initials through typed profile IPC only', () => {
    expect(layout).toContain('window.rex.getProfile()')
    expect(layout).toContain('sidebarProfile?.avatar_data')
    expect(layout).toContain("sidebarProfile?.initials ?? 'U'")
    expect(layout).not.toContain('/api/user/avatar')
    expect(profile).not.toContain('fetch(')
  })

  it('shows identity, permissions, preferences, voice, and data scope', () => {
    for (const label of [
      'Identity',
      'Permissions',
      'Preferences',
      'Voice Enrollment',
      'Private to This Profile'
    ]) {
      expect(profile).toContain(label)
    }
    expect(profile).toContain('profile.scope_labels')
    expect(profile).toContain('Shared Household Settings')
    expect(profile).toContain("navigate('/settings')")
  })

  it('uses typed IPC for avatar changes with bounded validation', () => {
    expect(profile).toContain('window.rex.setProfileAvatar(file.type, base64)')
    expect(profile).toContain('window.rex.removeProfileAvatar()')
    expect(profile).toContain("['image/jpeg', 'image/png']")
    expect(profile).toContain('2 * 1024 * 1024')
    expect(profile).toContain('avatarState.error')
  })

  it('keeps identity bound to the authenticated desktop session', () => {
    expect(profile).toContain('current authenticated desktop session')
    expect(profile).toContain('new authenticated desktop session')
    expect(profile).toContain('restart the application')
    expect(profile).not.toContain('switchUser')
    expect(profile).not.toContain('setActiveUser')
  })
})
