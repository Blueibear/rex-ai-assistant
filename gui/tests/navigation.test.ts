import { readFileSync } from 'fs'
import { join } from 'path'
import { describe, expect, it } from 'vitest'

const root = join(__dirname, '..')
const layout = readFileSync(join(root, 'src/layouts/AppLayout.tsx'), 'utf8')
const app = readFileSync(join(root, 'src/renderer/src/App.tsx'), 'utf8')

function navItemsSource(): string {
  const start = layout.indexOf('const navItems: NavItem[] = [')
  const end = layout.indexOf('/** Map each route path', start)
  return layout.slice(start, end)
}

describe('primary navigation contract', () => {
  it('omits SMS and duplicate Settings from scrolling navigation', () => {
    const nav = navItemsSource()
    expect(nav).not.toContain("path: '/sms'")
    expect(nav).not.toContain("path: '/settings'")
  })

  it('keeps Email visible without a beta readiness claim', () => {
    const nav = navItemsSource()
    expect(nav).toContain("path: '/email'")
    expect(nav).not.toContain('beta:')
    expect(layout).not.toContain('BETA badge')
  })

  it('preserves persistent Settings access and the direct SMS route', () => {
    expect(layout).toContain("navigate('/settings')")
    expect(layout).toContain('aria-label="Settings"')
    expect(app).toContain('path="/sms"')
  })

  it('preserves active-section and collapsed-sidebar behavior', () => {
    expect(layout).toContain('sectionNames[path]')
    expect(layout).toContain('title={narrow ? item.label : undefined}')
    expect(layout).toContain('style={{ width: narrow ? 64 : 240 }}')
  })
})
