import { readFileSync } from 'fs'
import { join } from 'path'
import { describe, expect, it } from 'vitest'

describe('guarded procedural memory UI', () => {
  it('keeps procedures separate from ordinary memory creation and exposes audit controls', () => {
    const root = join(__dirname, '..')
    const page = readFileSync(join(root, 'src/pages/MemoriesPage.tsx'), 'utf8')
    const modal = readFileSync(join(root, 'src/components/memory/ProceduresModal.tsx'), 'utf8')
    const types = readFileSync(join(root, 'src/types/ipc.ts'), 'utf8')
    const preload = readFileSync(join(root, 'src/preload/index.ts'), 'utf8')

    expect(page).toContain('Learned Procedures')
    expect(page).toContain('+ Add Memory')
    expect(modal).not.toContain('Add Procedure')
    expect(modal).toContain('verified action')
    expect(modal).toContain('Stored permissions are requirements, not grants')
    for (const label of ['Approve', 'Disable', 'Revoke', 'Delete', 'Provenance and audit']) {
      expect(modal).toContain(label)
    }
    for (const method of [
      'getProcedures',
      'approveProcedure',
      'disableProcedure',
      'revokeProcedure',
      'deleteProcedure'
    ]) {
      expect(types).toContain(method)
      expect(preload).toContain(method)
    }
  })

  it('requires Electron main-process confirmation before forwarding risky approval', () => {
    const root = join(__dirname, '..')
    const handlers = readFileSync(join(root, 'src/main/handlers/memories.ts'), 'utf8')

    expect(handlers).toContain('dialog.showMessageBox')
    expect(handlers).toContain("buttons: ['Cancel', 'Approve']")
    expect(handlers).toContain('if (response.response !== 1)')
    expect(handlers).toContain("mutateProcedure(session, id, 'procedures-approve', true)")
    expect(handlers).not.toContain("mutateProcedure(session, id, 'procedures-approve')")
  })
})
