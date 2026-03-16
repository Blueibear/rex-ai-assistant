import React, { useEffect, useState, useMemo, useRef } from 'react'
import type { Memory } from '../types/ipc'
import { Spinner } from '../components/ui/Spinner'
import { Badge } from '../components/ui/Badge'
import { EmptyState } from '../components/ui/EmptyState'
import { useToast } from '../components/ui/Toast'

const PAGE_SIZE = 20

function relativeDate(iso: string): string {
  const diffMs = Date.now() - new Date(iso).getTime()
  const diffDays = Math.floor(diffMs / 86400000)
  if (diffDays === 0) return 'Today'
  if (diffDays === 1) return 'Yesterday'
  if (diffDays < 30) return `${diffDays}d ago`
  const diffMonths = Math.floor(diffDays / 30)
  if (diffMonths < 12) return `${diffMonths}mo ago`
  return `${Math.floor(diffMonths / 12)}y ago`
}

interface EditState {
  text: string
  category: string
}

export function MemoriesPage(): React.ReactElement {
  const [memories, setMemories] = useState<Memory[]>([])
  const [loading, setLoading] = useState(true)
  const [search, setSearch] = useState('')
  const [categoryFilter, setCategoryFilter] = useState('all')
  const [page, setPage] = useState(1)
  // Which memory is currently being edited (id or null)
  const [editingId, setEditingId] = useState<string | null>(null)
  // Draft edit state
  const [editState, setEditState] = useState<EditState>({ text: '', category: '' })
  const [saving, setSaving] = useState(false)
  const addToast = useToast()

  useEffect(() => {
    window.rex
      .getMemories()
      .then((data) => setMemories(data))
      .catch(console.error)
      .finally(() => setLoading(false))
  }, [])

  const categories = useMemo(() => {
    const cats = new Set(memories.map((m) => m.category))
    return Array.from(cats).sort()
  }, [memories])

  const filtered = useMemo(() => {
    const q = search.toLowerCase()
    return memories.filter((m) => {
      const matchesSearch = q === '' || m.text.toLowerCase().includes(q)
      const matchesCategory = categoryFilter === 'all' || m.category === categoryFilter
      return matchesSearch && matchesCategory
    })
  }, [memories, search, categoryFilter])

  const totalPages = Math.max(1, Math.ceil(filtered.length / PAGE_SIZE))
  const currentPage = Math.min(page, totalPages)
  const paginated = filtered.slice((currentPage - 1) * PAGE_SIZE, currentPage * PAGE_SIZE)

  // Reset to page 1 when search/filter changes
  useEffect(() => {
    setPage(1)
  }, [search, categoryFilter])

  function isDirty(memory: Memory): boolean {
    return editState.text !== memory.text || editState.category !== memory.category
  }

  function openEditor(memory: Memory): void {
    // If another card is open and dirty, prompt before switching
    if (editingId !== null && editingId !== memory.id) {
      const current = memories.find((m) => m.id === editingId)
      if (current && isDirty(current)) {
        const confirmed = window.confirm('You have unsaved changes. Discard and edit this memory instead?')
        if (!confirmed) return
      }
    }
    setEditingId(memory.id)
    setEditState({ text: memory.text, category: memory.category })
  }

  function cancelEdit(): void {
    setEditingId(null)
    setEditState({ text: '', category: '' })
  }

  function saveEdit(memory: Memory): void {
    if (!editState.text.trim()) return
    setSaving(true)
    // Optimistic update
    const optimistic: Memory = { ...memory, text: editState.text, category: editState.category, updatedAt: new Date().toISOString() }
    setMemories((prev) => prev.map((m) => (m.id === memory.id ? optimistic : m)))
    setEditingId(null)

    window.rex
      .updateMemory(memory.id, { text: editState.text, category: editState.category })
      .then((updated) => {
        setMemories((prev) => prev.map((m) => (m.id === updated.id ? updated : m)))
      })
      .catch((err: unknown) => {
        // Revert on error
        setMemories((prev) => prev.map((m) => (m.id === memory.id ? memory : m)))
        addToast(err instanceof Error ? err.message : 'Failed to save memory', 'error')
      })
      .finally(() => setSaving(false))
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center h-full">
        <Spinner size="lg" />
      </div>
    )
  }

  return (
    <div className="flex flex-col h-full overflow-hidden">
      {/* Header + controls */}
      <div className="flex-none px-6 pt-6 pb-4 border-b border-border">
        <div className="flex items-center gap-3">
          <input
            type="text"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Search memories…"
            className="flex-1 bg-surface border border-border rounded-lg px-3 py-2 text-sm text-text-primary placeholder-text-secondary focus:outline-none focus:ring-2 focus:ring-accent"
          />
          <select
            value={categoryFilter}
            onChange={(e) => setCategoryFilter(e.target.value)}
            className="bg-surface border border-border rounded-lg px-3 py-2 text-sm text-text-primary focus:outline-none focus:ring-2 focus:ring-accent"
          >
            <option value="all">All categories</option>
            {categories.map((cat) => (
              <option key={cat} value={cat}>
                {cat}
              </option>
            ))}
          </select>
        </div>
        <p className="mt-2 text-xs text-text-secondary">
          {filtered.length} {filtered.length === 1 ? 'memory' : 'memories'}
          {search || categoryFilter !== 'all' ? ' matching filters' : ''}
          {saving && <span className="ml-2 text-accent">Saving…</span>}
        </p>
      </div>

      {/* List */}
      <div className="flex-1 overflow-y-auto px-6 py-4">
        {paginated.length === 0 ? (
          <EmptyState
            icon={<span>🧠</span>}
            heading={memories.length === 0 ? 'No memories yet' : 'No matching memories'}
            subtext={
              memories.length === 0
                ? 'Rex will remember things about you as you interact.'
                : 'Try adjusting your search or category filter.'
            }
          />
        ) : (
          <div className="flex flex-col gap-3">
            {paginated.map((memory) => (
              <MemoryCard
                key={memory.id}
                memory={memory}
                isEditing={editingId === memory.id}
                editState={editState}
                categories={categories}
                onOpen={() => openEditor(memory)}
                onCancel={cancelEdit}
                onSave={() => saveEdit(memory)}
                onChangeText={(t) => setEditState((s) => ({ ...s, text: t }))}
                onChangeCategory={(c) => setEditState((s) => ({ ...s, category: c }))}
              />
            ))}
          </div>
        )}
      </div>

      {/* Pagination */}
      {totalPages > 1 && (
        <div className="flex-none flex items-center justify-center gap-2 px-6 py-3 border-t border-border">
          <button
            disabled={currentPage === 1}
            onClick={() => setPage((p) => p - 1)}
            className="px-3 py-1 text-sm rounded bg-surface border border-border text-text-primary disabled:opacity-40 hover:bg-surface-raised transition-colors"
          >
            ← Prev
          </button>
          <span className="text-sm text-text-secondary">
            {currentPage} / {totalPages}
          </span>
          <button
            disabled={currentPage === totalPages}
            onClick={() => setPage((p) => p + 1)}
            className="px-3 py-1 text-sm rounded bg-surface border border-border text-text-primary disabled:opacity-40 hover:bg-surface-raised transition-colors"
          >
            Next →
          </button>
        </div>
      )}
    </div>
  )
}

interface MemoryCardProps {
  memory: Memory
  isEditing: boolean
  editState: EditState
  categories: string[]
  onOpen: () => void
  onCancel: () => void
  onSave: () => void
  onChangeText: (text: string) => void
  onChangeCategory: (category: string) => void
}

function MemoryCard({
  memory,
  isEditing,
  editState,
  categories,
  onOpen,
  onCancel,
  onSave,
  onChangeText,
  onChangeCategory
}: MemoryCardProps): React.ReactElement {
  const textareaRef = useRef<HTMLTextAreaElement>(null)
  const updatedLabel = relativeDate(memory.updatedAt)
  const createdLabel = relativeDate(memory.createdAt)
  const dateLabel = memory.updatedAt !== memory.createdAt ? `Updated ${updatedLabel}` : `Added ${createdLabel}`

  // Focus textarea when entering edit mode
  useEffect(() => {
    if (isEditing && textareaRef.current) {
      textareaRef.current.focus()
      textareaRef.current.selectionStart = textareaRef.current.value.length
    }
  }, [isEditing])

  if (isEditing) {
    return (
      <div className="bg-surface border border-accent/60 rounded-lg p-4">
        <textarea
          ref={textareaRef}
          value={editState.text}
          onChange={(e) => onChangeText(e.target.value)}
          rows={4}
          className="w-full bg-bg border border-border rounded-lg px-3 py-2 text-sm text-text-primary resize-none focus:outline-none focus:ring-2 focus:ring-accent"
        />
        <div className="flex items-center gap-3 mt-3">
          {/* Category: select from existing + allow typing a new one */}
          <input
            list="category-options"
            value={editState.category}
            onChange={(e) => onChangeCategory(e.target.value)}
            placeholder="Category"
            className="bg-bg border border-border rounded-lg px-3 py-1.5 text-sm text-text-primary focus:outline-none focus:ring-2 focus:ring-accent w-40"
          />
          <datalist id="category-options">
            {categories.map((cat) => (
              <option key={cat} value={cat} />
            ))}
          </datalist>
          <div className="flex-1" />
          <button
            onClick={onCancel}
            className="px-3 py-1.5 text-sm rounded-lg bg-surface border border-border text-text-secondary hover:text-text-primary hover:bg-surface-raised transition-colors"
          >
            Cancel
          </button>
          <button
            onClick={onSave}
            disabled={!editState.text.trim()}
            className="px-3 py-1.5 text-sm rounded-lg bg-accent text-white hover:bg-accent/90 disabled:opacity-40 transition-colors"
          >
            Save
          </button>
        </div>
      </div>
    )
  }

  return (
    <div
      role="button"
      tabIndex={0}
      onClick={onOpen}
      onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); onOpen() } }}
      className="bg-surface border border-border rounded-lg p-4 hover:border-accent/40 transition-colors cursor-pointer text-left"
    >
      <p className="text-text-primary text-sm leading-relaxed line-clamp-2">{memory.text}</p>
      <div className="flex items-center gap-2 mt-2">
        <Badge variant="default">{memory.category}</Badge>
        <span className="text-xs text-text-secondary">{dateLabel}</span>
      </div>
    </div>
  )
}
