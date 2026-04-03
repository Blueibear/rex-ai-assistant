import React, { useEffect, useRef, useState } from 'react'
import type { ShoppingItem } from '../types/ipc'

export function ShoppingListPage(): React.ReactElement {
  const [items, setItems] = useState<ShoppingItem[]>([])
  const [newName, setNewName] = useState('')
  const [newQty, setNewQty] = useState('1')
  const [newUnit, setNewUnit] = useState('')
  const [adding, setAdding] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null)

  const loadItems = (): void => {
    window.rex
      .getShoppingItems()
      .then((res) => {
        if (res.ok) setItems(res.items)
      })
      .catch(() => {})
  }

  useEffect(() => {
    loadItems()
    pollRef.current = setInterval(loadItems, 5000)
    return () => {
      if (pollRef.current !== null) clearInterval(pollRef.current)
    }
  }, [])

  const handleToggle = (item: ShoppingItem): void => {
    const op = item.checked
      ? window.rex.uncheckShoppingItem(item.id)
      : window.rex.checkShoppingItem(item.id)
    op.then(() => loadItems()).catch(() => {})
  }

  const handleAdd = (): void => {
    const name = newName.trim()
    if (!name) return
    const qty = parseFloat(newQty) || 1
    setAdding(true)
    setError(null)
    window.rex
      .addShoppingItem(name, qty, newUnit.trim())
      .then((res) => {
        if (res.ok) {
          setNewName('')
          setNewQty('1')
          setNewUnit('')
          loadItems()
        } else {
          setError(res.error ?? 'Failed to add item')
        }
      })
      .catch((err: unknown) => {
        setError(err instanceof Error ? err.message : 'Failed to add item')
      })
      .finally(() => setAdding(false))
  }

  const handleClearChecked = (): void => {
    window.rex
      .clearCheckedShoppingItems()
      .then(() => loadItems())
      .catch(() => {})
  }

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>): void => {
    if (e.key === 'Enter') handleAdd()
  }

  const toBuy = items.filter((i) => !i.checked)
  const gotIt = items.filter((i) => i.checked)

  return (
    <div className="p-6 max-w-lg mx-auto">
      <h2 className="text-xl font-semibold text-text-primary mb-6">Shopping List</h2>

      {/* Add item row */}
      <div className="flex gap-2 mb-6">
        <input
          type="text"
          placeholder="Item name"
          value={newName}
          onChange={(e) => setNewName(e.target.value)}
          onKeyDown={handleKeyDown}
          className="flex-1 px-3 py-2 rounded-lg bg-surface border border-border text-text-primary text-sm placeholder:text-text-secondary focus:outline-none focus:ring-2 focus:ring-accent/50"
        />
        <input
          type="number"
          placeholder="Qty"
          value={newQty}
          onChange={(e) => setNewQty(e.target.value)}
          className="w-16 px-3 py-2 rounded-lg bg-surface border border-border text-text-primary text-sm focus:outline-none focus:ring-2 focus:ring-accent/50"
          min={0.1}
          step={0.5}
        />
        <input
          type="text"
          placeholder="Unit"
          value={newUnit}
          onChange={(e) => setNewUnit(e.target.value)}
          onKeyDown={handleKeyDown}
          className="w-20 px-3 py-2 rounded-lg bg-surface border border-border text-text-primary text-sm placeholder:text-text-secondary focus:outline-none focus:ring-2 focus:ring-accent/50"
        />
        <button
          type="button"
          onClick={handleAdd}
          disabled={adding || !newName.trim()}
          className="px-4 py-2 rounded-lg bg-accent text-white text-sm font-medium hover:bg-accent/90 disabled:opacity-40 transition-colors"
        >
          Add
        </button>
      </div>

      {error && (
        <p className="mb-4 text-sm text-red-400">{error}</p>
      )}

      {/* To Buy section */}
      <section className="mb-6">
        <h3 className="text-xs font-semibold uppercase tracking-wide text-text-secondary mb-2">
          To Buy ({toBuy.length})
        </h3>
        {toBuy.length === 0 ? (
          <p className="text-sm text-text-secondary italic">Nothing to buy yet.</p>
        ) : (
          <ul className="space-y-1">
            {toBuy.map((item) => (
              <li
                key={item.id}
                className="flex items-center gap-3 px-3 py-2.5 rounded-lg bg-surface hover:bg-surface-raised transition-colors"
              >
                <input
                  type="checkbox"
                  checked={false}
                  onChange={() => handleToggle(item)}
                  className="w-4 h-4 accent-accent cursor-pointer"
                  aria-label={`Mark ${item.name} as bought`}
                />
                <span className="flex-1 text-sm text-text-primary">{item.name}</span>
                {(item.quantity !== 1 || item.unit) && (
                  <span className="text-xs text-text-secondary">
                    {item.quantity !== 1 ? item.quantity : ''} {item.unit}
                  </span>
                )}
              </li>
            ))}
          </ul>
        )}
      </section>

      {/* Got It section */}
      {gotIt.length > 0 && (
        <section>
          <div className="flex items-center justify-between mb-2">
            <h3 className="text-xs font-semibold uppercase tracking-wide text-text-secondary">
              Got It ({gotIt.length})
            </h3>
            <button
              type="button"
              onClick={handleClearChecked}
              className="text-xs text-text-secondary hover:text-text-primary transition-colors"
            >
              Clear checked
            </button>
          </div>
          <ul className="space-y-1">
            {gotIt.map((item) => (
              <li
                key={item.id}
                className="flex items-center gap-3 px-3 py-2.5 rounded-lg bg-surface hover:bg-surface-raised transition-colors"
              >
                <input
                  type="checkbox"
                  checked={true}
                  onChange={() => handleToggle(item)}
                  className="w-4 h-4 accent-accent cursor-pointer"
                  aria-label={`Unmark ${item.name}`}
                />
                <span className="flex-1 text-sm text-text-secondary line-through">{item.name}</span>
                {(item.quantity !== 1 || item.unit) && (
                  <span className="text-xs text-text-secondary">
                    {item.quantity !== 1 ? item.quantity : ''} {item.unit}
                  </span>
                )}
              </li>
            ))}
          </ul>
        </section>
      )}
    </div>
  )
}
