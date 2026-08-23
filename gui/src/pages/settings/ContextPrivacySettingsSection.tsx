import React, { useCallback, useEffect, useMemo, useState } from 'react'
import { useToast } from '../../components/ui/Toast'
import type { ContextPrivacyResponse } from '../../types/ipc'

type PrivacyCommand =
  | 'set_source_context'
  | 'update_upload_policy'
  | 'set_location_assist'
  | 'set_location_share'
  | 'set_proactive_assistance'

function sourceLabel(sourceId: string): string {
  const raw = sourceId.split(':').at(-1) ?? sourceId
  return raw
    .replaceAll('_', ' ')
    .replaceAll('-', ' ')
    .replace(/\b\w/g, (letter) => letter.toUpperCase())
}

export function ContextPrivacySettingsSection(): React.ReactElement {
  const addToast = useToast()
  const [state, setState] = useState<ContextPrivacyResponse | null>(null)
  const [loading, setLoading] = useState(true)
  const [busy, setBusy] = useState(false)
  const [shareRecipient, setShareRecipient] = useState('')

  const load = useCallback(async (): Promise<void> => {
    setLoading(true)
    try {
      const result = await window.rex.getContextPrivacy()
      if (!result.ok) throw new Error(result.error ?? 'Privacy settings are unavailable')
      setState(result)
    } catch (error) {
      addToast(`Context & Privacy: ${String(error)}`, 'error')
    } finally {
      setLoading(false)
    }
  }, [addToast])

  useEffect(() => {
    void load()
  }, [load])

  const connectedSources = useMemo(
    () =>
      (state?.sources ?? []).filter(
        (source) => source.source_type !== 'upload' && source.source_type !== 'location'
      ),
    [state]
  )

  async function mutate(command: PrivacyCommand, payload: Record<string, unknown>): Promise<void> {
    setBusy(true)
    try {
      const result = await window.rex.updateContextPrivacy(command, payload)
      if (!result.ok) throw new Error(result.error ?? 'Privacy setting could not be changed')
      await load()
    } catch (error) {
      addToast(`Privacy setting: ${String(error)}`, 'error')
    } finally {
      setBusy(false)
    }
  }

  if (loading && !state) {
    return <div className="p-6 text-sm text-text-secondary">Loading privacy settings…</div>
  }
  if (!state) {
    return <div className="p-6 text-sm text-text-secondary">Privacy settings are unavailable.</div>
  }

  return (
    <section className="max-w-3xl space-y-6 p-6">
      <div>
        <h2 className="text-lg font-semibold text-text-primary">Context & Privacy</h2>
        <p className="mt-1 text-sm text-text-secondary">
          Control what Rex may use to help you. These choices belong to your profile only.
        </p>
      </div>

      <div className="rounded-xl border border-border bg-surface-raised p-4">
        <div className="flex items-start justify-between gap-4">
          <div>
            <h3 className="text-sm font-semibold text-text-primary">Proactive assistance</h3>
            <p className="mt-1 text-xs text-text-secondary">
              Let Rex notice useful next steps from context you have already allowed.
            </p>
          </div>
          <input
            aria-label="Proactive assistance"
            type="checkbox"
            checked={Boolean(state.proactive_assistance)}
            disabled={busy}
            onChange={(event) =>
              void mutate('set_proactive_assistance', { enabled: event.target.checked })
            }
          />
        </div>
      </div>
      <div className="rounded-xl border border-border bg-surface-raised p-4">
        <h3 className="text-sm font-semibold text-text-primary">Connected context</h3>
        <p className="mt-1 text-xs text-text-secondary">
          Choose which connected sources Rex may use in ordinary future conversations.
        </p>
        <div className="mt-4 space-y-3">
          {connectedSources.length === 0 && (
            <div className="text-xs text-text-secondary">No connected context sources yet.</div>
          )}
          {connectedSources.map((source) => (
            <label key={source.source_id} className="flex items-center justify-between gap-4">
              <span>
                <span className="block text-sm text-text-primary">{sourceLabel(source.source_id)}</span>
                <span className="block text-xs text-text-secondary">Connected {source.source_type}</span>
              </span>
              <span className="flex items-center gap-2 text-xs text-text-secondary">
                Use this in future conversations
                <input
                  type="checkbox"
                  checked={source.context_enabled}
                  disabled={busy || !source.mutable}
                  onChange={(event) =>
                    void mutate('set_source_context', {
                      source_id: source.source_id,
                      enabled: event.target.checked
                    })
                  }
                />
              </span>
            </label>
          ))}
        </div>
      </div>
      <div className="rounded-xl border border-border bg-surface-raised p-4">
        <h3 className="text-sm font-semibold text-text-primary">Uploaded information</h3>
        <p className="mt-1 text-xs text-text-secondary">
          Decide whether each file may shape future context and who may use it.
        </p>
        <div className="mt-4 space-y-4">
          {(state.uploads ?? []).length === 0 && (
            <div className="text-xs text-text-secondary">No owned uploads yet.</div>
          )}
          {(state.uploads ?? []).map((upload) => (
            <div key={upload.doc_id} className="rounded-lg border border-border p-3">
              <div className="text-sm font-medium text-text-primary">{upload.title}</div>
              <div className="mt-3 grid gap-3 sm:grid-cols-2">
                <label className="text-xs text-text-secondary">
                  Who can use this
                  <select
                    value={upload.audience_scope}
                    disabled={busy}
                    onChange={(event) =>
                      void mutate('update_upload_policy', {
                        doc_id: upload.doc_id,
                        audience_scope: event.target.value,
                        context_enabled: upload.context_enabled
                      })
                    }
                    className="mt-1 w-full rounded-lg border border-border bg-bg px-2 py-2"
                  >
                    <option value="private">Private to me</option>
                    <option value="household">Shared household</option>
                  </select>
                </label>
                <label className="flex items-center gap-2 text-xs text-text-secondary sm:self-end sm:pb-2">
                  <input
                    type="checkbox"
                    checked={upload.context_enabled}
                    disabled={busy}
                    onChange={(event) =>
                      void mutate('update_upload_policy', {
                        doc_id: upload.doc_id,
                        audience_scope: upload.audience_scope,
                        context_enabled: event.target.checked
                      })
                    }
                  />
                  Use this in future conversations
                </label>
              </div>
            </div>
          ))}
        </div>
      </div>

      <div className="rounded-xl border border-border bg-surface-raised p-4">
        <h3 className="text-sm font-semibold text-text-primary">Location</h3>
        <label className="mt-3 flex items-center justify-between gap-4 text-sm text-text-primary">
          <span>
            Use my location to help me
            <span className="mt-1 block text-xs text-text-secondary">
              Used only when a task or enabled proactive rule needs it.
            </span>
          </span>
          <input
            aria-label="Use my location to help me"
            type="checkbox"
            checked={Boolean(state.location?.location_assist)}
            disabled={busy}
            onChange={(event) =>
              void mutate('set_location_assist', { enabled: event.target.checked })
            }
          />
        </label>

        <div className="mt-5 border-t border-border pt-4">
          <div className="text-sm font-medium text-text-primary">Share my location with</div>
          <p className="mt-1 text-xs text-text-secondary">
            This is separate from location assistance. Add only a person you want Rex to tell.
          </p>
          <div className="mt-3 flex gap-2">
            <input
              aria-label="Location share recipient"
              value={shareRecipient}
              onChange={(event) => setShareRecipient(event.target.value)}
              placeholder="User ID"
              className="min-w-0 flex-1 rounded-lg border border-border bg-bg px-3 py-2 text-sm"
            />
            <button
              type="button"
              disabled={busy || !shareRecipient.trim()}
              onClick={() => {
                const recipient = shareRecipient.trim()
                setShareRecipient('')
                void mutate('set_location_share', {
                  recipient_user_id: recipient,
                  enabled: true
                })
              }}
              className="rounded-lg bg-accent px-3 py-2 text-sm font-medium text-white disabled:opacity-50"
            >
              Add
            </button>
          </div>
          <div className="mt-3 flex flex-wrap gap-2">
            {(state.location?.shared_with ?? []).map((recipient) => (
              <button
                type="button"
                key={recipient}
                disabled={busy}
                onClick={() =>
                  void mutate('set_location_share', {
                    recipient_user_id: recipient,
                    enabled: false
                  })
                }
                className="rounded-full border border-border px-3 py-1 text-xs text-text-secondary"
              >
                {recipient} ×
              </button>
            ))}
          </div>
        </div>
      </div>
    </section>
  )
}
