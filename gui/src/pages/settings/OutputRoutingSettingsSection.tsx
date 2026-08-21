import React, { useCallback, useEffect, useMemo, useState } from 'react'
import { useToast } from '../../components/ui/Toast'
import type { AudioTargetInfo, Settings, SpeakerGroupInfo } from '../../types/ipc'
import type {
  MediaAccountInfo,
  OutputRoutingPolicy,
  OutputRoutingResponse,
  OutputRoutingRule,
  RoutingFallbackMode
} from '../../types/outputRouting'

const WEEKDAYS = [
  ['M', 0],
  ['T', 1],
  ['W', 2],
  ['T', 3],
  ['F', 4],
  ['S', 5],
  ['S', 6]
] as const

const OUTPUT_ROWS = [
  {
    label: 'Spoken responses',
    kind: 'spoken_response',
    target: 'spoken_response_target_id',
    fallback: 'spoken_response_fallback',
    fallbackTarget: 'spoken_response_fallback_target_id',
    volume: 'spoken_response_volume'
  },
  {
    label: 'Timers',
    kind: 'timer',
    target: 'timer_target_id',
    fallback: 'timer_fallback',
    fallbackTarget: 'timer_fallback_target_id',
    volume: 'timer_volume'
  },
  {
    label: 'Alarms',
    kind: 'alarm',
    target: 'alarm_target_id',
    fallback: 'alarm_fallback',
    fallbackTarget: 'alarm_fallback_target_id',
    volume: 'alarm_volume'
  },
  {
    label: 'Media',
    kind: 'media',
    target: 'media_target_id',
    fallback: 'media_fallback',
    fallbackTarget: 'media_fallback_target_id',
    volume: 'media_volume'
  }
] as const

type OutputRow = (typeof OUTPUT_ROWS)[number]
type TargetField = OutputRow['target']
type FallbackField = OutputRow['fallback']
type FallbackTargetField = OutputRow['fallbackTarget']
type VolumeField = OutputRow['volume']

interface GroupDraft {
  name: string
  memberIds: string[]
}

async function getRoutingPolicy(): Promise<OutputRoutingResponse> {
  return (await window.rex.getSettings('outputRouting')) as unknown as OutputRoutingResponse
}

async function getMediaAccounts(): Promise<OutputRoutingResponse> {
  return (await window.rex.getSettings('outputRoutingAccounts')) as unknown as OutputRoutingResponse
}

async function saveRoutingPolicy(policy: OutputRoutingPolicy): Promise<OutputRoutingResponse> {
  const result = await window.rex.setSettings('outputRouting', policy as unknown as Settings)
  return result.ok ? { ok: true, policy } : { ok: false, error: result.error }
}

async function testRoutingTarget(targetId: string): Promise<OutputRoutingResponse> {
  const result = await window.rex.setSettings('outputRoutingTest', {
    target_id: targetId
  } as Settings)
  return result.ok ? { ok: true, target_id: targetId } : { ok: false, error: result.error }
}

function targetLabel(target: AudioTargetInfo): string {
  const location = target.room ?? target.kind
  const status = target.online ? target.health : 'offline'
  return `${target.name} · ${location} · ${status}`
}

function toggleDay(days: number[], day: number): number[] {
  return days.includes(day) ? days.filter((value) => value !== day) : [...days, day].sort()
}

export function OutputRoutingSettingsSection(): React.ReactElement {
  const addToast = useToast()
  const [policy, setPolicy] = useState<OutputRoutingPolicy | null>(null)
  const [targets, setTargets] = useState<AudioTargetInfo[]>([])
  const [accounts, setAccounts] = useState<MediaAccountInfo[]>([])
  const [groups, setGroups] = useState<SpeakerGroupInfo[]>([])
  const [groupDrafts, setGroupDrafts] = useState<Record<string, GroupDraft>>({})
  const [newGroupName, setNewGroupName] = useState('')
  const [newGroupMembers, setNewGroupMembers] = useState<string[]>([])
  const [loading, setLoading] = useState(true)
  const [saving, setSaving] = useState(false)
  const [testingTarget, setTestingTarget] = useState<string | null>(null)

  const baseTargets = useMemo(
    () => targets.filter((target) => target.kind !== 'group'),
    [targets]
  )

  const load = useCallback(async (): Promise<void> => {
    setLoading(true)
    try {
      const [policyResult, targetResult, groupResult, accountResult] = await Promise.all([
        getRoutingPolicy(),
        window.rex.getAudioTargets(),
        window.rex.listSpeakerGroups(),
        getMediaAccounts()
      ])
      if (!policyResult.ok || !policyResult.policy) {
        throw new Error(policyResult.error ?? 'Routing policy is unavailable')
      }
      if (!targetResult.ok) throw new Error(targetResult.error ?? 'Audio targets are unavailable')
      if (!groupResult.ok) throw new Error(groupResult.error ?? 'Speaker groups are unavailable')
      if (!accountResult.ok) throw new Error(accountResult.error ?? 'Media accounts are unavailable')

      setPolicy(policyResult.policy)
      setTargets(targetResult.targets ?? [])
      setGroups(groupResult.groups ?? [])
      setAccounts(accountResult.accounts ?? [])
      setGroupDrafts(
        Object.fromEntries(
          (groupResult.groups ?? []).map((group) => [
            group.id,
            { name: group.name, memberIds: [...group.member_ids] }
          ])
        )
      )
    } catch (error) {
      addToast(`Output routing: ${String(error)}`, 'error')
    } finally {
      setLoading(false)
    }
  }, [addToast])

  useEffect(() => {
    void load()
  }, [load])

  function setField<K extends keyof OutputRoutingPolicy>(
    field: K,
    value: OutputRoutingPolicy[K]
  ): void {
    setPolicy((current) => (current ? { ...current, [field]: value } : current))
  }

  function updateRule(index: number, patch: Partial<OutputRoutingRule>): void {
    if (!policy) return
    setField(
      'rules',
      policy.rules.map((rule, ruleIndex) =>
        ruleIndex === index ? { ...rule, ...patch } : rule
      )
    )
  }

  function addRule(): void {
    const firstTarget = targets[0]?.id
    if (!policy || !firstTarget) return
    setField('rules', [
      ...policy.rules,
      {
        output_kind: 'media',
        target_id: firstTarget,
        days_of_week: [],
        start_local_time: null,
        end_local_time: null,
        target_volume: null,
        fallback_mode: null,
        fallback_target_id: null
      }
    ])
  }

  async function savePolicy(): Promise<void> {
    if (!policy) return
    setSaving(true)
    try {
      const result = await saveRoutingPolicy(policy)
      if (!result.ok || !result.policy) throw new Error(result.error ?? 'Save failed')
      setPolicy(result.policy)
      addToast('Output routing saved', 'success')
    } catch (error) {
      addToast(`Failed to save output routing: ${String(error)}`, 'error')
    } finally {
      setSaving(false)
    }
  }

  async function testTarget(targetId: string): Promise<void> {
    setTestingTarget(targetId)
    try {
      const result = await testRoutingTarget(targetId)
      if (!result.ok) throw new Error(result.error ?? 'Test playback failed')
      addToast('Test playback sent', 'success')
    } catch (error) {
      addToast(String(error), 'error')
    } finally {
      setTestingTarget(null)
    }
  }

  async function createGroup(): Promise<void> {
    const name = newGroupName.trim()
    if (!name || newGroupMembers.length === 0) return
    const result = await window.rex.createSpeakerGroup(name, newGroupMembers)
    if (!result.ok) {
      addToast(result.error ?? 'Failed to create speaker group', 'error')
      return
    }
    setNewGroupName('')
    setNewGroupMembers([])
    await load()
  }

  async function saveGroup(group: SpeakerGroupInfo): Promise<void> {
    const draft = groupDrafts[group.id]
    if (!draft || !draft.name.trim() || draft.memberIds.length === 0) return
    if (draft.name.trim() !== group.name) {
      const renamed = await window.rex.renameSpeakerGroup(group.id, draft.name.trim())
      if (!renamed.ok) {
        addToast(renamed.error ?? 'Failed to rename group', 'error')
        return
      }
    }
    const changed = await window.rex.setSpeakerGroupMembers(group.id, draft.memberIds)
    if (!changed.ok) {
      addToast(changed.error ?? 'Failed to update group members', 'error')
      return
    }
    await load()
  }

  async function deleteGroup(groupId: string): Promise<void> {
    const result = await window.rex.deleteSpeakerGroup(groupId)
    if (!result.ok) {
      addToast(result.error ?? 'Failed to delete group', 'error')
      return
    }
    await load()
  }

  if (loading && !policy) {
    return <div className="px-6 pb-8 text-sm text-text-secondary">Loading routing policy…</div>
  }
  if (!policy) {
    return <div className="px-6 pb-8 text-sm text-text-secondary">Output routing is unavailable.</div>
  }

  return (
    <section className="px-6 pb-10 max-w-3xl">
      <div className="border-t border-border pt-6">
        <div className="mb-5 flex items-start justify-between gap-4">
          <div>
            <h3 className="text-base font-semibold text-text-primary">Per-user routing</h3>
            <p className="mt-1 text-xs text-text-secondary">
              These defaults belong to this Rex profile. A target named in the request still wins.
            </p>
          </div>
          <button
            type="button"
            onClick={() => void savePolicy()}
            disabled={saving}
            className="rounded-lg bg-accent px-4 py-2 text-sm font-medium text-white disabled:opacity-50"
          >
            {saving ? 'Saving…' : 'Save routing'}
          </button>
        </div>

        <div className="space-y-4">
          {OUTPUT_ROWS.map((row) => {
            const targetField = row.target as TargetField
            const fallbackField = row.fallback as FallbackField
            const fallbackTargetField = row.fallbackTarget as FallbackTargetField
            const volumeField = row.volume as VolumeField
            const fallbackMode = policy[fallbackField] as RoutingFallbackMode
            return (
              <div
                key={row.kind}
                className="rounded-xl border border-border bg-surface-raised p-4"
              >
                <div className="mb-3 text-sm font-semibold text-text-primary">{row.label}</div>
                <div className="grid gap-3 md:grid-cols-3">
                  <label className="text-xs text-text-secondary">
                    Default target
                    <select
                      aria-label={`${row.label} default target`}
                      value={(policy[targetField] as string | null) ?? ''}
                      onChange={(event) => setField(targetField, event.target.value || null)}
                      className="mt-1 w-full rounded-lg border border-border bg-bg px-2 py-2 text-sm text-text-primary"
                    >
                      <option value="">No stored default</option>
                      {targets.map((target) => (
                        <option key={target.id} value={target.id}>
                          {targetLabel(target)}
                        </option>
                      ))}
                    </select>
                  </label>

                  <label className="text-xs text-text-secondary">
                    If unavailable
                    <select
                      aria-label={`${row.label} fallback mode`}
                      value={fallbackMode}
                      onChange={(event) => {
                        const mode = event.target.value as RoutingFallbackMode
                        setPolicy((current) =>
                          current
                            ? {
                                ...current,
                                [fallbackField]: mode,
                                [fallbackTargetField]:
                                  mode === 'named' ? current[fallbackTargetField] : null
                              }
                            : current
                        )
                      }}
                      className="mt-1 w-full rounded-lg border border-border bg-bg px-2 py-2 text-sm text-text-primary"
                    >
                      <option value="none">Do not silently reroute</option>
                      <option value="ask">Ask me</option>
                      <option value="named">Use another target</option>
                    </select>
                  </label>

                  <label className="text-xs text-text-secondary">
                    Target volume
                    <input
                      aria-label={`${row.label} target volume`}
                      type="number"
                      min={0}
                      max={100}
                      value={(policy[volumeField] as number | null) ?? ''}
                      placeholder="Keep current"
                      onChange={(event) =>
                        setField(
                          volumeField,
                          event.target.value === '' ? null : Number(event.target.value)
                        )
                      }
                      className="mt-1 w-full rounded-lg border border-border bg-bg px-2 py-2 text-sm text-text-primary"
                    />
                  </label>
                </div>

                {fallbackMode === 'named' && (
                  <label className="mt-3 block text-xs text-text-secondary">
                    Fallback target
                    <select
                      aria-label={`${row.label} fallback target`}
                      value={(policy[fallbackTargetField] as string | null) ?? ''}
                      onChange={(event) =>
                        setField(fallbackTargetField, event.target.value || null)
                      }
                      className="mt-1 w-full rounded-lg border border-border bg-bg px-2 py-2 text-sm text-text-primary"
                    >
                      <option value="">Choose a target</option>
                      {targets.map((target) => (
                        <option key={target.id} value={target.id}>
                          {targetLabel(target)}
                        </option>
                      ))}
                    </select>
                  </label>
                )}
              </div>
            )
          })}
        </div>

        <div className="mt-5 space-y-4 rounded-xl border border-border bg-surface-raised p-4">
          <label className="flex items-center justify-between gap-3 text-sm text-text-primary">
            Prefer the endpoint that received an interactive media request
            <input
              type="checkbox"
              checked={policy.prefer_media_request_origin}
              onChange={(event) => setField('prefer_media_request_origin', event.target.checked)}
            />
          </label>

          <label className="block text-xs text-text-secondary">
            Default media account
            <select
              aria-label="Default media account"
              value={
                policy.default_media_provider && policy.default_media_account_id
                  ? `${policy.default_media_provider}::${policy.default_media_account_id}`
                  : ''
              }
              onChange={(event) => {
                const [provider, accountId] = event.target.value.split('::')
                setPolicy((current) =>
                  current
                    ? {
                        ...current,
                        default_media_provider: provider || null,
                        default_media_account_id: accountId || null
                      }
                    : current
                )
              }}
              className="mt-1 w-full rounded-lg border border-border bg-bg px-2 py-2 text-sm text-text-primary"
            >
              <option value="">No account default</option>
              {accounts.map((account) => (
                <option
                  key={`${account.provider}:${account.account_id}`}
                  value={`${account.provider}::${account.account_id}`}
                >
                  {account.display_name} · {account.provider}
                </option>
              ))}
            </select>
          </label>

          <div>
            <label className="flex items-center gap-2 text-sm text-text-primary">
              <input
                type="checkbox"
                checked={policy.quiet_hours.enabled}
                onChange={(event) =>
                  setField('quiet_hours', {
                    ...policy.quiet_hours,
                    enabled: event.target.checked
                  })
                }
              />
              Quiet hours for optional spoken/media output
            </label>
            {policy.quiet_hours.enabled && (
              <div className="mt-3 space-y-3">
                <div className="grid gap-3 md:grid-cols-2">
                  <label className="text-xs text-text-secondary">
                    Start
                    <input
                      type="time"
                      value={policy.quiet_hours.start_local_time.slice(0, 5)}
                      onChange={(event) =>
                        setField('quiet_hours', {
                          ...policy.quiet_hours,
                          start_local_time: event.target.value
                        })
                      }
                      className="mt-1 w-full rounded-lg border border-border bg-bg px-2 py-2 text-sm text-text-primary"
                    />
                  </label>
                  <label className="text-xs text-text-secondary">
                    End
                    <input
                      type="time"
                      value={policy.quiet_hours.end_local_time.slice(0, 5)}
                      onChange={(event) =>
                        setField('quiet_hours', {
                          ...policy.quiet_hours,
                          end_local_time: event.target.value
                        })
                      }
                      className="mt-1 w-full rounded-lg border border-border bg-bg px-2 py-2 text-sm text-text-primary"
                    />
                  </label>
                </div>
                <div className="flex flex-wrap gap-3">
                  {WEEKDAYS.map(([label, day], index) => (
                    <label
                      key={`${label}-${index}`}
                      className="flex items-center gap-1 text-xs text-text-secondary"
                    >
                      <input
                        type="checkbox"
                        checked={policy.quiet_hours.days_of_week.includes(day)}
                        onChange={() =>
                          setField('quiet_hours', {
                            ...policy.quiet_hours,
                            days_of_week: toggleDay(policy.quiet_hours.days_of_week, day)
                          })
                        }
                      />
                      {label}
                    </label>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>

        <div className="mt-5 rounded-xl border border-border bg-surface-raised p-4">
          <div className="flex items-center justify-between gap-3">
            <div>
              <h4 className="text-sm font-semibold text-text-primary">Conditional routing</h4>
              <p className="mt-1 text-xs text-text-secondary">
                Matching day/time rules are checked before the stored default.
              </p>
            </div>
            <button
              type="button"
              onClick={addRule}
              disabled={targets.length === 0}
              className="rounded-lg border border-border px-3 py-1.5 text-xs text-text-primary disabled:opacity-50"
            >
              Add rule
            </button>
          </div>

          <div className="mt-4 space-y-3">
            {policy.rules.map((rule, index) => (
              <div
                key={`${rule.output_kind}-${index}`}
                className="rounded-lg border border-border bg-bg p-3"
              >
                <div className="grid gap-2 md:grid-cols-2">
                  <select
                    aria-label={`Rule ${index + 1} output kind`}
                    value={rule.output_kind}
                    onChange={(event) =>
                      updateRule(index, {
                        output_kind: event.target.value as OutputRoutingRule['output_kind']
                      })
                    }
                    className="rounded-lg border border-border bg-surface-raised px-2 py-2 text-sm text-text-primary"
                  >
                    <option value="spoken_response">Spoken responses</option>
                    <option value="timer">Timers</option>
                    <option value="alarm">Alarms</option>
                    <option value="media">Media</option>
                  </select>
                  <select
                    aria-label={`Rule ${index + 1} target`}
                    value={rule.target_id}
                    onChange={(event) => updateRule(index, { target_id: event.target.value })}
                    className="rounded-lg border border-border bg-surface-raised px-2 py-2 text-sm text-text-primary"
                  >
                    {targets.map((target) => (
                      <option key={target.id} value={target.id}>
                        {targetLabel(target)}
                      </option>
                    ))}
                  </select>
                </div>

                <div className="mt-3 flex flex-wrap items-center gap-2">
                  {WEEKDAYS.map(([label, day], dayIndex) => (
                    <label
                      key={`${label}-${dayIndex}`}
                      className="flex items-center gap-1 text-xs text-text-secondary"
                    >
                      <input
                        type="checkbox"
                        checked={rule.days_of_week.includes(day)}
                        onChange={() =>
                          updateRule(index, { days_of_week: toggleDay(rule.days_of_week, day) })
                        }
                      />
                      {label}
                    </label>
                  ))}
                  <input
                    aria-label={`Rule ${index + 1} start time`}
                    type="time"
                    value={rule.start_local_time?.slice(0, 5) ?? ''}
                    onChange={(event) =>
                      updateRule(index, {
                        start_local_time: event.target.value || null,
                        end_local_time: event.target.value
                          ? rule.end_local_time ?? '23:59'
                          : null
                      })
                    }
                    className="rounded border border-border bg-surface-raised px-2 py-1 text-xs text-text-primary"
                  />
                  <input
                    aria-label={`Rule ${index + 1} end time`}
                    type="time"
                    value={rule.end_local_time?.slice(0, 5) ?? ''}
                    onChange={(event) =>
                      updateRule(index, {
                        end_local_time: event.target.value || null,
                        start_local_time: event.target.value
                          ? rule.start_local_time ?? '00:00'
                          : null
                      })
                    }
                    className="rounded border border-border bg-surface-raised px-2 py-1 text-xs text-text-primary"
                  />
                  <input
                    aria-label={`Rule ${index + 1} volume`}
                    type="number"
                    min={0}
                    max={100}
                    value={rule.target_volume ?? ''}
                    placeholder="Volume"
                    onChange={(event) =>
                      updateRule(index, {
                        target_volume: event.target.value === '' ? null : Number(event.target.value)
                      })
                    }
                    className="w-20 rounded border border-border bg-surface-raised px-2 py-1 text-xs text-text-primary"
                  />
                  <select
                    aria-label={`Rule ${index + 1} fallback mode`}
                    value={rule.fallback_mode ?? ''}
                    onChange={(event) =>
                      updateRule(index, {
                        fallback_mode: event.target.value
                          ? (event.target.value as RoutingFallbackMode)
                          : null,
                        fallback_target_id:
                          event.target.value === 'named' ? rule.fallback_target_id : null
                      })
                    }
                    className="rounded border border-border bg-surface-raised px-2 py-1 text-xs text-text-primary"
                  >
                    <option value="">Use default fallback</option>
                    <option value="none">No fallback</option>
                    <option value="ask">Ask</option>
                    <option value="named">Named target</option>
                  </select>
                  {rule.fallback_mode === 'named' && (
                    <select
                      aria-label={`Rule ${index + 1} fallback target`}
                      value={rule.fallback_target_id ?? ''}
                      onChange={(event) =>
                        updateRule(index, { fallback_target_id: event.target.value || null })
                      }
                      className="rounded border border-border bg-surface-raised px-2 py-1 text-xs text-text-primary"
                    >
                      <option value="">Choose fallback</option>
                      {targets.map((target) => (
                        <option key={target.id} value={target.id}>
                          {target.name}
                        </option>
                      ))}
                    </select>
                  )}
                  <button
                    type="button"
                    onClick={() =>
                      setField(
                        'rules',
                        policy.rules.filter((_, ruleIndex) => ruleIndex !== index)
                      )
                    }
                    className="ml-auto text-xs text-danger"
                  >
                    Remove
                  </button>
                </div>
              </div>
            ))}
            {policy.rules.length === 0 && (
              <p className="text-xs text-text-secondary">No conditional routing rules.</p>
            )}
          </div>
        </div>

        <div className="mt-5 rounded-xl border border-border bg-surface-raised p-4">
          <h4 className="text-sm font-semibold text-text-primary">Targets and test playback</h4>
          <div className="mt-3 space-y-2">
            {targets.map((target) => (
              <div
                key={target.id}
                className="flex items-center justify-between gap-3 rounded-lg border border-border bg-bg p-3"
              >
                <div className="min-w-0">
                  <div className="truncate text-sm text-text-primary">{target.name}</div>
                  <div className="text-xs text-text-secondary">
                    {target.provider} · {target.online ? target.health : 'offline'}
                  </div>
                </div>
                <button
                  type="button"
                  disabled={!target.online || testingTarget === target.id}
                  onClick={() => void testTarget(target.id)}
                  className="rounded-lg border border-border px-3 py-1.5 text-xs text-text-primary disabled:opacity-50"
                >
                  {testingTarget === target.id ? 'Testing…' : 'Test'}
                </button>
              </div>
            ))}
          </div>
        </div>

        <div className="mt-5 rounded-xl border border-border bg-surface-raised p-4">
          <h4 className="text-sm font-semibold text-text-primary">Speaker groups</h4>
          <div className="mt-3 rounded-lg border border-border bg-bg p-3">
            <input
              value={newGroupName}
              onChange={(event) => setNewGroupName(event.target.value)}
              placeholder="New group name"
              className="w-full rounded-lg border border-border bg-surface-raised px-2 py-2 text-sm text-text-primary"
            />
            <div className="mt-2 flex flex-wrap gap-3">
              {baseTargets.map((target) => (
                <label
                  key={target.id}
                  className="flex items-center gap-1 text-xs text-text-secondary"
                >
                  <input
                    type="checkbox"
                    checked={newGroupMembers.includes(target.id)}
                    onChange={(event) =>
                      setNewGroupMembers((current) =>
                        event.target.checked
                          ? [...current, target.id]
                          : current.filter((id) => id !== target.id)
                      )
                    }
                  />
                  {target.name}
                </label>
              ))}
            </div>
            <button
              type="button"
              onClick={() => void createGroup()}
              disabled={!newGroupName.trim() || newGroupMembers.length === 0}
              className="mt-3 rounded-lg border border-border px-3 py-1.5 text-xs text-text-primary disabled:opacity-50"
            >
              Create group
            </button>
          </div>

          <div className="mt-3 space-y-3">
            {groups.map((group) => {
              const draft = groupDrafts[group.id] ?? {
                name: group.name,
                memberIds: [...group.member_ids]
              }
              return (
                <div key={group.id} className="rounded-lg border border-border bg-bg p-3">
                  <input
                    aria-label={`${group.name} group name`}
                    value={draft.name}
                    onChange={(event) =>
                      setGroupDrafts((current) => ({
                        ...current,
                        [group.id]: { ...draft, name: event.target.value }
                      }))
                    }
                    className="w-full rounded-lg border border-border bg-surface-raised px-2 py-2 text-sm text-text-primary"
                  />
                  <div className="mt-2 flex flex-wrap gap-3">
                    {baseTargets.map((target) => (
                      <label
                        key={target.id}
                        className="flex items-center gap-1 text-xs text-text-secondary"
                      >
                        <input
                          type="checkbox"
                          checked={draft.memberIds.includes(target.id)}
                          onChange={(event) =>
                            setGroupDrafts((current) => ({
                              ...current,
                              [group.id]: {
                                ...draft,
                                memberIds: event.target.checked
                                  ? [...draft.memberIds, target.id]
                                  : draft.memberIds.filter((id) => id !== target.id)
                              }
                            }))
                          }
                        />
                        {target.name}
                      </label>
                    ))}
                  </div>
                  <div className="mt-3 flex gap-2">
                    <button
                      type="button"
                      onClick={() => void saveGroup(group)}
                      className="rounded-lg border border-border px-3 py-1.5 text-xs text-text-primary"
                    >
                      Save group
                    </button>
                    <button
                      type="button"
                      onClick={() => void deleteGroup(group.id)}
                      className="rounded-lg border border-danger/40 px-3 py-1.5 text-xs text-danger"
                    >
                      Delete
                    </button>
                  </div>
                </div>
              )
            })}
          </div>
        </div>
      </div>
    </section>
  )
}
