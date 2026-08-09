import React, { useEffect, useState } from 'react'
import type { Memory, Settings, VoiceEnrollment } from '../../types/ipc'
import { PageLoadingFallback } from '../../components/ui/PageLoadingFallback'
import { SkeletonLine } from '../../components/ui/SkeletonLine'
import { useToast } from '../../components/ui/Toast'
import { ENROLLMENT_MIN_RMS, ENROLLMENT_PROMPT_PHRASE, ENROLLMENT_SAMPLE_RATE, ENROLLMENT_SAMPLE_TARGET, captureEnrollmentSample, computeRms, runEnrollmentCountdown, sleep } from './voice/voiceHelpers'

export function UsersSettingsSection(): React.ReactElement {
  const addToast = useToast()
  const [enrollments, setEnrollments] = useState<VoiceEnrollment[]>([])
  const [activeUserId, setActiveUserId] = useState('default')
  const [userNames, setUserNames] = useState<Record<string, string>>({})
  const [editingName, setEditingName] = useState<string | null>(null)
  const [editNameValue, setEditNameValue] = useState('')
  const [selectedUserId, setSelectedUserId] = useState<string | null>(null)
  const [memories, setMemories] = useState<Memory[]>([])
  const [memoriesLoading, setMemoriesLoading] = useState(false)
  const [addingUser, setAddingUser] = useState(false)
  const [newUserId, setNewUserId] = useState('')
  const [enrollingUserId, setEnrollingUserId] = useState<string | null>(null)
  const [enrollmentCountdown, setEnrollmentCountdown] = useState(0)
  const [capturedSamples, setCapturedSamples] = useState(0)
  const [enrollmentMessage, setEnrollmentMessage] = useState<string | null>(null)
  const [enrollmentError, setEnrollmentError] = useState<string | null>(null)
  const [deletingUserId, setDeletingUserId] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)

  function loadEnrollments(): void {
    window.rex
      .getVoiceEnrollments()
      .then((result) => {
        if (!result.ok) return
        setActiveUserId(result.active_user_id || 'default')
        setEnrollments(result.enrollments ?? [])
      })
      .catch(() => {
        addToast('Failed to load voice enrollments', 'error')
      })
  }

  function loadUserNames(): void {
    window.rex
      .getSettings('users')
      .then((s: Settings) => {
        const names = s.names && typeof s.names === 'object' ? (s.names as Record<string, string>) : {}
        setUserNames(names)
      })
      .catch(() => {
        // Non-fatal; names fall back to user IDs
      })
  }

  useEffect(() => {
    Promise.all([
      new Promise<void>((resolve) => {
        loadEnrollments()
        resolve()
      }),
      new Promise<void>((resolve) => {
        loadUserNames()
        resolve()
      })
    ]).finally(() => setLoading(false))
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    if (selectedUserId === null) {
      setMemories([])
      return
    }
    setMemoriesLoading(true)
    window.rex
      .getMemories()
      .then((mems) => setMemories(mems))
      .catch(() => setMemories([]))
      .finally(() => setMemoriesLoading(false))
  }, [selectedUserId])

  function saveName(userId: string, name: string): void {
    const updated = { ...userNames, [userId]: name }
    setUserNames(updated)
    window.rex
      .setSettings('users', { names: updated } as Settings)
      .then((result) => {
        if (!result.ok) {
          addToast(result.error ?? 'Failed to save user name', 'error')
        }
      })
      .catch(() => {
        addToast('Failed to save user name', 'error')
      })
  }

  function handleStartEditName(userId: string): void {
    setEditingName(userId)
    setEditNameValue(userNames[userId] ?? userId)
  }

  function handleSaveName(userId: string): void {
    saveName(userId, editNameValue.trim() || userId)
    setEditingName(null)
  }

  async function handleEnroll(userId: string): Promise<void> {
    setEnrollingUserId(userId)
    setEnrollmentCountdown(0)
    setCapturedSamples(0)
    setEnrollmentMessage(null)
    setEnrollmentError(null)

    let stream: MediaStream | null = null
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      const samples: number[][] = []
      let i = 0
      while (i < ENROLLMENT_SAMPLE_TARGET) {
        setEnrollmentMessage(
          `Sample ${i + 1} of ${ENROLLMENT_SAMPLE_TARGET}: say the phrase below when recording starts.`
        )
        await runEnrollmentCountdown(setEnrollmentCountdown)
        const sample = await captureEnrollmentSample(stream)
        const rms = computeRms(sample)
        const durationOk = sample.length >= ENROLLMENT_SAMPLE_RATE * 0.5
        if (!durationOk) {
          setEnrollmentError('Sample too short — please hold the microphone closer and try again.')
          await sleep(1500)
          setEnrollmentError(null)
          continue
        }
        if (rms < ENROLLMENT_MIN_RMS) {
          setEnrollmentError('Too quiet — please speak louder when recording starts.')
          await sleep(1500)
          setEnrollmentError(null)
          continue
        }
        samples.push(sample)
        i += 1
        setCapturedSamples(i)
        setEnrollmentMessage(`Sample ${i} of ${ENROLLMENT_SAMPLE_TARGET} captured.`)
        await sleep(250)
      }
      const result = await window.rex.enrollVoice(userId, samples)
      if (!result.ok) throw new Error(result.error ?? 'Voice enrollment failed')
      setEnrollmentMessage(`Voice enrollment saved for ${userId}.`)
      addToast('Voice enrollment saved', 'success')
      loadEnrollments()
    } catch (err) {
      const msg = err instanceof Error ? err.message : 'Voice enrollment failed'
      setEnrollmentError(msg)
      addToast(msg, 'error')
    } finally {
      setEnrollingUserId(null)
      setEnrollmentCountdown(0)
      stream?.getTracks().forEach((t) => t.stop())
    }
  }

  function handleDelete(userId: string): void {
    setDeletingUserId(userId)
    window.rex
      .deleteVoiceEnrollment(userId)
      .then((result) => {
        if (!result.ok) throw new Error(result.error ?? 'Failed to delete')
        addToast('Voice enrollment deleted', 'success')
        if (selectedUserId === userId) setSelectedUserId(null)
        loadEnrollments()
      })
      .catch((err: unknown) => {
        addToast(err instanceof Error ? err.message : 'Failed to delete', 'error')
      })
      .finally(() => setDeletingUserId(null))
  }

  function handleAddUser(): void {
    const id = newUserId.trim()
    if (!id) return
    setAddingUser(false)
    setNewUserId('')
    void handleEnroll(id)
  }

  if (loading) return <PageLoadingFallback lines={4} />

  return (
    <div className="p-6 max-w-lg">
      <div className="mb-6 flex items-center justify-between">
        <h2 className="text-lg font-semibold text-text-primary">Users</h2>
        {!addingUser && (
          <button
            type="button"
            onClick={() => setAddingUser(true)}
            className="flex items-center gap-1.5 rounded-lg bg-accent px-3 py-1.5 text-sm font-medium text-white transition-colors hover:bg-accent/90 focus:outline-none focus:ring-2 focus:ring-accent"
          >
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
              <line x1="12" y1="5" x2="12" y2="19" />
              <line x1="5" y1="12" x2="19" y2="12" />
            </svg>
            Add User
          </button>
        )}
      </div>

      {/* Add User form */}
      {addingUser && (
        <div className="mb-5 rounded-xl border border-accent/30 bg-accent/5 p-4">
          <p className="mb-3 text-sm font-medium text-text-primary">New User ID</p>
          <div className="flex gap-2">
            <input
              type="text"
              value={newUserId}
              placeholder="e.g. alice"
              onChange={(e) => setNewUserId(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter') handleAddUser()
                if (e.key === 'Escape') { setAddingUser(false); setNewUserId('') }
              }}
              autoFocus
              className="flex-1 bg-surface-raised border border-border rounded-lg px-3 py-2 text-sm text-text-primary placeholder:text-text-secondary focus:outline-none focus:ring-2 focus:ring-accent"
            />
            <button
              type="button"
              onClick={handleAddUser}
              disabled={!newUserId.trim()}
              className="rounded-lg bg-accent px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-accent/90 disabled:opacity-50"
            >
              Start Enrollment
            </button>
            <button
              type="button"
              onClick={() => { setAddingUser(false); setNewUserId('') }}
              className="rounded-lg border border-border px-3 py-2 text-sm text-text-secondary transition-colors hover:bg-surface-raised"
            >
              Cancel
            </button>
          </div>
          <p className="mt-2 text-xs text-text-secondary">
            You will be prompted to record {ENROLLMENT_SAMPLE_TARGET} voice samples.
          </p>
        </div>
      )}

      {/* Enrollment progress (shown during active enrollment) */}
      {enrollingUserId !== null && (
        <div className="mb-5 rounded-xl border border-border bg-surface-raised p-4">
          <p className="mb-3 text-sm font-medium text-text-primary">
            Enrolling: <span className="text-accent">{enrollingUserId}</span>
          </p>
          {/* Prompt phrase for the user to read aloud */}
          <div className="mb-3 rounded-lg bg-surface border border-accent/30 px-3 py-2">
            <p className="text-xs text-text-secondary mb-1 uppercase tracking-wide">Say aloud:</p>
            <p className="text-sm font-medium text-text-primary italic">"{ENROLLMENT_PROMPT_PHRASE}"</p>
          </div>
          <div className="h-2 overflow-hidden rounded-full bg-surface mb-2">
            <div
              className="h-full rounded-full bg-accent transition-all duration-300"
              style={{ width: `${(capturedSamples / ENROLLMENT_SAMPLE_TARGET) * 100}%` }}
            />
          </div>
          <div className="flex items-center justify-between text-sm">
            <span className="text-text-secondary">
              {enrollmentCountdown > 0 ? `Sample ${capturedSamples + 1} starts in` : 'Recording now'}
            </span>
            <span className={`text-2xl font-semibold tabular-nums ${enrollmentCountdown === 0 ? 'text-red-400 animate-pulse' : 'text-text-primary'}`}>
              {enrollmentCountdown > 0 ? enrollmentCountdown : '⏺ REC'}
            </span>
          </div>
          {enrollmentMessage && !enrollmentError && (
            <p className="mt-2 text-sm text-success">{enrollmentMessage}</p>
          )}
          {enrollmentError && (
            <p className="mt-2 text-sm text-danger">{enrollmentError}</p>
          )}
        </div>
      )}

      {/* Users list */}
      {enrollments.length === 0 ? (
        <div className="rounded-xl border border-dashed border-border bg-surface-raised/40 px-4 py-8 text-center text-sm text-text-secondary">
          No enrolled users. Click "Add User" to get started.
        </div>
      ) : (
        <div className="space-y-3">
          {enrollments.map((enrollment) => {
            const displayName = userNames[enrollment.user_id] ?? enrollment.user_id
            const initial = displayName.charAt(0).toUpperCase()
            const isSelected = selectedUserId === enrollment.user_id
            const isActive = enrollment.user_id === activeUserId
            return (
              <div key={enrollment.user_id} className="rounded-xl border border-border bg-surface-raised">
                <button
                  type="button"
                  onClick={() => setSelectedUserId(isSelected ? null : enrollment.user_id)}
                  className="w-full flex items-center gap-3 px-4 py-3 text-left focus:outline-none focus:ring-2 focus:ring-accent focus:ring-inset rounded-xl"
                >
                  <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-full bg-accent/20 text-accent font-semibold text-sm">
                    {initial}
                  </div>
                  <div className="flex-1 min-w-0">
                    {editingName === enrollment.user_id ? (
                      <input
                        type="text"
                        value={editNameValue}
                        onClick={(e) => e.stopPropagation()}
                        onChange={(e) => setEditNameValue(e.target.value)}
                        onBlur={() => handleSaveName(enrollment.user_id)}
                        onKeyDown={(e) => {
                          if (e.key === 'Enter') handleSaveName(enrollment.user_id)
                          if (e.key === 'Escape') setEditingName(null)
                        }}
                        autoFocus
                        className="w-full bg-bg border border-accent rounded px-2 py-0.5 text-sm text-text-primary focus:outline-none"
                      />
                    ) : (
                      <div className="text-sm font-medium text-text-primary truncate">{displayName}</div>
                    )}
                    <div className="text-xs text-text-secondary">
                      {enrollment.sample_count} samples · {enrollment.model_id}
                      {isActive && <span className="ml-1.5 text-accent">· active</span>}
                    </div>
                  </div>
                  <svg
                    width="14"
                    height="14"
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="2"
                    className={`shrink-0 text-text-secondary transition-transform ${isSelected ? 'rotate-180' : ''}`}
                  >
                    <polyline points="6 9 12 15 18 9" />
                  </svg>
                </button>

                {isSelected && (
                  <div className="border-t border-border px-4 pb-4 pt-3">
                    <div className="flex flex-wrap gap-2 mb-4">
                      <button
                        type="button"
                        onClick={(e) => { e.stopPropagation(); handleStartEditName(enrollment.user_id) }}
                        className="rounded-lg border border-border bg-bg px-3 py-1.5 text-xs font-medium text-text-primary transition-colors hover:bg-surface-raised"
                      >
                        Edit Name
                      </button>
                      <button
                        type="button"
                        onClick={(e) => { e.stopPropagation(); void handleEnroll(enrollment.user_id) }}
                        disabled={enrollingUserId !== null}
                        className="rounded-lg border border-border bg-bg px-3 py-1.5 text-xs font-medium text-text-primary transition-colors hover:bg-surface-raised disabled:opacity-50"
                      >
                        Re-enroll Voice
                      </button>
                      <button
                        type="button"
                        onClick={(e) => { e.stopPropagation(); handleDelete(enrollment.user_id) }}
                        disabled={deletingUserId === enrollment.user_id}
                        className="rounded-lg border border-danger/30 bg-bg px-3 py-1.5 text-xs font-medium text-danger transition-colors hover:bg-danger/10 disabled:opacity-50"
                      >
                        {deletingUserId === enrollment.user_id ? 'Deleting…' : 'Delete User'}
                      </button>
                    </div>

                    {/* Memory viewer */}
                    <div>
                      <h4 className="mb-2 text-xs font-semibold uppercase tracking-wider text-text-secondary">
                        Memory
                      </h4>
                      {memoriesLoading ? (
                        <div className="space-y-2">
                          <SkeletonLine width="100%" height="1rem" />
                          <SkeletonLine width="75%" height="1rem" />
                        </div>
                      ) : memories.length === 0 ? (
                        <p className="text-xs text-text-secondary">No stored memories.</p>
                      ) : (
                        <div className="max-h-48 overflow-y-auto space-y-2 rounded-lg border border-border bg-bg p-3">
                          {memories.map((mem) => (
                            <div key={mem.id} className="text-xs">
                              <span className="inline-block mr-1.5 rounded bg-surface-raised px-1.5 py-0.5 font-medium text-text-secondary">
                                {mem.category}
                              </span>
                              <span className="text-text-primary">{mem.text}</span>
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  </div>
                )}
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}
