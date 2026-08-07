import React, { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { useToast } from '../components/ui/Toast'
import type { UserProfile } from '../types/ipc'

interface AvatarUploadState {
  isLoading: boolean
  error?: string
}

function formatPreferenceValue(value: unknown): string {
  if (value === null) return 'Not set'
  if (typeof value === 'string' || typeof value === 'number' || typeof value === 'boolean') {
    return String(value)
  }
  return 'Configured'
}

export function ProfilePage(): React.ReactElement {
  const navigate = useNavigate()
  const addToast = useToast()
  const [profile, setProfile] = useState<UserProfile | null>(null)
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [avatarState, setAvatarState] = useState<AvatarUploadState>({ isLoading: false })

  useEffect(() => {
    const loadProfile = async (): Promise<void> => {
      try {
        const result = await window.rex.getProfile()
        if (result.ok && result.profile) {
          setProfile(result.profile)
          setError(null)
        } else {
          setError(result.error ?? 'Failed to load profile')
        }
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to load profile')
      } finally {
        setIsLoading(false)
      }
    }

    void loadProfile()
  }, [])

  const handleAvatarChange = async (e: React.ChangeEvent<HTMLInputElement>): Promise<void> => {
    const file = e.currentTarget.files?.[0]
    if (!file) return

    setAvatarState({ isLoading: true })

    try {
      // Validate file type
      const validTypes = ['image/jpeg', 'image/png']
      if (!validTypes.includes(file.type)) {
        setAvatarState({
          isLoading: false,
          error: 'Avatar must be JPEG or PNG'
        })
        return
      }

      // Validate file size (2 MB)
      const maxSize = 2 * 1024 * 1024
      if (file.size > maxSize) {
        setAvatarState({
          isLoading: false,
          error: 'Avatar must be less than 2 MB'
        })
        return
      }

      const reader = new FileReader()
      reader.onload = async (event): Promise<void> => {
        if (!event.target?.result) return

        const base64 = (event.target.result as string).split(',')[1]
        if (!base64) {
          setAvatarState({
            isLoading: false,
            error: 'Failed to read file'
          })
          return
        }

        try {
          const result = await window.rex.setProfileAvatar(file.type, base64)
          if (result.ok) {
            if (result.profile) {
              setProfile(result.profile)
            }
            addToast('Avatar updated', 'success')
            setAvatarState({ isLoading: false })
          } else {
            setAvatarState({
              isLoading: false,
              error: result.error ?? 'Failed to upload avatar'
            })
          }
        } catch (err) {
          setAvatarState({
            isLoading: false,
            error: err instanceof Error ? err.message : 'Upload failed'
          })
        }
      }

      reader.onerror = (): void => {
        setAvatarState({
          isLoading: false,
          error: 'Failed to read file'
        })
      }

      reader.readAsDataURL(file)
    } catch (err) {
      setAvatarState({
        isLoading: false,
        error: err instanceof Error ? err.message : 'Upload failed'
      })
    }
  }

  const handleRemoveAvatar = async (): Promise<void> => {
    setAvatarState({ isLoading: true })

    try {
      const result = await window.rex.removeProfileAvatar()
      if (result.ok) {
        if (result.profile) {
          setProfile(result.profile)
        }
        addToast('Avatar removed', 'success')
        setAvatarState({ isLoading: false })
      } else {
        setAvatarState({
          isLoading: false,
          error: result.error ?? 'Failed to remove avatar'
        })
      }
    } catch (err) {
      setAvatarState({
        isLoading: false,
        error: err instanceof Error ? err.message : 'Removal failed'
      })
    }
  }

  if (isLoading) {
    return (
      <div className="p-6 max-w-2xl mx-auto">
        <div className="animate-pulse space-y-4">
          <div className="h-8 bg-surface-raised rounded w-32" />
          <div className="h-4 bg-surface-raised rounded w-full" />
        </div>
      </div>
    )
  }

  if (error || !profile) {
    return (
      <div className="p-6 max-w-2xl mx-auto">
        <div className="p-4 bg-red-500/15 border border-red-500/30 rounded-lg">
          <p className="text-red-400 text-sm font-medium">{error ?? 'Profile not found'}</p>
        </div>
      </div>
    )
  }

  const permissionsText = profile.permissions.length > 0
    ? profile.permissions.join(', ')
    : 'No special permissions'

  return (
    <div className="p-6 max-w-2xl mx-auto space-y-8">
      {/* Avatar Section */}
      <section>
        <h2 className="text-xl font-semibold text-text-primary mb-4">Profile Picture</h2>

        {/* Avatar Display */}
        <div className="flex items-start gap-6 mb-6">
          <div className="flex-shrink-0">
            <div className="w-24 h-24 rounded-full bg-surface-raised flex items-center justify-center overflow-hidden text-2xl font-semibold">
              {profile.avatar_data && profile.avatar_mime_type ? (
                <img
                  src={`data:${profile.avatar_mime_type};base64,${profile.avatar_data}`}
                  alt="Profile avatar"
                  className="w-full h-full object-cover"
                />
              ) : (
                <span className="text-text-secondary">{profile.initials}</span>
              )}
            </div>
          </div>

          {/* Avatar Controls */}
          <div className="flex-1 space-y-3">
            <div>
              <label htmlFor="avatar-upload" className="flex gap-2">
                <input
                  id="avatar-upload"
                  type="file"
                  accept="image/jpeg,image/png"
                  onChange={(e) => {
                    void handleAvatarChange(e)
                  }}
                  disabled={avatarState.isLoading}
                  className="hidden"
                />
                <button
                  type="button"
                  onClick={() => document.getElementById('avatar-upload')?.click()}
                  disabled={avatarState.isLoading}
                  className="px-3 py-2 bg-accent text-white rounded-lg text-sm font-medium hover:bg-accent/90 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                >
                  {avatarState.isLoading ? 'Uploading…' : 'Upload Picture'}
                </button>
              </label>
              {profile.avatar_present && (
                <button
                  type="button"
                  onClick={() => {
                    void handleRemoveAvatar()
                  }}
                  disabled={avatarState.isLoading}
                  className="ml-2 px-3 py-2 bg-red-500/20 text-red-400 rounded-lg text-sm font-medium hover:bg-red-500/30 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                >
                  Remove
                </button>
              )}
            </div>

            {avatarState.error && (
              <p className="text-red-400 text-xs font-medium">{avatarState.error}</p>
            )}

            <p className="text-text-secondary text-xs">
              JPEG or PNG, up to 2 MB
            </p>
          </div>
        </div>
      </section>

      {/* Identity Section */}
      <section>
        <h2 className="text-xl font-semibold text-text-primary mb-4">Identity</h2>

        <div className="border border-border rounded-xl overflow-hidden divide-y divide-border">
          <div className="flex items-center justify-between px-4 py-3 bg-surface">
            <span className="text-text-secondary text-sm">Name</span>
            <span className="text-text-primary text-sm font-medium">{profile.name}</span>
          </div>
          <div className="flex items-center justify-between px-4 py-3 bg-surface">
            <span className="text-text-secondary text-sm">User ID</span>
            <span className="text-text-primary text-xs font-mono">{profile.user_id}</span>
          </div>
          <div className="flex items-center justify-between px-4 py-3 bg-surface">
            <span className="text-text-secondary text-sm">Role</span>
            <span className="text-text-primary text-sm font-medium capitalize">{profile.role}</span>
          </div>
        </div>
      </section>

      {/* Permissions Section */}
      <section>
        <h2 className="text-xl font-semibold text-text-primary mb-4">Permissions</h2>

        <div className="border border-border rounded-xl p-4 bg-surface">
          <p className="text-text-primary text-sm capitalize">{permissionsText}</p>
        </div>
      </section>


      {/* Preferences Section */}
      <section>
        <h2 className="text-xl font-semibold text-text-primary mb-4">Preferences</h2>
        <div className="border border-border rounded-xl overflow-hidden divide-y divide-border">
          {Object.keys(profile.preferences).length > 0 ? (
            Object.entries(profile.preferences).map(([key, value]) => (
              <div key={key} className="flex items-center justify-between gap-4 px-4 py-3 bg-surface">
                <span className="text-text-secondary text-sm capitalize">{key.replace(/_/g, ' ')}</span>
                <span className="text-text-primary text-sm text-right">{formatPreferenceValue(value)}</span>
              </div>
            ))
          ) : (
            <p className="px-4 py-3 bg-surface text-text-secondary text-sm">No profile preferences saved.</p>
          )}
        </div>
      </section>

      {/* Voice Enrollment Section */}
      <section>
        <h2 className="text-xl font-semibold text-text-primary mb-4">Voice</h2>

        <div className="border border-border rounded-xl overflow-hidden divide-y divide-border">
          <div className="flex items-center justify-between px-4 py-3 bg-surface">
            <span className="text-text-secondary text-sm">Voice Enrollment</span>
            <span className="text-text-primary text-sm font-medium">
              {profile.voice_enrolled ? 'Enrolled' : 'Not enrolled'}
            </span>
          </div>
          {profile.voice_enrolled && (
            <>
              <div className="flex items-center justify-between px-4 py-3 bg-surface">
                <span className="text-text-secondary text-sm">Samples Collected</span>
                <span className="text-text-primary text-sm font-medium">
                  {profile.voice_sample_count}
                </span>
              </div>
              <div className="flex items-center justify-between px-4 py-3 bg-surface">
                <span className="text-text-secondary text-sm">Last Updated</span>
                <span className="text-text-primary text-xs font-mono">
                  {profile.voice_updated_at
                    ? new Date(profile.voice_updated_at).toLocaleDateString()
                    : '—'}
                </span>
              </div>
            </>
          )}
        </div>
      </section>

      {/* Private Data Scope Section */}
      <section>
        <h2 className="text-xl font-semibold text-text-primary mb-4">Data Scope</h2>

        <div className="space-y-4">
          <div className="border border-border rounded-xl p-4 bg-surface">
            <h3 className="text-text-primary text-sm font-semibold mb-2">Private to This Profile</h3>
            <p className="text-text-secondary text-xs mb-3">
              The following data is stored only for this user profile:
            </p>
            <ul className="space-y-1 text-text-secondary text-xs">
              {Object.entries(profile.scope_labels)
                .filter(([, scope]) => scope === 'user-private')
                .map(([key]) => (
                  <li key={key} className="flex items-center gap-2">
                    <span className="w-1 h-1 rounded-full bg-text-secondary" />
                    <span className="capitalize">{key.replace(/_/g, ' ')}</span>
                  </li>
                ))}
            </ul>
          </div>

          {Object.values(profile.scope_labels).includes('shared') && (
            <div className="border border-border rounded-xl p-4 bg-surface">
              <h3 className="text-text-primary text-sm font-semibold mb-2">Shared Household Settings</h3>
              <p className="text-text-secondary text-xs mb-3">
                The following settings are shared across all profiles on this desktop:
              </p>
              <ul className="space-y-1 text-text-secondary text-xs mb-3">
                {Object.entries(profile.scope_labels)
                  .filter(([, scope]) => scope === 'shared')
                  .map(([key]) => (
                    <li key={key} className="flex items-center gap-2">
                      <span className="w-1 h-1 rounded-full bg-text-secondary" />
                      <span className="capitalize">{key.replace(/_/g, ' ')}</span>
                    </li>
                  ))}
              </ul>
              <button
                type="button"
                onClick={() => navigate('/settings')}
                className="text-accent text-xs font-medium hover:underline"
              >
                Go to Settings
              </button>
            </div>
          )}
        </div>
      </section>

      {/* Session Authority Info */}
      <section>
        <h2 className="text-xl font-semibold text-text-primary mb-4">Session</h2>

        <div className="border border-blue-500/30 rounded-xl bg-blue-500/5 p-4">
          <p className="text-text-secondary text-sm leading-relaxed">
            This profile is bound to your current authenticated desktop session.
            To switch to a different user profile, start a new authenticated desktop session
            or restart the application and select a different user during login.
          </p>
        </div>
      </section>
    </div>
  )
}
