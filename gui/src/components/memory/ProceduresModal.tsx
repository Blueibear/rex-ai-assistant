import React, { useEffect, useState } from 'react'
import type { Procedure, ProcedureStatus } from '../../types/ipc'
import { Badge, type BadgeProps } from '../ui/Badge'
import { EmptyState } from '../ui/EmptyState'
import { Modal } from '../ui/Modal'
import { PageLoadingFallback } from '../ui/PageLoadingFallback'
import { useToast } from '../ui/Toast'

function statusVariant(status: ProcedureStatus): BadgeProps['variant'] {
  if (status === 'active') return 'success'
  if (status === 'pending_approval') return 'warning'
  if (status === 'revoked') return 'danger'
  return 'default'
}

function riskVariant(risk: Procedure['risk']): BadgeProps['variant'] {
  if (risk === 'safe') return 'success'
  if (risk === 'sensitive') return 'warning'
  return 'danger'
}

function displayCode(value: string): string {
  return value.replaceAll('_', ' ')
}

function formatDate(value: string | null): string {
  if (!value) return 'Not set'
  const date = new Date(value)
  return Number.isNaN(date.getTime()) ? value : date.toLocaleString()
}

export interface ProceduresModalProps {
  onClose: () => void
}

export function ProceduresModal({ onClose }: ProceduresModalProps): React.ReactElement {
  const [procedures, setProcedures] = useState<Procedure[]>([])
  const [loading, setLoading] = useState(true)
  const [busyId, setBusyId] = useState<string | null>(null)
  const addToast = useToast()

  useEffect(() => {
    window.rex
      .getProcedures()
      .then(setProcedures)
      .catch((err: unknown) => {
        addToast(err instanceof Error ? err.message : 'Failed to load learned procedures', 'error')
      })
      .finally(() => setLoading(false))
  }, [addToast])

  function replaceProcedure(updated: Procedure): void {
    setProcedures((current) =>
      current.map((procedure) => (procedure.id === updated.id ? updated : procedure))
    )
  }

  function runUpdate(id: string, operation: () => Promise<Procedure>): void {
    setBusyId(id)
    operation()
      .then(replaceProcedure)
      .catch((err: unknown) => {
        const message = err instanceof Error ? err.message : 'Failed to update learned procedure'
        if (!message.toLowerCase().includes('cancelled')) addToast(message, 'error')
      })
      .finally(() => setBusyId(null))
  }

  function handleRevoke(procedure: Procedure): void {
    if (!window.confirm(`Revoke “${procedure.name}”? It cannot run again unless relearned.`)) return
    runUpdate(procedure.id, () => window.rex.revokeProcedure(procedure.id))
  }

  function handleDelete(procedure: Procedure): void {
    if (!window.confirm(`Delete “${procedure.name}” and its stored procedure record?`)) return
    setBusyId(procedure.id)
    window.rex
      .deleteProcedure(procedure.id)
      .then(() => setProcedures((current) => current.filter((item) => item.id !== procedure.id)))
      .catch((err: unknown) => {
        addToast(err instanceof Error ? err.message : 'Failed to delete learned procedure', 'error')
      })
      .finally(() => setBusyId(null))
  }

  return (
    <Modal
      title="Learned Procedures"
      onClose={onClose}
      footer={
        <button
          onClick={onClose}
          className="px-3 py-1.5 text-sm rounded-lg bg-surface border border-border text-text-secondary hover:text-text-primary hover:bg-surface-raised transition-colors"
        >
          Close
        </button>
      }
    >
      <div className="space-y-4">
        <div className="rounded-lg border border-border bg-bg/40 p-3 text-xs leading-relaxed">
          Procedures are separate from normal memories. Rex can only learn one from a verified action
          outcome. Stored permissions are requirements, not grants, and risky procedures require an
          explicit approval before activation.
        </div>

        {loading ? (
          <PageLoadingFallback lines={5} />
        ) : procedures.length === 0 ? (
          <EmptyState
            heading="No learned procedures"
            subtext="Verified repeated work can become a guarded reusable procedure. Ordinary memories cannot create one."
          />
        ) : (
          <div className="space-y-3">
            {procedures.map((procedure) => (
              <ProcedureCard
                key={procedure.id}
                procedure={procedure}
                busy={busyId === procedure.id}
                onApprove={() =>
                  runUpdate(procedure.id, () => window.rex.approveProcedure(procedure.id))
                }
                onDisable={() =>
                  runUpdate(procedure.id, () => window.rex.disableProcedure(procedure.id))
                }
                onRevoke={() => handleRevoke(procedure)}
                onDelete={() => handleDelete(procedure)}
              />
            ))}
          </div>
        )}
      </div>
    </Modal>
  )
}

interface ProcedureCardProps {
  procedure: Procedure
  busy: boolean
  onApprove: () => void
  onDisable: () => void
  onRevoke: () => void
  onDelete: () => void
}

function ProcedureCard({
  procedure,
  busy,
  onApprove,
  onDisable,
  onRevoke,
  onDelete
}: ProcedureCardProps): React.ReactElement {
  return (
    <div className="rounded-lg border border-border bg-surface p-3">
      <div className="flex flex-wrap items-start gap-2">
        <div className="min-w-0 flex-1">
          <h3 className="text-sm font-medium text-text-primary truncate">{procedure.name}</h3>
          {procedure.description && (
            <p className="mt-1 text-xs text-text-secondary leading-relaxed">
              {procedure.description}
            </p>
          )}
        </div>
        <Badge variant={statusVariant(procedure.status)}>{displayCode(procedure.status)}</Badge>
        <Badge variant={riskVariant(procedure.risk)}>{procedure.risk}</Badge>
        <Badge>{procedure.scope}</Badge>
      </div>

      {procedure.disabledReason && (
        <p className="mt-2 text-xs text-warning">Disabled reason: {procedure.disabledReason}</p>
      )}

      <dl className="mt-3 grid grid-cols-2 gap-x-3 gap-y-1 text-xs">
        <dt className="text-text-secondary">Operation</dt>
        <dd className="text-text-primary">{procedure.operation}</dd>
        <dt className="text-text-secondary">Verified successes</dt>
        <dd className="text-text-primary">{procedure.successCount}</dd>
        <dt className="text-text-secondary">Failures / unverified</dt>
        <dd className="text-text-primary">{procedure.failureCount}</dd>
        <dt className="text-text-secondary">Last validated</dt>
        <dd className="text-text-primary">{formatDate(procedure.lastValidatedAt)}</dd>
        <dt className="text-text-secondary">Expires</dt>
        <dd className="text-text-primary">{formatDate(procedure.expiresAt)}</dd>
        <dt className="text-text-secondary">Version</dt>
        <dd className="text-text-primary">{procedure.version}</dd>
      </dl>

      <div className="mt-3">
        <p className="text-xs text-text-secondary">Capabilities</p>
        <div className="mt-1 flex flex-wrap gap-1">
          {procedure.capabilities.map((capability) => (
            <code
              key={capability}
              className="rounded bg-bg px-1.5 py-0.5 text-[11px] text-text-primary"
            >
              {capability}
            </code>
          ))}
        </div>
      </div>

      <div className="mt-3">
        <p className="text-xs text-text-secondary">Required permissions</p>
        <div className="mt-1 flex flex-wrap gap-1">
          {procedure.requiredPermissions.length === 0 ? (
            <span className="text-xs text-text-secondary">None declared</span>
          ) : (
            procedure.requiredPermissions.map((permission) => (
              <code
                key={permission}
                className="rounded bg-bg px-1.5 py-0.5 text-[11px] text-text-primary"
              >
                {permission}
              </code>
            ))
          )}
        </div>
      </div>

      <details className="mt-3 rounded border border-border bg-bg/40 px-2 py-1.5 text-xs">
        <summary className="cursor-pointer text-text-secondary">Provenance and audit</summary>
        <div className="mt-2 space-y-1 text-text-secondary">
          <p>
            Verification:{' '}
            <code className="text-text-primary">{procedure.provenance.verificationId}</code>
          </p>
          <p>
            Audit: <code className="text-text-primary">{procedure.provenance.auditId}</code>
          </p>
          {procedure.auditHistory.slice(-6).map((event, index) => (
            <p key={`${event.timestamp}-${event.event}-${index}`}>
              {formatDate(event.timestamp)} — {displayCode(event.event)}
              {event.reason ? ` (${event.reason})` : ''}
            </p>
          ))}
        </div>
      </details>

      <div className="mt-3 flex flex-wrap justify-end gap-2">
        {procedure.status === 'pending_approval' && (
          <button
            disabled={busy}
            onClick={onApprove}
            className="px-2.5 py-1 text-xs rounded bg-accent text-white disabled:opacity-40"
          >
            Approve
          </button>
        )}
        {procedure.status === 'active' && (
          <button
            disabled={busy}
            onClick={onDisable}
            className="px-2.5 py-1 text-xs rounded bg-surface-raised border border-border text-text-primary disabled:opacity-40"
          >
            Disable
          </button>
        )}
        {procedure.status !== 'revoked' && (
          <button
            disabled={busy}
            onClick={onRevoke}
            className="px-2.5 py-1 text-xs rounded border border-danger/50 text-danger disabled:opacity-40"
          >
            Revoke
          </button>
        )}
        <button
          disabled={busy}
          onClick={onDelete}
          className="px-2.5 py-1 text-xs rounded text-text-secondary hover:text-danger disabled:opacity-40"
        >
          Delete
        </button>
      </div>
    </div>
  )
}

export default ProceduresModal
