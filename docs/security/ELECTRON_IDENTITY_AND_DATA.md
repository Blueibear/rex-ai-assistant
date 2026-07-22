# Electron identity and household data

Electron establishes one immutable user identity in the main process before it
registers private IPC handlers. The identity is resolved by the Python identity
bridge from the existing `rex identify` session/config chain, validated with the
canonical user-ID rules, and bound to the current operating-system login. The
renderer may display a profile label, but it cannot select or override the user
attached to a bridge request.

Launches without a valid active user fail closed with an actionable error. Set
the intended profile before launch:

```powershell
.\.venv\Scripts\rex.exe identify --user james
```

Chat, memories, reminders, tasks, command history, uploaded-document
extraction, email, calendar, SMS, quick actions, and voice operations carry the
private session identity. Voice enrollment changes must match that identity.
The shopping list remains an intentional household-shared store; its bridge
requires `shared_household` scope and records the authenticated actor in
`added_by`.

## Existing data migration

Legacy tasks and command-history rows without an owner are quarantined: James,
Cole, and other profiles cannot read or mutate them. Assign them only after a
human identifies the correct owner. The migration tool is dry-run by default:

```powershell
.\.venv\Scripts\python.exe scripts\migrate_electron_data_ownership.py --user james
.\.venv\Scripts\python.exe scripts\migrate_electron_data_ownership.py --user james --apply
```

`--apply` creates one-time `.pre-ownership-migration` backups beside each
changed store. If legacy data belongs to more than one person, do not run the
bulk migration; export and assign records individually after review.

The local OS-session binding protects against renderer-controlled profile
switching. It is not remote or multi-tenant authentication. Households where
people do not trust each other with the same Windows login must use separate OS
accounts.
