# Distribution & Installation

## Installation options

### Pip installation
```bash
pip install .
pip install .[sms,devtools]
```

### Full installation script
```bash
./install_full.sh
```

Environment overrides:
```bash
REX_SERVICE_PORT=8765 REX_SKIP_SERVICE=1 ./install_full.sh
```

### Lean installation script
```bash
./install_lean.sh
```

Environment overrides:
```bash
REX_SERVICES=event_bus,workflow_runner,memory_store,credential_manager ./install_lean.sh
```

## Optional dependencies
- `sms`: Twilio support (`twilio` package).
- `devtools`: Build and lint tooling (`build`, `ruff`, `black`, `mypy`).

## Remote node setup
Lean nodes run a trimmed service set and register with the Rex gateway.

See `node_installers/README.md` for platform-specific installer instructions.

## Customization
- Use `--services` with `rex.app` to control which services are supervised.
- Adjust health check port with `--port`.
- Modify `.env.node` for gateway registration details.

## Windows Authenticode signing

The packaged Windows Electron installer supports Authenticode signing that
activates only when a code-signing certificate is provisioned. Without the
certificate the build produces the same unsigned artifact as before, and
nothing claims to be signed.

**Certificate requirements:** a Windows code-signing certificate (OV or EV)
exported as a password-protected PFX. No certificate is bundled with the
repository and none is purchased by automation.

**CI secret names** (GitHub Actions, `windows-electron-artifact.yml`):

| Secret | Content |
|---|---|
| `WINDOWS_CSC_LINK` | Base64-encoded PFX (or a `file://`/`https://` link electron-builder accepts) |
| `WINDOWS_CSC_KEY_PASSWORD` | PFX password |

When `WINDOWS_CSC_LINK` is non-empty the workflow exports it as `CSC_LINK`
before `npm run dist`, and electron-builder signs the executables and NSIS
installer. Signed builds are RFC 3161 timestamped
(`gui/package.json` `build.win.signtoolOptions.rfc3161TimeStampServer`) so
signatures remain valid after certificate expiry.

**Local signing** (developer machine, from `gui/`):

```powershell
$env:CSC_LINK = "<base64 PFX or file path>"
$env:CSC_KEY_PASSWORD = "<pfx password>"
npm run dist
```

**Verification:** the workflow runs `Get-AuthenticodeSignature` on the
installer after the build. If signing was configured and the status is not
`Valid`, the job fails. If no secret is configured, the unsigned status is
reported truthfully and the job continues.

```powershell
Get-AuthenticodeSignature "gui/dist/AskRex Setup 1.0.0.exe" | Format-List Status, StatusMessage, SignerCertificate
```

**Unsigned behavior:** without a certificate the installer status is
`NotSigned`; Windows SmartScreen may warn on first run. The artifact remains
a private/developer release and must not be described as signed.
