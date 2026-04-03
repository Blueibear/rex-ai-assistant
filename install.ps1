# install.ps1 — single-command Rex installer for Windows
# Usage: .\install.ps1
[CmdletBinding()]
Param()

$ErrorActionPreference = "Stop"

$RepoDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$VenvDir = Join-Path $RepoDir ".venv"

function Fail {
    param([string]$Message)
    Write-Error "ERROR: $Message"
    exit 1
}

# Verify Python is available
$Python = Get-Command python -ErrorAction SilentlyContinue
if (-not $Python) {
    Fail "Python not found. Install Python 3.11 from https://python.org and ensure it is on your PATH."
}

# Require Python 3.11 exactly for the supported full install path
try {
    $VersionOutput = & python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>&1
    $Parts = $VersionOutput -split '\.'
    $Major = [int]$Parts[0]
    $Minor = [int]$Parts[1]
} catch {
    Fail "Could not determine Python version."
}
if ($Major -ne 3 -or $Minor -ne 11) {
    Fail "Unsupported Python $VersionOutput for the Rex full install path. Use Python 3.11. Fresh installs on Python 3.13/3.14 are known to fail in the ML/TTS dependency path."
}

Write-Host "Creating virtual environment in $VenvDir ..."
python -m venv $VenvDir
if ($LASTEXITCODE -ne 0) { Fail "Failed to create virtual environment." }

$Pip = Join-Path $VenvDir "Scripts\pip.exe"
$Rex = Join-Path $VenvDir "Scripts\rex.exe"

Write-Host "Upgrading pip ..."
& (Join-Path $VenvDir "Scripts\python.exe") -m pip install --upgrade pip setuptools wheel | Out-Null
if ($LASTEXITCODE -ne 0) { Fail "Failed to upgrade pip." }

Write-Host "Installing Rex with the supported full dependency set ..."
& $Pip install "$RepoDir[full]"
if ($LASTEXITCODE -ne 0) { Fail "pip install failed. Check the error above and re-run after resolving it." }

Write-Host "Verifying install ..."
& $Rex --help 2>&1 | Out-Null
if ($LASTEXITCODE -ne 0) {
    Fail "Rex was installed but the 'rex' command did not respond. Check the install log above."
}

Write-Host ""
Write-Host "Rex is installed. Run ``rex`` to start."
Write-Host ""
Write-Host "To activate the virtual environment manually:"
Write-Host "  $VenvDir\Scripts\Activate.ps1"
