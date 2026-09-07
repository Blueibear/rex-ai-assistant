[CmdletBinding()]
param(
    [switch]$RecreateVenv
)

$ErrorActionPreference = "Stop"

$RepoDir = [System.IO.Path]::GetFullPath($PSScriptRoot)
$VenvDir = Join-Path $RepoDir ".venv"
$VenvPython = Join-Path $VenvDir "Scripts\python.exe"
$VenvPip = Join-Path $VenvDir "Scripts\pip.exe"
$RexExe = Join-Path $VenvDir "Scripts\rex.exe"

function Fail {
    param([string]$Message)
    Write-Error "ERROR: $Message"
    exit 1
}

function Get-PythonVersion {
    param([string]$PythonExe)

    if (-not (Test-Path $PythonExe)) {
        return $null
    }

    try {
        $version = & $PythonExe -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')" 2>$null
        if ([string]::IsNullOrWhiteSpace($version)) {
            return $null
        }
        return $version.Trim()
    } catch {
        return $null
    }
}

function Test-Python311 {
    param([string]$VersionString)

    if ([string]::IsNullOrWhiteSpace($VersionString)) {
        return $false
    }

    return $VersionString -match '^3\.11(\.|$)'
}

function Get-PyLauncher311 {
    try {
        $null = & py -3.11 -c "import sys; print(sys.executable)" 2>$null
        if ($LASTEXITCODE -eq 0) {
            return $true
        }
        return $false
    } catch {
        return $false
    }
}

function Ensure-VenvFromPy311 {
    if (-not (Get-PyLauncher311)) {
        Fail "Python 3.11 was not found via the Windows py launcher. Install Python 3.11 and make sure the py launcher is available."
    }

    if ($RecreateVenv -and (Test-Path $VenvDir)) {
        Write-Host "Removing existing virtual environment at $VenvDir ..."
        Remove-Item -Recurse -Force $VenvDir
    }

    if (Test-Path $VenvPython) {
        $existingVersion = Get-PythonVersion -PythonExe $VenvPython
        if (Test-Python311 $existingVersion) {
            Write-Host "Reusing existing Python $existingVersion virtual environment at $VenvDir ..."
            return
        } else {
            Write-Host "Existing .venv is not Python 3.11. Recreating it ..."
            Remove-Item -Recurse -Force $VenvDir
        }
    }

    Write-Host "Creating Python 3.11 virtual environment in $VenvDir ..."
    & py -3.11 -m venv $VenvDir
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path $VenvPython)) {
        Fail "Failed to create the Python 3.11 virtual environment."
    }
}

function Upgrade-BootstrapTools {
    Write-Host "Upgrading pip, setuptools, and wheel ..."
    & $VenvPython -m pip install --upgrade pip setuptools wheel
    if ($LASTEXITCODE -ne 0) {
        Fail "Failed to upgrade pip, setuptools, and wheel."
    }
}

function Install-Rex {
    Write-Host "Installing Rex with the supported full dependency set ..."
    & $VenvPip install "$RepoDir[full]"
    if ($LASTEXITCODE -ne 0) {
        Fail "pip install failed. Check the error output above."
    }
}

function Bootstrap-Config {
    $EnvFile = Join-Path $RepoDir ".env"
    $EnvExample = Join-Path $RepoDir ".env.example"
    if (-not (Test-Path $EnvFile)) {
        if (Test-Path $EnvExample) {
            Copy-Item $EnvExample $EnvFile
            Write-Host "Created .env from .env.example - edit it to add your API keys."
        } else {
            New-Item -ItemType File -Path $EnvFile | Out-Null
            Write-Host "Created empty .env - edit it to add your API keys before running Rex."
        }
    } else {
        Write-Host ".env already exists - skipping."
    }
}

function Verify-Install {
    Write-Host "Running health check ..."
    & $RexExe doctor
    if ($LASTEXITCODE -ne 0) {
        Write-Host ""
        Write-Host "WARNING: 'rex doctor' reported one or more issues (see above)."
        Write-Host "Rex is installed but may need additional configuration."
        Write-Host "Edit .env with your API keys and re-run 'rex doctor' to clear warnings."
    }
}

$ActiveVenv = $env:VIRTUAL_ENV
if (-not [string]::IsNullOrWhiteSpace($ActiveVenv)) {
    $ActivePython = Join-Path $ActiveVenv "Scripts\python.exe"
    $ActiveVersion = Get-PythonVersion -PythonExe $ActivePython

    if (Test-Python311 $ActiveVersion) {
        Write-Host "Detected active Python $ActiveVersion virtual environment at $ActiveVenv"
        Write-Host "Using the active environment and skipping .venv creation."

        $VenvDir = $ActiveVenv
        $VenvPython = Join-Path $VenvDir "Scripts\python.exe"
        $VenvPip = Join-Path $VenvDir "Scripts\pip.exe"
        $RexExe = Join-Path $VenvDir "Scripts\rex.exe"
    } else {
        Write-Host "Active virtual environment is not Python 3.11. Ignoring it and using .venv instead."
        Ensure-VenvFromPy311
    }
} else {
    Ensure-VenvFromPy311
}

$ResolvedVersion = Get-PythonVersion -PythonExe $VenvPython
if (-not (Test-Python311 $ResolvedVersion)) {
    Fail "Resolved environment is Python $ResolvedVersion. Rex full install currently requires Python 3.11."
}

Upgrade-BootstrapTools
Install-Rex
Bootstrap-Config
Verify-Install

Write-Host ""
Write-Host "Rex is installed."
Write-Host ""
Write-Host "To activate the environment manually, run:"
Write-Host "  $VenvDir\Scripts\Activate.ps1"
Write-Host ""
Write-Host "Then start Rex with:"
Write-Host "  rex"