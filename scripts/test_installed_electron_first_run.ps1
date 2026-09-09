[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$Installer,
    [string]$DiagnosticsPath = ''
)

$ErrorActionPreference = 'Stop'
$installerPath = (Resolve-Path -LiteralPath $Installer).Path
$testRoot = Join-Path ([System.IO.Path]::GetTempPath()) ('arx-first-' + [guid]::NewGuid().ToString('N').Substring(0, 8))
$installPath = Join-Path $testRoot 'AskRex'
$localAppData = Join-Path $testRoot 'LocalAppData'
$appData = Join-Path $testRoot 'AppData'
$runtimeRoot = Join-Path $appData 'rex-first-run'
$smokeOutput = Join-Path $testRoot 'first-run-smoke.json'
$appProcess = $null

function Write-Diagnostics([string]$Status, [string]$Message = '') {
    if (-not $DiagnosticsPath) { return }
    $directory = Split-Path -Parent $DiagnosticsPath
    if ($directory) {
        New-Item -ItemType Directory -Force -Path $directory | Out-Null
    }
    $payload = [ordered]@{
        status = $Status
        phase = 'installed-first-run'
        message = $Message
        timestamp_utc = [DateTime]::UtcNow.ToString('o')
    } | ConvertTo-Json
    $utf8NoBom = New-Object System.Text.UTF8Encoding($false)
    [System.IO.File]::WriteAllText($DiagnosticsPath, $payload, $utf8NoBom)
}

function Stop-ProcessTree([System.Diagnostics.Process]$Process) {
    if ($null -eq $Process) { return }
    try {
        if (-not $Process.HasExited) {
            $taskkill = Join-Path $env:SystemRoot 'System32\taskkill.exe'
            & $taskkill /PID $Process.Id /T /F 2>$null | Out-Null
            $Process.WaitForExit(10000) | Out-Null
        }
    } catch {
        Write-Verbose "Process cleanup skipped for PID $($Process.Id): $($_.Exception.Message)"
    }
}

function Invoke-Installer {
    $process = Start-Process -FilePath $installerPath -ArgumentList @('/S', "/D=$installPath") -PassThru
    if (-not $process.WaitForExit(600000)) {
        Stop-ProcessTree $process
        throw 'First-run smoke installer timed out.'
    }
    if ($process.ExitCode -ne 0) {
        throw "First-run smoke installer exited with code $($process.ExitCode)."
    }
}

function Invoke-Uninstaller([string]$UninstallerPath) {
    if (-not (Test-Path -LiteralPath $UninstallerPath -PathType Leaf)) { return }
    $copyRoot = Join-Path ([System.IO.Path]::GetTempPath()) ('askrex-first-uninstall-' + [guid]::NewGuid().ToString('N'))
    $copyPath = Join-Path $copyRoot 'uninstaller.exe'
    New-Item -ItemType Directory -Force -Path $copyRoot | Out-Null
    Copy-Item -LiteralPath $UninstallerPath -Destination $copyPath -Force
    try {
        $process = Start-Process -FilePath $copyPath -ArgumentList @('/S', '/currentuser', "_?=$installPath") -PassThru
        if (-not $process.WaitForExit(300000)) {
            Stop-ProcessTree $process
            throw 'First-run smoke uninstaller timed out.'
        }
        if ($process.ExitCode -ne 0) {
            throw "First-run smoke uninstaller exited with code $($process.ExitCode)."
        }
    } finally {
        Remove-Item -LiteralPath $copyRoot -Recurse -Force -ErrorAction SilentlyContinue
    }
}

try {
    Write-Diagnostics 'running'
    New-Item -ItemType Directory -Force -Path $testRoot, $localAppData, $appData, $runtimeRoot | Out-Null
    Invoke-Installer

    $appExe = Join-Path $installPath 'AskRex.exe'
    $uninstaller = Join-Path $installPath 'Uninstall AskRex.exe'
    foreach ($required in @($appExe, $uninstaller)) {
        if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
            throw "First-run smoke installed artifact is missing: $required"
        }
    }

    $env:LOCALAPPDATA = $localAppData
    $env:APPDATA = $appData
    $env:ASKREX_RUNTIME_DIR = $runtimeRoot
    $env:ASKREX_ARTIFACT_SMOKE = '1'
    $env:ASKREX_ARTIFACT_SMOKE_FIRST_RUN = '1'
    $env:ASKREX_ARTIFACT_SMOKE_RUNTIME_ROOT = $runtimeRoot
    $env:ASKREX_ARTIFACT_SMOKE_OUTPUT = $smokeOutput
    $env:PATH = Join-Path $env:SystemRoot 'System32'

    $appProcess = Start-Process -FilePath $appExe -PassThru
    $deadline = [DateTime]::UtcNow.AddSeconds(120)
    while (-not (Test-Path -LiteralPath $smokeOutput) -and [DateTime]::UtcNow -lt $deadline) {
        Start-Sleep -Milliseconds 500
    }
    if (-not (Test-Path -LiteralPath $smokeOutput)) {
        Stop-ProcessTree $appProcess
        throw 'Fresh installed app did not produce the first-run smoke result within 120 seconds.'
    }

    $result = Get-Content -LiteralPath $smokeOutput -Raw | ConvertFrom-Json
    if (-not $result.ok -or -not $result.setup_ui -or -not $result.preauth_ipc -or
        -not $result.setup_completed -or -not $result.authenticated_ipc -or
        $result.background_voice_enabled) {
        throw "Installed first-run smoke failed: $(Get-Content -LiteralPath $smokeOutput -Raw)"
    }

    if (-not $appProcess.WaitForExit(15000)) {
        Stop-ProcessTree $appProcess
        throw 'Fresh installed Electron process did not exit after first-run smoke completion.'
    }

    Invoke-Uninstaller $uninstaller
    Write-Diagnostics 'success' 'Installed artifact reached setup from a clean runtime and transitioned to authenticated IPC.'
    Write-Host 'Installed AskRex first-run smoke passed (clean setup UI, pre-auth setup IPC, setup completion, authenticated transition).'
} catch {
    Write-Diagnostics 'failure' $_.Exception.Message
    throw
} finally {
    if ($null -ne $appProcess -and -not $appProcess.HasExited) {
        Stop-ProcessTree $appProcess
    }
    Remove-Item Env:ASKREX_ARTIFACT_SMOKE -ErrorAction SilentlyContinue
    Remove-Item Env:ASKREX_ARTIFACT_SMOKE_FIRST_RUN -ErrorAction SilentlyContinue
    Remove-Item Env:ASKREX_ARTIFACT_SMOKE_OUTPUT -ErrorAction SilentlyContinue
    Remove-Item Env:ASKREX_ARTIFACT_SMOKE_RUNTIME_ROOT -ErrorAction SilentlyContinue
    Remove-Item Env:ASKREX_RUNTIME_DIR -ErrorAction SilentlyContinue
    if (Test-Path -LiteralPath $testRoot) {
        Remove-Item -LiteralPath $testRoot -Recurse -Force -ErrorAction SilentlyContinue
    }
}
