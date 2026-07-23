[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$Installer,
    [string]$BuildPython = 'python'
)

$ErrorActionPreference = 'Stop'
$installerPath = (Resolve-Path -LiteralPath $Installer).Path
$testRoot = Join-Path ([System.IO.Path]::GetTempPath()) ('askrex-installed-smoke-' + [guid]::NewGuid())
$installPath = Join-Path $testRoot 'AskRex'
$localAppData = Join-Path $testRoot 'LocalAppData'
$smokeOutput = Join-Path $testRoot 'smoke.json'

function Invoke-Installer([string[]]$Arguments) {
    $process = Start-Process -FilePath $installerPath -ArgumentList $Arguments -Wait -PassThru
    if ($process.ExitCode -ne 0) { throw "Installer exited with code $($process.ExitCode)" }
}

try {
    New-Item -ItemType Directory -Force -Path $testRoot, $localAppData | Out-Null
    Invoke-Installer @('/S', "/D=$installPath")

    $appExe = Join-Path $installPath 'AskRex.exe'
    $uninstaller = Join-Path $installPath 'Uninstall AskRex.exe'
    $resources = Join-Path $installPath 'resources'
    $runtimePython = Join-Path $resources 'python\python.exe'
    foreach ($required in @($appExe, $uninstaller, $runtimePython)) {
        if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
            throw "Installed artifact is missing: $required"
        }
    }

    & $BuildPython (Join-Path $PSScriptRoot 'verify_electron_package_contents.py') $resources
    if ($LASTEXITCODE -ne 0) { throw 'Installed resource verification failed.' }

    $env:LOCALAPPDATA = $localAppData
    & $runtimePython -I -c "from rex.identity import set_session_user; set_session_user('artifact-ci-user')"
    if ($LASTEXITCODE -ne 0) { throw 'Managed runtime could not establish the smoke identity.' }

    $identityPayload = '{"action":"resolve_electron_session"}'
    $identityResult = $identityPayload | & $runtimePython -I (Join-Path $resources 'bridge\rex_identity_bridge.py')
    if ($LASTEXITCODE -ne 0) { throw 'Read-only identity bridge action failed.' }
    $identity = $identityResult | ConvertFrom-Json
    if (-not $identity.ok -or $identity.user_id -ne 'artifact-ci-user') {
        throw 'Read-only identity bridge returned an unexpected result.'
    }

    $env:ASKREX_ARTIFACT_SMOKE = '1'
    $env:ASKREX_ARTIFACT_SMOKE_OUTPUT = $smokeOutput
    $env:PATH = Join-Path $env:SystemRoot 'System32'
    $app = Start-Process -FilePath $appExe -PassThru
    $deadline = [DateTime]::UtcNow.AddSeconds(90)
    while (-not (Test-Path -LiteralPath $smokeOutput) -and [DateTime]::UtcNow -lt $deadline) {
        Start-Sleep -Milliseconds 500
    }
    if (-not (Test-Path -LiteralPath $smokeOutput)) {
        if (-not $app.HasExited) { $app.Kill() }
        throw 'Installed app did not produce the IPC smoke result within 90 seconds.'
    }
    $result = Get-Content -LiteralPath $smokeOutput -Raw | ConvertFrom-Json
    if (-not $result.ok -or -not $result.typed_ipc -or
        $result.chat -ne 'AskRex installed artifact chat verified' -or
        $result.memories_count -lt 0) {
        throw "Installed app smoke failed: $(Get-Content -LiteralPath $smokeOutput -Raw)"
    }
    $app.WaitForExit(15000) | Out-Null

    # A second silent install over the same target validates reinstall/upgrade behavior.
    Invoke-Installer @('/S', "/D=$installPath")
    if (-not (Test-Path -LiteralPath $runtimePython -PathType Leaf)) {
        throw 'Managed runtime was lost during reinstall.'
    }

    $uninstallProcess = Start-Process -FilePath $uninstaller -ArgumentList @('/S') -Wait -PassThru
    if ($uninstallProcess.ExitCode -ne 0) {
        throw "Uninstaller exited with code $($uninstallProcess.ExitCode)"
    }
    $deadline = [DateTime]::UtcNow.AddSeconds(30)
    while ((Test-Path -LiteralPath $appExe) -and [DateTime]::UtcNow -lt $deadline) {
        Start-Sleep -Milliseconds 250
    }
    if (Test-Path -LiteralPath $appExe) { throw 'Uninstall left the application executable behind.' }
    Write-Host 'Installed AskRex artifact smoke passed (IPC, managed bridge, chat, reinstall, uninstall).'
} finally {
    Remove-Item Env:ASKREX_ARTIFACT_SMOKE -ErrorAction SilentlyContinue
    Remove-Item Env:ASKREX_ARTIFACT_SMOKE_OUTPUT -ErrorAction SilentlyContinue
    if (Test-Path -LiteralPath $testRoot) {
        $resolvedTestRoot = (Resolve-Path -LiteralPath $testRoot).Path
        $tempBase = [System.IO.Path]::GetFullPath([System.IO.Path]::GetTempPath())
        if (-not $resolvedTestRoot.StartsWith($tempBase)) {
            throw "Refusing to clean test path outside the temp directory: $resolvedTestRoot"
        }
        Remove-Item -LiteralPath $resolvedTestRoot -Recurse -Force
    }
}
