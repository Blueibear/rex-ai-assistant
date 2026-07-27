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

function Stop-ProcessTree([System.Diagnostics.Process]$Process) {
    if ($null -eq $Process) { return }
    try {
        if (-not $Process.HasExited) {
            $taskkill = Join-Path $env:SystemRoot 'System32\taskkill.exe'
            & $taskkill /PID $Process.Id /T /F 2>$null | Out-Null
            $Process.WaitForExit(10000) | Out-Null
        }
    } catch {
        Write-Verbose "Process tree cleanup skipped for PID $($Process.Id): $($_.Exception.Message)"
    }
}

function Stop-InstalledProcesses([string]$RootPath) {
    $normalizedRoot = [System.IO.Path]::GetFullPath($RootPath).TrimEnd('\') + '\'
    Get-Process -ErrorAction SilentlyContinue | ForEach-Object {
        try {
            $processPath = $_.Path
            if ($processPath -and $processPath.StartsWith(
                $normalizedRoot,
                [System.StringComparison]::OrdinalIgnoreCase
            )) {
                Stop-ProcessTree $_
            }
        } catch {
            # Some system processes do not expose Path. They are unrelated to this install.
        }
    }
}

function Invoke-Installer(
    [string[]]$Arguments,
    [int]$TimeoutSeconds = 180
) {
    $process = Start-Process -FilePath $installerPath -ArgumentList $Arguments -PassThru
    if (-not $process.WaitForExit($TimeoutSeconds * 1000)) {
        Stop-ProcessTree $process
        throw "Installer timed out after $TimeoutSeconds seconds."
    }
    if ($process.ExitCode -ne 0) { throw "Installer exited with code $($process.ExitCode)" }
}

function Invoke-IdentityBridge(
    [string]$PythonExe,
    [string]$BridgeScript,
    [string]$Payload
) {
    # Windows PowerShell 5 can transcode native-command pipeline input and its
    # ProcessStartInfo lacks the modern encoding properties. Use exact UTF-8 file-backed
    # stdin redirection so the packaged bridge receives byte-for-byte valid JSON.
    $callRoot = Join-Path ([System.IO.Path]::GetTempPath()) ('askrex-identity-' + [guid]::NewGuid())
    $stdinPath = Join-Path $callRoot 'stdin.json'
    $stdoutPath = Join-Path $callRoot 'stdout.txt'
    $stderrPath = Join-Path $callRoot 'stderr.txt'
    New-Item -ItemType Directory -Force -Path $callRoot | Out-Null

    try {
        $utf8NoBom = New-Object System.Text.UTF8Encoding($false)
        [System.IO.File]::WriteAllText($stdinPath, $Payload, $utf8NoBom)
        $arguments = "-I `"$BridgeScript`""
        $startProcessArguments = @{
            FilePath = $PythonExe
            ArgumentList = $arguments
            RedirectStandardInput = $stdinPath
            RedirectStandardOutput = $stdoutPath
            RedirectStandardError = $stderrPath
            Wait = $true
            PassThru = $true
            NoNewWindow = $true
        }
        $process = Start-Process @startProcessArguments

        return [pscustomobject]@{
            ExitCode = $process.ExitCode
            Stdout = if (Test-Path -LiteralPath $stdoutPath) {
                Get-Content -LiteralPath $stdoutPath -Raw
            } else {
                ''
            }
            Stderr = if (Test-Path -LiteralPath $stderrPath) {
                Get-Content -LiteralPath $stderrPath -Raw
            } else {
                ''
            }
        }
    } finally {
        if (Test-Path -LiteralPath $callRoot) {
            Remove-Item -LiteralPath $callRoot -Recurse -Force
        }
    }
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
    $identityBridge = Join-Path $resources 'bridge\rex_identity_bridge.py'
    $identityResult = $null
    for ($attempt = 1; $attempt -le 3; $attempt++) {
        $identityResult = Invoke-IdentityBridge $runtimePython $identityBridge $identityPayload
        if ($identityResult.ExitCode -eq 0) { break }
        if ($attempt -lt 3) { Start-Sleep -Seconds 2 }
    }
    if ($null -eq $identityResult -or $identityResult.ExitCode -ne 0) {
        $exitCode = if ($null -eq $identityResult) { -1 } else { $identityResult.ExitCode }
        $diagnostic = if ($null -eq $identityResult) {
            'identity bridge did not start'
        } else {
            $parts = @($identityResult.Stdout.Trim(), $identityResult.Stderr.Trim()) |
                Where-Object { $_ }
            $parts -join ' '
        }
        throw "Read-only identity bridge action failed after 3 attempts (exit $exitCode): $diagnostic"
    }
    $identityJson = $identityResult.Stdout -split '\r?\n' |
        Where-Object { $_.TrimStart().StartsWith('{') } |
        Select-Object -Last 1
    if (-not $identityJson) {
        throw "Read-only identity bridge returned no JSON: $($identityResult.Stdout) $($identityResult.Stderr)"
    }
    $identity = $identityJson | ConvertFrom-Json
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
        Stop-ProcessTree $app
        throw 'Installed app did not produce the IPC smoke result within 90 seconds.'
    }
    $result = Get-Content -LiteralPath $smokeOutput -Raw | ConvertFrom-Json
    if (-not $result.ok -or -not $result.typed_ipc -or
        $result.chat -ne 'AskRex installed artifact chat verified' -or
        $result.memories_count -lt 0) {
        throw "Installed app smoke failed: $(Get-Content -LiteralPath $smokeOutput -Raw)"
    }

    $app.WaitForExit(15000) | Out-Null
    Stop-InstalledProcesses $installPath

    # A second silent install over the same target validates reinstall/upgrade behavior.
    Invoke-Installer @('/S', "/D=$installPath")
    if (-not (Test-Path -LiteralPath $runtimePython -PathType Leaf)) {
        throw 'Managed runtime was lost during reinstall.'
    }

    Stop-InstalledProcesses $installPath
    $uninstallProcess = Start-Process -FilePath $uninstaller -ArgumentList @('/S') -PassThru
    if (-not $uninstallProcess.WaitForExit(120000)) {
        Stop-ProcessTree $uninstallProcess
        throw 'Uninstaller timed out after 120 seconds.'
    }
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
    if (Test-Path -LiteralPath $installPath) {
        Stop-InstalledProcesses $installPath
    }
    if (Test-Path -LiteralPath $testRoot) {
        $resolvedTestRoot = (Resolve-Path -LiteralPath $testRoot).Path
        $tempBase = [System.IO.Path]::GetFullPath([System.IO.Path]::GetTempPath())
        if (-not $resolvedTestRoot.StartsWith($tempBase)) {
            throw "Refusing to clean test path outside the temp directory: $resolvedTestRoot"
        }
        Remove-Item -LiteralPath $resolvedTestRoot -Recurse -Force
    }
}
