#Requires -Version 5.1
<#
.SYNOPSIS
    Wrapper launched by Windows Task Scheduler that runs the Fortuna
    nightly training orchestrator.

.DESCRIPTION
    - Sets the working dir to the repo root.
    - Pipes all stdout/stderr to logs/nightly/<DATE>.log.
    - Calls ``uv run python scripts/nightly_train.py --sleep-after``.
    - Returns the orchestrator's exit code so Task Scheduler can record
      success/failure on the History tab.

.EXAMPLE
    powershell -NoProfile -ExecutionPolicy Bypass -File scripts/nightly_train.ps1
#>

$ErrorActionPreference = "Continue"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

$logDir = Join-Path $repo "logs\nightly"
if (-not (Test-Path $logDir)) {
    New-Item -ItemType Directory -Path $logDir | Out-Null
}

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$logFile = Join-Path $logDir "$timestamp.log"

"=== Fortuna nightly orchestrator started at $(Get-Date -Format 's') ===" | Out-File $logFile
"Repo:    $repo"                 | Add-Content $logFile
"Host:    $env:COMPUTERNAME"     | Add-Content $logFile
"User:    $env:USERNAME"         | Add-Content $logFile
"Args:    $($args -join ' ')"    | Add-Content $logFile
""                               | Add-Content $logFile

# Find uv on PATH; fall back to %USERPROFILE%\.local\bin\uv.exe (common install path).
$uv = (Get-Command uv -ErrorAction SilentlyContinue).Source
if (-not $uv) {
    $candidate = Join-Path $env:USERPROFILE ".local\bin\uv.exe"
    if (Test-Path $candidate) { $uv = $candidate }
}
if (-not $uv) {
    "ERROR: 'uv' was not found on PATH or in the default install location." | Add-Content $logFile
    exit 127
}

"Using uv: $uv" | Add-Content $logFile

# Force unbuffered Python stdout/stderr so phases stream into the log
# in real time. Combined with the script's own line-buffered fallback,
# this makes future hangs identifiable from the log alone.
$env:PYTHONUNBUFFERED = "1"

# Default args; the caller can override / extend via the script's $args.
# ``-u`` is python's belt-and-braces unbuffered switch.
$pyArgs = @(
    "run", "python", "-u", "scripts/nightly_train.py",
    "--data-source", "smartapi",
    "--policies", "CnnPolicy",
    "--device", "cuda",
    "--sleep-after"
) + $args

& $uv @pyArgs *>> $logFile
$rc = $LASTEXITCODE

"=== Orchestrator exited with code $rc at $(Get-Date -Format 's') ===" | Add-Content $logFile
exit $rc
